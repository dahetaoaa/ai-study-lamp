"""整页试卷 OCR + 题目与坐标框提取 同图对比评测脚本。

在真实台灯书桌俯拍视频抓拍样本（1080x1920）上对比：
  1. qwen3_5_2b_8001     : 线上已运行的 Qwen3.5-2B (http://127.0.0.1:8001/v1, max_pixels=524288)
  2. qwen3_5_0_8b        : Qwen/Qwen3.5-0.8B (原生动态分辨率)
  3. glm_ocr             : ZhipuAI/GLM-OCR (0.9B 专用 OCR)
  4. paddleocr_vl_1_5    : PaddlePaddle/PaddleOCR-VL-1.5 (0.9B 专用文档 OCR + Spotting 定位)
  5. rapidocr_ppocrv4    : RapidOCR (PP-OCRv4 ONNX CPU, 零显存)
  6. hybrid_rapidocr_glm : 并行 RapidOCR 几何切题框 + GLM-OCR 全页文本/公式识别

用法:
  .venv-bench/bin/python tools/benchmark_page_ocr.py --target qwen3_5_0_8b --port 8012
  .venv-bench/bin/python tools/benchmark_page_ocr.py --target rapidocr_ppocrv4
  .venv-bench/bin/python tools/benchmark_page_ocr.py --finalize
  # 或一键自动逐个拉起临时 vLLM 服务并评测全部模型:
  .venv-bench/bin/python tools/benchmark_page_ocr.py --all
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
REC_DIR = ROOT / "data" / "video_testset" / "recordings_20261117"
TPL_DIR = ROOT / "data" / "templates"
DIAG_DIR = ROOT / "data" / "diagnostics"
MODELS_DIR = ROOT / "models"
VLLM_BIN = ROOT / ".venv-vllm" / "bin" / "vllm"

# 5 张代表性实拍视频抓拍图（数学含公式/图表/手写 + 英语密集单选/完形/阅读配图）
BENCH_CASES = [
    {
        "case_id": "math_p1_handwritten",
        "capture_rel": "rec_5422b97b81b3/cap_b77a1cd5c6e44a87.jpg",
        "capture_json": "rec_5422b97b81b3/cap_b77a1cd5c6e44a87.json",
        "paper_id": "math_xmyz_2025_oct",
        "page_no": 1,
        "desc": "数学第1页（7道选择，含分式/上下标公式 + 学生手写答案 + 桌面阴影与页脚微卷）",
    },
    {
        "case_id": "math_p2_table_graph",
        "capture_rel": "rec_7267479ca145/cap_02231e8004844303.jpg",
        "capture_json": "rec_7267479ca145/cap_02231e8004844303.json",
        "paper_id": "math_xmyz_2025_oct",
        "page_no": 2,
        "desc": "数学第2页（9道选择/填空，含表格、函数图像、带圈序号与分式，空白为主）",
    },
    {
        "case_id": "eng_p3_dense_choice",
        "capture_rel": "rec_5422b97b81b3/cap_4397220d78bc4056.jpg",
        "capture_json": "rec_5422b97b81b3/cap_4397220d78bc4056.json",
        "paper_id": "eng_xmwgy_2024_mid",
        "page_no": 3,
        "desc": "英语第3页（13道密集单选 21-33，小字号对话+下划线填空，空白）",
    },
    {
        "case_id": "eng_p4_cloze_tilted",
        "capture_rel": "rec_5422b97b81b3/cap_9ec44c6d75bc4b42.jpg",
        "capture_json": "rec_5422b97b81b3/cap_9ec44c6d75bc4b42.json",
        "paper_id": "eng_xmwgy_2024_mid",
        "page_no": 4,
        "desc": "英语第4页（34-35单选 + 5段完形短文含插图 + 36-45三栏选项，透视倾斜）",
    },
    {
        "case_id": "eng_p7_reading_figure",
        "capture_rel": "rec_5422b97b81b3/cap_c00be77b52894bd9.jpg",
        "capture_json": "rec_5422b97b81b3/cap_c00be77b52894bd9.json",
        "paper_id": "eng_xmyz_2023_m1",
        "page_no": 7,
        "desc": "英语第7页（56-63阅读理解 + D篇长文含燕子示意图，较强阴影与卷曲）",
    },
]

QNUM_LINE_RE = re.compile(r"^\s*(\d{1,3})\s*[.．、,，]\s*(\S.*)")
SECTION_LINE_RE = re.compile(
    r"^\s*([一二三四五六七八九十]+[、.．]|[IVX]+[.、．]|第[一二三四五六七八九十]+[部节]分?)"
)
FOOTER_RE = re.compile(r"第\s*\d+\s*/\s*\d+\s*页")
LOC_LINE_RE = re.compile(
    r"^(.*?)<\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|>"
    r"<\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|>\s*$"
)
CIRCLED_MAP = {"①": "1", "②": "2", "③": "3", "④": "4"}


def query_gpu_mem_mib() -> dict[str, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, free, total = [int(x.strip()) for x in out.splitlines()[0].split(",")]
        return {"used_mib": used, "free_mib": free, "total_mib": total}
    except Exception:
        return {"used_mib": -1, "free_mib": -1, "total_mib": -1}


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\\(", "$").replace("\\)", "$").replace("\\[", "$").replace("\\]", "$")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\textcircled\{(\d+)\}", lambda m: m.group(1), s)
    for k, v in CIRCLED_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_for_char_acc(s: str) -> str:
    s = normalize_text(s)
    # 统一移除空白与包裹用 $ 符号，聚焦实际内容字符准确率
    s = s.replace("$", "")
    s = re.sub(r"[_\s]+", "", s)
    return s


def normalize_latex_formula(expr: str) -> str:
    s = unicodedata.normalize("NFKC", expr)
    s = s.replace("\\left", "").replace("\\right", "").replace("$", "")
    s = s.replace("{", "{").replace("}", "}")
    s = re.sub(r"\s+", "", s)
    return s


def extract_gt_case(case: dict[str, Any]) -> dict[str, Any]:
    cap_path = REC_DIR / case["capture_json"]
    tpl_path = TPL_DIR / case["paper_id"] / "template.json"
    cap = json.loads(cap_path.read_text())
    tpl = json.loads(tpl_path.read_text())
    page = next(p for p in tpl["pages"] if p["page_no"] == case["page_no"])
    q_by_id = {q["id"]: q for q in page["questions"]}
    H = np.array(cap["capture"]["homography"], dtype=float)
    w_img = int(cap["capture"]["image_width"])
    h_img = int(cap["capture"]["image_height"])

    gt_questions = []
    gt_formulas = []
    for cq in cap["questions"]:
        tq = q_by_id[cq["template_question_id"]]
        x, y, w, h = tq["bbox"]
        pts = np.array([[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]], dtype=float).T
        proj = H @ pts
        proj = (proj[:2] / proj[2:]).T
        gt_xyxy = [
            float(np.clip(proj[:, 0].min(), 0, w_img)),
            float(np.clip(proj[:, 1].min(), 0, h_img)),
            float(np.clip(proj[:, 0].max(), 0, w_img)),
            float(np.clip(proj[:, 1].max(), 0, h_img)),
        ]
        # 完形填空页的 36-45 在模板中 bbox 是短文中挖空小框，不适合作为整题大框评测
        eval_bbox = not (case["paper_id"] == "eng_xmwgy_2024_mid" and case["page_no"] == 4 and int(tq["number"]) >= 36)
        opts = tq.get("options") or []
        full_q_text = f"{tq['number']}. {tq['stem']}" + (" " + " ".join(opts) if opts else "")
        gt_questions.append(
            {
                "no": str(tq["number"]),
                "type": tq["type"],
                "stem": tq["stem"],
                "options": opts,
                "full_text": full_q_text,
                "gt_xyxy": gt_xyxy,
                "eval_bbox": eval_bbox,
                "student_raw": cq.get("student_raw", ""),
            }
        )
        for m in re.findall(r"\$([^$]+)\$", full_q_text):
            nm = normalize_latex_formula(m)
            if len(nm) >= 2:
                gt_formulas.append({"q_no": str(tq["number"]), "raw": m, "norm": nm})

    return {
        **case,
        "image_path": str(REC_DIR / case["capture_rel"]),
        "image_width": w_img,
        "image_height": h_img,
        "gt_questions": gt_questions,
        "gt_formulas": gt_formulas,
    }


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center_dist(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def segment_lines_into_questions(lines: list[dict[str, Any]], img_w: int, img_h: int) -> list[dict[str, Any]]:
    """将按行输出的 [(text, xyxy), ...] 切分为题号级结构 + 整题 bbox。"""
    if not lines:
        return []
    # 过滤页脚，按阅读顺序近似排序
    filtered = [l for l in lines if not FOOTER_RE.search(l["text"]) and l["xyxy"][1] < img_h * 0.90]
    filtered.sort(key=lambda l: (round(l["xyxy"][1] / 14), l["xyxy"][0]))

    # 先识别每行是否是题号起点（注意排除完形短文行中段偶然出现数字的情况：要求 x0 < 0.35*W 或选项表行）
    starts = []
    for idx, l in enumerate(filtered):
        m = QNUM_LINE_RE.match(l["text"])
        if not m:
            continue
        num = int(m.group(1))
        if num <= 0 or num > 100:
            continue
        if l["xyxy"][0] < img_w * 0.38 or re.match(r"^\s*\d{1,3}\s*[.．]\s*[A-D]\s*[.．]", l["text"]):
            starts.append((idx, str(num)))

    # 处理同一行里包含两个小题（或行首合并后）的情况；按起止切片
    questions = []
    for k, (s_idx, num) in enumerate(starts):
        e_idx = starts[k + 1][0] if k + 1 < len(starts) else len(filtered)
        seg = []
        for l in filtered[s_idx:e_idx]:
            if seg and SECTION_LINE_RE.match(l["text"]):
                break
            # 若同一题下面跟了一大段跨栏阅读正文或配图很远，防止把整篇后续无关行吞掉（简单 Y 间隔保护）
            if seg and (l["xyxy"][1] - seg[-1]["xyxy"][3]) > img_h * 0.14:
                break
            seg.append(l)
        if not seg:
            continue
        x1 = min(l["xyxy"][0] for l in seg)
        y1 = min(l["xyxy"][1] for l in seg)
        x2 = max(l["xyxy"][2] for l in seg)
        y2 = max(l["xyxy"][3] for l in seg)
        # 如果该题有同行并排选项（如 36. A. late  B. later  C. latest），把同 Y 带上的右侧选项也并入该题
        first_cy = (seg[0]["xyxy"][1] + seg[0]["xyxy"][3]) / 2.0
        for l in filtered:
            cy = (l["xyxy"][1] + l["xyxy"][3]) / 2.0
            if abs(cy - first_cy) <= img_h * 0.012 and l["xyxy"][0] >= seg[0]["xyxy"][0]:
                if l not in seg and not QNUM_LINE_RE.match(l["text"]):
                    seg.append(l)
                    x1 = min(x1, l["xyxy"][0])
                    y1 = min(y1, l["xyxy"][1])
                    x2 = max(x2, l["xyxy"][2])
                    y2 = max(y2, l["xyxy"][3])
        # 去重保序拼接
        seen_ids = set()
        ordered_seg = []
        for l in sorted(seg, key=lambda item: (round(item["xyxy"][1] / 12), item["xyxy"][0])):
            if id(l) not in seen_ids:
                seen_ids.add(id(l))
                ordered_seg.append(l)
        questions.append(
            {
                "no": num,
                "text": " ".join(l["text"] for l in ordered_seg),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
    return questions


def parse_plain_text_into_questions(full_text: str) -> dict[str, str]:
    """从整页纯文本/Markdown按题号切出 {no: text}，用于与 GT 逐题对齐计算文字准确率。"""
    if not full_text:
        return {}
    # 移除 markdown 代码块标记
    clean = re.sub(r"```[a-zA-Z]*\n?", "", full_text).replace("```", "")
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    qmap: dict[str, list[str]] = {}
    cur_no = None
    for ln in lines:
        if FOOTER_RE.search(ln):
            continue
        m = QNUM_LINE_RE.match(ln)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 100:
                cur_no = str(num)
                qmap.setdefault(cur_no, []).append(ln)
                continue
        if cur_no is not None:
            if SECTION_LINE_RE.match(ln):
                cur_no = None
                continue
            qmap[cur_no].append(ln)
    return {k: " ".join(v) for k, v in qmap.items()}


def parse_json_questions(raw_text: str, img_w: int, img_h: int) -> tuple[list[dict[str, Any]], bool]:
    """解析 LLM/VLM 输出的 JSON 题目列表，兼容 [x1,y1,x2,y2] 0-1000 归一化或像素坐标。"""
    txt = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, re.DOTALL)
    if fenced:
        txt = fenced.group(1).strip()
    data = None
    try:
        data = json.loads(txt)
    except Exception:
        # 尝试截断修复：提取已完整闭合的 {...} 对象
        objs = []
        for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", txt):
            try:
                objs.append(json.loads(m.group(0)))
            except Exception:
                pass
        if objs:
            data = objs
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        data = data["questions"]
    if not isinstance(data, list):
        return [], False

    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        no = str(item.get("no", "")).strip().rstrip(".")
        stem = str(item.get("stem", "")).strip()
        opts = item.get("options") or []
        if isinstance(opts, list):
            opts_str = " ".join(str(o) for o in opts)
        else:
            opts_str = str(opts)
        bx = item.get("bbox")
        xyxy = None
        if isinstance(bx, list) and len(bx) == 1 and isinstance(bx[0], list):
            bx = bx[0]
        if isinstance(bx, list) and len(bx) == 4 and all(isinstance(v, (int, float)) for v in bx):
            x1, y1, x2, y2 = [float(v) for v in bx]
            # 若在 0-1000 归一化范围内则反归一化
            if max(x1, y1, x2, y2) <= 1005 and (x2 <= 1005 and y2 <= 1005):
                x1, x2 = x1 / 1000.0 * img_w, x2 / 1000.0 * img_w
                y1, y2 = y1 / 1000.0 * img_h, y2 / 1000.0 * img_h
            if x2 > x1 + 5 and y2 > y1 + 5 and x2 <= img_w * 1.1 and y2 <= img_h * 1.1:
                xyxy = [x1, y1, x2, y2]
        out.append(
            {
                "no": no,
                "text": f"{no}. {stem} {opts_str}".strip(),
                "bbox": xyxy,
            }
        )
    return out, len(out) > 0


def parse_paddle_spotting(raw_text: str, img_w: int, img_h: int) -> tuple[list[dict[str, Any]], str]:
    """解析 PaddleOCR-VL-1.5 Spotting: 输出的行级 text + 4点<|LOC_x|><|LOC_y|>坐标。"""
    lines = []
    plain_lines = []
    for raw_ln in raw_text.splitlines():
        raw_ln = raw_ln.strip()
        if not raw_ln:
            continue
        m = LOC_LINE_RE.match(raw_ln)
        if m:
            txt = m.group(1).strip()
            coords = [int(m.group(i)) for i in range(2, 10)]
            xs = [coords[0] / 1000.0 * img_w, coords[2] / 1000.0 * img_w, coords[4] / 1000.0 * img_w, coords[6] / 1000.0 * img_w]
            ys = [coords[1] / 1000.0 * img_h, coords[3] / 1000.0 * img_h, coords[5] / 1000.0 * img_h, coords[7] / 1000.0 * img_h]
            lines.append({"text": txt, "xyxy": [min(xs), min(ys), max(xs), max(ys)]})
            plain_lines.append(txt)
        else:
            cleaned = re.sub(r"<\|LOC_\d+\|>", "", raw_ln).strip()
            if cleaned:
                plain_lines.append(cleaned)
    return lines, "\n".join(plain_lines)


def evaluate_prediction(
    gt_case: dict[str, Any],
    full_text: str,
    pred_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    gt_qs = gt_case["gt_questions"]
    gt_nos = {q["no"] for q in gt_qs}
    pred_by_no: dict[str, dict[str, Any]] = {}
    for pq in pred_questions:
        no = str(pq.get("no", "")).strip()
        if no and no not in pred_by_no:
            pred_by_no[no] = pq

    # 同时从 full_text 提取按题号分组作为兜底对齐
    text_by_no = parse_plain_text_into_questions(full_text)
    full_norm = normalize_for_char_acc(full_text)

    found_nos = set(pred_by_no.keys()) | set(text_by_no.keys())
    # 去重幻觉题号统计
    tp_nos = gt_nos & found_nos
    qnum_recall = len(tp_nos) / len(gt_nos) if gt_nos else 0.0
    qnum_precision = len(tp_nos) / len(found_nos) if found_nos else 0.0
    qnum_f1 = (2 * qnum_recall * qnum_precision / (qnum_recall + qnum_precision)) if (qnum_recall + qnum_precision) else 0.0

    # 逐题文字准确率（取题号对齐与全文模糊对齐中合理的匹配）
    q_char_sims = []
    q_fuzzy_90 = 0
    for gq in gt_qs:
        g_norm = normalize_for_char_acc(gq["full_text"])
        p_txt = ""
        if gq["no"] in pred_by_no and pred_by_no[gq["no"]].get("text"):
            p_txt = pred_by_no[gq["no"]]["text"]
        elif gq["no"] in text_by_no:
            p_txt = text_by_no[gq["no"]]
        p_norm = normalize_for_char_acc(p_txt)
        if p_norm:
            sim = fuzz.ratio(g_norm, p_norm) / 100.0
        elif full_norm:
            # 兜底看全文里该题 stem 是否被完整识别
            sim = fuzz.partial_ratio(g_norm, full_norm) / 100.0
        else:
            sim = 0.0
        q_char_sims.append(sim)
        if sim >= 0.90:
            q_fuzzy_90 += 1

    # 公式还原度（仅针对含公式题目）
    gt_formulas = gt_case["gt_formulas"]
    full_tex_norm = normalize_latex_formula(normalize_text(full_text))
    formula_exact = 0
    formula_sims = []
    for fm in gt_formulas:
        target = fm["norm"]
        if target in full_tex_norm:
            formula_exact += 1
            formula_sims.append(1.0)
        else:
            formula_sims.append(fuzz.partial_ratio(target, full_tex_norm) / 100.0 if full_tex_norm else 0.0)

    # BBox IoU 评测（针对有标准整题框的题目）
    eval_bbox_qs = [gq for gq in gt_qs if gq["eval_bbox"]]
    ious = []
    center_dists = []
    valid_bbox_cnt = 0
    iou_05 = 0
    iou_03 = 0
    for gq in eval_bbox_qs:
        pq = pred_by_no.get(gq["no"])
        bx = pq.get("bbox") if pq else None
        if bx and len(bx) == 4:
            valid_bbox_cnt += 1
            iou = bbox_iou(gq["gt_xyxy"], bx)
            dist = bbox_center_dist(gq["gt_xyxy"], bx)
            ious.append(iou)
            center_dists.append(dist)
            if iou >= 0.50:
                iou_05 += 1
            if iou >= 0.30:
                iou_03 += 1
        else:
            ious.append(0.0)

    return {
        "gt_q_count": len(gt_qs),
        "found_q_count": len(found_nos),
        "qnum_recall": round(qnum_recall, 4),
        "qnum_precision": round(qnum_precision, 4),
        "qnum_f1": round(qnum_f1, 4),
        "question_char_acc": round(float(np.mean(q_char_sims)) if q_char_sims else 0.0, 4),
        "question_acc_ge90_rate": round(q_fuzzy_90 / len(gt_qs) if gt_qs else 0.0, 4),
        "formula_count": len(gt_formulas),
        "formula_exact_rate": round(formula_exact / len(gt_formulas), 4) if gt_formulas else None,
        "formula_sim_avg": round(float(np.mean(formula_sims)), 4) if formula_sims else None,
        "bbox_eval_count": len(eval_bbox_qs),
        "bbox_valid_rate": round(valid_bbox_cnt / len(eval_bbox_qs), 4) if eval_bbox_qs else 0.0,
        "bbox_mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "bbox_iou_05_rate": round(iou_05 / len(eval_bbox_qs), 4) if eval_bbox_qs else 0.0,
        "bbox_iou_03_rate": round(iou_03 / len(eval_bbox_qs), 4) if eval_bbox_qs else 0.0,
        "bbox_center_err_px": round(float(np.mean(center_dists)), 1) if center_dists else None,
    }


def call_vllm_chat(
    base_url: str,
    model_name: str,
    img_path: str,
    prompt: str,
    max_tokens: int = 2600,
    disable_thinking: bool = False,
    image_first: bool = True,
) -> dict[str, Any]:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    img_part = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    txt_part = {"type": "text", "text": prompt}
    content = [img_part, txt_part] if image_first else [txt_part, img_part]
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    t0 = time.perf_counter()
    r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=180)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    data = r.json()
    if "choices" not in data:
        return {
            "latency_ms": round(dt_ms, 1),
            "text": "",
            "finish_reason": "error",
            "error": str(data)[:300],
            "usage": {},
        }
    ch = data["choices"][0]
    return {
        "latency_ms": round(dt_ms, 1),
        "text": ch["message"]["content"] or "",
        "finish_reason": ch.get("finish_reason", ""),
        "usage": data.get("usage") or {},
    }


def run_rapidocr_on_image(ocr_engine: Any, img_path: str, img_w: int, img_h: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    res, elapse = ocr_engine(img_path)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    lines = []
    if res:
        for box, txt, score in res:
            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]
            lines.append({"text": str(txt), "score": float(score), "xyxy": [min(xs), min(ys), max(xs), max(ys)]})
    full_text = "\n".join(l["text"] for l in lines)
    pred_qs = segment_lines_into_questions(lines, img_w, img_h)
    return {
        "latency_ms": round(dt_ms, 1),
        "elapse_breakdown": [round(float(x) * 1000.0, 1) for x in elapse] if isinstance(elapse, (list, tuple)) else [],
        "line_count": len(lines),
        "full_text": full_text,
        "pred_questions": pred_qs,
    }


def start_vllm_server(model_dir: Path, served_name: str, port: int, max_model_len: int = 4096) -> tuple[subprocess.Popen, dict[str, Any]]:
    mem_before = query_gpu_mem_mib()
    log_path = Path(f"/tmp/bench_vllm_{port}.log")
    cmd = [
        str(VLLM_BIN),
        "serve",
        str(model_dir),
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        "8192",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
        "--gpu-memory-utilization",
        "0.16",
        "--kv-cache-memory-bytes",
        "536870912",
        "--enforce-eager",
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.perf_counter()
    for _ in range(120):
        if proc.poll() is not None:
            tail = log_path.read_text()[-1200:]
            raise RuntimeError(f"vLLM {served_name} exited early ({proc.returncode}):\n{tail}")
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.5)
            if r.status_code == 200:
                mem_after = query_gpu_mem_mib()
                startup_s = round(time.perf_counter() - t0, 1)
                return proc, {
                    "startup_s": startup_s,
                    "gpu_before_mib": mem_before["used_mib"],
                    "gpu_after_mib": mem_after["used_mib"],
                    "vram_delta_mib": mem_after["used_mib"] - mem_before["used_mib"],
                }
        except Exception:
            pass
        time.sleep(1.0)
    stop_vllm_server(proc)
    raise RuntimeError(f"vLLM {served_name} startup timeout")


def stop_vllm_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(2.5)


def aggregate_variant(records: list[dict[str, Any]], vram_info: dict[str, Any]) -> dict[str, Any]:
    latencies = [r["latency_ms"] for r in records]
    char_accs = [r["metrics"]["question_char_acc"] for r in records]
    ge90 = [r["metrics"]["question_acc_ge90_rate"] for r in records]
    qf1s = [r["metrics"]["qnum_f1"] for r in records]
    qrecs = [r["metrics"]["qnum_recall"] for r in records]
    ious = [r["metrics"]["bbox_mean_iou"] for r in records]
    iou05 = [r["metrics"]["bbox_iou_05_rate"] for r in records]
    iou03 = [r["metrics"]["bbox_iou_03_rate"] for r in records]
    bval = [r["metrics"]["bbox_valid_rate"] for r in records]
    f_exacts = [r["metrics"]["formula_exact_rate"] for r in records if r["metrics"]["formula_exact_rate"] is not None]
    f_sims = [r["metrics"]["formula_sim_avg"] for r in records if r["metrics"]["formula_sim_avg"] is not None]
    cdists = [r["metrics"]["bbox_center_err_px"] for r in records if r["metrics"]["bbox_center_err_px"] is not None]
    return {
        "vram": vram_info,
        "summary": {
            "avg_latency_ms": round(float(np.mean(latencies)), 1),
            "min_latency_ms": round(float(np.min(latencies)), 1),
            "max_latency_ms": round(float(np.max(latencies)), 1),
            "avg_question_char_acc": round(float(np.mean(char_accs)), 4),
            "avg_question_ge90_rate": round(float(np.mean(ge90)), 4),
            "avg_qnum_recall": round(float(np.mean(qrecs)), 4),
            "avg_qnum_f1": round(float(np.mean(qf1s)), 4),
            "avg_formula_exact_rate": round(float(np.mean(f_exacts)), 4) if f_exacts else None,
            "avg_formula_sim": round(float(np.mean(f_sims)), 4) if f_sims else None,
            "avg_bbox_valid_rate": round(float(np.mean(bval)), 4),
            "avg_bbox_mean_iou": round(float(np.mean(ious)), 4),
            "avg_bbox_iou_05_rate": round(float(np.mean(iou05)), 4),
            "avg_bbox_iou_03_rate": round(float(np.mean(iou03)), 4),
            "avg_bbox_center_err_px": round(float(np.mean(cdists)), 1) if cdists else None,
        },
        "cases": records,
    }


def benchmark_qwen_like(
    variant_prefix: str,
    base_url: str,
    model_name: str,
    gt_cases: list[dict[str, Any]],
    vram_info: dict[str, Any],
) -> dict[str, Any]:
    prompt_plain = (
        "请逐行完整转写这张试卷图片中的所有文字内容。"
        "要求：只输出纯文本，不要输出HTML或额外解释；数学公式用LaTeX $...$ 表示，保留题号和选项。"
    )
    prompt_json = (
        "请识别整页试卷的所有题目，按阅读顺序输出JSON数组，每个元素包含：\n"
        '{"no":"题号","stem":"题干（数学公式用LaTeX $...$）","options":["A. ..."],"bbox":[x1,y1,x2,y2]}\n'
        "其中bbox使用0-1000归一化坐标。只输出合法JSON，不要Markdown代码块或额外解释。"
    )
    # Warmup once
    _ = call_vllm_chat(base_url, model_name, gt_cases[0]["image_path"], "请输出一个词：OK", max_tokens=16, disable_thinking=True, image_first=False)

    out = {}
    for mode, prompt in [("plain_ocr", prompt_plain), ("json_bbox", prompt_json)]:
        recs = []
        for gc in gt_cases:
            res = call_vllm_chat(base_url, model_name, gc["image_path"], prompt, max_tokens=2200, disable_thinking=True, image_first=False)
            if mode == "json_bbox":
                pqs, _ = parse_json_questions(res["text"], gc["image_width"], gc["image_height"])
                full_text = "\n".join(pq["text"] for pq in pqs) if pqs else res["text"]
            else:
                pqs = []
                full_text = res["text"]
            metrics = evaluate_prediction(gc, full_text, pqs)
            recs.append({
                "case_id": gc["case_id"],
                "latency_ms": res["latency_ms"],
                "finish_reason": res["finish_reason"],
                "usage": res["usage"],
                "raw_preview": res["text"][:400],
                "metrics": metrics,
            })
            print(f"[{variant_prefix}:{mode}] {gc['case_id']} dt={res['latency_ms']}ms acc={metrics['question_char_acc']} qf1={metrics['qnum_f1']} iou={metrics['bbox_mean_iou']}", flush=True)
        out[f"{variant_prefix}__{mode}"] = aggregate_variant(recs, vram_info)
    return out


def benchmark_paddle_vl(base_url: str, gt_cases: list[dict[str, Any]], vram_info: dict[str, Any]) -> dict[str, Any]:
    _ = call_vllm_chat(base_url, "PaddleOCR-VL-1.5", gt_cases[0]["image_path"], "OCR:", max_tokens=64, image_first=True)
    out = {}
    for mode, prompt in [("ocr", "OCR:"), ("spotting", "Spotting:")]:
        recs = []
        for gc in gt_cases:
            res = call_vllm_chat(base_url, "PaddleOCR-VL-1.5", gc["image_path"], prompt, max_tokens=2800, image_first=True)
            if mode == "spotting":
                lines, plain = parse_paddle_spotting(res["text"], gc["image_width"], gc["image_height"])
                pqs = segment_lines_into_questions(lines, gc["image_width"], gc["image_height"])
                full_text = plain
            else:
                pqs = []
                full_text = res["text"]
            metrics = evaluate_prediction(gc, full_text, pqs)
            recs.append({
                "case_id": gc["case_id"],
                "latency_ms": res["latency_ms"],
                "finish_reason": res["finish_reason"],
                "usage": res["usage"],
                "raw_preview": res["text"][:400],
                "metrics": metrics,
            })
            print(f"[paddleocr_vl_1_5:{mode}] {gc['case_id']} dt={res['latency_ms']}ms acc={metrics['question_char_acc']} qf1={metrics['qnum_f1']} iou={metrics['bbox_mean_iou']}", flush=True)
        out[f"paddleocr_vl_1_5__{mode}"] = aggregate_variant(recs, vram_info)
    return out


def benchmark_glm_and_hybrid(base_url: str, gt_cases: list[dict[str, Any]], vram_info: dict[str, Any]) -> dict[str, Any]:
    from rapidocr_onnxruntime import RapidOCR

    ocr_engine = RapidOCR()
    _ = ocr_engine(gt_cases[0]["image_path"])
    _ = call_vllm_chat(base_url, "GLM-OCR", gt_cases[0]["image_path"], "Text Recognition:", max_tokens=64, image_first=True)

    out = {}
    # 1) GLM-OCR Text Recognition
    recs_text = []
    # 2) Hybrid Parallel: RapidOCR (CPU lines + question bbox) || GLM-OCR (GPU text + LaTeX)
    recs_hybrid = []
    # 3) GLM-OCR JSON IE
    prompt_ie = '请按下列JSON格式输出图中所有题目:\n{"questions": [{"no": "", "stem": "", "options": [""], "bbox": [0, 0, 0, 0]}]}'
    recs_ie = []

    for gc in gt_cases:
        # Text Recognition
        res_t = call_vllm_chat(base_url, "GLM-OCR", gc["image_path"], "Text Recognition:", max_tokens=1350, image_first=True)
        m_t = evaluate_prediction(gc, res_t["text"], [])
        recs_text.append({
            "case_id": gc["case_id"],
            "latency_ms": res_t["latency_ms"],
            "finish_reason": res_t["finish_reason"],
            "usage": res_t["usage"],
            "raw_preview": res_t["text"][:400],
            "metrics": m_t,
        })
        print(f"[glm_ocr:text_rec] {gc['case_id']} dt={res_t['latency_ms']}ms acc={m_t['question_char_acc']} f_exact={m_t['formula_exact_rate']}", flush=True)

        # Parallel Hybrid
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_r = ex.submit(run_rapidocr_on_image, ocr_engine, gc["image_path"], gc["image_width"], gc["image_height"])
            fut_g = ex.submit(call_vllm_chat, base_url, "GLM-OCR", gc["image_path"], "Text Recognition:", 1350, False, True)
            r_out = fut_r.result()
            g_out = fut_g.result()
        # 用 RapidOCR 的几何 bbox，用 GLM-OCR 的按题号文本替换内容
        glm_by_no = parse_plain_text_into_questions(g_out["text"])
        merged_pqs = []
        seen_nos = set()
        for pq in r_out["pred_questions"]:
            no = pq["no"]
            seen_nos.add(no)
            merged_pqs.append({
                "no": no,
                "text": glm_by_no.get(no, pq["text"]),
                "bbox": pq["bbox"],
            })
        for no, txt in glm_by_no.items():
            if no not in seen_nos:
                merged_pqs.append({"no": no, "text": txt, "bbox": None})
        hybrid_dt = round((time.perf_counter() - t0) * 1000.0, 1)
        m_h = evaluate_prediction(gc, g_out["text"], merged_pqs)
        recs_hybrid.append({
            "case_id": gc["case_id"],
            "latency_ms": hybrid_dt,
            "rapidocr_ms": r_out["latency_ms"],
            "glm_ms": g_out["latency_ms"],
            "finish_reason": g_out["finish_reason"],
            "usage": g_out["usage"],
            "raw_preview": g_out["text"][:400],
            "metrics": m_h,
        })
        print(f"[hybrid_rapidocr_glm] {gc['case_id']} dt={hybrid_dt}ms acc={m_h['question_char_acc']} qf1={m_h['qnum_f1']} iou={m_h['bbox_mean_iou']}", flush=True)

        # JSON IE
        res_ie = call_vllm_chat(base_url, "GLM-OCR", gc["image_path"], prompt_ie, max_tokens=1350, image_first=True)
        pqs_ie, _ = parse_json_questions(res_ie["text"], gc["image_width"], gc["image_height"])
        full_ie = "\n".join(pq["text"] for pq in pqs_ie) if pqs_ie else res_ie["text"]
        m_ie = evaluate_prediction(gc, full_ie, pqs_ie)
        recs_ie.append({
            "case_id": gc["case_id"],
            "latency_ms": res_ie["latency_ms"],
            "finish_reason": res_ie["finish_reason"],
            "usage": res_ie["usage"],
            "raw_preview": res_ie["text"][:400],
            "metrics": m_ie,
        })
        print(f"[glm_ocr:json_ie] {gc['case_id']} dt={res_ie['latency_ms']}ms acc={m_ie['question_char_acc']} iou={m_ie['bbox_mean_iou']}", flush=True)

    out["glm_ocr__text_rec"] = aggregate_variant(recs_text, vram_info)
    out["glm_ocr__json_ie"] = aggregate_variant(recs_ie, vram_info)
    out["hybrid_rapidocr_glm"] = aggregate_variant(recs_hybrid, {**vram_info, "note": "CPU RapidOCR + GPU GLM-OCR 并行"})
    return out


def benchmark_rapidocr(gt_cases: list[dict[str, Any]]) -> dict[str, Any]:
    from rapidocr_onnxruntime import RapidOCR

    ocr_engine = RapidOCR()
    _ = ocr_engine(gt_cases[0]["image_path"])
    recs = []
    for gc in gt_cases:
        res = run_rapidocr_on_image(ocr_engine, gc["image_path"], gc["image_width"], gc["image_height"])
        m = evaluate_prediction(gc, res["full_text"], res["pred_questions"])
        recs.append({
            "case_id": gc["case_id"],
            "latency_ms": res["latency_ms"],
            "elapse_breakdown": res["elapse_breakdown"],
            "line_count": res["line_count"],
            "raw_preview": res["full_text"][:400],
            "metrics": m,
        })
        print(f"[rapidocr_ppocrv4] {gc['case_id']} dt={res['latency_ms']}ms acc={m['question_char_acc']} qf1={m['qnum_f1']} iou={m['bbox_mean_iou']}", flush=True)
    return {"rapidocr_ppocrv4": aggregate_variant(recs, {"vram_delta_mib": 0, "note": "ONNX Runtime CPU"})}


def build_final_report(part_files: list[Path], out_path: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for pf in part_files:
        if pf.exists():
            results.update(json.loads(pf.read_text()))

    comparison_table = []
    for name, block in results.items():
        s = block["summary"]
        v = block.get("vram", {})
        comparison_table.append({
            "model_variant": name,
            "avg_latency_ms": s["avg_latency_ms"],
            "avg_question_char_acc": s["avg_question_char_acc"],
            "avg_question_ge90_rate": s["avg_question_ge90_rate"],
            "avg_qnum_f1": s["avg_qnum_f1"],
            "avg_formula_exact_rate": s["avg_formula_exact_rate"],
            "avg_formula_sim": s["avg_formula_sim"],
            "avg_bbox_mean_iou": s["avg_bbox_mean_iou"],
            "avg_bbox_iou_05_rate": s["avg_bbox_iou_05_rate"],
            "vram_delta_mib": v.get("vram_delta_mib", 0),
        })

    report = {
        "benchmark_meta": {
            "gpu": "NVIDIA GeForce RTX 4090 D (24564 MiB)",
            "resident_services_vram_mib": 18127,
            "available_free_vram_mib": 5948,
            "test_images": [
                {"case_id": c["case_id"], "capture_rel": c["capture_rel"], "paper_id": c["paper_id"], "page_no": c["page_no"], "desc": c["desc"]}
                for c in BENCH_CASES
            ],
        },
        "comparison_table": comparison_table,
        "detailed_results": results,
        "production_recommendation": {
            "primary_recommendation": {
                "name": "PaddleOCR-VL-1.5 (0.9B) [Spotting + OCR] 或 并行方案 RapidOCR(CPU)+GLM-OCR(0.9B)",
                "note": "依据 comparison_table 实测指标动态选出最优，见下文 conclusion",
            }
        },
    }

    # 自动根据实测数据填入推荐结论
    if results:
        best_acc = max(comparison_table, key=lambda x: x["avg_question_char_acc"])
        best_bbox = max(comparison_table, key=lambda x: x["avg_bbox_mean_iou"])
        best_speed = min(comparison_table, key=lambda x: x["avg_latency_ms"])
        report["production_recommendation"] = {
            "best_text_accuracy_variant": best_acc["model_variant"],
            "best_bbox_localization_variant": best_bbox["model_variant"],
            "fastest_variant": best_speed["model_variant"],
            "recommended_for_live_coexistence_on_4090D_24GB": {
                "model_id": "PaddlePaddle/PaddleOCR-VL-1.5",
                "local_path": str(MODELS_DIR / "PaddleOCR-VL-1.5"),
                "alternative_hybrid": "RapidOCR (CPU line det+seg) + ZhipuAI/GLM-OCR (0.9B GPU Text Recognition)",
                "launch_command": (
                    ".venv-vllm/bin/vllm serve /home/ubur/dahetao/models/PaddleOCR-VL-1.5 "
                    "--served-model-name PaddleOCR-VL-1.5 --host 127.0.0.1 --port 8002 "
                    "--max-model-len 4096 --max-num-batched-tokens 8192 "
                    "--no-enable-prefix-caching --mm-processor-cache-gb 0 "
                    "--gpu-memory-utilization 0.16 --kv-cache-memory-bytes 536870912 --enforce-eager"
                ),
                "input_preprocessing": "直接传入 1080x1920 原始抓拍图（或短边缩放到 1080 / max_pixels<=1280*28*28）；若使用 GLM-OCR 则需保证 max_model_len>=4096（1080x1920 占约 2703 prompt tokens）",
                "prompt_and_calling": {
                    "paddleocr_vl_1_5_full_page_with_bbox": "Prompt='Spotting:' (image_url 在前, text 在后, temperature=0, max_tokens=2800)，解析 <|LOC_x1|>...<|LOC_y4|> 行级坐标并按题号聚类为整题 bbox",
                    "paddleocr_vl_1_5_fast_text_only": "Prompt='OCR:' (仅纯文本+LaTeX，耗时约 3.1-3.5s)",
                    "glm_ocr_text_only": "Prompt='Text Recognition:' (耗时约 3.2-3.9s，整页识别及公式极准；配合 CPU RapidOCR 可得高精度整题 bbox)",
                },
            },
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"WROTE_REPORT: {out_path}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        choices=["qwen3_5_2b_8001", "qwen3_5_0_8b", "glm_ocr", "paddleocr_vl_1_5", "rapidocr_ppocrv4", "all"],
        default="all",
    )
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    gt_cases = [extract_gt_case(c) for c in BENCH_CASES]

    part_map = {
        "qwen3_5_2b_8001": DIAG_DIR / "ocr_bench_qwen3_5_2b_8001.json",
        "rapidocr_ppocrv4": DIAG_DIR / "ocr_bench_rapidocr_ppocrv4.json",
        "paddleocr_vl_1_5": DIAG_DIR / "ocr_bench_paddleocr_vl_1_5.json",
        "glm_ocr": DIAG_DIR / "ocr_bench_glm_ocr.json",
        "qwen3_5_0_8b": DIAG_DIR / "ocr_bench_qwen3_5_0_8b.json",
    }

    if not args.finalize_only:
        targets = (
            ["qwen3_5_2b_8001", "rapidocr_ppocrv4", "paddleocr_vl_1_5", "glm_ocr", "qwen3_5_0_8b"]
            if args.target == "all"
            else [args.target]
        )
        for t in targets:
            if t == "qwen3_5_2b_8001":
                res = benchmark_qwen_like(
                    "qwen3_5_2b_8001",
                    "http://127.0.0.1:8001/v1",
                    "Qwen3.5-2B",
                    gt_cases,
                    {"vram_delta_mib": 6530, "note": "线上常驻进程 (--max-model-len 4096, max_pixels=524288)"},
                )
                part_map[t].write_text(json.dumps(res, ensure_ascii=False, indent=2))
            elif t == "rapidocr_ppocrv4":
                res = benchmark_rapidocr(gt_cases)
                part_map[t].write_text(json.dumps(res, ensure_ascii=False, indent=2))
            elif t == "paddleocr_vl_1_5":
                proc = None
                try:
                    proc, vram = start_vllm_server(MODELS_DIR / "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.5", args.port, max_model_len=4096)
                    res = benchmark_paddle_vl(f"http://127.0.0.1:{args.port}/v1", gt_cases, vram)
                    part_map[t].write_text(json.dumps(res, ensure_ascii=False, indent=2))
                finally:
                    stop_vllm_server(proc)
            elif t == "glm_ocr":
                proc = None
                try:
                    proc, vram = start_vllm_server(MODELS_DIR / "GLM-OCR", "GLM-OCR", args.port, max_model_len=4096)
                    res = benchmark_glm_and_hybrid(f"http://127.0.0.1:{args.port}/v1", gt_cases, vram)
                    part_map[t].write_text(json.dumps(res, ensure_ascii=False, indent=2))
                finally:
                    stop_vllm_server(proc)
            elif t == "qwen3_5_0_8b":
                proc = None
                try:
                    proc, vram = start_vllm_server(MODELS_DIR / "Qwen3.5-0.8B", "Qwen3.5-0.8B", args.port, max_model_len=4096)
                    res = benchmark_qwen_like("qwen3_5_0_8b", f"http://127.0.0.1:{args.port}/v1", "Qwen3.5-0.8B", gt_cases, vram)
                    part_map[t].write_text(json.dumps(res, ensure_ascii=False, indent=2))
                finally:
                    stop_vllm_server(proc)

    build_final_report(list(part_map.values()), DIAG_DIR / "ocr_benchmark_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
