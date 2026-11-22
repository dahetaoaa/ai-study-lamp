"""Task 4: PaddleOCR-VL-1.5 推理速度压测与分块并发实验脚本（无侵入，仅请求已运行的 8002 服务）。

对比项目：
A. 图像级调优（白边紧密裁剪、28px 对齐分辨率 784x1120 / 728x1036 / 672x952、光照归一化对比度增强）
   对 prompt_tokens、completion_tokens、噪声框（如 \\alpha）、耗时与题号/公式准确率的影响；
B. 水平分块并发 Spotting（单请求整页 vs 行缝切分 batch=2 并发 vs 60px重叠去重 batch=2 vs batch=3 并发）
   的真实墙钟耗时（wall-clock ms）、加速比、题号召回/F1、题干相似度与公式准确率。

输出：/home/ubur/dahetao/data/diagnostics/ocr_speed_parallel_exp.json
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import httpx
import numpy as np

ROOT = Path("/home/ubur/dahetao")
sys.path.insert(0, str(ROOT / "backend"))

from app.vision.matcher import TemplateLibrary, project, quad  # noqa: E402
from app.vision.page_ocr import (  # noqa: E402
    _parse_paddle_vl_blocks,
    build_freeform_page_from_lines,
    rectify_paper,
)

OCR_URL = "http://127.0.0.1:8002/v1/chat/completions"
OCR_MODEL = "PaddleOCR-VL-1.5"
DATA_DIR = ROOT / "data/video_testset/recordings_20261117"
OUT_PATH = ROOT / "data/diagnostics/ocr_speed_parallel_exp.json"

TEST_IMAGES = [
    ("math_p1", DATA_DIR / "rec_7267479ca145/cap_47d2afe7e71746d1.jpg"),
    ("eng_p3_choice", DATA_DIR / "rec_5422b97b81b3/cap_4397220d78bc4056.jpg"),
    ("eng_p4_cloze", DATA_DIR / "rec_5422b97b81b3/cap_9ec44c6d75bc4b42.jpg"),
]
ALPHA_NOISE_IMAGE = ("eng_p8_alpha", DATA_DIR / "rec_5422b97b81b3/cap_690166fbfe5449eb.jpg")


def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", "", s or "")
    s = s.replace("（", "(").replace("）", ")").replace("．", ".").replace("，", ",").replace("：", ":")
    s = re.sub(r"[_＿]{2,}", "_", s)
    s = s.replace("$", "")
    return s.lower()


def normalize_formula(s: str) -> str:
    s = re.sub(r"\s+", "", s or "")
    s = s.replace("\\left", "").replace("\\right", "").replace("（", "(").replace("）", ")")
    return s


def extract_math_segments(text: str) -> str:
    segs = re.findall(r"\$([^$]+)\$", text or "")
    if segs:
        return "".join(normalize_formula(x) for x in segs)
    return normalize_formula(text or "")


def align28(val: float, min_val: int = 28) -> int:
    return max(min_val, int(round(val / 28.0)) * 28)


def resize_to_max_pixels_28(img: np.ndarray, max_pixels: int, exact_wh: tuple[int, int] | None = None) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    if exact_wh is not None:
        tw, th = exact_wh
    else:
        cur = w * h
        if cur > max_pixels:
            scale = (max_pixels / float(cur)) ** 0.5
        else:
            scale = 1.0
        tw = align28(w * scale)
        th = align28(h * scale)
        while tw * th > max_pixels and tw > 28 and th > 28:
            if th >= tw:
                th -= 28
            else:
                tw -= 28
    if (tw, th) == (w, h):
        return img, tw, th
    interp = cv2.INTER_AREA if tw < w or th < h else cv2.INTER_CUBIC
    return cv2.resize(img, (tw, th), interpolation=interp), tw, th


def tight_crop_margins(warped: np.ndarray, pad: int = 24) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """紧密裁剪试卷四周纯白空白边与边缘暗影伪影，返回 (cropped_img, (x0, y0, x1, y1))。"""
    h, w = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 局部自适应阈值提取文字/线条墨迹，忽略渐进光照阴影
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 16)
    # 去掉最外圈 1.2% 避免拉平边线黑框干扰
    mx, my = max(4, int(w * 0.012)), max(4, int(h * 0.012))
    bw[:my, :] = 0
    bw[-my:, :] = 0
    bw[:, :mx] = 0
    bw[:, -mx:] = 0
    # 形态学开运算去孤立噪点
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    ys, xs = np.where(bw > 0)
    if len(xs) < 200:
        return warped, (0, 0, w, h)
    # 用 0.2%~99.8% 分位数抗偶发边缘噪点
    x0 = max(0, int(np.percentile(xs, 0.2)) - pad)
    x1 = min(w, int(np.percentile(xs, 99.8)) + pad)
    y0 = max(0, int(np.percentile(ys, 0.2)) - pad)
    y1 = min(h, int(np.percentile(ys, 99.8)) + pad)
    if (x1 - x0) < w * 0.45 or (y1 - y0) < h * 0.45:
        return warped, (0, 0, w, h)
    return warped[y0:y1, x0:x1], (x0, y0, x1, y1)


def enhance_contrast_paper(img: np.ndarray) -> np.ndarray:
    """光照归一化 + 纸面底噪漂白 + 轻微锐化：消除阴影渐变和浅色横线引发的幻觉框。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 局部大核形态学闭运算估计纸面背景亮度
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg_f = np.maximum(bg.astype(np.float32), 32.0)
    norm = np.clip((gray.astype(np.float32) / bg_f) * 255.0, 0, 255).astype(np.uint8)
    # 将浅灰纸底 (>226) 提纯至纯白 255，同时增强深色墨迹对比度
    lut = np.arange(256, dtype=np.float32)
    lut = np.clip((lut - 35.0) * (255.0 / (226.0 - 35.0)), 0, 255).astype(np.uint8)
    clean = cv2.LUT(norm, lut)
    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)


def find_blank_split_ys(img: np.ndarray, num_chunks: int) -> list[int]:
    """在图像水平方向寻找墨迹最稀疏的行间空白带作为切分 Y 坐标。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 16)
    # 排除左右 3% 边缘装订线/阴影
    x_l, x_r = int(w * 0.03), int(w * 0.97)
    row_ink = (bw[:, x_l:x_r] > 0).sum(axis=1).astype(np.float32)
    # 轻微平滑避免单个标点间隙被误判为大空白
    row_smooth = cv2.GaussianBlur(row_ink.reshape(-1, 1), (1, 15), 0).flatten()

    splits = []
    for k in range(1, num_chunks):
        target = k / float(num_chunks)
        lo = max(40, int(h * (target - 0.09)))
        hi = min(h - 40, int(h * (target + 0.09)))
        sub = row_smooth[lo:hi]
        # 找最小墨迹区域；若有多处全白，取最靠近中心空白带的中点
        min_v = float(sub.min())
        cand = np.where(sub <= min_v + max(2.0, w * 0.002))[0]
        # 找最长连续空白段的中点
        if len(cand) > 0:
            breaks = np.where(np.diff(cand) > 3)[0]
            segs = np.split(cand, breaks + 1)
            best_seg = max(segs, key=lambda s: (len(s), -abs((s.mean() + lo) - h * target)))
            best_y = int(round(lo + best_seg.mean()))
        else:
            best_y = int(lo + np.argmin(sub))
        splits.append(best_y)
    return splits


def merge_chunk_blocks(
    chunk_results: list[tuple[int, int, list[dict[str, Any]]]],
    page_w: int,
    page_h: int,
    dedup_overlap: bool = True,
) -> list[dict[str, Any]]:
    """将多段并发块还原到整页坐标，并对重叠带内的重复检测行去重。"""
    all_items: list[dict[str, Any]] = []
    for ci, (cy0, cy1, blocks) in enumerate(chunk_results):
        for b in blocks:
            x0, y0, x1, y1 = b["box"]
            py0, py1 = cy0 + y0, cy0 + y1
            px0, px1 = x0, x1
            dist_to_cut = min(
                (py0 - cy0) if ci > 0 else 1e9,
                (cy1 - py1) if ci < len(chunk_results) - 1 else 1e9,
            )
            all_items.append(
                {
                    "label": b.get("label", "line"),
                    "box": [px0, py0, px1, py1],
                    "text": b["text"],
                    "chunk_idx": ci,
                    "dist_to_cut": float(dist_to_cut),
                }
            )

    if not dedup_overlap or len(chunk_results) <= 1:
        return [{"label": it["label"], "box": it["box"], "text": it["text"]} for it in all_items]

    kept: list[dict[str, Any]] = []
    dropped = [False] * len(all_items)
    for i in range(len(all_items)):
        if dropped[i]:
            continue
        ai = all_items[i]
        ax0, ay0, ax1, ay1 = ai["box"]
        ah = max(1, ay1 - ay0)
        aw = max(1, ax1 - ax0)
        for j in range(i + 1, len(all_items)):
            if dropped[j]:
                continue
            aj = all_items[j]
            if ai["chunk_idx"] == aj["chunk_idx"]:
                continue
            bx0, by0, bx1, by1 = aj["box"]
            bh = max(1, by1 - by0)
            bw = max(1, bx1 - bx0)
            iy = max(0, min(ay1, by1) - max(ay0, by0))
            ix = max(0, min(ax1, bx1) - max(ax0, bx0))
            y_ov = iy / min(ah, bh)
            x_ov = ix / min(aw, bw)
            if y_ov >= 0.45 and x_ov >= 0.35:
                # 同一行被相邻两段重叠扫到：保留离各自切分边缘更远（更完整）的那一条
                score_i = (ai["dist_to_cut"] > 4.0, len(ai["text"]), ai["dist_to_cut"])
                score_j = (aj["dist_to_cut"] > 4.0, len(aj["text"]), aj["dist_to_cut"])
                if score_j > score_i:
                    dropped[i] = True
                    break
                else:
                    dropped[j] = True
        if not dropped[i]:
            kept.append({"label": ai["label"], "box": ai["box"], "text": ai["text"]})
    return kept


async def call_spotting(client: httpx.AsyncClient, img: np.ndarray, jpeg_q: int = 85, max_tokens: int = 2800) -> dict[str, Any]:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    raw_bytes = enc.tobytes()
    b64_url = "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode()
    payload = {
        "model": OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": b64_url}},
                    {"type": "text", "text": "Spotting:"},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = perf_counter()
    resp = await client.post(OCR_URL, json=payload, timeout=60.0)
    ms = round((perf_counter() - t0) * 1000.0, 1)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    text = data["choices"][0]["message"]["content"] or ""
    return {
        "latency_ms": ms,
        "raw_text": text,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "jpeg_kb": round(len(raw_bytes) / 1024.0, 1),
        "finish_reason": data["choices"][0].get("finish_reason", ""),
    }


def evaluate_blocks(
    blocks: list[dict[str, Any]],
    warped: np.ndarray,
    pw: int,
    ph: int,
    gt_qs: dict[str, dict],
    gt_boxes_warped: dict[str, list[float]],
    is_math: bool,
) -> dict[str, Any]:
    _, pred_page, _ = build_freeform_page_from_lines(blocks, warped, pw, ph)
    pred_qs = {str(q["number"]): q for q in pred_page["questions"]}
    matched = [n for n in gt_qs if n in pred_qs]
    recall = round(len(matched) / max(1, len(gt_qs)), 4)
    prec = round(len(matched) / max(1, len(pred_qs)), 4)
    f1 = round((2 * recall * prec / (recall + prec)) if (recall + prec) > 0 else 0.0, 4)

    stem_sims = []
    formula_sims = []
    ious = []
    for n in matched:
        gq, pq = gt_qs[n], pred_qs[n]
        gt_full = gq["stem"] + (" " + " ".join(gq.get("options") or []) if gq.get("options") else "")
        pred_full = pq["stem"] + (" " + " ".join(pq.get("options") or []) if pq.get("options") else "")
        stem_sims.append(difflib.SequenceMatcher(None, normalize_text(gt_full), normalize_text(pred_full)).ratio())
        if is_math:
            f_gt = extract_math_segments(gt_full)
            f_pr = extract_math_segments(pred_full)
            if f_gt:
                formula_sims.append(difflib.SequenceMatcher(None, f_gt, f_pr).ratio())
        if n in gt_boxes_warped:
            bx = pq["bbox"]
            pb = [bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]]
            gb = gt_boxes_warped[n]
            ix0, iy0 = max(pb[0], gb[0]), max(pb[1], gb[1])
            ix1, iy1 = min(pb[2], gb[2]), min(pb[3], gb[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            a1 = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
            a2 = max(1.0, (gb[2] - gb[0]) * (gb[3] - gb[1]))
            ious.append(inter / (a1 + a2 - inter))

    return {
        "gt_q_count": len(gt_qs),
        "pred_q_count": len(pred_qs),
        "matched_q_count": len(matched),
        "missing_nums": [n for n in gt_qs if n not in pred_qs],
        "extra_nums": [n for n in pred_qs if n not in gt_qs],
        "q_recall": recall,
        "q_precision": prec,
        "q_f1": f1,
        "mean_stem_sim": round(float(np.mean(stem_sims)) if stem_sims else 0.0, 4),
        "mean_formula_char_acc": round(float(np.mean(formula_sims)) if formula_sims else 0.0, 4) if is_math else None,
        "bbox_mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
    }


def prepare_case_gt(gt_lib: TemplateLibrary, ipath: Path) -> dict[str, Any]:
    img = cv2.imread(str(ipath))
    gt_match = gt_lib.match(img)
    warped, h_page_to_cam, _, pw, ph = rectify_paper(img)
    # Template -> Camera -> Warped 投影，用于算 bbox_mean_iou
    h_cam_to_page = np.linalg.inv(h_page_to_cam)
    h_tpl_to_warped = h_cam_to_page @ gt_match.homography

    gt_qs = {str(q["number"]): q for q in gt_match.template.page["questions"]}
    gt_boxes_warped = {}
    for n, gq in gt_qs.items():
        if gt_match.template.paper["paper_id"] == "eng_xmwgy_2024_mid" and gt_match.template.page["page_no"] == 4 and int(n) >= 36:
            continue
        pts = np.float32(project(quad(gq["bbox"]), h_tpl_to_warped))
        gt_boxes_warped[n] = [
            float(np.clip(pts[:, 0].min(), 0, pw)),
            float(np.clip(pts[:, 1].min(), 0, ph)),
            float(np.clip(pts[:, 0].max(), 0, pw)),
            float(np.clip(pts[:, 1].max(), 0, ph)),
        ]
    return {
        "img": img,
        "warped": warped,
        "pw": pw,
        "ph": ph,
        "gt_qs": gt_qs,
        "gt_boxes_warped": gt_boxes_warped,
    }


async def run_experiments():
    gt_lib = TemplateLibrary(ROOT / "data/templates")
    gt_lib.load()

    async with httpx.AsyncClient() as client:
        # Warmup 1 次
        w_img = cv2.imread(str(TEST_IMAGES[0][1]))
        w_warp, _, _, _, _ = rectify_paper(w_img)
        w_small, _, _ = resize_to_max_pixels_28(w_warp, 878080, (784, 1120))
        await call_spotting(client, w_small, jpeg_q=85, max_tokens=64)

        cases = {}
        for label, ipath in TEST_IMAGES + [ALPHA_NOISE_IMAGE]:
            cases[label] = prepare_case_gt(gt_lib, ipath)

        # ======================================================================
        # 实验 A：图像紧密裁剪 + 28px 对齐分辨率 + 对比度增强
        # ======================================================================
        print("\n=== 实验 A：图像裁剪、28px 对齐分辨率与对比度增强（8002 单请求） ===")
        exp_a_configs = [
            ("baseline_1540_q90", False, False, 1540, None, 90),
            ("res_784x1120_q85", False, False, None, (784, 1120), 85),
            ("res_728x1036_q85", False, False, None, (728, 1036), 85),
            ("res_672x952_q85", False, False, None, (672, 952), 85),
            ("tight_784x1120_q85", True, False, 878080, None, 85),
            ("tight_contrast_784x1120_q85", True, True, 878080, None, 85),
            ("tight_contrast_728x1036_q85", True, True, 754208, None, 85),
        ]

        exp_a_runs = []
        for label, ipath in TEST_IMAGES + [ALPHA_NOISE_IMAGE]:
            c = cases[label]
            warped, pw, ph = c["warped"], c["pw"], c["ph"]
            is_math = label == "math_p1"
            for cfg_name, do_tight, do_contrast, limit_val, exact_wh, jq in exp_a_configs:
                # 仅在 baseline 和 tight_contrast 上额外跑 eng_p8_alpha（节省时间）
                if label == "eng_p8_alpha" and cfg_name not in (
                    "baseline_1540_q90",
                    "res_784x1120_q85",
                    "tight_784x1120_q85",
                    "tight_contrast_784x1120_q85",
                ):
                    continue

                if do_tight:
                    work_img, (cx0, cy0, cx1, cy1) = tight_crop_margins(warped, pad=24)
                else:
                    work_img, (cx0, cy0, cx1, cy1) = warped, (0, 0, pw, ph)
                crop_w, crop_h = cx1 - cx0, cy1 - cy0

                if do_contrast:
                    work_img = enhance_contrast_paper(work_img)

                if exact_wh is not None:
                    ocr_img, ow, oh = resize_to_max_pixels_28(work_img, 1003520, exact_wh)
                elif cfg_name == "baseline_1540_q90":
                    sc = min(1.0, float(limit_val) / max(crop_w, crop_h))
                    ow, oh = round(crop_w * sc), round(crop_h * sc)
                    ocr_img = cv2.resize(work_img, (ow, oh), interpolation=cv2.INTER_AREA) if sc < 1.0 else work_img
                else:
                    ocr_img, ow, oh = resize_to_max_pixels_28(work_img, limit_val)

                res = await call_spotting(client, ocr_img, jpeg_q=jq, max_tokens=2800)
                blocks = _parse_paddle_vl_blocks(res["raw_text"], crop_w, crop_h)
                for b in blocks:
                    b["box"] = [b["box"][0] + cx0, b["box"][1] + cy0, b["box"][2] + cx0, b["box"][3] + cy0]
                alpha_lines = sum(1 for ln in res["raw_text"].splitlines() if "\\alpha" in ln or "α" in ln)
                metrics = evaluate_blocks(blocks, warped, pw, ph, c["gt_qs"], c["gt_boxes_warped"], is_math)

                row = {
                    "config": cfg_name,
                    "label": label,
                    "crop_box": [cx0, cy0, cx1, cy1],
                    "ocr_wh": [ow, oh],
                    "ocr_pixels": ow * oh,
                    "jpeg_kb": res["jpeg_kb"],
                    "latency_ms": res["latency_ms"],
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "tok_per_s": round(res["completion_tokens"] / max(0.001, res["latency_ms"] / 1000.0), 1),
                    "blocks_count": len(blocks),
                    "alpha_noise_lines": alpha_lines,
                    **metrics,
                }
                exp_a_runs.append(row)
                print(
                    f"  [{label:13s}] {cfg_name:28s} ({ow}x{oh}) | "
                    f"{res['latency_ms']:6.1f}ms | p={res['prompt_tokens']:4d} c={res['completion_tokens']:4d} "
                    f"({row['tok_per_s']:5.1f} tok/s) | Q={metrics['matched_q_count']}/{metrics['gt_q_count']} "
                    f"F1={metrics['q_f1']:.3f} stem={metrics['mean_stem_sim']:.3f} "
                    f"form={metrics['mean_formula_char_acc']} alpha={alpha_lines}"
                )

        # ======================================================================
        # 实验 B：上下两半页 / 三段并发 Spotting (batch=2 / batch=3)
        # ======================================================================
        print("\n=== 实验 B：行缝分块并发 Spotting (整页 vs batch=2 vs batch=3, 8002 实测) ===")
        exp_b_modes = [
            ("full_baseline_1540_q90", 1, 0, False, 1540, 90),
            ("full_784x1120_q85", 1, 0, False, 878080, 85),
            ("batch2_gap_no_overlap_784w", 2, 0, False, 784, 85),
            ("batch2_gap_overlap60_784w", 2, 60, False, 784, 85),
            ("batch2_contrast_overlap60_784w", 2, 60, True, 784, 85),
            ("batch3_gap_no_overlap_784w", 3, 0, False, 784, 85),
            ("batch3_gap_overlap60_784w", 3, 60, False, 784, 85),
        ]

        exp_b_runs = []
        for label, ipath in TEST_IMAGES:
            c = cases[label]
            warped, pw, ph = c["warped"], c["pw"], c["ph"]
            is_math = label == "math_p1"

            for mode_name, n_chunks, overlap_px, do_contrast, param_w, jq in exp_b_modes:
                base_img = enhance_contrast_paper(warped) if do_contrast else warped

                if n_chunks == 1:
                    if mode_name == "full_baseline_1540_q90":
                        sc = min(1.0, float(param_w) / max(pw, ph))
                        ow, oh = round(pw * sc), round(ph * sc)
                        ocr_img = cv2.resize(base_img, (ow, oh), interpolation=cv2.INTER_AREA)
                    else:
                        ocr_img, ow, oh = resize_to_max_pixels_28(base_img, param_w, (784, 1120))
                    t0 = perf_counter()
                    res = await call_spotting(client, ocr_img, jpeg_q=jq, max_tokens=2800)
                    wall_ms = round((perf_counter() - t0) * 1000.0, 1)
                    blocks = _parse_paddle_vl_blocks(res["raw_text"], pw, ph)
                    metrics = evaluate_blocks(blocks, warped, pw, ph, c["gt_qs"], c["gt_boxes_warped"], is_math)
                    row = {
                        "mode": mode_name,
                        "label": label,
                        "n_chunks": 1,
                        "overlap_px": 0,
                        "wall_ms": wall_ms,
                        "chunk_latencies_ms": [res["latency_ms"]],
                        "chunk_whs": [[ow, oh]],
                        "total_prompt_tokens": res["prompt_tokens"],
                        "total_completion_tokens": res["completion_tokens"],
                        "max_chunk_completion_tokens": res["completion_tokens"],
                        "effective_tok_per_s": round(res["completion_tokens"] / max(0.001, wall_ms / 1000.0), 1),
                        "blocks_count": len(blocks),
                        **metrics,
                    }
                else:
                    splits = find_blank_split_ys(warped, n_chunks)
                    half_ov = overlap_px // 2
                    bounds = []
                    prev_y = 0
                    for s_y in splits:
                        bounds.append((max(0, prev_y - half_ov), min(ph, s_y + half_ov)))
                        prev_y = s_y
                    bounds.append((max(0, prev_y - half_ov), ph))

                    chunk_imgs = []
                    chunk_whs = []
                    for cy0, cy1 in bounds:
                        sub = base_img[cy0:cy1, :]
                        sub_h, sub_w = sub.shape[:2]
                        # 保持宽度 784（28对齐），高度按比例 28 对齐
                        tw = align28(param_w)
                        th = align28(sub_h * (tw / float(sub_w)))
                        sub_res, tw, th = resize_to_max_pixels_28(sub, 1003520, (tw, th))
                        chunk_imgs.append((cy0, cy1, sub_w, sub_h, sub_res))
                        chunk_whs.append([tw, th])

                    t0 = perf_counter()
                    chunk_resps = await asyncio.gather(
                        *(call_spotting(client, ci[4], jpeg_q=jq, max_tokens=1800) for ci in chunk_imgs)
                    )
                    wall_ms = round((perf_counter() - t0) * 1000.0, 1)

                    parsed_chunks = []
                    for (cy0, cy1, sub_w, sub_h, _), resp in zip(chunk_imgs, chunk_resps):
                        c_blocks = _parse_paddle_vl_blocks(resp["raw_text"], sub_w, sub_h)
                        parsed_chunks.append((cy0, cy1, c_blocks))

                    merged_blocks = merge_chunk_blocks(parsed_chunks, pw, ph, dedup_overlap=(overlap_px > 0))
                    metrics = evaluate_blocks(merged_blocks, warped, pw, ph, c["gt_qs"], c["gt_boxes_warped"], is_math)

                    tot_p = sum(r["prompt_tokens"] for r in chunk_resps)
                    tot_c = sum(r["completion_tokens"] for r in chunk_resps)
                    max_c = max(r["completion_tokens"] for r in chunk_resps)
                    row = {
                        "mode": mode_name,
                        "label": label,
                        "n_chunks": n_chunks,
                        "overlap_px": overlap_px,
                        "split_ys": splits,
                        "chunk_bounds": bounds,
                        "wall_ms": wall_ms,
                        "chunk_latencies_ms": [r["latency_ms"] for r in chunk_resps],
                        "chunk_whs": chunk_whs,
                        "total_prompt_tokens": tot_p,
                        "total_completion_tokens": tot_c,
                        "max_chunk_completion_tokens": max_c,
                        "chunk_completion_tokens": [r["completion_tokens"] for r in chunk_resps],
                        "effective_tok_per_s": round(tot_c / max(0.001, wall_ms / 1000.0), 1),
                        "blocks_count": len(merged_blocks),
                        **metrics,
                    }

                exp_b_runs.append(row)
                print(
                    f"  [{label:13s}] {mode_name:30s} | wall={row['wall_ms']:6.1f}ms "
                    f"c_tok={row['total_completion_tokens']} (max={row['max_chunk_completion_tokens']}) "
                    f"eff={row['effective_tok_per_s']:5.1f} tok/s | "
                    f"Q={metrics['matched_q_count']}/{metrics['gt_q_count']} F1={metrics['q_f1']:.3f} "
                    f"stem={metrics['mean_stem_sim']:.3f} iou={metrics['bbox_mean_iou']:.3f} form={metrics['mean_formula_char_acc']}"
                )

    # 汇总表
    def summarize_group(runs: list[dict], key: str, only_primary: bool = True) -> dict[str, Any]:
        primary_labels = {x[0] for x in TEST_IMAGES}
        groups: dict[str, list[dict]] = {}
        for r in runs:
            if only_primary and r["label"] not in primary_labels:
                continue
            groups.setdefault(r[key], []).append(r)
        out = {}
        for k, items in groups.items():
            lat_k = "wall_ms" if "wall_ms" in items[0] else "latency_ms"
            math_forms = [x["mean_formula_char_acc"] for x in items if x["mean_formula_char_acc"] is not None]
            out[k] = {
                "n_cases": len(items),
                f"avg_{lat_k}": round(float(np.mean([x[lat_k] for x in items])), 1),
                "avg_q_recall": round(float(np.mean([x["q_recall"] for x in items])), 4),
                "avg_q_f1": round(float(np.mean([x["q_f1"] for x in items])), 4),
                "avg_stem_sim": round(float(np.mean([x["mean_stem_sim"] for x in items])), 4),
                "avg_bbox_iou": round(float(np.mean([x["bbox_mean_iou"] for x in items])), 4),
                "math_formula_char_acc": round(float(np.mean(math_forms)), 4) if math_forms else None,
            }
            if "effective_tok_per_s" in items[0]:
                out[k]["avg_effective_tok_per_s"] = round(float(np.mean([x["effective_tok_per_s"] for x in items])), 1)
                out[k]["avg_total_completion_tokens"] = round(float(np.mean([x["total_completion_tokens"] for x in items])), 1)
                out[k]["avg_max_chunk_completion_tokens"] = round(float(np.mean([x["max_chunk_completion_tokens"] for x in items])), 1)
            else:
                out[k]["avg_prompt_tokens"] = round(float(np.mean([x["prompt_tokens"] for x in items])), 1)
                out[k]["avg_completion_tokens"] = round(float(np.mean([x["completion_tokens"] for x in items])), 1)
                out[k]["avg_tok_per_s"] = round(float(np.mean([x["tok_per_s"] for x in items])), 1)
        return out

    exp_a_summary = summarize_group(exp_a_runs, "config", only_primary=True)
    exp_b_summary = summarize_group(exp_b_runs, "mode", only_primary=True)
    base_wall = exp_b_summary["full_baseline_1540_q90"]["avg_wall_ms"]
    for k, v in exp_b_summary.items():
        v["speedup_vs_baseline_1540"] = round(base_wall / max(1.0, v["avg_wall_ms"]), 3)

    alpha_runs = [r for r in exp_a_runs if r["label"] == "eng_p8_alpha"]

    report = {
        "env": {
            "ocr_url": OCR_URL,
            "ocr_model": OCR_MODEL,
            "note": "Measured live against 8002 (currently running with --enforce-eager, before restart)",
        },
        "exp_a_summary": exp_a_summary,
        "exp_a_alpha_noise_check": alpha_runs,
        "exp_a_runs": exp_a_runs,
        "exp_b_summary": exp_b_summary,
        "exp_b_runs": exp_b_runs,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Exp A Summary (3 primary pages) ===")
    print(json.dumps(exp_a_summary, ensure_ascii=False, indent=2))
    print("\n=== Exp B Summary (Parallel Chunked Spotting) ===")
    print(json.dumps(exp_b_summary, ensure_ascii=False, indent=2))
    print(f"\nSaved report to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run_experiments())
