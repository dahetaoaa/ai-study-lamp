"""Subtask B 专项审计脚本：纸面透视拉平几何精度 + OCR 分辨率速度权衡 + 空白门控与 HWR 根因审计。

在不修改 backend/app/ 生产代码、不重启常驻服务的前提下，针对
data/video_testset/recordings_20261117/rec_*/cap_*.jpg 共 9 张实拍图执行三项定量实验，
并将完整结果输出至 data/diagnostics/probe_rectify_and_speed.json。
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import re
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.grading.client import ModelClients  # noqa: E402
from app.grading.hwr_client import HWRClient  # noqa: E402
from app.vision.matcher import TemplateLibrary, project, quad  # noqa: E402
from app.vision.page_ocr import (  # noqa: E402
    BRACKET_RE,
    NORM_H,
    NORM_W,
    QNUM_RE,
    _parse_paddle_vl_blocks,
    build_freeform_page_from_lines,
    detect_paper_quad,
    order_quad,
    rectify_paper,
)

DATA_DIR = ROOT / "data/video_testset/recordings_20261117"
OUT_PATH = ROOT / "data/diagnostics/probe_rectify_and_speed.json"
WRITTEN_MATH_P1 = {"1": "B", "2": "C", "3": "A", "4": "A", "5": "B", "6": "B", "7": "B"}

OCR_BENCH_IMAGES = [
    ("math_p1", DATA_DIR / "rec_7267479ca145/cap_47d2afe7e71746d1.jpg"),
    ("eng_p3_choice", DATA_DIR / "rec_5422b97b81b3/cap_4397220d78bc4056.jpg"),
    ("eng_p4_cloze", DATA_DIR / "rec_5422b97b81b3/cap_9ec44c6d75bc4b42.jpg"),
]


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


def polygon_iou(q1: np.ndarray, q2: np.ndarray, img_h: int, img_w: int) -> tuple[float, float, float]:
    """返回 (iou, recall_of_q1, extra_outside_q1_ratio)。"""
    m1 = np.zeros((img_h, img_w), np.uint8)
    m2 = np.zeros((img_h, img_w), np.uint8)
    cv2.fillPoly(m1, [np.int32(np.round(q1))], 1)
    cv2.fillPoly(m2, [np.int32(np.round(q2))], 1)
    inter = float(np.count_nonzero(m1 & m2))
    union = float(np.count_nonzero(m1 | m2))
    a1 = float(np.count_nonzero(m1))
    a2 = float(np.count_nonzero(m2))
    iou = inter / max(1.0, union)
    recall = inter / max(1.0, a1)
    extra = (a2 - inter) / max(1.0, a2)
    return round(iou, 4), round(recall, 4), round(extra, 4)


def box_iou(b1: list[float], b2: list[float]) -> float:
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


# ==============================================================================
# 实验 1：改进版纸面四角检测 detect_paper_quad_v2
# ==============================================================================

def _fit_line_ransac(pts: np.ndarray, iters: int = 120, thresh: float = 2.0):
    if len(pts) < 8:
        return None
    rng = np.random.default_rng(42)
    best_inliers = None
    best_cnt = -1
    n = len(pts)
    for _ in range(iters):
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        v = p2 - p1
        norm = float(np.hypot(v[0], v[1]))
        if norm < 5.0:
            continue
        nx, ny = -v[1] / norm, v[0] / norm
        dist = np.abs((pts[:, 0] - p1[0]) * nx + (pts[:, 1] - p1[1]) * ny)
        inliers = dist < thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers
    if best_inliers is None or best_cnt < 6:
        return None
    in_pts = pts[best_inliers].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(in_pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return float(vx), float(vy), float(x0), float(y0)


def _intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    det = vx1 * (-vy2) - vy1 * (-vx2)
    if abs(det) < 1e-6:
        return None
    dx, dy = x2 - x1, y2 - y1
    t = (dx * (-vy2) - dy * (-vx2)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def detect_paper_quad_v2(image: np.ndarray) -> tuple[np.ndarray, float]:
    """改进版纸面四角检测：
    1) Lab 空间去字迹（L 闭运算）+ 极速低频光照归一化，消除桌面台灯明暗梯度；
    2) 结合局部亮度差与中性低色度（a/b 偏白）+ 强梯度边缘阻隔，防止与邻近桌面/底纸粘连；
    3) 沿粗框四边法向搜索由暗桌面向亮纸面的单向梯度峰值点，RANSAC + Huber 拟合四条直线求交点。
    """
    h, w = image.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    sw, sh = max(16, round(w * scale)), max(16, round(h * scale))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # 1. 形态学闭运算擦除黑字与笔迹，仅保留干净纸面与外边界
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    L_clean = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k_close)
    L_blur = cv2.GaussianBlur(L_clean, (5, 5), 0).astype(np.float32)

    # 2. 低频背景估计（1/16 下采样后高斯模糊再上采样，耗时 <0.5ms）
    tiny_w, tiny_h = max(8, sw // 16), max(8, sh // 16)
    bg_tiny = cv2.GaussianBlur(cv2.resize(L_blur, (tiny_w, tiny_h), interpolation=cv2.INTER_AREA), (9, 9), 0)
    bg_L = cv2.resize(bg_tiny, (sw, sh), interpolation=cv2.INTER_LINEAR)

    gx = cv2.Sobel(L_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_blur, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)

    # 相对局部亮度 + 纸面中性色度（桌面通常偏黄/绿）
    white_score = (L_blur - bg_L) + 1.8 * (a - 126.5) - 1.5 * (b - 138.0)
    edge_barrier = gmag > np.percentile(gmag, 88)
    score_u8 = np.clip((white_score + 15.0) * 5.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[edge_barrier] = 0
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_open, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return detect_paper_quad(image)

    cx_img, cy_img = sw * 0.5, sh * 0.5

    def cnt_score(c):
        ar = cv2.contourArea(c)
        if ar < 0.12 * sw * sh:
            return -1.0
        M = cv2.moments(c)
        if M["m00"] <= 0:
            return ar
        mx, my = M["m10"] / M["m00"], M["m01"] / M["m00"]
        dist = np.hypot((mx - cx_img) / sw, (my - cy_img) / sh)
        return ar * max(0.2, 1.0 - dist)

    cnt = max(contours, key=cnt_score)
    ratio = float(cv2.contourArea(cnt)) / float(sw * sh)
    if ratio < 0.12:
        return detect_paper_quad(image)

    hull = cv2.convexHull(cnt)
    init_q = order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))

    # 3. 沿四边内法向搜寻“桌面暗 -> 纸面亮”最强单向阶跃梯度，RANSAC 直线拟合求交
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    fitted_lines = []
    band = max(14.0, 0.07 * min(sw, sh))
    for i0, i1 in edges:
        p0, p1 = init_q[i0], init_q[i1]
        ev = p1 - p0
        elen = float(np.hypot(ev[0], ev[1]))
        if elen < 20:
            fitted_lines.append(None)
            continue
        tx, ty = ev[0] / elen, ev[1] / elen
        nx, ny = -ty, tx
        pts = []
        for s_t in np.linspace(0.08, 0.92, 64):
            cx = p0[0] + s_t * ev[0]
            cy = p0[1] + s_t * ev[1]
            best_val = -1e9
            best_pt = None
            for d in range(-int(band), int(band) + 1):
                px = int(round(cx + d * nx))
                py = int(round(cy + d * ny))
                if 3 <= px < sw - 3 and 3 <= py < sh - 3:
                    dg = gx[py, px] * nx + gy[py, px] * ny
                    if dg > best_val:
                        best_val = dg
                        best_pt = (cx + d * nx, cy + d * ny)
            if best_pt is not None and best_val > 18.0:
                pts.append(best_pt)
        if len(pts) >= 12:
            fitted_lines.append(_fit_line_ransac(np.array(pts, dtype=np.float32), iters=120, thresh=2.0))
        else:
            fitted_lines.append(None)

    refined = []
    for k in range(4):
        l_prev = fitted_lines[(k - 1) % 4]
        l_curr = fitted_lines[k]
        pt = _intersect_lines(l_prev, l_curr) if (l_prev is not None and l_curr is not None) else None
        if pt is None or np.linalg.norm(pt - init_q[k]) > band * 3.0:
            pt = init_q[k]
        refined.append(pt)

    q_out = order_quad(np.array(refined, dtype=np.float32) / scale)
    q_out[:, 0] = np.clip(q_out[:, 0], 0, w - 1)
    q_out[:, 1] = np.clip(q_out[:, 1], 0, h - 1)
    return q_out, round(ratio, 3)


def rectify_paper_v2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    q, _ = detect_paper_quad_v2(image)
    top_w = float(np.hypot(q[1][0] - q[0][0], q[1][1] - q[0][1]))
    bot_w = float(np.hypot(q[2][0] - q[3][0], q[2][1] - q[3][1]))
    left_h = float(np.hypot(q[3][0] - q[0][0], q[3][1] - q[0][1]))
    right_h = float(np.hypot(q[2][0] - q[1][0], q[2][1] - q[1][1]))
    avg_w = max(320.0, (top_w + bot_w) * 0.5)
    avg_h = max(320.0, (left_h + right_h) * 0.5)
    if avg_w > avg_h * 1.15:
        dst_w = NORM_H
        dst_h = max(900, round(NORM_H * (avg_h / avg_w)))
    else:
        dst_w = NORM_W
        dst_h = max(1100, min(2400, round(NORM_W * (avg_h / avg_w))))
    src_page = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    h_page_to_cam = cv2.getPerspectiveTransform(src_page, q)
    h_cam_to_page = cv2.getPerspectiveTransform(q, src_page)
    warped = cv2.warpPerspective(image, h_cam_to_page, (dst_w, dst_h), borderValue=(255, 255, 255))
    return warped, h_page_to_cam, q, dst_w, dst_h


def run_exp1_rectify_audit(gt_lib: TemplateLibrary) -> dict:
    print("\n=== 实验 1：纸面四角透视拉平精度审计 (Baseline vs v2) ===")
    cases = []
    for jpath in sorted(DATA_DIR.glob("rec_*/cap_*.json")):
        ipath = jpath.with_suffix(".jpg")
        img = cv2.imread(str(ipath))
        h, w = img.shape[:2]
        m = gt_lib.match(img)
        tpl_w, tpl_h = m.template.image.shape[1], m.template.image.shape[0]
        gt_q = order_quad(np.float32(project(quad([0, 0, tpl_w, tpl_h]), m.homography)))
        left_slot_q = np.float32(project(quad([0, 0, 110, tpl_h]), m.homography))

        # Baseline
        t0 = perf_counter()
        q_base, ratio_base = detect_paper_quad(img)
        ms_base = round((perf_counter() - t0) * 1000, 2)
        iou_b, rec_b, extra_b = polygon_iou(gt_q, q_base, h, w)
        _, left_cov_b, _ = polygon_iou(left_slot_q, q_base, h, w)
        errs_b = np.linalg.norm(q_base - gt_q, axis=1)

        # v2
        t1 = perf_counter()
        q_v2, ratio_v2 = detect_paper_quad_v2(img)
        ms_v2 = round((perf_counter() - t1) * 1000, 2)
        iou_v2, rec_v2, extra_v2 = polygon_iou(gt_q, q_v2, h, w)
        _, left_cov_v2, _ = polygon_iou(left_slot_q, q_v2, h, w)
        errs_v2 = np.linalg.norm(q_v2 - gt_q, axis=1)

        # 灰度诊断统计
        m_gt = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m_gt, [np.int32(np.round(gt_q))], 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        p_in, p_out = m_gt > 0, m_gt == 0

        row = {
            "case": f"{jpath.parent.name}/{ipath.name}",
            "paper_id": m.template.paper["paper_id"],
            "page_no": m.template.page["page_no"],
            "image_size": [w, h],
            "gt_quad": np.round(gt_q, 1).tolist(),
            "intensity_diagnostics": {
                "gray_in_p10_p50_p90": np.percentile(gray[p_in], [10, 50, 90]).round(1).tolist(),
                "gray_out_p10_p50_p90": np.percentile(gray[p_out], [10, 50, 90]).round(1).tolist(),
                "gray_all_p45": round(float(np.percentile(gray, 45)), 1),
                "lab_a_in_p50": round(float(np.median(lab[:, :, 1][p_in])), 1),
                "lab_a_out_p50": round(float(np.median(lab[:, :, 1][p_out])), 1),
                "lab_b_in_p50": round(float(np.median(lab[:, :, 2][p_in])), 1),
                "lab_b_out_p50": round(float(np.median(lab[:, :, 2][p_out])), 1),
            },
            "baseline": {
                "quad": np.round(q_base, 1).tolist(),
                "paper_ratio": ratio_base,
                "iou": iou_b,
                "paper_recall": rec_b,
                "extra_bg_ratio": extra_b,
                "left_margin_x0_110_cov": left_cov_b,
                "corner_errors_px": np.round(errs_b, 1).tolist(),
                "mean_corner_err_px": round(float(errs_b.mean()), 1),
                "max_corner_err_px": round(float(errs_b.max()), 1),
                "runtime_ms": ms_base,
            },
            "v2": {
                "quad": np.round(q_v2, 1).tolist(),
                "paper_ratio": ratio_v2,
                "iou": iou_v2,
                "paper_recall": rec_v2,
                "extra_bg_ratio": extra_v2,
                "left_margin_x0_110_cov": left_cov_v2,
                "corner_errors_px": np.round(errs_v2, 1).tolist(),
                "mean_corner_err_px": round(float(errs_v2.mean()), 1),
                "max_corner_err_px": round(float(errs_v2.max()), 1),
                "runtime_ms": ms_v2,
            },
        }
        cases.append(row)
        print(
            f"  [{row['case']}] {row['paper_id']} p{row['page_no']} | "
            f"base IoU={iou_b:.3f} err={row['baseline']['mean_corner_err_px']}px left={left_cov_b:.3f} | "
            f"v2 IoU={iou_v2:.3f} err={row['v2']['mean_corner_err_px']}px left={left_cov_v2:.3f} ({ms_v2}ms)"
        )

    summary = {
        "baseline_mean_iou": round(float(np.mean([c["baseline"]["iou"] for c in cases])), 4),
        "baseline_min_iou": round(float(np.min([c["baseline"]["iou"] for c in cases])), 4),
        "baseline_mean_corner_err_px": round(float(np.mean([c["baseline"]["mean_corner_err_px"] for c in cases])), 1),
        "baseline_max_corner_err_px": round(float(np.max([c["baseline"]["max_corner_err_px"] for c in cases])), 1),
        "baseline_min_left_margin_cov": round(float(np.min([c["baseline"]["left_margin_x0_110_cov"] for c in cases])), 4),
        "baseline_mean_extra_bg_ratio": round(float(np.mean([c["baseline"]["extra_bg_ratio"] for c in cases])), 4),
        "baseline_mean_ms": round(float(np.mean([c["baseline"]["runtime_ms"] for c in cases])), 2),
        "v2_mean_iou": round(float(np.mean([c["v2"]["iou"] for c in cases])), 4),
        "v2_min_iou": round(float(np.min([c["v2"]["iou"] for c in cases])), 4),
        "v2_mean_corner_err_px": round(float(np.mean([c["v2"]["mean_corner_err_px"] for c in cases])), 1),
        "v2_max_corner_err_px": round(float(np.max([c["v2"]["max_corner_err_px"] for c in cases])), 1),
        "v2_min_left_margin_cov": round(float(np.min([c["v2"]["left_margin_x0_110_cov"] for c in cases])), 4),
        "v2_mean_extra_bg_ratio": round(float(np.mean([c["v2"]["extra_bg_ratio"] for c in cases])), 4),
        "v2_mean_ms": round(float(np.mean([c["v2"]["runtime_ms"] for c in cases])), 2),
    }
    return {"summary": summary, "cases": cases}


# ==============================================================================
# 实验 2：OCR 分辨率与 JPEG 质量速度/准确率实测 (127.0.0.1:8002)
# ==============================================================================

async def run_exp2_ocr_speed(gt_lib: TemplateLibrary, clients: ModelClients) -> dict:
    print("\n=== 实验 2：PaddleOCR-VL-1.5 分辨率与 JPEG 质量速度/准确率测试 ===")
    s = clients.settings

    # Warmup 1 次避免首包 JIT 干扰
    w_img = cv2.imread(str(OCR_BENCH_IMAGES[0][1]))
    w_warp, _, _, w_pw, w_ph = rectify_paper_v2(w_img)
    _, w_enc = cv2.imencode(".jpg", cv2.resize(w_warp, (round(w_pw * 1280 / max(w_pw, w_ph)), round(w_ph * 1280 / max(w_pw, w_ph)))), [cv2.IMWRITE_JPEG_QUALITY, 90])
    await clients.http.post(
        s.ocr_base_url.rstrip("/") + "/chat/completions",
        json={
            "model": s.ocr_model,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(w_enc.tobytes()).decode()}}, {"type": "text", "text": "Spotting:"}]}],
            "max_tokens": 64,
            "temperature": 0.0,
        },
        timeout=35.0,
    )

    configs = [
        (1540, 90),
        (1540, 75),
        (1280, 90),
        (1280, 75),
        (1120, 90),
        (1120, 75),
        (960, 90),
        (960, 75),
    ]

    results = []
    for label, ipath in OCR_BENCH_IMAGES:
        img = cv2.imread(str(ipath))
        gt_match = gt_lib.match(img)
        gt_qs = {str(q["number"]): q for q in gt_match.template.page["questions"]}
        warped, _, _, pw, ph = rectify_paper_v2(img)

        for long_edge, jpeg_q in configs:
            scale = min(1.0, float(long_edge) / max(pw, ph))
            ow, oh = round(pw * scale), round(ph * scale)
            ocr_img = cv2.resize(warped, (ow, oh), interpolation=cv2.INTER_AREA) if scale < 1.0 else warped
            ok, enc = cv2.imencode(".jpg", ocr_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            raw_bytes = enc.tobytes()
            b64_url = "data:image/jpeg;base64," + base64.b64encode(raw_bytes).decode()

            t0 = perf_counter()
            resp = await clients.http.post(
                s.ocr_base_url.rstrip("/") + "/chat/completions",
                json={
                    "model": s.ocr_model,
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64_url}}, {"type": "text", "text": "Spotting:"}]}],
                    "max_tokens": 2800,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=45.0,
            )
            elapsed_ms = round((perf_counter() - t0) * 1000, 1)
            data = resp.json()
            usage = data.get("usage") or {}
            raw_text = data["choices"][0]["message"]["content"]
            blocks = _parse_paddle_vl_blocks(raw_text, pw, ph)
            _, pred_page, _ = build_freeform_page_from_lines(blocks, warped, pw, ph)
            pred_qs = {str(q["number"]): q for q in pred_page["questions"]}

            matched = [n for n in gt_qs if n in pred_qs]
            recall = round(len(matched) / max(1, len(gt_qs)), 4)
            prec = round(len(matched) / max(1, len(pred_qs)), 4)
            f1 = round((2 * recall * prec / (recall + prec)) if (recall + prec) > 0 else 0.0, 4)

            stem_sims = []
            formula_sims = []
            for n in matched:
                gq, pq = gt_qs[n], pred_qs[n]
                gt_full = gq["stem"] + (" " + " ".join(gq.get("options") or []) if gq.get("options") else "")
                pred_full = pq["stem"] + (" " + " ".join(pq.get("options") or []) if pq.get("options") else "")
                stem_sims.append(difflib.SequenceMatcher(None, normalize_text(gt_full), normalize_text(pred_full)).ratio())
                if label == "math_p1":
                    f_gt = extract_math_segments(gt_full)
                    f_pr = extract_math_segments(pred_full)
                    if f_gt:
                        formula_sims.append(difflib.SequenceMatcher(None, f_gt, f_pr).ratio())

            comp_toks = int(usage.get("completion_tokens") or 0)
            prompt_toks = int(usage.get("prompt_tokens") or 0)
            tok_per_s = round(comp_toks / max(0.001, elapsed_ms / 1000.0), 1)

            row = {
                "label": label,
                "image": ipath.name,
                "long_edge": long_edge,
                "jpeg_quality": jpeg_q,
                "ocr_wh": [ow, oh],
                "jpeg_kb": round(len(raw_bytes) / 1024.0, 1),
                "latency_ms": elapsed_ms,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "decode_tok_per_s_e2e": tok_per_s,
                "gt_q_count": len(gt_qs),
                "pred_q_count": len(pred_qs),
                "matched_q_count": len(matched),
                "missing_nums": [n for n in gt_qs if n not in pred_qs],
                "q_recall": recall,
                "q_f1": f1,
                "mean_stem_sim": round(float(np.mean(stem_sims)) if stem_sims else 0.0, 4),
                "mean_formula_char_acc": round(float(np.mean(formula_sims)) if formula_sims else 0.0, 4) if label == "math_p1" else None,
            }
            results.append(row)
            f_str = f" formula={row['mean_formula_char_acc']}" if row["mean_formula_char_acc"] is not None else ""
            print(
                f"  [{label}] {long_edge}@q{jpeg_q} ({ow}x{oh}, {row['jpeg_kb']}KB): "
                f"{elapsed_ms}ms | p_tok={prompt_toks} c_tok={comp_toks} ({tok_per_s} tok/s) | "
                f"Q={len(matched)}/{len(gt_qs)} F1={f1} stem={row['mean_stem_sim']}{f_str}"
            )

    # 按配置汇总均值
    by_cfg = {}
    for long_edge, jpeg_q in configs:
        sub = [r for r in results if r["long_edge"] == long_edge and r["jpeg_quality"] == jpeg_q]
        math_row = next((r for r in sub if r["label"] == "math_p1"), None)
        by_cfg[f"{long_edge}_q{jpeg_q}"] = {
            "avg_latency_ms": round(float(np.mean([r["latency_ms"] for r in sub])), 1),
            "avg_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in sub])), 1),
            "avg_completion_tokens": round(float(np.mean([r["completion_tokens"] for r in sub])), 1),
            "avg_q_recall": round(float(np.mean([r["q_recall"] for r in sub])), 4),
            "avg_q_f1": round(float(np.mean([r["q_f1"] for r in sub])), 4),
            "avg_stem_sim": round(float(np.mean([r["mean_stem_sim"] for r in sub])), 4),
            "math_formula_char_acc": math_row["mean_formula_char_acc"] if math_row else None,
        }

    # 读取 start_ocr.sh 与 ocr.log / hwr.log 的启动配置信息
    ocr_log = (ROOT / "deploy/logs/ocr.log").read_text(errors="ignore")
    hwr_log = (ROOT / "deploy/logs/hwr.log").read_text(errors="ignore")
    start_ocr_sh = (ROOT / "deploy/start_ocr.sh").read_text(errors="ignore")
    deploy_notes = {
        "start_ocr_flags": [ln.strip() for ln in start_ocr_sh.splitlines() if "--" in ln],
        "ocr_log_observations": {
            "enforce_eager": "--enforce-eager" in start_ocr_sh,
            "gpu_memory_utilization": 0.16,
            "kv_cache_bytes": 536870912,
            "model_weights_gib": 1.82,
            "initial_free_mem_at_ocr_start_gib": 16.41,
            "encoder_cache_tokens_budget": 8192,
            "flash_attn_used": "Using FlashAttention version 2" in ocr_log,
            "cudagraph_disabled": "Cudagraph is disabled under eager mode" in ocr_log,
        },
        "hwr_cudagraph_overhead_from_hwr_log": "Graph capturing finished in 1 secs, took 0.09 GiB" if "took 0.09 GiB" in hwr_log else "unknown",
    }
    return {"by_config": by_cfg, "runs": results, "deploy_analysis": deploy_notes}


# ==============================================================================
# 实验 3：空白题 HWR 误报/uncertain 审计 & 数学手写题错题几何根因审计
# ==============================================================================

def _crop_features(crop_bgr: np.ndarray, ans_bbox: list[int], ocr_blocks: list[dict], pad: int = 14) -> dict:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]
    ax, ay, aw, ah = ans_bbox
    cx0, cy0 = max(0, ax - pad), max(0, ay - pad)
    cx1, cy1 = cx0 + cw, cy0 + ch

    # 自适应二值墨迹像素统计（区分边框 14px 与中心区域）
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 13) > 0
    dark120 = gray < 120
    border_mask = np.zeros_like(bw)
    b_t = min(14, ch // 4)
    b_w = min(14, cw // 4)
    border_mask[:b_t, :] = True
    border_mask[ch - b_t :, :] = True
    border_mask[:, :b_w] = True
    border_mask[:, cw - b_w :] = True
    interior_mask = ~border_mask

    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(bw.astype(np.uint8))
    cc_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_cc) if int(stats[i, cv2.CC_STAT_AREA]) >= 4]

    # 检查裁剪区是否与 PaddleOCR 识别出的印刷文本块重叠
    overlapping_texts = []
    for blk in ocr_blocks:
        bx0, by0, bx1, by1 = blk["box"]
        ix0, iy0 = max(cx0, bx0), max(cy0, by0)
        ix1, iy1 = min(cx1, bx1), min(cy1, by1)
        if ix1 > ix0 and iy1 > iy0:
            inter_area = (ix1 - ix0) * (iy1 - iy0)
            if inter_area >= 12:
                overlapping_texts.append({"text": blk["text"][:40], "box": blk["box"], "inter_px": int(inter_area)})

    return {
        "crop_wh": [int(cw), int(ch)],
        "crop_canvas_box": [int(cx0), int(cy0), int(cx1), int(cy1)],
        "min_gray": int(gray.min()),
        "p5_gray": round(float(np.percentile(gray, 5)), 1),
        "p50_gray": round(float(np.percentile(gray, 50)), 1),
        "dark120_total": int(np.count_nonzero(dark120)),
        "adaptive_bw_total": int(np.count_nonzero(bw)),
        "adaptive_bw_border": int(np.count_nonzero(bw & border_mask)),
        "adaptive_bw_interior": int(np.count_nonzero(bw & interior_mask)),
        "cc_ge4_count": len(cc_areas),
        "max_cc_area": max(cc_areas) if cc_areas else 0,
        "overlapping_ocr_blocks": overlapping_texts,
    }


def _fast_blank_gate_decision(
    warped: np.ndarray, ans_bbox: list[int], ocr_blocks: list[dict]
) -> tuple[bool, int]:
    """原型无模板快速空白门控：
    在 ans_bbox 内部（不外扩 pad=14 吃到题号），mask 掉与已知 OCR 印刷块重叠的区域后，
    检查是否存在显著深色墨迹连通域（area>=8 且局部对比度>28）。返回 (is_blank, ink_px)。
    """
    ph, pw = warped.shape[:2]
    ax, ay, aw, ah = ans_bbox
    x0, y0 = max(0, int(ax)), max(0, int(ay))
    x1, y1 = min(pw, int(ax + aw)), min(ph, int(ay + ah))
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return True, 0
    sub = cv2.cvtColor(warped[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 16) > 0
    # 屏蔽掉任何与 OCR 印刷框重叠的像素（如题号、印刷括号边）
    for blk in ocr_blocks:
        bx0, by0, bx1, by1 = blk["box"]
        # 仅屏蔽印刷块（非单字母手写块）
        t_unw = blk["text"].strip().strip("$").strip()
        if re.fullmatch(r"[A-Ha-h]", t_unw) and (bx1 - bx0) < pw * 0.08:
            continue
        ix0, iy0 = max(x0, bx0 - 2), max(y0, by0 - 2)
        ix1, iy1 = min(x1, bx1 + 2), min(y1, by1 + 2)
        if ix1 > ix0 and iy1 > iy0:
            bw[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = False
    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(bw.astype(np.uint8))
    ink_px = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_cc) if int(stats[i, cv2.CC_STAT_AREA]) >= 8)
    return ink_px < 12, int(ink_px)


async def run_exp3_hwr_and_blank_audit(gt_lib: TemplateLibrary, clients: ModelClients) -> dict:
    print("\n=== 实验 3：空白门控误判与手写数学错题定位根因审计 ===")
    s = clients.settings
    hwr = HWRClient(clients)

    blank_cases = []
    blank_problem_crops = []
    written_cases = []

    all_blank_count = 0
    all_blank_hwr_ms = []
    gate_blank_skipped = 0
    gate_written_kept = 0

    for jpath in sorted(DATA_DIR.glob("rec_*/cap_*.json")):
        ipath = jpath.with_suffix(".jpg")
        obs = json.loads(jpath.read_text())
        paper_id = obs["capture"]["paper_id"]
        page_no = obs["capture"]["page_no"]
        is_written = paper_id == "math_xmyz_2025_oct" and page_no == 1

        img = cv2.imread(str(ipath))
        gt_match = gt_lib.match(img)
        gt_qs = {str(q["number"]): q for q in gt_match.template.page["questions"]}

        # 使用当前生产 rectify_paper + PaddleOCR-VL-1.5 + build_freeform_page_from_lines 完整复现
        warped, H_free, q_free, pw, ph = rectify_paper(img)
        scale = min(1.0, 1540.0 / max(pw, ph))
        ocr_img = cv2.resize(warped, (round(pw * scale), round(ph * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else warped
        _, enc = cv2.imencode(".jpg", ocr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()
        resp = await clients.http.post(
            s.ocr_base_url.rstrip("/") + "/chat/completions",
            json={
                "model": s.ocr_model,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64}}, {"type": "text", "text": "Spotting:"}]}],
                "max_tokens": 2800,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=35.0,
        )
        raw_ocr = resp.json()["choices"][0]["message"]["content"]
        ocr_blocks = _parse_paddle_vl_blocks(raw_ocr, pw, ph)
        _, pred_page, crops = build_freeform_page_from_lines(ocr_blocks, warped, pw, ph)

        # 将 GT answer_areas 投影到当前 freeform warped 坐标系，供几何对比
        H_tpl_to_warped = np.linalg.inv(H_free) @ gt_match.homography

        t_hwr0 = perf_counter()

        async def _eval_q(pq: dict):
            qid = pq["id"]
            cr = crops[qid]
            if pq["type"] == "listening":
                return qid, "", False, "listening", 0.0
            if cr.get("is_blank"):
                return qid, "", False, "blank", 0.0
            t0 = perf_counter()
            blocks = cr.get("blocks") or []
            raw, low = ("", False)
            if blocks:
                raw, low = await hwr.recognize(blocks, pq["type"], freeform=True)
            pre = str(cr.get("pre_ocr_handwriting") or "").strip()
            if not raw.strip() and pre:
                raw, low = pre, False
            if not raw.strip() and pq["type"] in ("solution", "fill", "writing") and cr.get("question_jpeg"):
                raw2, low2 = await hwr.recognize(
                    [{"jpeg": cr["question_jpeg"], "hint": "full_question", "area_index": 0, "is_blank": False, "ink_pixels": 64}],
                    pq["type"],
                    freeform=True,
                )
                if raw2.strip():
                    raw, low = raw2, low2
            qms = round((perf_counter() - t0) * 1000, 1)
            st = ("uncertain" if low else "blank") if not raw.strip() else ("uncertain" if low else "recognized")
            return qid, raw, low, st, qms

        hwr_res = await asyncio.gather(*(_eval_q(pq) for pq in pred_page["questions"]))
        hwr_wall_ms = round((perf_counter() - t_hwr0) * 1000, 1)
        res_by_id = {x[0]: x[1:] for x in hwr_res}

        if not is_written:
            st_counts = {"blank": 0, "uncertain": 0, "recognized_fp": 0}
            for pq in pred_page["questions"]:
                all_blank_count += 1
                raw, low, st, qms = res_by_id[pq["id"]]
                all_blank_hwr_ms.append(qms)
                ab = pq["answer_areas"][0]["bbox"]
                gate_blank, gate_ink = _fast_blank_gate_decision(warped, ab, ocr_blocks)
                if gate_blank:
                    gate_blank_skipped += 1
                if st == "blank":
                    st_counts["blank"] += 1
                elif st == "uncertain":
                    st_counts["uncertain"] += 1
                else:
                    st_counts["recognized_fp"] += 1

                if st != "blank":
                    b0 = crops[pq["id"]]["blocks"][0]["jpeg"]
                    arr = cv2.imdecode(np.frombuffer(b0, np.uint8), cv2.IMREAD_COLOR)
                    feats = _crop_features(arr, ab, ocr_blocks)
                    blank_problem_crops.append(
                        {
                            "case": f"{jpath.parent.name}/{ipath.name}",
                            "paper_id": paper_id,
                            "page_no": page_no,
                            "qno": pq["number"],
                            "qtype": pq["type"],
                            "hwr_status": st,
                            "pred_ans": raw,
                            "low_conf": low,
                            "hwr_ms": qms,
                            "ans_bbox": ab,
                            "gate_would_skip": gate_blank,
                            "gate_ink_px": gate_ink,
                            "crop_features": feats,
                        }
                    )
            blank_cases.append(
                {
                    "case": f"{jpath.parent.name}/{ipath.name}",
                    "paper_id": paper_id,
                    "page_no": page_no,
                    "pred_q_count": len(pred_page["questions"]),
                    "hwr_wall_ms": hwr_wall_ms,
                    **st_counts,
                }
            )
            print(
                f"  [BLANK {jpath.parent.name}/{ipath.name}] {paper_id} p{page_no}: "
                f"Q={len(pred_page['questions'])} blank={st_counts['blank']} uncertain={st_counts['uncertain']} "
                f"fp={st_counts['recognized_fp']} HWR={hwr_wall_ms}ms"
            )
        else:
            # 详细审计手写数学卷 Q1~Q7：对比 GT answer_area 投影框 vs page_ocr ans_bbox 及排序/切题行为
            # 同时追踪 valid.sort 与 seg.sort 索引定位问题
            footer_y = int(ph * 0.955)
            valid = [l for l in ocr_blocks if l.get("text", "").strip() and l["box"][1] < footer_y]
            valid_sorted = sorted(valid, key=lambda l: (l["box"][1] // 22, l["box"][0]))

            q_details = []
            for pq in pred_page["questions"]:
                qno = str(pq["number"])
                raw, low, st, qms = res_by_id[pq["id"]]
                ab = pq["answer_areas"][0]["bbox"]
                ab_x1y1 = [ab[0], ab[1], ab[0] + ab[2], ab[1] + ab[3]]
                gate_blank, gate_ink = _fast_blank_gate_decision(warped, ab, ocr_blocks)
                if not gate_blank:
                    gate_written_kept += 1

                gt_ans_warped = None
                ans_iou = None
                center_dist = None
                if qno in gt_qs and gt_qs[qno].get("answer_areas"):
                    ga = gt_qs[qno]["answer_areas"][0]["bbox"]
                    pts = np.float32(project(quad(ga), H_tpl_to_warped))
                    gt_ans_warped = [
                        round(float(pts[:, 0].min()), 1),
                        round(float(pts[:, 1].min()), 1),
                        round(float(pts[:, 0].max()), 1),
                        round(float(pts[:, 1].max()), 1),
                    ]
                    ans_iou = round(box_iou(gt_ans_warped, ab_x1y1), 4)
                    gcx = 0.5 * (gt_ans_warped[0] + gt_ans_warped[2])
                    gcy = 0.5 * (gt_ans_warped[1] + gt_ans_warped[3])
                    pcx = 0.5 * (ab_x1y1[0] + ab_x1y1[2])
                    pcy = 0.5 * (ab_x1y1[1] + ab_x1y1[3])
                    center_dist = round(float(np.hypot(gcx - pcx, gcy - pcy)), 1)

                # 查找该题原始 OCR 块（包含题干行和独立手写字母行）
                near_blocks = [
                    {
                        "box": b["box"],
                        "y0_div22": b["box"][1] // 22,
                        "y0_div14": b["box"][1] // 14,
                        "text": b["text"],
                    }
                    for b in valid_sorted
                    if gt_ans_warped is not None and abs(0.5 * (b["box"][1] + b["box"][3]) - 0.5 * (gt_ans_warped[1] + gt_ans_warped[3])) < 80
                ]

                b0 = crops[pq["id"]]["blocks"][0]["jpeg"]
                arr = cv2.imdecode(np.frombuffer(b0, np.uint8), cv2.IMREAD_COLOR)
                feats = _crop_features(arr, ab, ocr_blocks)

                gt_ans = WRITTEN_MATH_P1.get(qno, "")
                q_details.append(
                    {
                        "qno": qno,
                        "gt_ans": gt_ans,
                        "pred_ans": raw,
                        "low_conf": low,
                        "hwr_status": st,
                        "exact_match": (st == "recognized" and raw.upper() == gt_ans),
                        "pre_ocr_handwriting": crops[pq["id"]].get("pre_ocr_handwriting", ""),
                        "pred_ans_bbox_xywh": ab,
                        "pred_ans_bbox_xyxy": ab_x1y1,
                        "gt_ans_bbox_in_warped_xyxy": gt_ans_warped,
                        "ans_bbox_iou": ans_iou,
                        "ans_bbox_center_dist_px": center_dist,
                        "stem_preview": pq["stem"][:100],
                        "near_ocr_blocks": near_blocks,
                        "overlapping_ocr_in_crop": feats["overlapping_ocr_blocks"],
                        "hwr_ms": qms,
                    }
                )

            written_cases.append(
                {
                    "case": f"{jpath.parent.name}/{ipath.name}",
                    "paper_id": paper_id,
                    "page_no": page_no,
                    "hwr_wall_ms": hwr_wall_ms,
                    "correct_count": sum(1 for d in q_details if d["exact_match"]),
                    "questions": q_details,
                }
            )
            print(
                f"  [WRITTEN {jpath.parent.name}/{ipath.name}] {paper_id} p{page_no}: "
                f"correct={written_cases[-1]['correct_count']}/7 HWR={hwr_wall_ms}ms"
            )
            for d in q_details:
                print(
                    f"    Q{d['qno']}: GT={d['gt_ans']} pred={d['pred_ans']!r}(low={d['low_conf']}) "
                    f"pre={d['pre_ocr_handwriting']!r} ans_IoU={d['ans_bbox_iou']} dist={d['ans_bbox_center_dist_px']}px"
                )

    summary = {
        "blank_pages_count": len(blank_cases),
        "blank_questions_observed": all_blank_count,
        "blank_exact": sum(c["blank"] for c in blank_cases),
        "blank_uncertain": sum(c["uncertain"] for c in blank_cases),
        "blank_false_positive": sum(c["recognized_fp"] for c in blank_cases),
        "blank_avg_hwr_wall_ms": round(float(np.mean([c["hwr_wall_ms"] for c in blank_cases])), 1),
        "blank_total_hwr_wall_ms": round(sum(c["hwr_wall_ms"] for c in blank_cases), 1),
        "fast_gate_blank_skip_count": gate_blank_skipped,
        "fast_gate_blank_skip_rate": round(gate_blank_skipped / max(1, all_blank_count), 4),
        "written_questions_total": 14,
        "written_correct_baseline": sum(c["correct_count"] for c in written_cases),
    }
    return {
        "summary": summary,
        "blank_pages": blank_cases,
        "blank_problem_crops": blank_problem_crops,
        "written_math_pages": written_cases,
    }


async def main():
    gt_lib = TemplateLibrary(ROOT / "data/templates")
    gt_lib.load()

    with tempfile.TemporaryDirectory() as empty_dir:
        settings = Settings(
            templates_dir=Path(empty_dir),
            freeform_ocr_enabled=True,
            ocr_base_url="http://127.0.0.1:8002/v1",
            ocr_model="PaddleOCR-VL-1.5",
            hwr_base_url="http://127.0.0.1:8001/v1",
            hwr_model="Qwen3.5-2B",
            hwr_api_key="",
        )
        clients = ModelClients(settings)
        await clients.warmup()
        try:
            exp1 = run_exp1_rectify_audit(gt_lib)
            exp2 = await run_exp2_ocr_speed(gt_lib, clients)
            exp3 = await run_exp3_hwr_and_blank_audit(gt_lib, clients)
        finally:
            await clients.http.aclose()

    report = {
        "exp1_rectify_audit": exp1,
        "exp2_ocr_speed_resolution": exp2,
        "exp3_blank_gate_and_hwr_audit": exp3,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
