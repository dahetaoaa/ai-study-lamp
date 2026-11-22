"""无模板自由试卷答题区毫秒级手写墨迹门控与智能裁剪。

在调用本地 Qwen3.5-2B HWR 之前，先在每个候选答题框内剔除：
1. 纸张边缘阴影与左侧题号（left_margin 右界严格不超过 first_x0 - 3）；
2. PaddleOCR-VL-1.5 已知印刷文本行框（保留独立单字母手写框与括号内部）；
3. 印刷水平/微倾斜横线及表格/坐标轴长直线；
4. 紧贴裁剪边缘的残缺印刷字符与孤立噪点。

仅当干净有效手写墨迹超过安全阈值（或框内存在可靠手写单字母块）时才生成软去边裁剪图送往 HWR；
可靠空白答题区直接标记 is_blank=True，0ms 跳过 HWR 推理并杜绝幻觉误判。
"""

from __future__ import annotations

from functools import lru_cache
import math
import re

import cv2
import numpy as np

DEFAULT_INK_THRESHOLD = 35
PAPER_EDGE_MARGIN_PX = 12

_SINGLE_HW_LETTER_RE = re.compile(r"^[A-Ha-h]$")
_PRINTED_SHORT_TOKEN_RE = re.compile(r"^(?:\d{1,3}[.．]?|[?？!！。，,;；:：\"'“”‘’]+)$")
_CJK_RE = re.compile(r"[一-鿿]")
_BRACKET_SPAN_RE = re.compile(r"[（(]\s*[A-Ha-h]?\s*[）)]")
_OPEN_BRACKET_TAIL_RE = re.compile(r"[（(]\s*$")


@lru_cache(maxsize=32)
def _line_kernels(length: int) -> tuple[np.ndarray, ...]:
    """预计算并缓存 -15°~+15° 直线开运算核，消除微倾斜下划线残留。"""
    kernels: list[np.ndarray] = []
    for deg in (-15, -10, -5, 0, 5, 10, 15):
        rad = math.radians(deg)
        dx = math.cos(rad) * (length * 0.5)
        dy = math.sin(rad) * (length * 0.5)
        h = max(3, int(math.ceil(abs(dy) * 2)) + 3)
        w = max(3, int(math.ceil(abs(dx) * 2)) + 3)
        k = np.zeros((h, w), np.uint8)
        cx, cy = w // 2, h // 2
        p1 = (int(round(cx - dx)), int(round(cy - dy)))
        p2 = (int(round(cx + dx)), int(round(cy + dy)))
        cv2.line(k, p1, p2, 1, 2)
        kernels.append(k)
    return tuple(kernels)


def _normalize_ocr_box(blk: dict | list | tuple) -> tuple[list[int], str] | None:
    if isinstance(blk, dict):
        box = blk.get("box") or blk.get("bbox")
        text = str(blk.get("text") or "")
    elif isinstance(blk, (list, tuple)) and len(blk) >= 4:
        box = blk[:4]
        text = ""
    else:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    bx0, by0, bx1, by1 = [int(round(float(v))) for v in box]
    if bx1 <= bx0 or by1 <= by0:
        return None
    return [bx0, by0, bx1, by1], text


def is_single_handwriting_letter_box(blk: dict | list | tuple, page_w: int) -> bool:
    """判断是否为 Spotting 独立检出的手写单字母框（如 '$ B $'），避免被当作印刷体抹除。"""
    parsed = _normalize_ocr_box(blk)
    if parsed is None:
        return False
    (bx0, _, bx1, _), text = parsed
    t_unwrapped = text.strip().strip("$").strip()
    return bool(_SINGLE_HW_LETTER_RE.fullmatch(t_unwrapped)) and (bx1 - bx0) < page_w * 0.08


def _printed_erase_spans(box: list[int], text: str) -> list[list[int]]:
    """计算一个印刷 OCR 框中应当抹除的矩形区间；含括号的题干只抹除括号外侧印刷文字。"""
    bx0, by0, bx1, by1 = box
    if not text:
        return [[bx0 - 2, by0 - 2, bx1 + 2, by1 + 2]]
    bm = _BRACKET_SPAN_RE.search(text)
    if bm:
        n_chars = max(1, len(text))
        left_x1 = int(bx0 + (bx1 - bx0) * (bm.start() / n_chars))
        right_x0 = int(bx0 + (bx1 - bx0) * (bm.end() / n_chars))
        spans: list[list[int]] = []
        if left_x1 > bx0 + 2:
            spans.append([bx0 - 2, by0 - 2, left_x1, by1 + 2])
        if bx1 > right_x0 + 4:
            spans.append([right_x0, by0 - 2, bx1 + 2, by1 + 2])
        return spans
    om = _OPEN_BRACKET_TAIL_RE.search(text)
    if om:
        n_chars = max(1, len(text))
        left_x1 = int(bx0 + (bx1 - bx0) * (om.start() / n_chars))
        if left_x1 > bx0 + 2:
            return [[bx0 - 2, by0 - 2, left_x1, by1 + 2]]
        return []
    return [[bx0 - 2, by0 - 2, bx1 + 2, by1 + 2]]


def _extract_clean_handwriting_mask(
    gray: np.ndarray,
    box_xyxy: list[int],
    qtype: str,
    parsed_ocr: list[tuple[list[int], str, bool]],
) -> tuple[np.ndarray, int, int, list[int] | None]:
    """在指定框内构建净手写掩码，返回 (clean_mask, clean_pixels, cc_count, ink_bbox_xywh)。"""
    x0, y0, x1, y1 = box_xyxy
    sub = gray[y0:y1, x0:x1]
    sh, sw = sub.shape
    if sh < 4 or sw < 4:
        return np.zeros((max(1, sh), max(1, sw)), np.uint8), 0, 0, None

    # 局部光照归一化 + 对比度二值化（同时兼容真实拍照偏灰纸面与纯白合成图）
    bg = cv2.GaussianBlur(sub, (31, 31), 0)
    diff = bg.astype(np.int16) - sub.astype(np.int16)
    bw = ((diff > 18) & (sub < 235)).astype(np.uint8) * 255

    # 1. 擦除印刷水平/微倾斜横线（-15°~+15°）
    h_lines_dil = np.zeros_like(bw)
    hk_len = max(24, min(44, int(max(sw, 60) * 0.35)))
    if sw >= hk_len:
        bw_bridge = cv2.dilate(bw, np.ones((2, 2), np.uint8))
        h_lines = np.zeros_like(bw)
        for kernel in _line_kernels(hk_len):
            if kernel.shape[0] <= sh and kernel.shape[1] <= sw:
                h_lines = cv2.bitwise_or(h_lines, cv2.morphologyEx(bw_bridge, cv2.MORPH_OPEN, kernel))
        h_lines_dil = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
        bw[h_lines_dil > 0] = 0
    h_tip_zone = cv2.dilate(h_lines_dil, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))) > 0

    # 2. 擦除贯穿全高的细垂直表格线/坐标轴线（保留字母 B/D 的竖笔）
    vk_len = max(36, sh - 4)
    if sh >= vk_len:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk_len))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
        v_n, v_lbl, v_st, _ = cv2.connectedComponentsWithStats((v_lines > 0).astype(np.uint8))
        v_mask = np.zeros_like(bw)
        for vi in range(1, v_n):
            if v_st[vi, cv2.CC_STAT_WIDTH] <= 6 and v_st[vi, cv2.CC_STAT_HEIGHT] >= sh - 6:
                v_mask[v_lbl == vi] = 255
        if np.any(v_mask):
            v_mask = cv2.dilate(v_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
            bw[v_mask > 0] = 0

    # 3. 擦除已知印刷 OCR 文本区域（保留独立单字母手写框、括号内部及框内窄符号候选）
    for box, text, is_hw_letter in parsed_ocr:
        if is_hw_letter:
            continue
        if box[2] <= x0 or box[0] >= x1 or box[3] <= y0 or box[1] >= y1:
            continue
        # 若窄 OCR 框完全/主要落入答题框内部且非纯题号/中文/标点（例如将手写 C 误识为 '$ \odot $'），不作印刷擦除
        bw_ocr = box[2] - box[0]
        bcx = 0.5 * (box[0] + box[2])
        bcy = 0.5 * (box[1] + box[3])
        t_clean = text.strip().strip("$").strip()
        if (
            bw_ocr < gray.shape[1] * 0.055
            and (x0 - 4) <= bcx <= (x1 + 4)
            and (y0 - 4) <= bcy <= (y1 + 4)
            and t_clean
            and not _PRINTED_SHORT_TOKEN_RE.fullmatch(t_clean)
            and not _CJK_RE.search(t_clean)
        ):
            continue
        for ex0, ey0, ex1, ey1 in _printed_erase_spans(box, text):
            ix0, iy0 = max(x0, ex0), max(y0, ey0)
            ix1, iy1 = min(x1, ex1), min(y1, ey1)
            if ix1 > ix0 and iy1 > iy0:
                bw[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = 0

    # 4. 连通域滤波：剔除孤立小噪点、横线残端、贴边残切印刷字符及纯印刷圆括号弧
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8))
    clean_mask = np.zeros_like(bw)
    kept_px = 0
    kept_cnt = 0
    min_x, min_y = sw, sh
    max_x, max_y = -1, -1

    for i in range(1, n_cc):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP])
        cw_c = int(stats[i, cv2.CC_STAT_WIDTH])
        ch_c = int(stats[i, cv2.CC_STAT_HEIGHT])
        mcx, mcy = float(centroids[i][0]), float(centroids[i][1])

        if area < 10:
            continue
        if ch_c <= 4 and cw_c <= 14 and area <= 28:
            continue
        if ch_c <= 5 and cw_c >= 28:
            continue
        if ch_c <= 9 and np.any(h_tip_zone[cy : cy + ch_c, cx : cx + cw_c] & (labels[cy : cy + ch_c, cx : cx + cw_c] == i)):
            continue
        touches_edge = cx <= 1 or cy <= 1 or (cx + cw_c) >= sw - 1 or (cy + ch_c) >= sh - 1
        if touches_edge and (area < 65 or cw_c <= 5 or ch_c <= 5):
            continue
        if (cy <= 2 and mcy < 0.20 * sh) or (cy + ch_c >= sh - 2 and mcy > 0.88 * sh):
            continue
        if (cx <= 2 and mcx < 0.12 * sw) or (cx + cw_c >= sw - 2 and mcx > 0.90 * sw):
            continue
        if qtype == "choice" and cw_c <= 11 and 10 <= ch_c <= 32 and area <= 110 and ch_c >= cw_c * 1.7:
            continue
        if (
            qtype == "fill"
            and cw_c <= 8
            and 12 <= ch_c <= 24
            and area <= 70
            and (cx >= sw * 0.80 or (cx + cw_c) <= sw * 0.20)
        ):
            continue

        clean_mask[labels == i] = 255
        kept_px += area
        kept_cnt += 1
        min_x = min(min_x, cx)
        min_y = min(min_y, cy)
        max_x = max(max_x, cx + cw_c)
        max_y = max(max_y, cy + ch_c)

    ink_bbox = [x0 + min_x, y0 + min_y, max_x - min_x, max_y - min_y] if kept_cnt > 0 else None
    return clean_mask, kept_px, kept_cnt, ink_bbox


def _has_reliable_pre_ocr_hw_in_box(
    pre_ocr_handwriting: str,
    box_xyxy: list[int],
    parsed_ocr: list[tuple[list[int], str, bool]],
    clean_ink_pixels: int,
    ink_threshold: int,
) -> bool:
    """仅当 pre_ocr_handwriting 有对应位置的单字母块或框内真实墨迹支撑时，才视为可靠。"""
    pre = (pre_ocr_handwriting or "").strip().upper()
    if not pre:
        return False
    if clean_ink_pixels >= ink_threshold:
        return True
    x0, y0, x1, y1 = box_xyxy
    for box, text, is_hw in parsed_ocr:
        if not is_hw:
            continue
        t_unwrapped = text.strip().strip("$").strip().upper()
        if t_unwrapped != pre:
            continue
        bcx = 0.5 * (box[0] + box[2])
        bcy = 0.5 * (box[1] + box[3])
        if (x0 - 8) <= bcx <= (x1 + 8) and (y0 - 8) <= bcy <= (y1 + 8):
            return True
    return False


def _soft_clean_crop_borders(
    crop: np.ndarray,
    crop_x0: int,
    crop_y0: int,
    crop_x1: int,
    crop_y1: int,
    parsed_ocr: list[tuple[list[int], str, bool]],
    ink_bbox: list[int] | None,
) -> np.ndarray:
    """将紧贴裁剪块边缘、已越界侵入的印刷文字区域涂白，避免干扰 HWR 模型。"""
    if not parsed_ocr:
        return crop
    ch, cw = crop.shape[:2]
    safe_x0 = safe_y0 = safe_x1 = safe_y1 = -1
    if ink_bbox is not None:
        ix, iy, iw, ih = ink_bbox
        safe_x0, safe_y0 = ix - 4, iy - 4
        safe_x1, safe_y1 = ix + iw + 4, iy + ih + 4

    for box, text, is_hw_letter in parsed_ocr:
        if is_hw_letter:
            continue
        if box[2] <= crop_x0 or box[0] >= crop_x1 or box[3] <= crop_y0 or box[1] >= crop_y1:
            continue
        t_clean = text.strip().strip("$").strip()
        if (
            (box[2] - box[0]) < (crop_x1 - crop_x0) * 0.75
            and t_clean
            and not _PRINTED_SHORT_TOKEN_RE.fullmatch(t_clean)
            and not _CJK_RE.search(t_clean)
            and ink_bbox is not None
            and (ink_bbox[0] - 4) <= 0.5 * (box[0] + box[2]) <= (ink_bbox[0] + ink_bbox[2] + 4)
        ):
            continue
        for ex0, ey0, ex1, ey1 in _printed_erase_spans(box, text):
            rx0 = max(crop_x0, ex0)
            ry0 = max(crop_y0, ey0)
            rx1 = min(crop_x1, ex1)
            ry1 = min(crop_y1, ey1)
            if rx1 <= rx0 or ry1 <= ry0:
                continue
            # 若待涂白区域与有效手写 ink_bbox 垂直重叠，只涂白 ink_bbox 外侧部分
            if ink_bbox is not None and min(ry1, safe_y1) > max(ry0, safe_y0):
                if rx0 < safe_x0 < rx1:
                    rx1 = min(rx1, safe_x0)
                elif rx0 < safe_x1 < rx1:
                    rx0 = max(rx0, safe_x1)
                elif rx0 >= safe_x0 and rx1 <= safe_x1:
                    continue
            if rx1 > rx0 and ry1 > ry0:
                lx0 = max(0, rx0 - crop_x0)
                ly0 = max(0, ry0 - crop_y0)
                lx1 = min(cw, rx1 - crop_x0)
                ly1 = min(ch, ry1 - crop_y0)
                if lx1 > lx0 and ly1 > ly0:
                    crop[ly0:ly1, lx0:lx1] = 255
    return crop


def prepare_freeform_answer_blocks(
    warped: np.ndarray,
    gray: np.ndarray | None,
    ans_areas: list[dict | list | tuple],
    qtype: str,
    ocr_text_boxes: list[dict | list | tuple] | None = None,
    pre_ocr_handwriting: str = "",
    *,
    first_x0: int | None = None,
    ink_threshold: int = DEFAULT_INK_THRESHOLD,
) -> dict:
    """无模板答题框手写墨迹门控与智能裁剪。

    返回结构与 cropper.crop_answers 单题输出对齐：
    {
        "blocks": [...],           # 若全部为空白则保留全区占位块，否则仅保留非空白块
        "all_blocks": [...],
        "is_blank": bool,
        "primary_area_index": int,
        "ink_bbox": [x, y, w, h] | None,
        "clean_ink_pixels": int,
        "area_diagnostics": [...],
    }
    """
    if warped is None or warped.size == 0:
        return {
            "blocks": [],
            "all_blocks": [],
            "is_blank": True,
            "primary_area_index": 0,
            "ink_bbox": None,
            "clean_ink_pixels": 0,
            "area_diagnostics": [],
        }

    if gray is None:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    page_h, page_w = gray.shape[:2]

    parsed_ocr: list[tuple[list[int], str, bool]] = []
    for blk in ocr_text_boxes or []:
        norm = _normalize_ocr_box(blk)
        if norm is None:
            continue
        box, text = norm
        parsed_ocr.append((box, text, is_single_handwriting_letter_box(blk, page_w)))

    all_blocks: list[dict] = []
    diagnostics: list[dict] = []

    for ai, area in enumerate(ans_areas or []):
        if isinstance(area, dict):
            bbox = area.get("bbox")
            hint = str(area.get("hint") or ("choice_letter" if qtype == "choice" else "math_steps"))
        else:
            bbox = area
            hint = "choice_letter" if qtype == "choice" else "math_steps"
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        ax, ay, aw, ah = [int(round(float(v))) for v in bbox]
        x0, y0 = max(0, ax), max(0, ay)
        x1, y1 = min(page_w, ax + aw), min(page_h, ay + ah)

        # 识别是否为题号左侧边距槽（left_margin）
        right_cands = [
            box[0]
            for box, _, is_hw in parsed_ocr
            if not is_hw and box[1] < y1 + 4 and box[3] > y0 - 4 and box[0] >= x0 + max(12, int(aw * 0.4))
        ]
        nearest_right_x0 = (
            first_x0
            if first_x0 is not None
            else (min(right_cands) if right_cands else None)
        )
        is_left_margin = (
            hint == "left_margin"
            or (
                qtype == "choice"
                and ax < page_w * 0.16
                and nearest_right_x0 is not None
                and x1 <= nearest_right_x0 + 18
            )
        )

        if is_left_margin:
            if nearest_right_x0 is not None:
                x1 = min(x1, int(nearest_right_x0) - 3)
            x0 = max(x0, min(PAPER_EDGE_MARGIN_PX, max(0, x1 - 16)))

        if x1 <= x0 + 4 or y1 <= y0 + 4:
            diagnostics.append(
                {
                    "index": ai,
                    "clean_ink_pixels": 0,
                    "cc_count": 0,
                    "is_blank": True,
                    "ink_bbox": None,
                    "left_margin": is_left_margin,
                }
            )
            continue

        _, clean_px, cc_cnt, ink_bbox = _extract_clean_handwriting_mask(
            gray, [x0, y0, x1, y1], qtype, parsed_ocr
        )
        reliable_pre = _has_reliable_pre_ocr_hw_in_box(
            pre_ocr_handwriting, [x0, y0, x1, y1], parsed_ocr, clean_px, ink_threshold
        )
        area_blank = (clean_px < ink_threshold) and not reliable_pre

        diagnostics.append(
            {
                "index": ai,
                "clean_ink_pixels": int(clean_px),
                "cc_count": int(cc_cnt),
                "is_blank": bool(area_blank),
                "ink_bbox": ink_bbox,
                "left_margin": is_left_margin,
            }
        )

        # 生成裁剪框（left_margin 严格不越过 first_x0 - 3 且避开外侧 12px 黑边）
        if is_left_margin:
            crop_x0 = max(PAPER_EDGE_MARGIN_PX, x0 - 4)
            crop_x1 = min(page_w, (int(nearest_right_x0) - 3) if nearest_right_x0 is not None else x1, x1 + 2)
            if crop_x1 <= crop_x0 + 8:
                crop_x0 = max(0, min(crop_x0, crop_x1 - 12))
            crop_y0 = max(0, y0 - 8)
            crop_y1 = min(page_h, y1 + 8)
        else:
            pad_x = 6 if aw > 400 else 8
            pad_y = 8
            crop_x0 = max(0 if ax >= PAPER_EDGE_MARGIN_PX else PAPER_EDGE_MARGIN_PX, x0 - pad_x)
            crop_x1 = min(page_w, x1 + pad_x)
            if crop_x1 <= crop_x0 + 8:
                crop_x0 = max(0, x0)
            crop_y0 = max(0, y0 - pad_y)
            crop_y1 = min(page_h, y1 + pad_y)

        crop = warped[crop_y0:crop_y1, crop_x0:crop_x1].copy()
        if not area_blank and crop.size > 0:
            crop = _soft_clean_crop_borders(
                crop, crop_x0, crop_y0, crop_x1, crop_y1, parsed_ocr, ink_bbox
            )

        ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94]) if crop.size > 0 else (False, None)
        if ok and enc is not None:
            all_blocks.append(
                {
                    "jpeg": enc.tobytes(),
                    "hint": hint,
                    "area_index": ai,
                    "alignment_uncertain": False,
                    "is_blank": bool(area_blank),
                    "ink_pixels": int(clean_px),
                    "ink_bbox": ink_bbox,
                }
            )

    nonblank_blocks = [b for b in all_blocks if not b["is_blank"]]
    all_blank = bool(diagnostics) and all(d["is_blank"] for d in diagnostics)
    if not diagnostics:
        all_blank = True

    primary = max(
        (d for d in diagnostics if not d["is_blank"]),
        key=lambda d: d["clean_ink_pixels"],
        default=None,
    )
    total_ink = sum(d["clean_ink_pixels"] for d in diagnostics)

    return {
        "blocks": nonblank_blocks or all_blocks,
        "all_blocks": all_blocks,
        "is_blank": all_blank,
        "primary_area_index": primary["index"] if primary else 0,
        "ink_bbox": primary["ink_bbox"] if primary else None,
        "clean_ink_pixels": int(total_ink),
        "area_diagnostics": diagnostics,
    }
