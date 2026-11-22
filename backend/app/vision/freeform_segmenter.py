"""全版式 Spotting 结构化切题器（自由试卷无模板题目、阅读正文与答题区提取）。

支持：
1. PaddleOCR-VL-1.5 Spotting 四点多边形斜率估计与局部透视去斜聚类（y_corr = cy - k(cy)*(cx - x_ref)），
   彻底消除微倾斜导致的跨题串行与 seg[0] 篡改；
2. 完形填空（Cloze）短文提取、文内空位数字融合及紧凑选项表切分；
3. 七选五 / 五选五嵌入式题号拆分、共享选项池关联与行内空位精确子框定位；
4. 跨页阅读分篇标记（居中 A~E）、页首孤立残余选项过滤、页尾截断选择题判定与长横线情景交际识别；
5. 数学手写页半开括号/末尾单字母精准定位（排除图象选项标签与居中篇目号），以及函数图象坐标轴与填空横线分离。
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

import cv2
import numpy as np

from app.vision.freeform_ink_gate import prepare_freeform_answer_blocks

QNUM_RE = re.compile(
    r"^\s*(?:[\(（]?(\d{1,3})[\)）]|[第](\d{1,3})[题小题]|(\d{1,3})\s*[.．、,，])\s*(.*)$"
)
SECTION_RE = re.compile(
    r"^\s*(?:"
    r"[一二三四五六七八九十]+[.、．\s]|"
    r"[\(（][一二三四五六七八九十]+[\)）]|"
    r"(?:I{1,3}|IV|VI{0,3}|IX|X)[.、．]|"
    r"Part\s+[A-Z0-9]|Section\s+[A-Z0-9]|"
    r"第[一二三四五六七八九十]+[部节大]分?|"
    r"[\(（]?\s*(?:每小题|共\s*\d+\s*[小大]?题|满分|小题\s*\d+\s*分)"
    r")"
)
SECTION_CONT_RE = re.compile(
    r"^\s*(?:[\(（]?\s*(?:每小题|小题\s*\d+\s*分|满分)|完整\s*[。.）)]|从每小题|选出)"
)
EXAM_HEADER_RE = re.compile(
    r"(?:月考|期中|期末|质检|摸底|联考|模拟|学年|学期|试卷|答题卡|注意事项|考试时间|"
    r"(?:学校|姓名|班级|考号|座号|准考证)\s*[:：])"
)
FOOTER_RE = re.compile(
    r"(?:试卷\s*)?第\s*\d+\s*(?:/\s*\d+\s*页|页\s*[,，]?\s*[\(（]?\s*共\s*\d+\s*页\s*[\)）]?)"
)
PAIRED_BRACKET_RE = re.compile(r"[（(]\s*([A-Ha-h]?)\s*[）)]")
OPEN_BRACKET_END_RE = re.compile(r"[（(]\s*$")
STRAY_PAREN_SYM_RE = re.compile(r"^(?:[）)]|\\rightarrow|\\to|→|\s)*$")
GREEK_NOISE_RE = re.compile(r"^\\(?:alpha|beta|gamma|theta|circ|emptyset)$")
ENG_LETTER_RE = re.compile(r"[A-Za-z]{3,}")
MATH_HINT_RE = re.compile(
    r"[=+\-×÷<>≤≥√△∠⊥∥⊙π]|\\[a-zA-Z]+|\^|_|\b(?:sin|cos|tan|x|y)\b"
)
CHOICE_STEM_HINT_RE = re.compile(
    r"(?:Which of the following|What does|What is|According to|best title|underlined word|"
    r"下列|正确的是|不正确的是|下列说法|故选)",
    re.IGNORECASE,
)
OPT_MARKER_RE = re.compile(r"(?:^|\s)([A-F])\s*[.．、\)）]\s*")
OPT_LOOSE_LINE_RE = re.compile(r"^\s*([A-F])\s+(?=[A-Za-z一-鿿$])")

LOC_LINE_RE = re.compile(
    r"^(.*?)<\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|>"
    r"<\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|><\|LOC_(\d+)\|>\s*$"
)


def _strip_ocr_tags(text: str) -> str:
    s = re.sub(r"<\|[^>]+?\|>", "", text or "")
    s = s.replace("\\(", "$").replace("\\)", "$").replace("\\[", "$$").replace("\\]", "$$")
    return s.strip()


def _infer_subject(texts: list[str]) -> str:
    joined = "\n".join(texts)
    en_words = len(ENG_LETTER_RE.findall(joined))
    math_hits = len(MATH_HINT_RE.findall(joined))
    if en_words >= 12 and en_words > math_hits * 1.5:
        return "en"
    return "math"


def _box_iou_xyxy(b1: list[int | float], b2: list[int | float]) -> float:
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


def _rect_contains_center(rect_xywh: list[int], cx: float, cy: float, pad: int = 4) -> bool:
    rx, ry, rw, rh = rect_xywh
    return (rx - pad) <= cx <= (rx + rw + pad) and (ry - pad) <= cy <= (ry + rh + pad)


def find_underlines(
    gray: np.ndarray,
    roi: list[int],
    figures: list[list[int]] | None = None,
) -> list[list[int]]:
    """在题目局部 ROI 内检测印刷填空横线（支持轻微倾斜与长横线，排除页眉页脚线与插图内部坐标轴）。"""
    H, W = gray.shape[:2]
    x, y, w, h = roi
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return []
    bw = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    k_short = max(26, round(W * 0.020))
    k_len = max(38, round(W * 0.035))
    h_runs = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k_short, 1)))
    bridged = cv2.morphologyEx(h_runs, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bridged)
    out = []
    for i in range(1, n):
        sx, sy, sw, sh, _ = stats[i]
        abs_sx = int(x0 + sx)
        abs_sy = int(y0 + sy)
        abs_ex = abs_sx + int(sw)
        cy = abs_sy + sh * 0.5
        if cy < H * 0.06 or cy > H * 0.93:
            continue
        if not (k_len <= sw <= W * 0.90 and sh <= max(14, int(sw * 0.05) + 8)):
            continue
        patch = h_runs[sy : sy + sh, sx : sx + sw]
        col_cov = float((patch > 0).any(axis=0).mean())
        col_sum = int((patch > 0).sum())
        col_any = int((patch > 0).any(axis=0).sum())
        mean_th = col_sum / max(1, col_any)
        # 空白横线上方不应是连续密集的印刷字底笔画
        ay0, ay1 = max(0, sy - 14), max(0, sy - 2)
        above_cov = float((bw[ay0:ay1, sx : sx + sw] > 0).any(axis=0).mean()) if ay1 > ay0 else 0.0
        if col_cov < 0.72 or mean_th > 5.5 or above_cov > 0.45:
            continue
        # 排除落入插图矩形框内部的坐标轴横线
        if figures:
            cx = (abs_sx + abs_ex) * 0.5
            if any(_rect_contains_center(fig, cx, cy, pad=2) for fig in figures):
                continue
        out.append([abs_sx, int(round(cy)), abs_ex])
    return sorted(out, key=lambda s: (s[1] // 18, s[0]))


def detect_figure_regions(gray: np.ndarray) -> list[list[int]]:
    """检测页面上的几何插图/函数图象/结构图区域 [x, y, w, h]。"""
    H, W = gray.shape[:2]
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    figs = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < W * 0.075 or h < H * 0.032:
            continue
        if w > W * 0.88 and h > H * 0.65:
            continue
        aspect = w / max(1, h)
        if 0.25 <= aspect <= 4.2 and h >= 65:
            figs.append([int(x), int(y), int(w), int(h)])
    return figs


def _parse_paddle_vl_blocks(raw_text: str, page_w: int, page_h: int) -> list[dict]:
    """解析 PaddleOCR-VL-1.5 Spotting 四点坐标行（保留原 polygon 与 raw_text）、JSON 或 grounding 块。"""
    blocks = []

    for raw_ln in raw_text.splitlines():
        raw_ln = raw_ln.strip()
        if not raw_ln:
            continue
        m_loc = LOC_LINE_RE.match(raw_ln)
        if m_loc:
            raw_part = m_loc.group(1).strip()
            txt = _strip_ocr_tags(raw_part)
            txt_clean_footer = txt.replace("\\_", "_")
            if not txt or FOOTER_RE.search(txt_clean_footer):
                continue
            coords = [int(m_loc.group(i)) for i in range(2, 10)]
            poly = [
                [coords[0] * page_w / 1000.0, coords[1] * page_h / 1000.0],
                [coords[2] * page_w / 1000.0, coords[3] * page_h / 1000.0],
                [coords[4] * page_w / 1000.0, coords[5] * page_h / 1000.0],
                [coords[6] * page_w / 1000.0, coords[7] * page_h / 1000.0],
            ]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bx1, by1 = int(round(min(xs))), int(round(min(ys)))
            bx2, by2 = int(round(max(xs))), int(round(max(ys)))
            if bx2 > bx1 and by2 > by1:
                blocks.append(
                    {
                        "label": "line",
                        "box": [bx1, by1, bx2, by2],
                        "poly": poly,
                        "text": txt,
                        "raw_text": raw_part,
                    }
                )
    if blocks:
        return blocks

    stripped = raw_text.strip()
    if stripped.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if m:
            stripped = m.group(1).strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
            items = parsed.get("questions") or parsed.get("blocks") or (parsed if isinstance(parsed, list) else [])
            if isinstance(items, list) and items:
                return items
        except json.JSONDecodeError:
            pass

    pattern = re.compile(
        r"(?:<\|ref\|>(.*?)<\|/ref\|>)?\s*<\|det\|>\s*\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*\]\s*<\|/det\|>\s*(.*?)(?=(?:<\|ref\|>|<\|det\|>|$))",
        re.DOTALL,
    )
    for m in pattern.finditer(raw_text):
        label, x1, y1, x2, y2, content = m.groups()
        txt = _strip_ocr_tags(content or label)
        if not txt or FOOTER_RE.search(txt.replace("\\_", "_")):
            continue
        bx1 = round(int(x1) * page_w / 1000)
        by1 = round(int(y1) * page_h / 1000)
        bx2 = round(int(x2) * page_w / 1000)
        by2 = round(int(y2) * page_h / 1000)
        if bx2 > bx1 and by2 > by1:
            blocks.append(
                {
                    "label": (label or "text").strip(),
                    "box": [bx1, by1, bx2, by2],
                    "text": txt,
                    "raw_text": (content or label or "").strip(),
                }
            )
    return blocks


def _filter_text_covered_figures(figs: list[list[int]], blocks: list[dict]) -> list[list[int]]:
    """过滤掉由密集英文段落粘连形成的伪插图框，保留真正几何/函数/树状结构图区域。"""
    kept = []
    for fx, fy, fw, fh in figs:
        fig_area = max(1.0, float(fw * fh))
        prose_cover = 0.0
        prose_hits = 0
        for b in blocks:
            bx0, by0, bx1, by1 = b["box"]
            ix0, iy0 = max(fx, bx0), max(fy, by0)
            ix1, iy1 = min(fx + fw, bx1), min(fy + fh, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            txt = b["text"].strip("$").strip()
            # 排除图内单字母/轴标签/数学噪声（如 O, x, y, B., \alpha）
            if len(txt) <= 5 or GREEK_NOISE_RE.match(txt) or re.fullmatch(r"[A-Ha-h]\.?", txt):
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            prose_cover += inter / fig_area
            if inter / max(1.0, (bx1 - bx0) * (by1 - by0)) > 0.35:
                prose_hits += 1
        if prose_cover > 0.25 or prose_hits >= 2:
            continue
        kept.append([fx, fy, fw, fh])
    return kept


def _prepare_blocks_and_rows(
    lines: list[dict], page_w: int, page_h: int
) -> tuple[list[dict], list[dict], float, callable]:
    """预处理块：去脚标、去重、过滤英文结构图噪声、拟合局部斜率 k(y) 并聚类成去斜视觉行。"""
    footer_y = int(page_h * 0.96)
    raw_blocks = []
    for l in lines:
        txt = (l.get("text") or "").strip()
        if not txt:
            continue
        if FOOTER_RE.search(txt.replace("\\_", "_")):
            continue
        box = l["box"]
        if box[1] >= footer_y:
            continue
        poly = l.get("poly")
        if poly and len(poly) == 4:
            cx = sum(p[0] for p in poly) * 0.25
            cy = sum(p[1] for p in poly) * 0.25
            dx_top = poly[1][0] - poly[0][0]
            dx_bot = poly[2][0] - poly[3][0]
            if dx_top > 25 and dx_bot > 25:
                slope = 0.5 * (
                    (poly[1][1] - poly[0][1]) / dx_top + (poly[2][1] - poly[3][1]) / dx_bot
                )
            else:
                slope = 0.0
        else:
            cx = (box[0] + box[2]) * 0.5
            cy = (box[1] + box[3]) * 0.5
            slope = 0.0
        w = max(1, box[2] - box[0])
        h = max(1, box[3] - box[1])
        raw_blocks.append(
            {
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "poly": poly,
                "text": txt,
                "raw_text": l.get("raw_text", txt),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "slope": float(slope),
            }
        )

    # 去重完全重合或高度重合的同一文本块（如 Spotting 对倾斜行偶发双报）
    deduped = []
    for b in raw_blocks:
        dup = False
        b_norm = re.sub(r"\s+", "", b["text"])
        for prev in deduped:
            if _box_iou_xyxy(b["box"], prev["box"]) >= 0.70 and b_norm == re.sub(r"\s+", "", prev["text"]):
                dup = True
                break
        if not dup:
            deduped.append(b)

    subj_hint = _infer_subject([b["text"] for b in deduped])
    greek_cnt = sum(1 for b in deduped if GREEK_NOISE_RE.match(b["text"].strip("$").strip()))
    blocks = []
    for b in deduped:
        unwrapped = b["text"].strip("$").strip()
        if GREEK_NOISE_RE.match(unwrapped) and (subj_hint == "en" or greek_cnt >= 3):
            continue
        blocks.append(b)

    if not blocks:
        return [], [], 32.0, (lambda _y: 0.0)

    # 拟合局部斜率函数 k_slope(y)（处理透视梯形畸变导致上下行斜率渐变的情况）
    wide = [
        b
        for b in blocks
        if b["w"] >= page_w * 0.18 and b["h"] <= page_h * 0.08 and abs(b["slope"]) < 0.35 and b["poly"] is not None
    ]
    if len(wide) >= 4:
        ys = np.array([b["cy"] for b in wide], dtype=np.float64)
        ss = np.array([b["slope"] for b in wide], dtype=np.float64)
        if float(ys.max() - ys.min()) >= page_h * 0.20:
            a0, b0 = np.polyfit(ys, ss, 1)
            res = np.abs(ss - (a0 * ys + b0))
            mad = float(np.median(res))
            keep = res <= max(0.025, 2.5 * mad)
            if int(keep.sum()) >= 3:
                a_fit, b_fit = np.polyfit(ys[keep], ss[keep], 1)
            else:
                a_fit, b_fit = a0, b0
            k_func = lambda y, _a=float(a_fit), _b=float(b_fit): _a * y + _b
        else:
            med_s = float(np.median(ss))
            k_func = lambda _y, _s=med_s: _s
    elif wide:
        med_s = float(np.median([b["slope"] for b in wide]))
        k_func = lambda _y, _s=med_s: _s
    else:
        k_func = lambda _y: 0.0

    x_ref = page_w * 0.10
    for b in blocks:
        k = k_func(b["cy"])
        b["y_corr"] = b["cy"] - k * (b["cx"] - x_ref)

    normal_hs = [b["h"] for b in blocks if page_h * 0.005 <= b["h"] <= page_h * 0.06]
    med_h = float(np.median(normal_hs)) if normal_hs else 32.0
    row_thresh = max(12.0, min(28.0, 0.45 * med_h))

    blocks.sort(key=lambda b: (b["y_corr"], b["cx"]))
    rows_raw: list[list[dict]] = []
    row_ycs: list[float] = []
    for b in blocks:
        if rows_raw and abs(b["y_corr"] - row_ycs[-1]) <= row_thresh:
            rows_raw[-1].append(b)
            # 更新当前行基准 y_corr（优先使用正常高度印刷行，避免 tall 符号拉偏）
            norm_ycs = [x["y_corr"] for x in rows_raw[-1] if x["h"] <= 1.6 * med_h]
            row_ycs[-1] = float(np.mean(norm_ycs)) if norm_ycs else float(np.mean([x["y_corr"] for x in rows_raw[-1]]))
        else:
            rows_raw.append([b])
            row_ycs.append(b["y_corr"])

    rows = []
    for r_idx, r_blocks in enumerate(rows_raw):
        r_blocks.sort(key=lambda x: x["cx"])
        yc = float(np.mean([x["y_corr"] for x in r_blocks]))
        rows.append(
            {
                "idx": r_idx,
                "blocks": r_blocks,
                "y_corr": yc,
                "x0": min(x["box"][0] for x in r_blocks),
                "y0": min(x["box"][1] for x in r_blocks),
                "x1": max(x["box"][2] for x in r_blocks),
                "y1": max(x["box"][3] for x in r_blocks),
                "text": " ".join(x["text"] for x in r_blocks),
            }
        )
    return blocks, rows, med_h, k_func


def _extract_options_from_texts(texts: list[str]) -> list[str]:
    """从文本行/块列表中稳健提取 A.~F. 选项（支持单块无点号如 'A 电动'、'C John'、粘连 'C100' 及多选项同行）。"""
    has_dotted = any(OPT_MARKER_RE.search(t) for t in texts)
    opts: list[str] = []
    unmatched_texts: list[str] = []
    for ln in texts:
        s = ln.strip()
        if not s:
            continue
        matches = list(OPT_MARKER_RE.finditer(s))
        if matches:
            for idx, m in enumerate(matches):
                letter = m.group(1)
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(s)
                val = s[start:end].strip()
                if val or len(s) <= 8:
                    opts.append(f"{letter}. {val}".strip())
        else:
            m_loose = OPT_LOOSE_LINE_RE.match(s)
            if m_loose and (has_dotted or len(s) <= 90):
                opts.append(f"{m_loose.group(1)}. {s[m_loose.end():].strip()}")
            else:
                unmatched_texts.append(s)

    present_letters = {o[0] for o in opts if len(o) >= 2 and o[1] == "."}
    # 紧凑粘连选项容错：如 A./B./D. 已存在而中间缺 C.，且存在 'C100'/'Cloo' 等无点号粘连块或尾随词
    if has_dotted and len(present_letters) >= 2:
        max_ord = max(ord(c) for c in present_letters)
        missing_letters = [
            chr(code) for code in range(ord("A"), min(ord("F"), max_ord) + 1) if chr(code) not in present_letters
        ]
        if missing_letters:
            # 1) 检查未命中块（如独立块 'C100' / 'Cloo'）
            for miss in list(missing_letters):
                glued_pat = re.compile(rf"\b{miss}([1lI0-9A-Za-z]{{2,}})\b")
                found = False
                for ut in unmatched_texts:
                    gm = glued_pat.search(ut)
                    if gm:
                        opts.append(f"{miss}. {gm.group(1)}")
                        present_letters.add(miss)
                        found = True
                        break
                if found:
                    continue
                # 2) 检查是否被粘在前一个选项（如 B. ... C100）的末尾
                prev_ch = chr(ord(miss) - 1)
                tail_pat = re.compile(rf"\s+{miss}([1lI0-9]{{2,}}|[a-z]{{2,}})$")
                for oi, o in enumerate(opts):
                    if o.startswith(f"{prev_ch}."):
                        tm = tail_pat.search(o)
                        if tm:
                            opts[oi] = o[: tm.start()].strip()
                            opts.append(f"{miss}. {tm.group(1)}")
                            present_letters.add(miss)
                            break

    seen = set()
    uniq = []
    for o in opts:
        key = o[:2]
        if key not in seen:
            seen.add(key)
            uniq.append(o)
    if len(uniq) >= 2 and all(len(o) >= 2 and o[0] in "ABCDEF" and o[1] == "." for o in uniq):
        uniq.sort(key=lambda o: o[0])
    return uniq


def _is_option_row_text(txt: str) -> bool:
    s = txt.strip()
    if re.match(r"^[A-F]\s*[.．、\)）](?:\s|$|[A-Za-z一-鿿$])", s):
        return True
    if OPT_LOOSE_LINE_RE.match(s) and len(s) <= 90:
        return True
    return False


def _is_single_letter_token(txt: str) -> str | None:
    unwrapped = txt.strip().strip("$").strip()
    if re.fullmatch(r"[A-Ha-h]", unwrapped):
        return unwrapped.upper()
    return None


def _row_has_multi_labels(row: dict) -> bool:
    """判断一行是否为图表/选项多标签并排行（如 A. O B O C O D O x 或 B. C. D.）。"""
    cnt = 0
    for b in row["blocks"]:
        unwrapped = b["text"].strip("$").strip()
        if re.fullmatch(r"[A-Ha-hOxXyY]\.?", unwrapped):
            cnt += 1
    return cnt >= 2


def _find_cloze_qno_span(text: str, qn: str) -> tuple[int, int] | None:
    """在完形短文文本中定位题号 qn（兼容 _36_、__36__、_37_.、“41、$ 42 $、$ ^{43} $、裸数字等）。"""
    esc = re.escape(qn)
    m_tex = re.search(rf"\$\s*(?:\^\s*\{{?\s*{esc}\s*\}}?|{esc})\s*\$", text)
    if m_tex:
        return m_tex.span()
    m_plain = re.search(
        rf"(?:^|[\s\"“'‘(（])((?:\\?_)*{esc}(?:\\?_)*)(?=$|[\s.,?!;:，。？！；：\"”'’)）])",
        text,
    )
    if m_plain:
        return m_plain.span(1)
    return None


def _sub_cloze_qno(text: str, qn: str, repl: str) -> str:
    """将完形短文内的各形态题号替换为统一占位符 repl（保留句末标点与前导引号）。"""
    esc = re.escape(qn)
    out = re.sub(rf"\$\s*(?:\^\s*\{{?\s*{esc}\s*\}}?|{esc})\s*\$", repl, text)
    if out != text:
        return out
    return re.sub(
        rf"(^|[\s\"“'‘(（])(?:\\?_)*{esc}(?:\\?_)*(?=$|[\s.,?!;:，。？！；：\"”'’)）])",
        rf"\1{repl}",
        out,
    )


def _extract_cloze_context(passage_text: str, qno: str, *, is_cloze: bool = False) -> str:
    """从完形/七选五短文中提取包含 __qno__ 的完整句子上下文（忽略行内换行）。"""
    flat = re.sub(r"\s+", " ", passage_text or "").strip()
    marker = f"__{qno}__"
    pos = flat.find(marker)
    if pos < 0:
        return ""
    left = flat[:pos]
    # 句末标点后紧跟空格，或句末标点+闭引号后再跟空格（如 `made it." People...`）
    end_cands: list[tuple[int, int]] = []
    for sep in (". ", "! ", "? "):
        idx = left.rfind(sep)
        if idx >= 0:
            end_cands.append((idx, idx + 1))
    for qsep in ('." ', '.” ', '!" ', '!” ', '?" ', '?” '):
        idx = left.rfind(qsep)
        if idx >= 0:
            end_cands.append((idx, idx + 2))
    l_idx, l_after = max(end_cands) if end_cands else (-1, 0)
    if is_cloze:
        # 对话引语逗号分隔（如 `...reporters, "When I won...`）也可作为新句起点
        q_starts = [m.start(1) for m in re.finditer(r'[,，]\s*([“"][A-Z_])', left)]
        cand_quote = max(q_starts) if q_starts else -1
        sent_start = max(l_after if l_idx >= 0 else 0, cand_quote)
    else:
        # 七选五空位紧跟在句号后（如 "...teachers say. __68__"）时，向前多取前一句
        if l_idx >= 0 and len(left[l_after:].strip()) <= 2:
            prev_left = left[:l_idx]
            p_cands = [prev_left.rfind(sep) for sep in (". ", "! ", "? ")]
            p_idx = max(p_cands)
            l_after = p_idx + 1 if p_idx >= 0 else 0
        sent_start = l_after if l_idx >= 0 else 0

    right = flat[pos + len(marker) :]
    r_cands = [
        i
        for i in (
            right.find(". "),
            right.find('."'),
            right.find(".”"),
            right.find("! "),
            right.find("? "),
        )
        if i >= 0
    ]
    if is_cloze:
        mq_end = re.search(r'[,，]\s*[“"][A-Z_]', right)
        if mq_end:
            r_cands.append(mq_end.start())
    sent_end = pos + len(marker) + (min(r_cands) + 1 if r_cands else len(right))
    snippet = flat[sent_start:sent_end].strip().replace(marker, "______")
    return snippet


def _locate_cloze_blank_boxes(
    prose_rows: list[dict],
    cloze_qnos: set[str],
    gray: np.ndarray,
    valid_figs: list[list[int]],
    page_w: int,
    page_h: int,
) -> dict[str, tuple[list[int], list[int]]]:
    """在完形短文 prose_rows 中精确定位每个空位题号的行内空位框 (bbox_xyxy, ans_xyxy)。"""
    sorted_qnos = sorted(cloze_qnos, key=lambda s: int(s))
    located: dict[str, tuple[int, list[int], list[int]]] = {}
    claimed_uls: list[list[int]] = []

    for qn in sorted_qnos:
        for ri, r_sub in enumerate(prose_rows):
            hit_blk = None
            hit_span = None
            for b in r_sub["blocks"]:
                sp = _find_cloze_qno_span(b["text"], qn)
                if sp is not None:
                    hit_blk = b
                    hit_span = sp
                    break
            if hit_blk is None or hit_span is None:
                continue

            lx0, ly0, lx1, ly1 = hit_blk["box"]
            txt_len = max(1, len(hit_blk["text"]))
            s_idx, e_idx = hit_span
            ratio = (s_idx + e_idx) * 0.5 / txt_len
            x_est = lx0 + (lx1 - lx0) * ratio
            if hit_blk.get("poly"):
                _, _, p2, p3 = hit_blk["poly"]
                bot_y = p3[1] + (p2[1] - p3[1]) * ratio
            else:
                bot_y = float(ly1)

            sx0 = int(max(0, x_est - 200))
            sy0 = int(max(0, bot_y - 16))
            sw = int(min(page_w - sx0, 400))
            sh = int(min(page_h - sy0, 34))
            uls = find_underlines(gray, [sx0, sy0, sw, sh], figures=valid_figs)
            cands = [
                u
                for u in uls
                if abs(u[1] - bot_y) <= 16 and abs(0.5 * (u[0] + u[2]) - x_est) <= 180
            ]
            if cands:
                ul_sx, ul_y, ul_ex = min(
                    cands,
                    key=lambda u: (abs(0.5 * (u[0] + u[2]) - x_est), abs(u[1] - bot_y)),
                )
                claimed_uls.append([ul_sx, ul_y, ul_ex])
                bx0 = int(max(0, ul_sx - 4))
                bx1 = int(min(page_w, ul_ex + 4))
                by0 = int(max(0, ul_y - 38))
                by1 = int(min(page_h, ul_y + 8))
                ax0, ay0, ax1, ay1 = bx0, int(max(0, ul_y - 36)), bx1, int(min(page_h, ul_y + 6))
            else:
                bx0 = int(max(0, x_est - 60))
                bx1 = int(min(page_w, bx0 + 120))
                by0 = int(max(0, bot_y - 38))
                by1 = int(min(page_h, bot_y + 8))
                ax0, ay0, ax1, ay1 = bx0, by0, bx1, by1
            located[qn] = (ri, [bx0, by0, bx1, by1], [ax0, ay0, ax1, ay1])
            break

    # 若偶发某个中间空位数字被 OCR 漏印，基于前后已定位行号间的未认领下划线进行插值定位
    if prose_rows and len(located) < len(sorted_qnos):
        for idx_q, qn in enumerate(sorted_qnos):
            if qn in located:
                continue
            prev_ri = 0
            prev_y = prose_rows[0]["y0"] - 10
            for pj in range(idx_q - 1, -1, -1):
                pq = sorted_qnos[pj]
                if pq in located:
                    prev_ri = located[pq][0]
                    prev_y = located[pq][1][1]
                    break
            next_ri = len(prose_rows) - 1
            next_y = prose_rows[-1]["y1"] + 10
            for nj in range(idx_q + 1, len(sorted_qnos)):
                nq = sorted_qnos[nj]
                if nq in located:
                    next_ri = located[nq][0]
                    next_y = located[nq][1][3]
                    break
            band_rows = prose_rows[prev_ri : next_ri + 1]
            if not band_rows:
                continue
            rx0 = max(0, min(r["x0"] for r in band_rows) - 20)
            ry0 = max(0, min(r["y0"] for r in band_rows) - 10)
            rx1 = min(page_w, max(r["x1"] for r in band_rows) + 120)
            ry1 = min(page_h, max(r["y1"] for r in band_rows) + 16)
            band_uls = find_underlines(gray, [rx0, ry0, rx1 - rx0, ry1 - ry0], figures=valid_figs)
            unclaimed = [
                u
                for u in band_uls
                if prev_y - 6 <= u[1] <= next_y + 6
                and not any(abs(u[0] - c[0]) < 15 and abs(u[1] - c[1]) < 12 for c in claimed_uls)
            ]
            if unclaimed:
                ul_sx, ul_y, ul_ex = unclaimed[0]
                claimed_uls.append([ul_sx, ul_y, ul_ex])
                bx0 = int(max(0, ul_sx - 4))
                bx1 = int(min(page_w, ul_ex + 4))
                by0 = int(max(0, ul_y - 38))
                by1 = int(min(page_h, ul_y + 8))
                located[qn] = (
                    prev_ri,
                    [bx0, by0, bx1, by1],
                    [bx0, int(max(0, ul_y - 36)), bx1, int(min(page_h, ul_y + 6))],
                )

    return {qn: (bbox_xyxy, ans_xyxy) for qn, (_, bbox_xyxy, ans_xyxy) in located.items()}


def _locate_embedded_blank_box(
    row: dict,
    qno: str,
    gray: np.ndarray,
    page_w: int,
    page_h: int,
) -> tuple[list[int], list[int]]:
    """精确定位七选五/五选五嵌入式题号所在行内的水平空位 bbox 与 answer_bbox。"""
    blocks = row["blocks"]
    target_idx = -1
    match_span = None
    pat = re.compile(rf"(?<![\d$])({re.escape(qno)})\s*[.．]?")
    for bi, b in enumerate(blocks):
        for m in pat.finditer(b["text"]):
            # 确认匹配在句首/空白/标点后，且后面是空白/句末/大写
            before = b["text"][: m.start()]
            after = b["text"][m.end() :]
            if (not before or re.search(r"[\s.?!,;:\"”'’]$", before)) and (
                not after or re.match(r"^\s", after)
            ):
                target_idx = bi
                match_span = m.span()
                break
        if target_idx >= 0:
            break

    if target_idx < 0:
        # 兜底：使用行首局部框
        rx0, ry0, _, ry1 = row["x0"], row["y0"], row["x1"], row["y1"]
        cy = (ry0 + ry1) * 0.5
        bx0 = int(max(0, rx0))
        bx1 = int(min(page_w, bx0 + 150))
        by0 = int(max(0, cy - 29))
        by1 = int(min(page_h, cy + 29))
        return [bx0, by0, bx1, by1], [min(page_w - 20, bx0 + 36), by0, bx1, by1]

    blk = blocks[target_idx]
    lx0, ly0, lx1, ly1 = blk["box"]
    txt_len = max(1, len(blk["text"]))
    s_idx, e_idx = match_span

    # 情形1：独立编号块后紧跟同行的下一文本块，中间留有空白/下划线缝隙（如 "things. 67." + "Knowing..." 或 "70." + "You..."）
    if e_idx >= txt_len - 1 and target_idx + 1 < len(blocks):
        next_blk = blocks[target_idx + 1]
        gap = next_blk["box"][0] - lx1
        if gap >= 22:
            num_x0 = lx0 + (lx1 - lx0) * (s_idx / txt_len)
            sub_y0 = int(max(0, min(ly0, next_blk["box"][1]) - 14))
            sub_y1 = int(min(page_h, max(ly1, next_blk["box"][3]) + 18))
            roi_w = int(min(page_w - num_x0 + 10, next_blk["box"][0] - num_x0 + 25))
            uls = find_underlines(gray, [int(max(0, num_x0 - 8)), sub_y0, max(60, roi_w), sub_y1 - sub_y0])
            if uls:
                ul_sx, ul_y, ul_ex = uls[0]
                bx0 = int(max(0, min(num_x0 - 4, ul_sx - 38)))
                bx1 = int(min(page_w, max(ul_ex + 8, bx0 + 125)))
                cy = ul_y - 12
                by0, by1 = int(max(0, cy - 29)), int(min(page_h, cy + 29))
                ax0 = int(max(bx0, ul_sx - 6))
                ax1 = int(min(page_w, max(ax0 + 70, ul_ex + 6)))
                return [bx0, by0, bx1, by1], [ax0, by0, ax1, by1]
            bx0 = int(max(0, num_x0 - 4))
            bx1 = int(min(page_w, max(bx0 + 125, next_blk["box"][0] - 4)))
            cy = (ly0 + ly1) * 0.5
            by0, by1 = int(max(0, cy - 29)), int(min(page_h, cy + 29))
            ax0 = int(min(bx1 - 40, lx1 - 2))
            return [bx0, by0, bx1, by1], [ax0, by0, bx1, by1]

    # 情形2：题号位于完整融合行内部（如 "66. It doesn't..." 或 "...books. 69. Those..."）或行末（"...say. 68."）
    num_x0 = lx0 + (lx1 - lx0) * (s_idx / txt_len)
    num_x1 = lx0 + (lx1 - lx0) * (e_idx / txt_len)
    # 若该行存在下划线（如 66. ____），优先吸附到邻近下划线
    cy_est = (ly0 + ly1) * 0.5
    if blk.get("poly"):
        # 用字符比例沿上/下边插值得到精确 y
        ratio = (s_idx + e_idx) * 0.5 / txt_len
        p0, p1, p2, p3 = blk["poly"]
        top_y = p0[1] + (p1[1] - p0[1]) * ratio
        bot_y = p3[1] + (p2[1] - p3[1]) * ratio
        cy_est = 0.5 * (top_y + bot_y)
    search_x0 = int(max(0, num_x0 - 15))
    search_w = int(min(page_w - search_x0, max(160, (num_x1 - num_x0) + 150)))
    uls = find_underlines(gray, [search_x0, int(max(0, cy_est - 25)), search_w, 60])
    if uls:
        ul_sx, ul_y, ul_ex = min(uls, key=lambda u: abs(u[0] - num_x1))
        bx0 = int(max(0, min(num_x0 - 6, ul_sx - 40)))
        bx1 = int(min(page_w, max(ul_ex + 10, bx0 + 125)))
        by0 = int(max(0, ul_y - 42))
        by1 = int(min(page_h, by0 + 58))
        ax0 = int(max(bx0, ul_sx - 6))
        ax1 = int(min(page_w, max(ax0 + 70, ul_ex + 6)))
        return [bx0, by0, bx1, by1], [ax0, by0, ax1, by1]

    # 无下划线（如句末 "68."）：按字符比例外扩 ~125..160px
    if e_idx >= txt_len - 2:
        bx0 = int(max(0, num_x0 - 6))
        bx1 = int(min(page_w, max(bx0 + 145, lx1 + 100)))
    else:
        bx0 = int(max(0, num_x0 - 6))
        bx1 = int(min(page_w, bx0 + 145))
    by0 = int(max(0, cy_est - 29))
    by1 = int(min(page_h, cy_est + 29))
    ax0 = int(min(bx1 - 40, num_x1 - 2))
    return [bx0, by0, bx1, by1], [ax0, by0, bx1, by1]


def build_freeform_page_from_vlm_json(
    parsed: dict, warped: np.ndarray, page_w: int, page_h: int
) -> tuple[dict, dict, dict]:
    """将结构化 questions 列表（归一化 0~1000 坐标）组装为标准页面与裁剪包。"""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    figs = detect_figure_regions(gray)
    raw_qs = parsed.get("questions") or []
    subject = (
        parsed.get("subject")
        if parsed.get("subject") in ("math", "en")
        else _infer_subject([str(q.get("stem", "")) for q in raw_qs])
    )
    title = parsed.get("title") or ("数学自由识别卷" if subject == "math" else "英语自由识别卷")
    paper_id = "free_" + uuid4().hex[:8]

    def scale_box(b, default_box):
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            return default_box
        try:
            x1, y1, x2, y2 = [float(v) for v in b]
        except (TypeError, ValueError):
            return default_box
        if max(x1, y1, x2, y2) <= 1005:
            px1 = int(np.clip(round(x1 * page_w / 1000), 0, page_w - 8))
            py1 = int(np.clip(round(y1 * page_h / 1000), 0, page_h - 8))
            px2 = int(np.clip(round(x2 * page_w / 1000), px1 + 24, page_w))
            py2 = int(np.clip(round(y2 * page_h / 1000), py1 + 24, page_h))
        else:
            px1 = int(np.clip(round(x1), 0, page_w - 8))
            py1 = int(np.clip(round(y1), 0, page_h - 8))
            px2 = int(np.clip(round(x2), px1 + 24, page_w))
            py2 = int(np.clip(round(y2), py1 + 24, page_h))
        return [px1, py1, px2 - px1, py2 - py1]

    # 优先使用调用方过滤后的 figures（若未传入则用检测值）
    valid_figs = parsed.get("_figures") if isinstance(parsed.get("_figures"), list) else figs

    questions = []
    crops = {}
    n_total = max(1, len(raw_qs))
    for idx, rq in enumerate(raw_qs):
        qno = str(rq.get("number") or rq.get("no") or (idx + 1)).strip()
        qid = f"{paper_id}_q{idx + 1}"
        band_y = int(page_h * (0.08 + 0.82 * idx / n_total))
        band_h = max(90, int(page_h * 0.78 / n_total))
        default_bbox = [int(page_w * 0.06), band_y, int(page_w * 0.88), band_h]
        q_bbox = scale_box(rq.get("bbox"), default_bbox)
        qx, qy, qw, qh = q_bbox

        qtype = str(rq.get("type") or "fill").lower()
        if qtype not in ("choice", "fill", "solution", "writing", "listening"):
            qtype = "choice" if rq.get("options") else "solution"

        has_diagram = bool(rq.get("has_figure") or rq.get("figure_bbox"))
        fig_box = scale_box(rq.get("figure_bbox"), None) if rq.get("figure_bbox") else None
        if fig_box is None and not rq.get("_skip_auto_fig"):
            # 若明确关联了插图列表则扩入总框，避免跨题误吞右栏下一题配图
            next_qy = None
            if idx + 1 < len(raw_qs) and isinstance(raw_qs[idx + 1].get("bbox"), (list, tuple)):
                nb = scale_box(raw_qs[idx + 1]["bbox"], None)
                if nb:
                    next_qy = nb[1]
            for fx, fy, fw, fh in valid_figs:
                # 若插图位于右半页且主跨度落入下一题文字纵向带，留给下一题
                if (
                    next_qy is not None
                    and fx > page_w * 0.55
                    and fy + fh > next_qy + 15
                    and qy + qh < fy + fh * 0.55
                ):
                    continue
                if fx < qx + qw + 80 and fx + fw > qx - 40 and fy < qy + qh + 60 and fy + fh > qy - 20:
                    has_diagram = True
                    nx0, ny0 = min(qx, fx), min(qy, fy)
                    nx1, ny1 = max(qx + qw, fx + fw), max(qy + qh, fy + fh)
                    q_bbox = [nx0, ny0, nx1 - nx0, ny1 - ny0]
                    qx, qy, qw, qh = q_bbox
        elif fig_box is not None:
            fx, fy, fw, fh = fig_box
            has_diagram = True
            nx0, ny0 = min(qx, fx), min(qy, fy)
            nx1, ny1 = max(qx + qw, fx + fw), max(qy + qh, fy + fh)
            q_bbox = [nx0, ny0, nx1 - nx0, ny1 - ny0]
            qx, qy, qw, qh = q_bbox

        ans_areas = []
        if isinstance(rq.get("answer_bboxes"), list):
            for ab_raw in rq["answer_bboxes"]:
                ab = scale_box(ab_raw, None)
                if ab:
                    hint = "choice_letter" if qtype == "choice" else ("math_steps" if qtype == "solution" else "fill_blank")
                    ans_areas.append({"id": f"a{len(ans_areas) + 1}", "bbox": ab, "hint": hint})
        if not ans_areas and rq.get("answer_bbox"):
            ab = scale_box(rq.get("answer_bbox"), None)
            if ab:
                hint = "choice_letter" if qtype == "choice" else ("math_steps" if qtype == "solution" else "fill_blank")
                ans_areas.append({"id": "a1", "bbox": ab, "hint": hint})

        if not ans_areas:
            underlines = find_underlines(gray, [qx, qy, qw, qh], figures=valid_figs)
            if underlines and qtype in ("fill", "choice"):
                for ui, (sx, sy, ex) in enumerate(underlines[:4]):
                    h_box = 64 if (ex - sx) > 260 else 54
                    y_off = 52 if (ex - sx) > 260 else 44
                    ans_areas.append(
                        {"id": f"a{ui + 1}", "bbox": [sx, max(0, sy - y_off), ex - sx, h_box], "hint": "fill_blank"}
                    )
            else:
                if qtype == "choice":
                    ans_areas.append(
                        {"id": "a1", "bbox": [qx, qy, qw, min(qh, max(80, int(qh * 0.65)))], "hint": "choice_letter"}
                    )
                else:
                    ans_areas.append({"id": "a1", "bbox": [qx, qy, qw, qh], "hint": "math_steps"})

        stem = str(rq.get("stem") or "").strip() or f"第 {qno} 题"
        options = rq.get("options") if isinstance(rq.get("options"), list) else None
        q_dict = {
            "id": qid,
            "number": qno,
            "type": qtype,
            "stem": stem,
            "options": options,
            "tag": str(rq.get("tag") or ("几何与图象题" if has_diagram else "")),
            "passage_id": rq.get("passage_id"),
            "bbox": q_bbox,
            "answer_areas": ans_areas,
            "has_diagram": has_diagram,
        }
        questions.append(q_dict)

        pre_raw = str(rq.get("student_handwriting") or "").strip()
        gate_res = prepare_freeform_answer_blocks(
            warped,
            gray,
            ans_areas,
            qtype,
            ocr_text_boxes=parsed.get("_ocr_blocks"),
            pre_ocr_handwriting=pre_raw,
            first_x0=rq.get("_first_x0"),
        )
        is_blank_hint = rq.get("is_blank")
        if is_blank_hint is True and gate_res["clean_ink_pixels"] < 35:
            final_is_blank = True
        elif "_ocr_blocks" not in parsed and is_blank_hint is False and pre_raw:
            final_is_blank = False
        else:
            final_is_blank = gate_res["is_blank"]

        full_crop = warped[max(0, qy - 10) : min(page_h, qy + qh + 10), max(0, qx - 10) : min(page_w, qx + qw + 10)]
        ok_full, full_jpeg = cv2.imencode(".jpg", full_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        crops[qid] = {
            "blocks": gate_res["blocks"],
            "all_blocks": gate_res["all_blocks"],
            "question_jpeg": full_jpeg.tobytes() if ok_full else None,
            "pre_ocr_handwriting": pre_raw,
            "pre_ocr_confident": bool(rq.get("_pre_ocr_confident", False)),
            "clean_ink_pixels": gate_res["clean_ink_pixels"],
            "is_blank": final_is_blank,
            "primary_area_index": gate_res["primary_area_index"],
            "ink_bbox": gate_res["ink_bbox"],
            "area_diagnostics": gate_res["area_diagnostics"],
            "freeform": True,
        }

    passages = dict(parsed.get("passages") or {})
    if not passages and parsed.get("passage"):
        passages = {"P_main": str(parsed["passage"])}
        for q in questions:
            if not q.get("passage_id"):
                q["passage_id"] = "P_main"

    paper = {
        "paper_id": paper_id,
        "title": title,
        "subject": subject,
        "passages": passages,
        "pages": [],
    }
    page = {
        "page_no": int(parsed.get("page_no") or 1),
        "width_px": page_w,
        "height_px": page_h,
        "questions": questions,
    }
    paper["pages"].append(page)
    return paper, page, crops


def build_freeform_page_from_lines(
    lines: list[dict], warped: np.ndarray, page_w: int, page_h: int
) -> tuple[dict, dict, dict]:
    """全版式自由试卷几何切题主入口。"""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blocks, rows, med_h, _ = _prepare_blocks_and_rows(lines, page_w, page_h)
    raw_figs = detect_figure_regions(gray)
    valid_figs = _filter_text_covered_figures(raw_figs, blocks)

    if not rows:
        return build_freeform_page_from_vlm_json(
            {"questions": [], "_figures": valid_figs, "_ocr_blocks": blocks},
            warped,
            page_w,
            page_h,
        )

    n_rows = len(rows)
    row_tags = ["text"] * n_rows

    # 1) 标记大题分节标题及紧随的换行指令说明、试卷考号头部
    for i, r in enumerate(rows):
        t = r["text"].strip()
        if SECTION_RE.match(t):
            row_tags[i] = "section"
            continue
        if i > 0 and row_tags[i - 1] == "section" and len(t) <= 45 and SECTION_CONT_RE.match(t):
            row_tags[i] = "section"
            continue
        if EXAM_HEADER_RE.search(t) and i <= 8:
            row_tags[i] = "header"
            continue

    # 头部连续短标签（如“数学试卷”）也归为 header
    seen_section_or_q = False
    for i, r in enumerate(rows):
        t = r["text"].strip()
        if row_tags[i] == "section" or QNUM_RE.match(t):
            seen_section_or_q = True
            break
        if row_tags[i] == "header":
            continue
        if len(ENG_LETTER_RE.findall(t)) < 4 and len(t) <= 24:
            row_tags[i] = "header"

    # 2) 标记居中阅读篇目标记（A/B/C/D/E）：居中窄单字母且紧接下方为连续英文正文
    for i, r in enumerate(rows):
        if row_tags[i] != "text":
            continue
        t = r["text"].strip()
        if len(r["blocks"]) == 1 and re.fullmatch(r"[A-E]", t):
            b = r["blocks"][0]
            if page_w * 0.38 <= b["cx"] <= page_w * 0.62 and b["w"] < page_w * 0.06:
                # 确认下方紧邻行是正文段落（非选项、非题号）
                for j in range(i + 1, min(n_rows, i + 4)):
                    nt = rows[j]["text"].strip()
                    if not nt:
                        continue
                    if (
                        not _is_option_row_text(nt)
                        and not QNUM_RE.match(nt)
                        and len(ENG_LETTER_RE.findall(nt)) >= 4
                    ):
                        row_tags[i] = "passage_label"
                    break

    # 3) 检测页首在第一道显式题号之前的孤立残余选项（如 p8 顶部的 B. ... / C. ... / D. ...）
    first_q_row = None
    for i, r in enumerate(rows):
        if row_tags[i] != "text":
            continue
        m = QNUM_RE.match(r["text"])
        if m:
            nv = int(next((g for g in m.groups()[:3] if g), "0"))
            if 1 <= nv <= 110 and (r["blocks"][0]["box"][0] < page_w * 0.42 or re.match(r"^\s*\d{1,3}\s*[.．]\s*[A-D]\s*[.．]", r["text"])):
                first_q_row = i
                break
    if first_q_row is not None:
        for i in range(first_q_row):
            if row_tags[i] == "text" and _is_option_row_text(rows[i]["text"]):
                # 确认前面没有同题题干，而是纯孤立选项
                if i == 0 or all(row_tags[k] in ("orphan_opt", "header", "section") for k in range(i)):
                    row_tags[i] = "orphan_opt"

    # 4) 识别连续英文正文区段（区分：七选五/五选五嵌入式短文 vs 完形填空短文 vs 阅读理解正文）
    def _is_prose_candidate(i: int) -> bool:
        if row_tags[i] != "text":
            return False
        t = rows[i]["text"].strip()
        if _is_option_row_text(t) or _row_has_multi_labels(rows[i]):
            return False
        # 绝不从 "题号. " 行起步误判成阅读短文（避免双行对话选择题如 23. 被吞成 passage）
        if QNUM_RE.match(t):
            return False
        en_cnt = len(ENG_LETTER_RE.findall(t))
        if en_cnt >= 3:
            return True
        return False

    # 扫描连续正文区间
    passages: dict[str, str] = {}
    passage_spans: list[dict] = []  # {"pid", "start_row", "end_row", "kind": "embed"|"cloze"|"read", ...}
    p_counter = 0

    i = 0
    while i < n_rows:
        if row_tags[i] == "passage_label" or _is_prose_candidate(i):
            has_label = row_tags[i] == "passage_label"
            start_r = i + 1 if has_label else i
            j = start_r
            emb_pat = re.compile(r"(?:^|[\s.?!,;:\"”'’])(\d{1,3})\s*[.．]?(?=\s+[A-Z]|$|\n)")
            emb_vals_seen: list[int] = []
            while j < n_rows:
                if row_tags[j] in ("section", "header", "passage_label", "orphan_opt"):
                    break
                if _is_option_row_text(rows[j]["text"]) or _row_has_multi_labels(rows[j]):
                    break
                mj = QNUM_RE.match(rows[j]["text"])
                if mj:
                    nv = int(next((g for g in mj.groups()[:3] if g), "0"))
                    nxt = next((k for k in range(j + 1, n_rows) if rows[k]["text"].strip()), None)
                    nxt_is_opt = nxt is not None and _is_option_row_text(rows[nxt]["text"])
                    nxt_is_q = nxt is not None and bool(QNUM_RE.match(rows[nxt]["text"]))
                    nxt_is_sec = nxt is not None and row_tags[nxt] == "section"
                    # 若当前段落已收集到至少 2 个递增嵌入题号，且本行延续该递增序列（如 70. You will...）并紧接选项池 A..E，则仍属该嵌入式短文！
                    continues_embed_seq = (
                        len(emb_vals_seen) >= 2
                        and emb_vals_seen[-1] < nv <= emb_vals_seen[-1] + 2
                        and (
                            len(ENG_LETTER_RE.findall(rows[j]["text"])) >= 4
                            or re.fullmatch(r"\d{1,3}\.?", rows[j]["text"].strip())
                        )
                    )
                    if continues_embed_seq:
                        pass
                    elif nxt is not None and (nxt_is_opt or nxt_is_q or nxt_is_sec):
                        break
                    elif len(ENG_LETTER_RE.findall(rows[j]["text"])) < 4 and not re.fullmatch(r"\d{1,3}\.?", rows[j]["text"].strip()):
                        break
                if j > start_r and (rows[j]["y0"] - rows[j - 1]["y1"]) > page_h * 0.06:
                    break
                for em in emb_pat.finditer(rows[j]["text"]):
                    ev = int(em.group(1))
                    if 1 <= ev <= 110 and (not emb_vals_seen or emb_vals_seen[-1] < ev <= emb_vals_seen[-1] + 2):
                        emb_vals_seen.append(ev)
                j += 1

            prose_rows = rows[start_r:j]
            total_en_words = sum(len(ENG_LETTER_RE.findall(r["text"])) for r in prose_rows)
            if prose_rows and total_en_words >= 20 and (len(prose_rows) >= 3 or (has_label and len(prose_rows) >= 1) or (len(prose_rows) >= 2 and total_en_words >= 28)):
                # 检查是否为七选五/五选五：短文内存在 >=3 个递增的嵌入题号，且紧随 >=4 个无题号选项行 A..E/G
                emb_hits = []
                for r_sub in prose_rows:
                    for em in emb_pat.finditer(r_sub["text"]):
                        val = int(em.group(1))
                        if 1 <= val <= 110:
                            emb_hits.append((val, r_sub))
                seq_hits: list[tuple[int, dict]] = []
                for val, r_sub in emb_hits:
                    if not seq_hits:
                        seq_hits.append((val, r_sub))
                    elif val == seq_hits[-1][0] + 1:
                        seq_hits.append((val, r_sub))
                    elif val > seq_hits[-1][0] and (val - seq_hits[-1][0]) <= 2:
                        seq_hits.append((val, r_sub))

                # 检查 j 之后是否紧跟无题号选项池 A. ... E. ...
                opt_pool_end = j
                while opt_pool_end < n_rows and row_tags[opt_pool_end] == "text" and _is_option_row_text(rows[opt_pool_end]["text"]):
                    opt_pool_end += 1
                pool_opts = _extract_options_from_texts([rows[k]["text"] for k in range(j, opt_pool_end)])

                if len(seq_hits) >= 3 and len(pool_opts) >= 4:
                    p_counter += 1
                    pid = f"P_{p_counter}"
                    for k in range(start_r, opt_pool_end):
                        row_tags[k] = "consumed_embed"
                    norm_prose_lines = []
                    seq_nums = {str(v) for v, _ in seq_hits}
                    for r_sub in prose_rows:
                        txt_p = r_sub["text"]
                        for qn in seq_nums:
                            txt_p = re.sub(
                                rf"(^|[\s.?!,;:\"”'’]){re.escape(qn)}\s*[.．]?(?=\s|$)",
                                rf"\1__{qn}__",
                                txt_p,
                            )
                        norm_prose_lines.append(txt_p)
                    passages[pid] = "\n".join(norm_prose_lines)
                    passage_spans.append(
                        {
                            "pid": pid,
                            "start_row": start_r,
                            "end_row": opt_pool_end,
                            "kind": "embed",
                            "seq_hits": seq_hits,
                            "pool_opts": pool_opts,
                            "passage_text": passages[pid],
                        }
                    )
                    i = opt_pool_end
                    continue

                # 检查是否为完形填空（后面紧跟带有题号+选项的行如 "36. A. ... B. ... C. ..."）或普通阅读短文
                is_cloze = False
                cloze_qnos = set()
                for k in range(j, n_rows):
                    if row_tags[k] in ("section", "passage_label"):
                        if k > j:
                            break
                        continue
                    mc = re.match(r"^\s*(\d{1,3})\s*[.．]\s*[A-F]\s*[.．]", rows[k]["text"])
                    if mc:
                        is_cloze = True
                        cloze_qnos.add(mc.group(1))
                    elif is_cloze and not QNUM_RE.match(rows[k]["text"]):
                        break

                p_counter += 1
                pid = f"P_{p_counter}"
                for k in range(start_r, j):
                    row_tags[k] = "passage"
                clean_p_lines = []
                for r_sub in prose_rows:
                    txt_p = r_sub["text"]
                    if is_cloze and cloze_qnos:
                        for qn in cloze_qnos:
                            txt_p = _sub_cloze_qno(txt_p, qn, f"__{qn}__")
                    clean_p_lines.append(txt_p)
                passages[pid] = "\n".join(clean_p_lines)
                cloze_boxes = (
                    _locate_cloze_blank_boxes(prose_rows, cloze_qnos, gray, valid_figs, page_w, page_h)
                    if is_cloze and cloze_qnos
                    else {}
                )
                passage_spans.append(
                    {
                        "pid": pid,
                        "start_row": start_r,
                        "end_row": j,
                        "kind": "cloze" if is_cloze else "read",
                        "passage_text": passages[pid],
                        "cloze_boxes": cloze_boxes,
                    }
                )
                i = j
                continue
        i += 1

    # 5) 定位所有常规题目的锚点行（严格绑定 qno 到锚点行）
    q_starts: list[tuple[int, str]] = []
    for i, r in enumerate(rows):
        if row_tags[i] != "text":
            continue
        # 锚点行必须在左侧合理区间内，或是标准完形选项行 "36. A. ..."
        b0 = r["blocks"][0]
        m = QNUM_RE.match(b0["text"]) or QNUM_RE.match(r["text"])
        if not m:
            continue
        num_str = next((g for g in m.groups()[:3] if g), "")
        if not num_str:
            continue
        num_val = int(num_str)
        if not (1 <= num_val <= 110):
            continue
        if b0["box"][0] < page_w * 0.42 or re.match(r"^\s*\d{1,3}\s*[.．]\s*[A-D]\s*[.．]", r["text"]):
            q_starts.append((i, num_str))

    # 若整页既无嵌入式题也无常规题号，整页有效文本兜底作第 1 题
    if not q_starts and not any(ps["kind"] == "embed" for ps in passage_spans):
        first_valid = next((i for i in range(n_rows) if row_tags[i] == "text"), None)
        if first_valid is not None:
            q_starts = [(first_valid, "1")]

    # 预先计算每张图所属的最佳题目索引（避免 Q15 吞入 Q16 的右半栏图象）
    # 先确定每道常规题的大致文本纵坐标范围 [q_y0, q_y1_raw]
    q_prelim: list[dict] = []
    for k, (r_start, qno) in enumerate(q_starts):
        r_end_limit = q_starts[k + 1][0] if k + 1 < len(q_starts) else n_rows
        seg_rows = []
        for ri in range(r_start, r_end_limit):
            if seg_rows and row_tags[ri] in ("section", "passage_label", "passage", "consumed_embed", "header"):
                break
            if seg_rows and (rows[ri]["y0"] - seg_rows[-1]["y1"]) > page_h * 0.14:
                break
            if row_tags[ri] == "text":
                seg_rows.append(rows[ri])
        if seg_rows:
            q_prelim.append(
                {
                    "qno": qno,
                    "r_start": r_start,
                    "r_end_limit": r_end_limit,
                    "seg_rows": seg_rows,
                    "text_y0": min(r["y0"] for r in seg_rows),
                    "text_y1": max(r["y1"] for r in seg_rows),
                }
            )

    # 预先计算短文区间纵坐标，排除落入短文内部的插图（如完形/阅读配图）
    passage_ybands: list[tuple[int, int]] = []
    for ps in passage_spans:
        sr = ps["start_row"]
        er = min(len(rows), ps["end_row"])
        if sr < er:
            passage_ybands.append((min(r["y0"] for r in rows[sr:er]), max(r["y1"] for r in rows[sr:er])))

    fig_owner: dict[int, int] = {}  # fig_idx -> q_prelim idx
    for fi, (fx, fy, fw, fh) in enumerate(valid_figs):
        f_cx = fx + fw * 0.5
        f_cy = fy + fh * 0.5
        if any((py0 - 10) <= f_cy <= (py1 + 10) for py0, py1 in passage_ybands):
            continue
        best_qi = None
        best_score = 1e9
        for qi, qp in enumerate(q_prelim):
            qy0, qy1 = qp["text_y0"], qp["text_y1"]
            # 插图可以与题干同行、位于题干右侧/左侧、或紧接在题干下方且在下一题/分节/短文边界之前
            next_y0 = q_prelim[qi + 1]["text_y0"] if qi + 1 < len(q_prelim) else page_h * 0.95
            barrier_y0 = next_y0
            for ri in range(qp["r_start"] + 1, qp["r_end_limit"]):
                if row_tags[ri] in ("section", "passage_label", "passage", "consumed_embed", "header"):
                    barrier_y0 = min(barrier_y0, float(rows[ri]["y0"]))
                    break
            if fy + fh < qy0 - 30 or fy >= barrier_y0 - 10:
                continue
            # 若插图主跨度位于下一题文字高度带且在右半栏（如 Q16 的图像），优先归属下一题
            if f_cy >= qy0 - 20 and f_cy <= qy1 + 25:
                dist = 0.0
            elif fy >= qy1 - 15 and fy + fh <= barrier_y0 + 25:
                dist = float(fy - qy1)
            else:
                dist = abs(f_cy - 0.5 * (qy0 + qy1)) + 80.0
            if dist < best_score:
                best_score = dist
                best_qi = qi
        if best_qi is not None:
            fig_owner[fi] = best_qi

    # 构建所有已占用的行号和未归属短字母块（用于防串行/跨题吸附手写单字母）
    assigned_row_indices = set()
    for qp in q_prelim:
        for r in qp["seg_rows"]:
            assigned_row_indices.add(r["idx"])

    # 6) 组装常规题 + 嵌入式题，按页面阅读顺序统一输出
    parsed_items: list[tuple[float, dict]] = []

    # 6a) 嵌入式题组（七选五/五选五）
    for ps in passage_spans:
        if ps["kind"] != "embed":
            continue
        pid = ps["pid"]
        pool_opts = ps["pool_opts"]
        ptext = ps["passage_text"]
        for val, r_sub in ps["seq_hits"]:
            qno = str(val)
            (bx0, by0, bx1, by1), (ax0, ay0, ax1, ay1) = _locate_embedded_blank_box(
                r_sub, qno, gray, page_w, page_h
            )
            ctx = _extract_cloze_context(ptext, qno) or f"第 {qno} 题"
            parsed_items.append(
                (
                    float(by0),
                    {
                        "number": qno,
                        "type": "choice",
                        "stem": ctx,
                        "options": pool_opts,
                        "passage_id": pid,
                        "bbox": [
                            round(bx0 * 1000 / page_w),
                            round(by0 * 1000 / page_h),
                            round(bx1 * 1000 / page_w),
                            round(by1 * 1000 / page_h),
                        ],
                        "answer_bbox": [
                            round(ax0 * 1000 / page_w),
                            round(ay0 * 1000 / page_h),
                            round(ax1 * 1000 / page_w),
                            round(ay1 * 1000 / page_h),
                        ],
                        "student_handwriting": "",
                        "is_blank": None,
                        "_skip_auto_fig": True,
                    },
                )
            )

    # 辅助函数：查找某行所属最近前驱 passage_id（遇到中间有 section 分隔则清零）
    def _resolve_passage_id(start_r: int) -> str | None:
        best_ps = None
        for ps in passage_spans:
            if ps["kind"] in ("cloze", "read") and ps["end_row"] <= start_r:
                best_ps = ps
        if best_ps is None:
            return None
        # 若短文结束行与本题起始行之间有大题分节标题（如“第二部分 / V. 情景交际”），则不再关联
        for ri in range(best_ps["end_row"], start_r):
            if row_tags[ri] == "section":
                return None
        return best_ps["pid"]

    cloze_boxes_by_pid = {
        ps["pid"]: ps.get("cloze_boxes") or {}
        for ps in passage_spans
        if ps.get("kind") == "cloze"
    }

    # 6b) 常规题目逐题解析
    prev_qtype = None
    for qi, qp in enumerate(q_prelim):
        qno = qp["qno"]
        seg_rows = qp["seg_rows"]
        anchor_row = seg_rows[0]
        anchor_yc = anchor_row["y_corr"]
        anchor_x0 = anchor_row["x0"]
        pid = _resolve_passage_id(qp["r_start"])

        # 归属本题的插图
        my_figs = [valid_figs[fi] for fi, owner_qi in fig_owner.items() if owner_qi == qi]
        other_figs = [valid_figs[fi] for fi, owner_qi in fig_owner.items() if owner_qi != qi]

        # 从 seg_rows 中分离：印刷题干行、选项行、图内轴标签、以及题干末尾手写单字母块
        stem_lines_txt: list[str] = []
        opt_lines_txt: list[str] = []
        stem_blocks_all: list[dict] = []
        all_non_other_fig_blocks: list[dict] = []
        cloze_opt_row = False

        for r in seg_rows:
            row_multi_lbl = _row_has_multi_labels(r)
            kept_row_blocks = []
            for b in r["blocks"]:
                # 若块落在归属其他题目的插图内部（如 Q16 图内的 $y$ / $x=1$ 落到 Q15 纵向高度），剔除
                if any(_rect_contains_center(of, b["cx"], b["cy"], pad=6) for of in other_figs):
                    continue
                kept_row_blocks.append(b)
            if not kept_row_blocks:
                continue
            all_non_other_fig_blocks.extend(kept_row_blocks)
            row_str = " ".join(b["text"] for b in kept_row_blocks).strip()
            # 判断本行是否主要是选项行或图内多标签行
            if _is_option_row_text(row_str) and r is not anchor_row:
                opt_lines_txt.extend(b["text"].strip() for b in kept_row_blocks)
            elif row_multi_lbl and r is not anchor_row:
                # 可能是 Q10 "A. O B O C O D O x" 或 Q65 "B. C. D."，尝试提取选项但绝不当成手写答案
                if OPT_MARKER_RE.search(row_str) or re.search(r"\b[A-D]\.", row_str):
                    opt_lines_txt.extend(b["text"].strip() for b in kept_row_blocks)
            else:
                # 检查题干行内部是否有行内选项（如完形 "36. A. late B. later C. latest"）
                if r is anchor_row and re.match(r"^\s*\d{1,3}\s*[.．]\s*[A-F]\s*[.．]", row_str):
                    cloze_opt_row = True
                    m_prefix = re.match(r"^\s*\d{1,3}\s*[.．]\s*(.*)$", kept_row_blocks[0]["text"].strip())
                    if m_prefix:
                        opt_lines_txt.append(m_prefix.group(1))
                    for kb in kept_row_blocks[1:]:
                        opt_lines_txt.append(kb["text"].strip())
                else:
                    stem_blocks_all.extend(kept_row_blocks)

        # 如果尚未进入选项行且题干末尾右侧（或紧邻下一行的孤立单字母）有手写字母块，进行精确匹配
        # 先从 stem_blocks_all 中找出最后一块“真正的印刷题干块”（排除末尾孤立单字母与 stray paren）
        printed_stem_blocks = []
        candidate_hw_blocks = []
        for bi, b in enumerate(stem_blocks_all):
            tok = _is_single_letter_token(b["text"])
            if tok and b is not stem_blocks_all[0]:
                # 排除属于本题插图内部的轴/选项字母（如 Q10 图下的 $B$/$C$/$D$）
                in_my_fig = any(_rect_contains_center(mf, b["cx"], b["cy"], pad=8) for mf in my_figs)
                parent_row = next((r for r in seg_rows if b in r["blocks"]), None)
                multi_lbl = _row_has_multi_labels(parent_row) if parent_row else False
                if not in_my_fig and not multi_lbl and b["w"] < page_w * 0.08:
                    candidate_hw_blocks.append(b)
                    continue
            if STRAY_PAREN_SYM_RE.match(b["text"].strip("$").strip()) and b is not stem_blocks_all[0]:
                continue
            printed_stem_blocks.append(b)

        if not printed_stem_blocks:
            printed_stem_blocks = list(stem_blocks_all)

        stem_last_blk = printed_stem_blocks[-1] if printed_stem_blocks else anchor_row["blocks"][-1]
        stem_end_yc = stem_last_blk["y_corr"]
        stem_last_x1 = stem_last_blk["box"][2]

        # 校验 candidate_hw_blocks 是否紧邻 stem 末尾（或左侧边距答题槽）
        hw_blk = None
        pre_hw_letter = ""
        for cb in candidate_hw_blocks:
            right_of_stem = (
                abs(cb["y_corr"] - stem_end_yc) <= 0.65 * med_h
                and cb["box"][0] >= stem_last_x1 - 60
                and cb["box"][0] <= stem_last_x1 + page_w * 0.14
            )
            left_margin = (
                abs(cb["y_corr"] - anchor_yc) <= 0.65 * med_h
                and cb["box"][2] <= anchor_x0 + 15
            )
            if right_of_stem or left_margin:
                tok = _is_single_letter_token(cb["text"])
                if tok:
                    hw_blk = cb
                    pre_hw_letter = tok
                    break

        # 拼装干净题干文本（按行聚合 printed_stem_blocks）
        line_groups: list[list[dict]] = []
        for b in printed_stem_blocks:
            if line_groups and abs(b["y_corr"] - line_groups[-1][0]["y_corr"]) <= 0.55 * med_h:
                line_groups[-1].append(b)
            else:
                line_groups.append([b])
        for lg in line_groups:
            lg.sort(key=lambda x: x["cx"])
            ln_txt = " ".join(x["text"] for x in lg).strip()
            # 清理单独被转成 $ ? $ 的标点
            if ln_txt == "$ ? $":
                ln_txt = "?"
            stem_lines_txt.append(ln_txt)

        # 提取选项
        opts = _extract_options_from_texts(opt_lines_txt)
        # 检查是否因完形填空选项行需要从短文关联句子作题干
        is_cloze_q = False
        if cloze_opt_row and pid and pid in passages and opts:
            ctx = _extract_cloze_context(passages[pid], qno, is_cloze=True)
            if ctx:
                stem_lines_txt = [f"{qno}. {ctx}"]
                is_cloze_q = True
            else:
                stem_lines_txt = [anchor_row["text"].strip()]
                is_cloze_q = True

        raw_stem = "\n".join(stem_lines_txt).strip() or f"{qno}."
        # 将未闭合左括号“（”补全显示为“（ ）”并清除括号内手写字母，保持题干干净
        clean_stem = re.sub(r"([（(])\s*[A-Ha-h]?\s*([）)])", r"\1 \2", raw_stem)
        if OPEN_BRACKET_END_RE.search(clean_stem):
            clean_stem = re.sub(r"[（(]\s*$", "（ ）", clean_stem)

        # 检查题干里是否已有内嵌单字母如 ( B )
        if not pre_hw_letter:
            for b in printed_stem_blocks:
                inner_m = re.search(r"[（(]\s*([A-Ha-h])\s*[）)]", b["text"])
                if inner_m:
                    pre_hw_letter = inner_m.group(1).upper()
                    break

        # 坐标范围
        use_blocks = all_non_other_fig_blocks or [b for r in seg_rows for b in r["blocks"]]
        x0 = min(b["box"][0] for b in use_blocks)
        y0 = min(b["box"][1] for b in use_blocks)
        x1 = max(b["box"][2] for b in use_blocks)
        y1_text = max(b["box"][3] for b in use_blocks)
        for fx, fy, fw, fh in my_figs:
            x0 = min(x0, fx)
            y0 = min(y0, fy)
            x1 = max(x1, fx + fw)
            y1_text = max(y1_text, fy + fh)

        # 查找下界用于 solution 题留白判断与 fill 局部横线不越界
        next_r_idx = qp["r_end_limit"]
        next_top = rows[next_r_idx]["y0"] if next_r_idx < n_rows else int(page_h * 0.93)

        # 检测题目局部横线（排除插图内部坐标轴，并限制不向下越入下一题）
        ul_roi_y0 = max(0, y0 - 10)
        ul_max_h = min(y1_text - y0 + 75, max(36, next_top - ul_roi_y0 - 2))
        ul_roi = [
            max(0, x0 - 20),
            ul_roi_y0,
            min(page_w - max(0, x0 - 20), max(x1 - x0 + 120, int(page_w * 0.72))),
            min(page_h - ul_roi_y0, ul_max_h),
        ]
        underlines = find_underlines(gray, ul_roi, figures=valid_figs)

        # 题型判定
        has_choice_bracket = bool(
            PAIRED_BRACKET_RE.search(raw_stem)
            or OPEN_BRACKET_END_RE.search(stem_last_blk["text"])
            or hw_blk is not None
        )
        is_last_q = qi == len(q_prelim) - 1
        has_single_opt_choice = len(opts) == 1 and (
            bool(CHOICE_STEM_HINT_RE.search(raw_stem))
            or prev_qtype == "choice"
            or (is_last_q and opts[0].startswith("A."))
        )
        # Q65 / Q10 图象选择题（无完整文本选项但含图内 A/B/C/D 标签或 B./C./D.）
        has_fig_opts = len(my_figs) >= 2 and (len(opts) >= 1 or "结构" in raw_stem or "大致为" in raw_stem)

        if len(opts) >= 2 or has_choice_bracket or has_single_opt_choice or has_fig_opts:
            qtype = "choice"
        elif "___" in raw_stem or len(underlines) > 0:
            qtype = "fill"
        else:
            qtype = "solution"
        prev_qtype = qtype

        y1 = y1_text if qtype in ("choice", "fill") else max(y1_text, min(next_top - 10, y1_text + int(page_h * 0.16)))

        # 答题区 answer_bbox 精确定位
        ans_bbox = None
        if qtype == "choice":
            # 1) 优先：题干右侧/槽内已定位的手写单字母块
            if hw_blk is not None:
                hx0, hy0, hx1, hy1 = hw_blk["box"]
                hcx = (hx0 + hx1) * 0.5
                ans_bbox = [
                    round(max(0, hcx - 55) * 1000 / page_w),
                    round(max(0, hy0 - 16) * 1000 / page_h),
                    round(min(page_w, hcx + 55) * 1000 / page_w),
                    round(min(page_h, hy1 + 16) * 1000 / page_h),
                ]
            # 2) 次优：成对括号 （ ） 或行末半开左括号 （
            if ans_bbox is None:
                for b in printed_stem_blocks:
                    bm = PAIRED_BRACKET_RE.search(b["text"])
                    if bm:
                        lx0, ly0, lx1, ly1 = b["box"]
                        cx = lx0 + (lx1 - lx0) * (bm.start() + max(1, (bm.end() - bm.start()) / 2.0)) / max(1, len(b["text"]))
                        ans_bbox = [
                            round(max(0, cx - 55) * 1000 / page_w),
                            round(max(0, ly0 - 16) * 1000 / page_h),
                            round(min(page_w, cx + 55) * 1000 / page_w),
                            round(min(page_h, ly1 + 16) * 1000 / page_h),
                        ]
                        break
                    om = OPEN_BRACKET_END_RE.search(b["text"])
                    if om:
                        lx0, ly0, lx1, ly1 = b["box"]
                        ans_bbox = [
                            round(max(0, lx1 - 15) * 1000 / page_w),
                            round(max(0, ly0 - 16) * 1000 / page_h),
                            round(min(page_w, lx1 + 98) * 1000 / page_w),
                            round(min(page_h, ly1 + 16) * 1000 / page_h),
                        ]
                        break
            # 3) 再次：无括号选择题（如英语单选/完形），取题号左侧空白作答槽
            if ans_bbox is None:
                fx0, fy0, _, fy1 = anchor_row["blocks"][0]["box"]
                if fx0 >= page_w * 0.055:
                    ans_bbox = [
                        round(max(0, fx0 - page_w * 0.085) * 1000 / page_w),
                        round(max(0, fy0 - 12) * 1000 / page_h),
                        round(max(24, fx0 - 4) * 1000 / page_w),
                        round(min(page_h, fy1 + 20) * 1000 / page_h),
                    ]
        elif qtype == "fill":
            if underlines:
                # 优先选择与题干基线一致或紧贴题干下方的横线
                stem_y1_max = max(b["box"][3] for b in printed_stem_blocks)
                sx, sy, ex = min(
                    underlines,
                    key=lambda u: (0 if u[1] <= stem_y1_max + 55 else 1, abs(u[1] - stem_y1_max)),
                )
                h_box = 64 if (ex - sx) > 260 else 54
                y_off = 52 if (ex - sx) > 260 else 44
                ans_bbox = [
                    round(sx * 1000 / page_w),
                    round(max(0, sy - y_off) * 1000 / page_h),
                    round(ex * 1000 / page_w),
                    round(min(page_h, sy - y_off + h_box) * 1000 / page_h),
                ]
        elif qtype == "solution" and y1 > y1_text + 36:
            ans_bbox = [
                round(x0 * 1000 / page_w),
                round(y1_text * 1000 / page_h),
                round(min(page_w * 0.94, x1 + 120) * 1000 / page_w),
                round(y1 * 1000 / page_h),
            ]

        # 题框精确收口：
        # - 完形填空若成功定位到短文内嵌空位，bbox 设在短文空位子框，并保留双候选答题框 [短文空位框, 下方选项行左侧边距槽]
        # - fill 填空题 right 取 max(text_x1, underline_ex) + 16（不强拉至 0.72*page_w），单行高 <48px 对称扩至 54px
        cloze_loc = cloze_boxes_by_pid.get(pid or "", {}).get(qno) if is_cloze_q else None
        if cloze_loc is not None:
            (cbx0, cby0, cbx1, cby1), (cax0, cay0, cax1, cay1) = cloze_loc
            q_bbox_norm = [
                round(cbx0 * 1000 / page_w),
                round(cby0 * 1000 / page_h),
                round(cbx1 * 1000 / page_w),
                round(cby1 * 1000 / page_h),
            ]
            cloze_ans_norm = [
                round(cax0 * 1000 / page_w),
                round(cay0 * 1000 / page_h),
                round(cax1 * 1000 / page_w),
                round(cay1 * 1000 / page_h),
            ]
            ans_bboxes_norm = [cloze_ans_norm]
            if ans_bbox is not None:
                ans_bboxes_norm.append(ans_bbox)
            ans_bbox = cloze_ans_norm
        else:
            ans_bboxes_norm = None
            out_y0, out_y1 = y0, y1
            if qtype == "fill":
                ul_ex_max = max((u[2] for u in underlines), default=x1)
                out_x1 = min(page_w * 0.95, max(x1, ul_ex_max) + 16)
                if (out_y1 - out_y0) < 48:
                    cy_fill = 0.5 * (out_y0 + out_y1)
                    out_y0 = max(0, int(round(cy_fill - 27)))
                    out_y1 = min(page_h, out_y0 + 54)
            elif is_cloze_q:
                out_x1 = x1
            else:
                out_x1 = min(page_w * 0.95, max(x1, int(page_w * 0.72)))
            q_bbox_norm = [
                round(x0 * 1000 / page_w),
                round(out_y0 * 1000 / page_h),
                round(out_x1 * 1000 / page_w),
                round(out_y1 * 1000 / page_h),
            ]

        q_entry = {
            "number": qno,
            "type": qtype,
            "stem": clean_stem,
            "options": opts or None,
            "passage_id": pid,
            "bbox": q_bbox_norm,
            "answer_bbox": ans_bbox,
            "student_handwriting": pre_hw_letter,
            "is_blank": False if pre_hw_letter else None,
            "_first_x0": int(anchor_x0),
            "_pre_ocr_confident": bool(hw_blk is not None),
            "_skip_auto_fig": True,
        }
        if ans_bboxes_norm is not None:
            q_entry["answer_bboxes"] = ans_bboxes_norm
        if my_figs:
            # 合并所有属于本题的插图外接框
            fx0 = min(f[0] for f in my_figs)
            fy0 = min(f[1] for f in my_figs)
            fx1 = max(f[0] + f[2] for f in my_figs)
            fy1 = max(f[1] + f[3] for f in my_figs)
            q_entry["figure_bbox"] = [
                round(fx0 * 1000 / page_w),
                round(fy0 * 1000 / page_h),
                round(fx1 * 1000 / page_w),
                round(fy1 * 1000 / page_h),
            ]
            q_entry["has_figure"] = True

        parsed_items.append((float(y0), q_entry))

    parsed_items.sort(key=lambda pair: pair[0])
    parsed_qs = [q for _, q in parsed_items]

    return build_freeform_page_from_vlm_json(
        {
            "passages": passages,
            "questions": parsed_qs,
            "_figures": valid_figs,
            "_ocr_blocks": blocks,
        },
        warped,
        page_w,
        page_h,
    )
