"""高精度纸面透视拉平与文本行微倾角精校（Lab 低频光照归一化 + RANSAC/Huber 四边拟合 + Deskew）。

提供生产级接口：
- `detect_paper_quad(image)` -> `(quad_in_cam, paper_ratio)`
- `rectify_paper(image)` -> `(warped_bgr, h_page_to_cam, page_quad_in_cam, page_w, page_h)`
- `estimate_text_line_skew(warped)` -> `(theta_deg, line_count)`
- `resize_for_ocr(warped, max_long_edge=OCR_MAX_LONG_EDGE)` -> `(resized_bgr, scale)`
"""

from __future__ import annotations

import cv2
import numpy as np

NORM_W = 1400
NORM_H = 1980
MIN_PORTRAIT_H = 1750
MAX_PORTRAIT_H = 2150
OCR_MAX_LONG_EDGE = 1280

DESKEW_MIN_DEG = 0.4
DESKEW_MAX_DEG = 12.0
DESKEW_MIN_LINES = 6


def order_quad(pts: np.ndarray) -> np.ndarray:
    """将任意四个顶点排成 [tl, tr, br, bl] 顺时针顺序。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(4)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _fallback_paper_quad(image: np.ndarray) -> tuple[np.ndarray, float]:
    """亮色纸面轮廓兜底检测；若整幅已是近景纸面则退回全图留边四边形。"""
    h, w = image.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(image, (max(16, round(w * scale)), max(16, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (blur > max(130, int(np.percentile(blur, 45)))).astype(np.uint8) * 255
    mask = cv2.bitwise_or(otsu, bright)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(small.shape[0] * small.shape[1])
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        area = float(cv2.contourArea(cnt))
        ratio = area / max(1.0, total_area)
        if ratio < 0.16:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = order_quad(approx.reshape(4, 2) / scale)
        else:
            rect = cv2.minAreaRect(cnt)
            q = order_quad(cv2.boxPoints(rect) / scale)
        q[:, 0] = np.clip(q[:, 0], 0, w - 1)
        q[:, 1] = np.clip(q[:, 1], 0, h - 1)
        return q, round(ratio, 3)

    margin_x, margin_y = w * 0.02, h * 0.02
    fallback_q = np.array(
        [[margin_x, margin_y], [w - margin_x, margin_y], [w - margin_x, h - margin_y], [margin_x, h - margin_y]],
        dtype=np.float32,
    )
    return fallback_q, 0.96


def _fit_line_ransac(pts: np.ndarray, iters: int = 120, thresh: float = 2.0) -> tuple[float, float, float, float] | None:
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


def _intersect_lines(
    l1: tuple[float, float, float, float],
    l2: tuple[float, float, float, float],
) -> np.ndarray | None:
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    det = vx1 * (-vy2) - vy1 * (-vx2)
    if abs(det) < 1e-6:
        return None
    dx, dy = x2 - x1, y2 - y1
    t = (dx * (-vy2) - dy * (-vx2)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def detect_paper_quad(image: np.ndarray) -> tuple[np.ndarray, float]:
    """高精度纸面四角检测：
    1) Lab 空间去字迹（L 闭运算）+ 低频光照归一化，消除桌面台灯明暗梯度；
    2) 结合局部亮度差与中性低色度（a/b 偏白）+ 强梯度边缘阻隔，防止与邻近桌面/底纸粘连；
    3) 沿粗框四边内法向搜寻“暗桌面 -> 亮纸面”单向梯度峰值点，RANSAC + Huber 拟合四条直线求交。
    """
    if image is None or not image.size:
        raise ValueError("输入图像为空")
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    h, w = bgr.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    sw, sh = max(16, round(w * scale)), max(16, round(h * scale))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # 1. 形态学闭运算擦除黑字与笔迹，仅保留干净纸面与外边界
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    L_clean = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k_close)
    L_blur = cv2.GaussianBlur(L_clean, (5, 5), 0).astype(np.float32)

    # 2. 低频背景估计（1/16 下采样后高斯模糊再上采样）
    tiny_w, tiny_h = max(8, sw // 16), max(8, sh // 16)
    bg_tiny = cv2.GaussianBlur(cv2.resize(L_blur, (tiny_w, tiny_h), interpolation=cv2.INTER_AREA), (9, 9), 0)
    bg_L = cv2.resize(bg_tiny, (sw, sh), interpolation=cv2.INTER_LINEAR)

    gx = cv2.Sobel(L_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_blur, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)

    # 自适应白纸色度参考（高亮度低梯度像素中位 a/b），消除相机 AWB 漂移导致的固定偏移误切
    l_hi = float(np.percentile(L_blur, 75))
    g_lo = float(np.percentile(gmag, 75))
    ref_mask = (L_blur >= l_hi) & (gmag <= g_lo)
    if int(np.count_nonzero(ref_mask)) < 20:
        ref_mask = L_blur >= l_hi
    a_ref = float(np.median(a[ref_mask]))
    b_ref = float(np.median(b[ref_mask]))
    chroma_dist = np.hypot(a - a_ref, b - b_ref)

    white_score = (L_blur - bg_L) + 2.0 * (1.8 - chroma_dist)
    edge_barrier = gmag > np.percentile(gmag, 88)
    score_u8 = np.clip((white_score + 15.0) * 5.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[edge_barrier] = 0
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_open, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_paper_quad(bgr)

    cx_img, cy_img = sw * 0.5, sh * 0.5

    def _cnt_score(c: np.ndarray) -> float:
        ar = cv2.contourArea(c)
        if ar < 0.12 * sw * sh:
            return -1.0
        moments = cv2.moments(c)
        if moments["m00"] <= 0:
            return float(ar)
        mx, my = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        dist = float(np.hypot((mx - cx_img) / sw, (my - cy_img) / sh))
        return float(ar * max(0.2, 1.0 - dist))

    cnt = max(contours, key=_cnt_score)
    ratio = float(cv2.contourArea(cnt)) / float(sw * sh)
    if ratio < 0.12:
        return _fallback_paper_quad(bgr)

    hull = cv2.convexHull(cnt)
    init_q = order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))

    # 2b. A4 物理纵横比校验与侧暗/贴边截断恢复：
    # 竖版试卷在侧向强光照衰减或画面贴边时，低频背景与梯度阻隔可能将纸面偏暗侧切断。
    # 若主轮廓偏窄 (H/W > 1.48，标准 A4 为 1.414)，合并同色度侧向碎片并在 H/W > 1.58 时沿中性纸面列向外扩展。
    top_w = float(np.hypot(init_q[1][0] - init_q[0][0], init_q[1][1] - init_q[0][1]))
    bot_w = float(np.hypot(init_q[2][0] - init_q[3][0], init_q[2][1] - init_q[3][1]))
    left_h = float(np.hypot(init_q[3][0] - init_q[0][0], init_q[3][1] - init_q[0][1]))
    right_h = float(np.hypot(init_q[2][0] - init_q[1][0], init_q[2][1] - init_q[1][1]))
    avg_w = 0.5 * (top_w + bot_w)
    avg_h = 0.5 * (left_h + right_h)

    border_pinned_edges: set[int] = set()
    if avg_w > 20.0 and (avg_h / avg_w) > 1.48:
        main_mask = np.zeros((sh, sw), np.uint8)
        cv2.drawContours(main_mask, [cnt], -1, 1, -1)
        paper_pixels = main_mask > 0
        paper_cd_p90 = float(np.percentile(chroma_dist[paper_pixels], 90))
        cd_tol = max(2.4, paper_cd_p90 + 0.7)
        paper_L_med = float(np.median(L_blur[paper_pixels]))
        bg_pixels = ~paper_pixels
        desk_L_est = (
            float(np.percentile(L_blur[bg_pixels], 35))
            if int(np.count_nonzero(bg_pixels)) > 50
            else paper_L_med - 40.0
        )
        l_floor = max(desk_L_est + 6.0, paper_L_med - 75.0)

        target_w = avg_h / 1.414
        need_w = max(0.0, target_w - avg_w)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        pts_list = [cnt]
        for c2 in contours:
            if c2 is cnt or cv2.contourArea(c2) < 0.008 * sw * sh:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(c2)
            oy = max(0, min(by + bh, y2 + h2) - max(by, y2))
            gap_x = max(0, max(bx, x2) - min(bx + bw, x2 + w2))
            if oy >= 0.40 * min(bh, h2) and gap_x <= max(16.0, need_w * 1.25):
                c2_mask = np.zeros((sh, sw), np.uint8)
                cv2.drawContours(c2_mask, [c2], -1, 1, -1)
                c2_px = c2_mask > 0
                if float(np.median(chroma_dist[c2_px])) <= cd_tol and float(np.median(L_blur[c2_px])) >= l_floor:
                    pts_list.append(c2)

        if len(pts_list) > 1:
            hull = cv2.convexHull(np.vstack(pts_list))
            init_q = order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))
            top_w = float(np.hypot(init_q[1][0] - init_q[0][0], init_q[1][1] - init_q[0][1]))
            bot_w = float(np.hypot(init_q[2][0] - init_q[3][0], init_q[2][1] - init_q[3][1]))
            avg_w = 0.5 * (top_w + bot_w)
            need_w = max(0.0, avg_h / 1.414 - avg_w)

        if (avg_h / max(1.0, avg_w)) > 1.58 and need_w > 8.0:
            y0_s = max(0, int(min(init_q[0][1], init_q[1][1]) + 0.1 * avg_h))
            y1_s = min(sh, int(max(init_q[2][1], init_q[3][1]) - 0.1 * avg_h))
            if y1_s > y0_s + 10:
                best_side: int | None = None
                best_span = 0
                for side_dir in (1, -1):
                    x_start = (
                        int(round(0.5 * (init_q[1][0] + init_q[2][0])))
                        if side_dir == 1
                        else int(round(0.5 * (init_q[0][0] + init_q[3][0])))
                    )
                    if not (0 <= x_start < sw):
                        continue
                    prev_L = float(np.median(L_blur[y0_s:y1_s, x_start]))
                    max_steps = int(round(need_w * 1.15))
                    span = 0
                    misses = 0
                    for step in range(1, max_steps + 1):
                        xc = x_start + side_dir * step
                        if xc < 0 or xc >= sw:
                            break
                        col_cd = float(np.median(chroma_dist[y0_s:y1_s, xc]))
                        col_L = float(np.median(L_blur[y0_s:y1_s, xc]))
                        if col_cd <= cd_tol and col_L >= l_floor and col_L >= prev_L - 18.0:
                            span = step
                            misses = 0
                            prev_L = 0.7 * prev_L + 0.3 * col_L
                        else:
                            misses += 1
                            if misses >= 4:
                                break
                    if span > best_span:
                        best_span = span
                        best_side = side_dir
                if best_side is not None and best_span >= max(8, int(need_w * 0.35)):
                    if best_side == 1:
                        ev_top, ev_bot = init_q[1] - init_q[0], init_q[2] - init_q[3]
                        init_q[1] += (ev_top / max(1.0, float(np.hypot(ev_top[0], ev_top[1])))) * best_span
                        init_q[2] += (ev_bot / max(1.0, float(np.hypot(ev_bot[0], ev_bot[1])))) * best_span
                        if init_q[1][0] >= sw - 2.0 and init_q[2][0] >= sw - 2.0:
                            border_pinned_edges.add(1)
                    else:
                        ev_top, ev_bot = init_q[0] - init_q[1], init_q[3] - init_q[2]
                        init_q[0] += (ev_top / max(1.0, float(np.hypot(ev_top[0], ev_top[1])))) * best_span
                        init_q[3] += (ev_bot / max(1.0, float(np.hypot(ev_bot[0], ev_bot[1])))) * best_span
                        if init_q[0][0] <= 1.0 and init_q[3][0] <= 1.0:
                            border_pinned_edges.add(3)
                    init_q[:, 0] = np.clip(init_q[:, 0], 0, sw - 1)
                    init_q[:, 1] = np.clip(init_q[:, 1], 0, sh - 1)

    # 3. 沿四边内法向向量化搜寻“暗桌面 -> 亮纸面”最强单向阶跃梯度，RANSAC + Huber 拟合直线
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    fitted_lines: list[tuple[float, float, float, float] | None] = []
    band = max(14.0, 0.07 * min(sw, sh))
    d_offsets = np.arange(-int(band), int(band) + 1, dtype=np.float32)
    s_samples = np.linspace(0.08, 0.92, 64, dtype=np.float32)

    for ei, (i0, i1) in enumerate(edges):
        if ei in border_pinned_edges:
            fitted_lines.append(None)
            continue
        p0, p1 = init_q[i0], init_q[i1]
        ev = p1 - p0
        elen = float(np.hypot(ev[0], ev[1]))
        if elen < 20.0:
            fitted_lines.append(None)
            continue
        tx, ty = ev[0] / elen, ev[1] / elen
        nx, ny = -ty, tx

        cx = p0[0] + s_samples[:, None] * ev[0]
        cy = p0[1] + s_samples[:, None] * ev[1]
        fx = cx + d_offsets[None, :] * nx
        fy = cy + d_offsets[None, :] * ny
        px = np.rint(fx).astype(np.int32)
        py = np.rint(fy).astype(np.int32)
        valid = (px >= 3) & (px < sw - 3) & (py >= 3) & (py < sh - 3)
        px_safe = np.clip(px, 0, sw - 1)
        py_safe = np.clip(py, 0, sh - 1)
        dg = gx[py_safe, px_safe] * nx + gy[py_safe, px_safe] * ny
        dg = np.where(valid, dg, -1e9)

        best_idx = np.argmax(dg, axis=1)
        best_vals = dg[np.arange(len(s_samples)), best_idx]
        keep = best_vals > 18.0
        if int(keep.sum()) >= 12:
            pts = np.stack([fx[keep, best_idx[keep]], fy[keep, best_idx[keep]]], axis=1).astype(np.float32)
            fitted_lines.append(_fit_line_ransac(pts, iters=120, thresh=2.0))
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
    pad_x, pad_y = w * 0.01, h * 0.01
    q_out[:, 0] = np.clip(q_out[:, 0], -pad_x, w - 1 + pad_x)
    q_out[:, 1] = np.clip(q_out[:, 1], -pad_y, h - 1 + pad_y)
    return q_out, round(ratio, 3)


def estimate_text_line_skew(warped: np.ndarray) -> tuple[float, int]:
    """在拉平后的正视图上通过缩略图黑帽 + 横向闭运算提取文本行连通域，估计中位数倾角（度）。

    返回 `(theta_deg, valid_line_count)`：
    - `theta_deg > 0` 表示文本行从左到右略微向下倾斜，对应 `cv2.getRotationMatrix2D(..., theta_deg, 1.0)` 可将其旋平。
    """
    if warped is None or not warped.size:
        return 0.0, 0
    h, w = warped.shape[:2]
    scale = min(1.0, 700.0 / max(h, w))
    sw, sh = max(16, round(w * scale)), max(16, round(h * scale))
    small = cv2.resize(warped, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    k_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bh)
    _, bw = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    lines = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_line)

    # 屏蔽外周 4% 边缘，避免纸张边框或桌面边缘线干扰
    mx, my = max(4, int(sw * 0.04)), max(4, int(sh * 0.04))
    lines[:my, :] = 0
    lines[-my:, :] = 0
    lines[:, :mx] = 0
    lines[:, -mx:] = 0

    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(lines)
    angles: list[float] = []
    weights: list[float] = []
    min_cw = sw * 0.08
    max_ch = sh * 0.06
    for i in range(1, n_cc):
        x, y, cw, ch, _ = stats[i]
        if cw < min_cw or ch < 3 or ch > max_ch or cw / max(1, ch) < 2.5:
            continue
        ys, xs = np.nonzero(labels[y : y + ch, x : x + cw] == i)
        if len(xs) < 12:
            continue
        dx = xs.astype(np.float64) - float(xs.mean())
        dy = ys.astype(np.float64) - float(ys.mean())
        cov_xx = float(np.dot(dx, dx))
        cov_xy = float(np.dot(dx, dy))
        cov_yy = float(np.dot(dy, dy))
        if cov_xx <= cov_yy:
            continue
        ang = float(np.degrees(0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)))
        if abs(ang) <= DESKEW_MAX_DEG:
            angles.append(ang)
            weights.append(float(cw))

    if len(angles) < DESKEW_MIN_LINES or sum(weights) < sw:
        return 0.0, len(angles)

    order = np.argsort(angles)
    sa = np.asarray(angles, dtype=np.float64)[order]
    sw_arr = np.asarray(weights, dtype=np.float64)[order]
    cum = np.cumsum(sw_arr)
    med = float(sa[np.searchsorted(cum, cum[-1] * 0.5)])
    med = max(-DESKEW_MAX_DEG, min(DESKEW_MAX_DEG, med))
    return round(med, 3), len(angles)


def resize_for_ocr(warped: np.ndarray, max_long_edge: int = OCR_MAX_LONG_EDGE) -> tuple[np.ndarray, float]:
    """将正视图按长边 `1280` 档位友好比例等比缩放以送入整页 OCR。"""
    ph, pw = warped.shape[:2]
    scale = min(1.0, float(max_long_edge) / max(1, pw, ph))
    if scale >= 1.0:
        return warped, 1.0
    ow, oh = max(16, round(pw * scale)), max(16, round(ph * scale))
    return cv2.resize(warped, (ow, oh), interpolation=cv2.INTER_AREA), scale


def rectify_paper(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """将检测到的试卷四边形透视拉平为标准正视图，并执行文本行微倾角精校（同步合成到 Homography）。

    返回: `(warped_bgr, homography_page_to_cam, page_quad_in_cam, page_w, page_h)`
    其中 `homography_page_to_cam` 始终满足:
    `project(point_on_warped_page, homography_page_to_cam) -> point_on_original_camera_image`
    """
    q, _ = detect_paper_quad(image)
    top_w = float(np.hypot(q[1][0] - q[0][0], q[1][1] - q[0][1]))
    bot_w = float(np.hypot(q[2][0] - q[3][0], q[2][1] - q[3][1]))
    left_h = float(np.hypot(q[3][0] - q[0][0], q[3][1] - q[0][1]))
    right_h = float(np.hypot(q[2][0] - q[1][0], q[2][1] - q[1][1]))
    avg_w = max(320.0, (top_w + bot_w) * 0.5)
    avg_h = max(320.0, (left_h + right_h) * 0.5)

    # 竖版试卷统一到接近 A4 比例（NORM_W=1400, 高度限制在 [1750, 2150] 避免过度拉伸）
    if avg_w > avg_h * 1.15:
        dst_w = NORM_H
        dst_h = max(1100, min(1600, round(NORM_H * (avg_h / avg_w))))
    else:
        dst_w = NORM_W
        dst_h = max(MIN_PORTRAIT_H, min(MAX_PORTRAIT_H, round(NORM_W * (avg_h / avg_w))))

    src_page = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    h_page_to_cam = cv2.getPerspectiveTransform(src_page, q).astype(np.float64)
    h_cam_to_page = cv2.getPerspectiveTransform(q, src_page).astype(np.float64)
    warped = cv2.warpPerspective(image, h_cam_to_page, (dst_w, dst_h), borderValue=(255, 255, 255))

    # 二次精校：文本行微倾角检测与 Homography 同步合成
    theta, n_lines = estimate_text_line_skew(warped)
    if n_lines >= DESKEW_MIN_LINES and DESKEW_MIN_DEG <= abs(theta) <= DESKEW_MAX_DEG:
        cx, cy = dst_w * 0.5, dst_h * 0.5
        r_2x3 = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        m_deskew = np.vstack([r_2x3, [0.0, 0.0, 1.0]]).astype(np.float64)
        m_deskew_inv = np.linalg.inv(m_deskew)
        h_cam_to_page = m_deskew @ h_cam_to_page
        h_page_to_cam = h_page_to_cam @ m_deskew_inv
        warped = cv2.warpPerspective(image, h_cam_to_page, (dst_w, dst_h), borderValue=(255, 255, 255))

    return warped, h_page_to_cam, q, dst_w, dst_h
