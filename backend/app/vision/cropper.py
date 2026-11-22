"""局部印刷对齐后的空白判定；送HWR的是原始彩色笔迹裁剪，不是差分图。"""
import cv2
import numpy as np


def ink_mask(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 13) > 0


def _patch(image, x, y, w, h, fill=0):
    """带边界填充的整数裁剪，允许纸边附近做对称的局部配准搜索。"""
    out = np.full((h, w), fill, dtype=image.dtype)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x1 > x0 and y1 > y0:
        out[y0 - y:y1 - y, x0 - x:x1 - x] = image[y0:y1, x0:x1]
    return out


def _local_shift(baseline, observed, box):
    x, y, w, h = box
    context, radius = 48, 24
    reference = _patch(baseline, x-context, y-context, w+2*context, h+2*context)
    if np.count_nonzero(reference) < 16:
        return 0, 0, 1.0  # 纯白模板区域无需依靠不存在的印刷字来配准
    search = _patch(observed, x-context-radius, y-context-radius,
                    w+2*(context+radius), h+2*(context+radius))
    scores = cv2.matchTemplate(search.astype(np.float32), reference.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    return location[0]-radius, location[1]-radius, float(score)


def _affine_print_mask(baseline, observed, box, dx, dy):
    """少量残差时用答题区外的印刷上下文细化仿射，中央手写不参与拟合。"""
    x, y, w, h = box
    pad = 64
    reference = _patch(baseline, x-pad, y-pad, w+2*pad, h+2*pad)
    sample = _patch(observed, x-pad, y-pad, w+2*pad, h+2*pad)
    mask = np.full(reference.shape, 255, np.uint8)
    mask[pad-12:pad+h+12, pad-12:pad+w+12] = 0
    if np.count_nonzero(reference & (mask > 0)) < 40:
        return None
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    try:
        score, matrix = cv2.findTransformECC(
            cv2.GaussianBlur(reference.astype(np.float32), (5, 5), 0),
            cv2.GaussianBlur(sample.astype(np.float32), (5, 5), 0), matrix,
            cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-4), mask,
        )
    except cv2.error:
        return None
    if score < 0.65 or not 0.85 < np.linalg.det(matrix[:, :2]) < 1.15 or np.max(np.abs(matrix[:, :2]-np.eye(2))) > 0.1:
        return None
    aligned = cv2.warpAffine(reference, matrix, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_NEAREST)
    return cv2.dilate(aligned, np.ones((5, 5), np.uint8))[pad:pad+h, pad:pad+w] > 0


def _retained(mask):
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    retained = np.zeros_like(mask)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= 4:
            retained |= components == component
    return retained


def crop_answers(image, match):
    template = match.template
    height, width = template.image.shape
    warped = cv2.warpPerspective(
        image, np.linalg.inv(match.homography), (width, height), borderValue=(255, 255, 255)
    )
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    visibility = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8),
                                     np.linalg.inv(match.homography), (width, height),
                                     flags=cv2.INTER_NEAREST, borderValue=0) > 0
    baseline = ink_mask(template.image).astype(np.uint8)
    observed = ink_mask(gray).astype(np.uint8)
    result = {}
    for question in template.page["questions"]:
        areas, diagnostics = [], []
        for index, area in enumerate(question.get("answer_areas", [])):
            x, y, w, h = map(int, area["bbox"])
            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
            if x2 <= x1 or y2 <= y1:
                continue
            dx, dy, score = _local_shift(baseline, observed, (x, y, w, h))
            # 保持观察图的坐标，移动模板印刷遮罩去匹配局部纸张弯曲造成的位移。
            ref = _patch(baseline, x1-dx-2, y1-dy-2, x2-x1+4, y2-y1+4)
            printed = cv2.dilate(ref, np.ones((5, 5), np.uint8))[2:-2, 2:-2] > 0
            added = (observed[y1:y2, x1:x2] > 0) & ~printed
            # 不再只看中心84%：边缘的浅笔迹也不能因位置而被丢掉。
            retained = _retained(added)
            pixels = int(np.count_nonzero(retained))
            refined = False
            if 0 < pixels < retained.size * 0.03:
                affine = _affine_print_mask(baseline, observed, (x1, y1, x2-x1, y2-y1), dx, dy)
                if affine is not None:
                    retained = _retained((observed[y1:y2, x1:x2] > 0) & ~affine)
                    pixels = int(np.count_nonzero(retained))
                    refined = True
            coverage = float(np.count_nonzero(visibility[y1:y2, x1:x2]) / max(1, w*h))
            uncertain = score < 0.35 or coverage < 0.98
            # 配准后只在没有任何成片新增墨迹时确认空白，不用整框面积稀释细负号/小数字。
            blank = not uncertain and pixels == 0
            ys, xs = np.where(retained)
            ink_bbox = [x1+int(xs.min()), y1+int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)] if pixels else None
            diagnostics.append({"index": index, "shift": [dx, dy], "alignment_score": round(score, 3), "affine_refined": refined,
                                "ink_pixels": pixels, "is_blank": blank, "alignment_uncertain": uncertain,
                                "visible_ratio": round(coverage, 3), "ink_bbox": ink_bbox})
            crop_box = ink_bbox if ink_bbox and not blank and not uncertain else [x, y, w, h]
            cx, cy, cw, ch = crop_box
            pad = 10 if crop_box is ink_bbox else 16
            crop = warped[max(0, cy-pad):min(height, cy+ch+pad), max(0, cx-pad):min(width, cx+cw+pad)]
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if ok:
                areas.append({"jpeg": encoded.tobytes(), "hint": area.get("hint", "math_steps"),
                              "area_index": index, "alignment_uncertain": uncertain, "is_blank": blank,
                              "ink_pixels": pixels})
        nonblank = [area for area in areas if not area["is_blank"]]
        primary = max((d for d in diagnostics if not d["is_blank"]), key=lambda d: d["ink_pixels"], default=None)
        result[question["id"]] = {
            "blocks": nonblank or areas,
            "is_blank": bool(diagnostics) and all(d["is_blank"] for d in diagnostics),
            "primary_area_index": primary["index"] if primary else 0,
            "ink_bbox": primary["ink_bbox"] if primary and not primary["alignment_uncertain"] else None,
            "area_diagnostics": diagnostics,
        }
    return result
