"""自然图像裁剪与保守空白预判；不向 HWR 发送背景差分图。"""

import cv2
import numpy as np


def ink_mask(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 13) > 0


def crop_answers(image, match):
    template = match.template
    height, width = template.image.shape
    warped = cv2.warpPerspective(
        image, np.linalg.inv(match.homography), (width, height), borderValue=(255, 255, 255)
    )
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 轻微失焦/透视插值会改变印刷边缘；模板膨胀只供预判，避免把边缘误当笔迹。
    baseline = ink_mask(template.image)
    observed = ink_mask(gray)
    dilated = cv2.dilate(baseline.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    result = {}
    for question in template.page["questions"]:
        blocks, blank_flags = [], []
        for area in question.get("answer_areas", []):
            x, y, w, h = map(int, area["bbox"])
            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            # 只用中心 84% 测墨迹，括号与横线仍保留给识别模型。
            dx, dy = max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)
            sl = np.s_[y1 + dy : y2 - dy, x1 + dx : x2 - dx]
            added = observed[sl] & ~dilated[sl]
            ratio = float(added.mean()) if added.size else 1.0
            blank_flags.append(ratio < 0.003)
            crop = warped[max(0, y - 16) : min(height, y + h + 16), max(0, x - 16) : min(width, x + w + 16)]
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if ok:
                blocks.append({"jpeg": encoded.tobytes(), "hint": area.get("hint", "math_steps")})
        result[question["id"]] = {"blocks": blocks, "is_blank": bool(blank_flags) and all(blank_flags)}
    return result
