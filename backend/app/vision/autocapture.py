"""稳定候选验证与同页答题区去重；纯视觉，不访问云端。"""
from time import perf_counter

import cv2
import numpy as np

from app.vision.matcher import project, quad
from app.vision.stability import DetectionConfig, Stability


class AutoDetector:
    def __init__(self, library, config=None):
        self.library = library
        self.stability = Stability(config or DetectionConfig())

    def step(self, frame, now):
        state, candidate = self.stability.update(frame, now)
        if candidate is None:
            return state, None, None
        started = perf_counter()
        match = self.library.match(candidate)
        state["match_ms"] = round((perf_counter() - started) * 1000, 2)
        reason = "unmatched"
        if match:
            h, w = candidate.shape[:2]
            corners = np.float32(project(quad([0, 0, match.template.page["width_px"],
                                              match.template.page["height_px"]]), match.homography))
            visible, _ = cv2.intersectConvexConvex(corners, np.float32(quad([0, 0, w, h])))
            ratio = float(visible / max(1, cv2.contourArea(corners)))
            state.update(paper_id=match.template.paper["paper_id"], page_no=match.template.page["page_no"],
                         inliers=match.inliers, visible_ratio=round(ratio, 3))
            if ratio >= 0.9:
                state["reason"] = "ready"
                return state, candidate, match
            reason = "incomplete_page"
        self.stability.reject(reason, now)
        state.update(stability="rejected", reason=reason)
        return state, None, None


def page_key(match):
    return match.template.paper["paper_id"], match.template.page["page_no"]


def answer_fingerprint(image, match):
    """匹配页身份后对齐答题区，只比较实际笔迹图像，不比较模板答案。"""
    height, width = match.template.image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aligned = cv2.warpPerspective(gray, np.linalg.inv(match.homography), (width, height), borderValue=255)
    ink = cv2.adaptiveThreshold(aligned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 13)
    patches = []
    for q in match.template.page["questions"]:
        if q["type"] == "listening":
            continue
        areas = q.get("answer_areas") or [{"bbox": q["bbox"]}]
        for area in areas:
            x, y, w, h = map(int, area["bbox"])
            patch = ink[max(0, y - 8):min(height, y + h + 8), max(0, x - 8):min(width, x + w + 8)]
            if not patch.size:
                continue
            ph, pw = patch.shape
            factor = min(1, 320 / max(ph, pw))
            patch = cv2.resize(patch, (max(1, round(pw * factor)), max(1, round(ph * factor)))) > 100
            patches.append(patch)
    return patches


def same_answers(a, b):
    """保守去重：允许小配准偏移；任一局部答题内容明显改变即保留新批改。"""
    if not a or len(a) != len(b):
        return False
    for first, second in zip(a, b):
        if first.shape != second.shape:
            return False
        h, w = first.shape
        padded = np.pad(second, 2)
        shifted = min((padded[y:y + h, x:x + w] for y in range(5) for x in range(5)),
                      key=lambda v: np.count_nonzero(first != v))
        # 透视插值会让同一笔画边缘偏移一像素；只把容差范围之外的新/缺失墨迹算变化。
        kernel = np.ones((3, 3), np.uint8)
        first_near = cv2.dilate(first.astype(np.uint8), kernel) > 0
        second_near = cv2.dilate(shifted.astype(np.uint8), kernel) > 0
        diff = (first & ~second_near) | (shifted & ~first_near)
        # 按小块比较，不能用整张试卷的巨大印刷背景淹没一个改写的选项/数字。
        for y in range(0, h, 24):
            for x in range(0, w, 24):
                tile = np.s_[y:y + 24, x:x + 24]
                count = int(np.count_nonzero(diff[tile]))
                ink = max(np.count_nonzero(first[tile]), np.count_nonzero(shifted[tile]))
                if count > max(8, ink * 0.18):
                    return False
    return True
