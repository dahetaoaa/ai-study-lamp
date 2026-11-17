"""AKAZE 特征常驻内存，所有接口由编排器在线程池中调用。"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LONG_EDGE = 1280
MAX_POINTS = 2500


def features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, LONG_EDGE / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.AKAZE_create().detectAndCompute(small, None)
    if descriptors is None:
        return np.empty((0, 2), np.float32), np.empty((0, 61), np.uint8)
    indices = sorted(range(len(keypoints)), key=lambda i: keypoints[i].response, reverse=True)[:MAX_POINTS]
    return np.float32([keypoints[i].pt for i in indices]) / scale, descriptors[indices]


def build_features(image_path: Path, output_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取模板图：{image_path.name}")
    points, descriptors = features(image)
    np.savez_compressed(
        output_path,
        points=points,
        descriptors=descriptors,
        image_shape=np.array(image.shape),
        version=np.array([1]),
    )
    return points, descriptors


def quad(bbox):
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def project(points, homography):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), homography).reshape(-1, 2).tolist()


@dataclass
class TemplatePage:
    paper: dict
    page: dict
    image: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray


@dataclass
class Match:
    template: TemplatePage
    homography: np.ndarray
    inliers: int
    ratio: float


class TemplateLibrary:
    def __init__(self, directory: Path, fixture: Path | None = None):
        self.directory, self.fixture = directory, fixture
        self.papers: dict[str, dict] = {}
        self.pages: list[TemplatePage] = []

    def load(self):
        from app.telemetry import log

        files = sorted(self.directory.glob("*/template.json"))
        if self.fixture:
            files.append(self.fixture)
        for path in files:
            try:
                paper = json.loads(path.read_text())
                loaded = []
                for page in paper["pages"]:
                    image_path = (path.parent / page["image_file"]).resolve()
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise ValueError("模板图不存在")
                    output = path.parent / page.get("features_file", f"page_{page['page_no']:02d}_akaze.npz")
                    try:
                        with np.load(output, allow_pickle=False) as saved:
                            points, desc = saved["points"], saved["descriptors"]
                            if tuple(saved["image_shape"]) != image.shape:
                                raise ValueError("特征尺寸已失效")
                    except (OSError, ValueError, KeyError):
                        points, desc = build_features(image_path, output)
                    loaded.append(TemplatePage(paper, page, image, points, desc))
                self.papers[paper["paper_id"]] = paper
                self.pages.extend(loaded)
            except (OSError, ValueError, KeyError, cv2.error) as error:
                log.warning("跳过无效模板 %s：%s", path.parent.name, type(error).__name__)

    def match(self, image) -> Match | None:
        points, desc = features(image)
        if len(desc) < 45:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        candidates = []
        for page in self.pages:
            if len(page.descriptors) < 2:
                continue
            pairs = bf.knnMatch(page.descriptors, desc, k=2)
            good = [
                a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.78 * b.distance
            ]
            candidates.append((len(good), page, good))
        best = None
        for _, page, good in sorted(candidates, key=lambda c: c[0], reverse=True)[:3]:
            if len(good) < 45:
                continue
            src = np.float32([page.points[m.queryIdx] for m in good])
            dst = np.float32([points[m.trainIdx] for m in good])
            h, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0)
            if h is None or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-8:
                continue
            h /= h[2, 2]
            n = int(mask.sum())
            ratio = n / len(good)
            determinant = np.linalg.det(h)
            corners = np.float32(project(quad([0, 0, page.image.shape[1], page.image.shape[0]]), h))
            area = cv2.contourArea(corners)
            if (
                n >= 45
                and ratio >= 0.25
                and 1e-5 < determinant < 100
                and cv2.isContourConvex(corners)
                and 0.03 * image.shape[0] * image.shape[1] < area < 2 * image.shape[0] * image.shape[1]
            ):
                if best is None or n > best.inliers:
                    best = Match(page, h, n, ratio)
        return best
