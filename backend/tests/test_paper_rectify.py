"""Regression tests for freeform AutoDetector filtering and paper_rectify geometry/deskew."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.vision.autocapture import AutoDetector
from app.vision.matcher import TemplateLibrary, project, quad
from app.vision.paper_rectify import (
    OCR_MAX_LONG_EDGE,
    detect_paper_quad,
    estimate_text_line_skew,
    order_quad,
    rectify_paper,
    resize_for_ocr,
)
from app.vision.stability import DetectionConfig

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(os.environ.get("VIDEO_CASES_ROOT", ROOT / "data/video_testset/recordings_20261117"))
REC_DIR = ROOT / "data/capture_recordings"


def _make_exam_frame(*, with_text: bool = True, text_skew_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((1080, 1920, 3), 42, dtype=np.uint8)
    paper = np.full((1400, 1000, 3), 245, dtype=np.uint8)
    if with_text:
        for y in range(120, 1260, 70):
            cv2.putText(
                paper,
                f"1. Example printed question line y={y} A.1 B.2 C.3 D.4",
                (80, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (30, 30, 30),
                2,
            )
    if abs(text_skew_deg) > 1e-3:
        # 仅旋转纸内印刷文字，使纸张外框保持正矩形而内部文字产生倾角，验证二次精校
        r = cv2.getRotationMatrix2D((500.0, 700.0), -text_skew_deg, 1.0)
        paper = cv2.warpAffine(paper, r, (1000, 1400), borderValue=(245, 245, 245))
    src = np.float32([[0, 0], [1000, 0], [1000, 1400], [0, 1400]])
    dst = np.float32([[520, 50], [1340, 80], [1390, 1020], [480, 995]])
    h_p2c = cv2.getPerspectiveTransform(src, dst)
    cv2.warpPerspective(paper, h_p2c, (1920, 1080), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
    return img, dst


def test_autocapture_default_disables_freeform_and_filters_blank_transition():
    empty_lib = SimpleNamespace(match=lambda _img: None)
    default_det = AutoDetector(empty_lib, DetectionConfig())
    assert default_det.allow_freeform is False

    # 1) 轻微失焦/无文字的空白纸过渡帧与 2) 满屏纯白无文字纸面均必须被 freeform 过滤
    blank_paper_center = np.full((1080, 1920, 3), 240, dtype=np.uint8)
    blank_paper_center[:, :120] = 42
    blank_paper_center[:, -120:] = 42
    blur_frame, _ = _make_exam_frame(with_text=True)
    blur_frame = cv2.GaussianBlur(blur_frame, (41, 41), 12.0)
    for bad_frame in (blank_paper_center, blur_frame):
        det = AutoDetector(empty_lib, DetectionConfig(), allow_freeform=True)
        triggers = []
        for idx in range(25):
            state, cand, _ = det.step(bad_frame, idx / 15.0)
            if cand is not None:
                triggers.append(state)
        assert triggers == []

    # 放入有清晰印刷文字的试卷帧，应正常放行 freeform
    text_frame, _ = _make_exam_frame(with_text=True)
    det2 = AutoDetector(empty_lib, DetectionConfig(), allow_freeform=True)
    hits = []
    for idx in range(25):
        state, cand, _ = det2.step(text_frame, idx / 15.0)
        if cand is not None:
            hits.append(state)
    assert len(hits) == 1
    assert hits[0]["mode"] == "freeform"
    assert hits[0]["edge_ratio"] >= 0.006


def test_rectify_paper_deskew_and_homography_roundtrip():
    skewed_img, _ = _make_exam_frame(with_text=True, text_skew_deg=4.5)
    warped, h_page_to_cam, _, dst_w, dst_h = rectify_paper(skewed_img)

    assert dst_w == 1400
    assert 1750 <= dst_h <= 2150

    res_skew, n_lines = estimate_text_line_skew(warped)
    assert n_lines >= 6
    assert abs(res_skew) < 0.6

    ocr_img, _ = resize_for_ocr(warped)
    assert max(ocr_img.shape[:2]) == OCR_MAX_LONG_EDGE

    h_cam_to_page = np.linalg.inv(h_page_to_cam)
    pts_page = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h], [321.5, 654.25]])
    pts_cam = np.float32(project(pts_page, h_page_to_cam))
    pts_back = np.float32(project(pts_cam, h_cam_to_page))
    rt_err = float(np.max(np.linalg.norm(pts_page - pts_back, axis=1)))
    assert rt_err < 0.01


def _poly_iou(q1: np.ndarray, q2: np.ndarray, h: int, w: int) -> tuple[float, float]:
    m1 = np.zeros((h, w), np.uint8)
    m2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m1, [np.int32(np.round(q1))], 1)
    cv2.fillPoly(m2, [np.int32(np.round(q2))], 1)
    inter = float(np.count_nonzero(m1 & m2))
    union = float(np.count_nonzero(m1 | m2))
    a1 = float(np.count_nonzero(m1))
    return inter / max(1.0, union), inter / max(1.0, a1)


@pytest.mark.skipif(not CORPUS.exists(), reason="用户视频样本未随源码提交")
def test_real_captures_rectify_accuracy_and_deskew():
    gt_lib = TemplateLibrary(ROOT / "data/templates")
    gt_lib.load()
    cases = sorted(CORPUS.glob("rec_*/cap_*.json"))
    assert len(cases) == 9
    for jpath in cases:
        img = cv2.imread(str(jpath.with_suffix(".jpg")))
        h, w = img.shape[:2]
        m = gt_lib.match(img)
        assert m is not None
        tpl_w, tpl_h = m.template.image.shape[1], m.template.image.shape[0]
        gt_q = order_quad(np.float32(project(quad([0, 0, tpl_w, tpl_h]), m.homography)))
        left_slot_q = np.float32(project(quad([0, 0, 110, tpl_h]), m.homography))

        warped, h_page_to_cam, q, dst_w, dst_h = rectify_paper(img)
        assert dst_w == 1400
        assert 1750 <= dst_h <= 2150

        res_skew, _ = estimate_text_line_skew(warped)
        assert abs(res_skew) < 0.6, (jpath.name, res_skew)

        h_cam_to_page = np.linalg.inv(h_page_to_cam)
        pts_page = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h], [555.5, 1111.1]])
        pts_back = np.float32(project(project(pts_page, h_page_to_cam), h_cam_to_page))
        assert float(np.max(np.linalg.norm(pts_page - pts_back, axis=1))) < 0.01

        warped_cam_q = order_quad(np.float32(project(quad([0, 0, dst_w, dst_h]), h_page_to_cam)))
        iou, _ = _poly_iou(gt_q, warped_cam_q, h, w)
        _, left_cov = _poly_iou(left_slot_q, warped_cam_q, h, w)
        assert iou >= 0.90, (jpath.name, iou)
        assert left_cov >= 0.995, (jpath.name, left_cov)

        q_det, _ = detect_paper_quad(img)
        assert q_det.shape == (4, 2)


def test_rectify_paper_lateral_lighting_gradient_clipped_edge():
    h, w = 1920, 1080
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (95, 115, 130)
    paper = np.full((1400, 1000, 3), 240, dtype=np.uint8)
    for y in range(120, 1280, 75):
        cv2.putText(paper, f"6. Question line y={y}", (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2)
        cv2.putText(paper, "( B )", (820, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2)
    src = np.float32([[0, 0], [1000, 0], [1000, 1400], [0, 1400]])
    dst = np.float32([[110, 220], [1079, 225], [1065, 1590], [85, 1570]])
    h_p2c = cv2.getPerspectiveTransform(src, dst)
    cv2.warpPerspective(paper, h_p2c, (w, h), dst=img, borderMode=cv2.BORDER_TRANSPARENT)

    xs = np.linspace(1.0, 0.56, w, dtype=np.float32)[None, :, None]
    dip = 1.0 - 0.12 * np.exp(-0.5 * ((np.arange(w, dtype=np.float32) - 760.0) / 55.0) ** 2)[None, :, None]
    shaded = np.clip(img.astype(np.float32) * xs * dip, 0, 255).astype(np.uint8)

    warped, h_page_to_cam, q_det, dst_w, dst_h = rectify_paper(shaded)
    assert dst_w == 1400
    assert 1750 <= dst_h <= 2050
    warped_cam_q = order_quad(np.float32(project(quad([0, 0, dst_w, dst_h]), h_page_to_cam)))
    iou, rec = _poly_iou(dst, warped_cam_q, h, w)
    assert iou >= 0.90 and rec >= 0.98
    assert float(q_det[:, 0].max()) >= w * 0.97

    h_cam_to_page = np.linalg.inv(h_page_to_cam)
    right_bracket_cam = np.float32(project(quad([810, 800, 130, 60]), h_p2c))
    right_bracket_warped = np.float32(project(right_bracket_cam, h_cam_to_page))
    assert float(right_bracket_warped[:, 0].min()) >= 0.0
    assert float(right_bracket_warped[:, 0].max()) <= float(dst_w)


@pytest.mark.skipif(not CORPUS.exists(), reason="用户视频样本未随源码提交")
def test_real_replay_triggers_rectify_accuracy_and_deskew():
    gt_lib = TemplateLibrary(ROOT / "data/templates")
    gt_lib.load()
    cases = sorted(CORPUS.glob("baseline_rec_*/capture_*.jpg"))
    assert len(cases) == 9
    ious = []
    for ipath in cases:
        img = cv2.imread(str(ipath))
        h, w = img.shape[:2]
        m = gt_lib.match(img)
        assert m is not None
        tpl_w, tpl_h = m.template.image.shape[1], m.template.image.shape[0]
        gt_q = order_quad(np.float32(project(quad([0, 0, tpl_w, tpl_h]), m.homography)))

        warped, h_page_to_cam, _, dst_w, dst_h = rectify_paper(img)
        assert dst_w == 1400
        assert 1750 <= dst_h <= 2150
        res_skew, _ = estimate_text_line_skew(warped)
        assert abs(res_skew) < 0.6, (ipath.name, res_skew)

        warped_cam_q = order_quad(np.float32(project(quad([0, 0, dst_w, dst_h]), h_page_to_cam)))
        iou, _ = _poly_iou(gt_q, warped_cam_q, h, w)
        ious.append(iou)
        assert iou >= 0.89, (ipath.name, iou)
        if ipath.name == "capture_000086.jpg":
            assert iou >= 0.90, iou
            h_tpl_to_warped = np.linalg.inv(h_page_to_cam) @ m.homography
            for qitem in m.template.page["questions"]:
                aa = qitem["answer_areas"][0]["bbox"]
                pts = np.float32(project(quad(aa), h_tpl_to_warped))
                assert float(pts[:, 0].min()) >= 0.0 and float(pts[:, 0].max()) <= float(dst_w)
                assert float(pts[:, 1].min()) >= 0.0 and float(pts[:, 1].max()) <= float(dst_h)
    assert float(np.mean(ious)) >= 0.915


@pytest.mark.skipif(not (REC_DIR / "rec_7267479ca145/video.avi").exists(), reason="录制视频未随源码提交")
def test_real_video_replay_zero_spurious_triggers(tmp_path):
    gt_lib = TemplateLibrary(ROOT / "data/templates")
    gt_lib.load()
    empty_lib = TemplateLibrary(tmp_path, None)
    empty_lib.load()

    expected_pages = {"rec_5422b97b81b3": 5, "rec_7267479ca145": 4}
    for clip, expected_count in expected_pages.items():
        cap = cv2.VideoCapture(str(REC_DIR / clip / "video.avi"))
        meta = [json.loads(ln) for ln in (REC_DIR / clip / "frames.jsonl").read_text().splitlines() if ln.strip()]
        det = AutoDetector(empty_lib, DetectionConfig(), allow_freeform=True)
        gt_pages = set()
        spurious = 0
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = float(meta[idx]["t"]) if idx < len(meta) else idx / 15.0
                _, cand, _ = det.step(frame, now)
                if cand is not None:
                    gm = gt_lib.match(cand)
                    if gm is None:
                        spurious += 1
                    else:
                        gt_pages.add((gm.template.paper["paper_id"], gm.template.page["page_no"]))
                    det.stability.lock(cand)
                idx += 1
        finally:
            cap.release()
        assert spurious == 0, clip
        assert len(gt_pages) == expected_count, (clip, gt_pages)
