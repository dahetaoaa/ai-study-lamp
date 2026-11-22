"""Unit & corpus regressions for freeform_ink_gate and HWR freeform enhancements."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pytest

from app.grading.hwr_client import merge_freeform_choice_result, parse_recognition
from app.vision.freeform_ink_gate import prepare_freeform_answer_blocks
from app.vision.matcher import TemplateLibrary, project, quad
from app.vision.page_ocr import rectify_paper

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(os.environ.get("VIDEO_CASES_ROOT", ROOT / "data/video_testset/recordings_20261117"))


def test_left_margin_clips_before_printed_qnum_and_paper_edge():
    # White page with dark left physical shadow (0..10) and printed "21." starting at x=120
    warped = np.full((300, 800, 3), 245, np.uint8)
    warped[:, :10] = 35
    cv2.putText(warped, "21. Look at this", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Deliberately oversized left_margin ans_bbox that overlaps both left shadow and "21."
    res = prepare_freeform_answer_blocks(
        warped,
        gray,
        [{"id": "a0", "bbox": [2, 75, 130, 50], "hint": "choice_letter"}],
        "choice",
        ocr_text_boxes=[{"box": [120, 85, 360, 118], "text": "21. Look at this"}],
        first_x0=120,
    )
    assert res["is_blank"] is True
    assert res["clean_ink_pixels"] == 0
    # Crop must not include x < 12 or x >= 117
    crop = cv2.imdecode(np.frombuffer(res["blocks"][0]["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[1] <= (117 - 12 + 2)
    assert float(crop.min()) > 200.0


def test_tilted_underline_and_brackets_without_ink_are_blank():
    warped = np.full((300, 800, 3), 245, np.uint8)
    gray_img = np.full((300, 800), 245, np.uint8)
    # Tilted printed underline (~5 deg)
    cv2.line(gray_img, (210, 150), (330, 160), 30, 2)
    # Empty printed parentheses
    cv2.putText(gray_img, "(   )", (450, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 30, 2)
    warped = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    res_fill = prepare_freeform_answer_blocks(
        warped,
        gray_img,
        [{"id": "a0", "bbox": [200, 120, 140, 55], "hint": "fill_blank"}],
        "fill",
        ocr_text_boxes=[],
    )
    assert res_fill["is_blank"] is True

    res_choice = prepare_freeform_answer_blocks(
        warped,
        gray_img,
        [{"id": "a0", "bbox": [440, 125, 100, 48], "hint": "choice_letter"}],
        "choice",
        ocr_text_boxes=[{"box": [200, 130, 535, 162], "text": "1. 下列结论正确的是 (   )"}],
    )
    assert res_choice["is_blank"] is True


def test_handwriting_inside_brackets_and_crossing_underline_is_kept():
    img = np.full((300, 800), 245, np.uint8)
    cv2.putText(img, "1. Equation", (100, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 30, 2)
    # Printed brackets & handwritten "B" inside
    cv2.putText(img, "(", (310, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 30, 2)
    cv2.putText(img, ")", (380, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 30, 2)
    cv2.putText(img, "B", (335, 156), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 25, 3)
    # Also test fill underline with handwritten "3" crossing the line
    cv2.line(img, (480, 160), (600, 160), 30, 2)
    cv2.putText(img, "3", (530, 163), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 25, 3)
    warped = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res_choice = prepare_freeform_answer_blocks(
        warped,
        img,
        [{"id": "a0", "bbox": [300, 120, 100, 52], "hint": "choice_letter"}],
        "choice",
        ocr_text_boxes=[
            {"box": [100, 130, 318, 162], "text": "1. Equation ("},
            {"box": [332, 124, 362, 164], "text": "$ B $"},
        ],
        pre_ocr_handwriting="B",
    )
    assert res_choice["is_blank"] is False
    assert res_choice["clean_ink_pixels"] >= 100
    assert res_choice["ink_bbox"] is not None

    res_fill = prepare_freeform_answer_blocks(
        warped,
        img,
        [{"id": "a0", "bbox": [475, 120, 130, 55], "hint": "fill_blank"}],
        "fill",
        ocr_text_boxes=[],
    )
    assert res_fill["is_blank"] is False
    assert res_fill["clean_ink_pixels"] >= 100


def test_mis_ocr_symbol_inside_choice_box_is_not_erased():
    img = np.full((300, 800), 245, np.uint8)
    cv2.putText(img, "C", (445, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 25, 3)
    warped = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    res = prepare_freeform_answer_blocks(
        warped,
        img,
        [{"id": "a0", "bbox": [410, 125, 95, 50], "hint": "choice_letter"}],
        "choice",
        ocr_text_boxes=[{"box": [440, 130, 476, 173], "text": "$ \\odot $"}],
        pre_ocr_handwriting="",
    )
    assert res["is_blank"] is False
    assert res["clean_ink_pixels"] >= 100


def test_spurious_far_pre_ocr_handwriting_does_not_override_blank_gate():
    warped = np.full((400, 800, 3), 245, np.uint8)
    cv2.putText(warped, "D", (520, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 30, 2)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    res = prepare_freeform_answer_blocks(
        warped,
        gray,
        [{"id": "a0", "bbox": [40, 100, 100, 50], "hint": "choice_letter"}],
        "choice",
        ocr_text_boxes=[{"box": [515, 280, 540, 305], "text": "D"}],
        pre_ocr_handwriting="D",
    )
    assert res["is_blank"] is True
    assert res["clean_ink_pixels"] == 0


def test_parse_recognition_freeform_and_fusion():
    # Template mode (freeform=False) unchanged
    assert parse_recognition('{"student_raw":"A)","low_confidence":false}', "choice", "stop") == ("", True)
    assert parse_recognition('{"student_raw":"A. absent","low_confidence":false}', "choice", "stop") == ("", True)

    # Freeform mode normalizes bracketed single letters, rejects options/footers/commentary
    assert parse_recognition('{"student_raw":"A)","low_confidence":false}', "choice", "stop", freeform=True) == ("A", False)
    assert parse_recognition('{"student_raw":"（B）","low_confidence":false}', "choice", "stop", freeform=True) == ("B", False)
    assert parse_recognition('{"student_raw":"A. absent","low_confidence":false}', "choice", "stop", freeform=True) == ("", True)
    assert parse_recognition('{"student_raw":"试卷第8页，共11页","low_confidence":false}', "solution", "stop", freeform=True) == ("", False)
    assert parse_recognition('{"student_raw":"故A错误；判别式大于0。故选C。","low_confidence":false}', "fill", "stop", freeform=True) == ("", False)

    # Fusion helper
    assert merge_freeform_choice_result("B", False, "C") == ("B", False)
    assert merge_freeform_choice_result("", True, "B", ink_pixels=200) == ("B", False)
    assert merge_freeform_choice_result("", False, "D", ink_pixels=0) == ("", False)
    assert merge_freeform_choice_result("A", True, "A", ink_pixels=180) == ("A", False)


@pytest.mark.skipif(not CORPUS.exists(), reason="用户视频样本未随源码提交")
def test_real_corpus_gt_blank_and_written_ink_gate_speed_and_accuracy():
    cache_path = ROOT / "data/diagnostics/spotting_raw_cache.json"
    if not cache_path.exists():
        pytest.skip("spotting_raw_cache.json 不存在")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    gt_lib = TemplateLibrary(ROOT / "data/templates")
    gt_lib.load()

    blank_total = blank_ok = written_total = written_ok = 0
    gate_ms_total = 0.0
    for item in cache["items"]:
        img = cv2.imread(item["img_path"])
        warped = cv2.imread(item["warped_path"])
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        h_free = np.asarray(item["homography"], dtype=np.float64) if "homography" in item else rectify_paper(img)[1]
        gt_m = gt_lib.match(img)
        h_tpl_to_warped = np.linalg.inv(h_free) @ gt_m.homography
        is_written = item["paper_id"] == "math_xmyz_2025_oct" and item["page_no"] == 1

        for q in gt_m.template.page["questions"]:
            areas = []
            for a in q.get("answer_areas", []):
                pts = np.float32(project(quad(a["bbox"]), h_tpl_to_warped))
                x0, y0 = int(round(pts[:, 0].min())), int(round(pts[:, 1].min()))
                x1, y1 = int(round(pts[:, 0].max())), int(round(pts[:, 1].max()))
                areas.append({"id": a.get("id", "a0"), "bbox": [x0, y0, x1 - x0, y1 - y0], "hint": a.get("hint", "")})

            t0 = perf_counter()
            res = prepare_freeform_answer_blocks(warped, gray, areas, q["type"], item["blocks"])
            gate_ms_total += (perf_counter() - t0) * 1000.0

            if is_written:
                written_total += 1
                if not res["is_blank"]:
                    written_ok += 1
            else:
                blank_total += 1
                if res["is_blank"]:
                    blank_ok += 1

    avg_ms = gate_ms_total / max(1, blank_total + written_total)
    assert blank_total == 75 and blank_ok == 75
    assert written_total == 14 and written_ok == 14
    assert avg_ms < 3.0, f"Gate too slow: {avg_ms:.2f}ms/q"
