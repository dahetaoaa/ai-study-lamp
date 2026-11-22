"""Blank-vs-handwriting regressions; optional real-video cases remain outside Git."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.grading.hwr_client import parse_recognition
from app.vision.cropper import crop_answers
from app.vision.matcher import TemplateLibrary

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(os.environ.get("VIDEO_CASES_ROOT", ROOT / "data/video_testset/recordings_20261117"))


def fixture(stroke=None, shift=(0, 0), at_edge=False):
    clean = np.full((300, 500), 255, np.uint8)
    cv2.putText(clean, "Example printed question", (95, 105), cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 1)
    cv2.line(clean, (155, 175), (285, 175), 30, 1)
    page = {"questions": [{"id": "q", "type": "fill", "answer_areas": [{"bbox": [150, 125, 140, 58]}]}]}
    dx, dy = shift
    observed = cv2.warpAffine(clean, np.float32([[1, 0, dx], [0, 1, dy]]), (500, 300), borderValue=255)
    if stroke is not None:
        x = 151 if at_edge else 198
        cv2.line(observed, (x, 155), (x + 16, 155), stroke, 1)
    image = cv2.cvtColor(observed, cv2.COLOR_GRAY2BGR)
    match = SimpleNamespace(template=SimpleNamespace(image=clean, page=page), homography=np.eye(3))
    return image, match


@pytest.mark.parametrize("shift", [(0, 0), (9, 12), (-10, 16), (15, -8)])
def test_print_displacement_is_not_handwriting(shift):
    image, match = fixture(shift=shift)
    result = crop_answers(image, match)["q"]
    assert result["is_blank"]
    assert result["ink_bbox"] is None


@pytest.mark.parametrize("intensity", [20, 180, 225])
@pytest.mark.parametrize("edge", [False, True])
def test_faint_thin_and_edge_strokes_are_not_blank(intensity, edge):
    image, match = fixture(stroke=intensity, at_edge=edge)
    result = crop_answers(image, match)["q"]
    assert not result["is_blank"]
    x, y, w, h = result["ink_bbox"]
    assert y <= 155 < y + h
    assert x <= (151 if edge else 198) + 8 < x + w


def test_ink_bbox_stays_in_observed_template_coordinates():
    image, match = fixture(stroke=180, shift=(9, 12))
    result = crop_answers(image, match)["q"]
    assert not result["is_blank"]
    x, y, w, h = result["ink_bbox"]
    assert x <= 206 < x + w and y <= 155 < y + h
    assert result["area_diagnostics"][0]["shift"] != [0, 0]


def test_primary_area_follows_handwriting_not_first_candidate():
    image, match = fixture(stroke=180)
    match.template.page["questions"][0]["answer_areas"].insert(0, {"bbox": [10, 125, 65, 58]})
    result = crop_answers(image, match)["q"]
    assert not result["is_blank"] and result["primary_area_index"] == 1
    assert [block["area_index"] for block in result["blocks"]] == [1]


def test_unobserved_warp_border_is_not_confirmed_blank():
    image, match = fixture()
    match.homography = np.float32([[1, 0, 600], [0, 1, 0], [0, 0, 1]])
    result = crop_answers(image, match)["q"]
    assert not result["is_blank"]
    assert result["area_diagnostics"][0]["visible_ratio"] == 0
    assert result["blocks"][0]["alignment_uncertain"]


def test_obscured_print_context_is_not_confirmed_blank():
    _, match = fixture()
    image = np.full((300, 500, 3), 220, np.uint8)
    result = crop_answers(image, match)["q"]
    assert not result["is_blank"]
    assert result["blocks"][0]["alignment_uncertain"]


@pytest.mark.parametrize("content,qtype,finish", [
    ('{"student_raw":"1234567890123456789', "fill", "length"),
    ('{"student_raw":null,"low_confidence":false}', "fill", "stop"),
    ('{"student_raw":"A","low_confidence":"false"}', "choice", "stop"),
    ('{"student_raw":"A. absent","low_confidence":false}', "choice", "stop"),
    ('{"student_raw":"33","low_confidence":false}', "choice", "stop"),
    ('{"student_raw":"' + '1234567890' * 40 + '","low_confidence":false}', "fill", "stop"),
])
def test_malformed_or_printed_output_is_not_reliable_answer(content, qtype, finish):
    assert parse_recognition(content, qtype, finish) == ("", True)


@pytest.mark.parametrize("content,expected", [
    ('{"student_raw":"B","low_confidence":false}', ("B", False)),
    ('{"student_raw":"","low_confidence":false}', ("", False)),
    ('{"student_raw":"A","low_confidence":true}', ("A", True)),
    ('B', ("B", False)),
    ('E', ("E", False)),
    ('h', ("H", False)),
    ('{"student_raw":"E","low_confidence":false}', ("E", False)),
    ('```json\n{"student_raw":"C","low_confidence":false}\n```', ("C", False)),
])
def test_valid_recognition_contract(content, expected):
    assert parse_recognition(content, "choice", "stop") == expected


@pytest.mark.skipif(not CORPUS.exists(), reason="用户视频样本未随源码提交")
def test_real_video_captures_keep_all_blank_and_written_cases():
    blank_count = written_count = 0
    for path in sorted(CORPUS.glob("rec_*/cap_*.json")):
        observed = json.loads(path.read_text())
        cap = observed["capture"]
        template_dir = ROOT / "data/templates" / cap["paper_id"]
        paper = json.loads((template_dir / "template.json").read_text())
        page = next(p for p in paper["pages"] if p["page_no"] == cap["page_no"])
        clean = cv2.imread(str(template_dir / page["image_file"]), 0)
        image = cv2.imread(str(path.with_suffix(".jpg")))
        match = SimpleNamespace(template=SimpleNamespace(image=clean, page=page), homography=np.array(cap["homography"]))
        expected_blank = not (cap["paper_id"] == "math_xmyz_2025_oct" and cap["page_no"] == 1)
        for qid, result in crop_answers(image, match).items():
            assert result["is_blank"] == expected_blank, (path.name, qid, result["area_diagnostics"])
            if expected_blank:
                blank_count += 1
                assert result["ink_bbox"] is None
            else:
                written_count += 1
                assert result["ink_bbox"] is not None
    assert blank_count == 75 and written_count == 14


@pytest.mark.skipif(not list(CORPUS.glob("baseline_rec_*/report.json")), reason="用户视频回放候选未随源码提交")
def test_decoded_replay_candidates_rematch_and_keep_blank_state():
    library = TemplateLibrary(ROOT / "data/templates")
    library.load()
    blank_count = written_count = captures = 0
    for path in sorted(CORPUS.glob("baseline_rec_*/report.json")):
        for hit in json.loads(path.read_text())["triggers"]:
            if "image" not in hit:
                continue
            image = cv2.imread(str(path.parent / hit["image"]))
            match = library.match(image)
            assert match and match.template.paper["paper_id"] == hit["paper_id"]
            assert match.template.page["page_no"] == hit["page_no"]
            captures += 1
            expected_blank = not (hit["paper_id"] == "math_xmyz_2025_oct" and hit["page_no"] == 1)
            for qid, result in crop_answers(image, match).items():
                assert result["is_blank"] == expected_blank, (path.parent.name, hit["frame"], qid, result["area_diagnostics"])
                blank_count += expected_blank
                written_count += not expected_blank
    assert captures == 9 and blank_count == 75 and written_count == 14
