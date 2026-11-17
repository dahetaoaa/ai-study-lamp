from time import perf_counter

import numpy as np
import pytest
from conftest import FIXTURE
from synth_capture import synthesize

from app.vision.cropper import crop_answers
from app.vision.matcher import TemplateLibrary, project, quad
from app.vision.stability import Stability


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    value = TemplateLibrary(tmp_path_factory.mktemp("empty_templates"), FIXTURE)
    value.load()
    return value


@pytest.mark.parametrize("seed,angle", [(0, -15), (1, -8), (2, 0), (3, 8), (4, 15)])
def test_match_projection(library, seed, angle):
    image, truth = synthesize(FIXTURE, seed=seed, angle=angle)
    started = perf_counter()
    match = library.match(image)
    ms = (perf_counter() - started) * 1000
    assert match is not None
    assert match.template.paper["paper_id"] == truth["paper_id"]
    errors = []
    for q in match.template.page["questions"]:
        expected = np.array(project(quad(q["bbox"]), np.array(truth["homography"])))
        actual = np.array(project(quad(q["bbox"]), match.homography))
        errors.extend(np.linalg.norm(actual - expected, axis=1))
    error = max(errors)
    print(f"\n匹配 seed={seed} 旋转={angle}° 耗时={ms:.1f}ms 内点={match.inliers} 题框最大误差={error:.2f}px")
    assert error < 12
    crops = crop_answers(image, match)
    assert not crops["fixture_math_q1"]["is_blank"]
    assert not crops["fixture_math_q2"]["is_blank"]
    assert crops["fixture_math_q3"]["is_blank"]


def test_blank_capture(library):
    image, _ = synthesize(FIXTURE, mode="blank", seed=10)
    match = library.match(image)
    assert match
    assert all(value["is_blank"] for value in crop_answers(image, match).values())


def test_unknown(library):
    assert library.match(np.zeros((1080, 1920, 3), np.uint8)) is None


def test_stability_locks():
    image, _ = synthesize(FIXTURE, seed=3)
    stability = Stability()
    captured = []
    for i in range(30):
        state, frame = stability.update(image, i / 15)
        if frame is not None:
            captured.append(frame)
    assert len(captured) == 1
    assert state["stability"] == "locked"


def test_stability_rearm_with_mock_motion():
    stability = Stability()
    stability.state = "locked"
    for i in range(6):
        now = i / 15
        moving = np.full((180, 320, 3), 20 + int(now * 750) % 180, np.uint8)
        state, _ = stability.update(moving, now)
    assert state["stability"] == "idle"
