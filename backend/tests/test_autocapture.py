import asyncio
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import FIXTURE
from synth_capture import synthesize

from app.config import Settings
from app.engine import Engine
from app.state import Session
from app.vision.autocapture import AutoDetector, answer_fingerprint, same_answers
from app.vision.matcher import TemplateLibrary
from app.vision.stability import DetectionConfig, Stability


def paper():
    image = np.full((400, 600, 3), 45, np.uint8)
    cv2.rectangle(image, (130, 40), (470, 360), (215, 215, 215), -1)
    for y in range(70, 340, 22):
        cv2.putText(image, "ABCD 1234 example", (150, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (25, 25, 25), 1)
    return image


def feed(detector, frames, start=0):
    hits = []
    for index, frame in enumerate(frames):
        state, image = detector.update(frame, start + index / 15)
        if image is not None:
            hits.append(index)
    return hits, state


def test_still_once_and_global_light_change():
    d = Stability()
    image = paper()
    hits, _ = feed(d, [image] * 30)
    assert len(hits) == 1
    lighter = cv2.add(image, np.full_like(image, 10))
    assert not feed(d, [lighter] * 30, 2)[0]


def test_slow_motion_must_finish_before_capture():
    image = paper()
    frames = [cv2.warpAffine(image, np.float32([[1, 0, x], [0, 1, 0]]), (600, 400), borderValue=(45, 45, 45))
              for x in range(-100, 1, 2)]
    d = Stability()
    moving, _ = feed(d, frames)
    settled, _ = feed(d, [frames[-1]] * 20, len(frames) / 15)
    assert not moving
    assert len(settled) == 1


def test_remove_replace_and_minor_occlusion():
    image = paper()
    d = Stability()
    assert len(feed(d, [image] * 20)[0]) == 1
    hand = image.copy()
    cv2.rectangle(hand, (220, 140), (380, 260), (110, 110, 110), -1)
    assert not feed(d, [hand] * 4 + [image] * 20, 2)[0]
    blank = np.full_like(image, 45)
    feed(d, [blank] * 7, 4)
    assert len(feed(d, [image] * 20, 5)[0]) == 1  # 页面内容去重由后面的共同校验负责


def test_blank_blur_and_manual_lock_reset():
    d = Stability()
    assert not feed(d, [np.full((400, 600, 3), 240, np.uint8)] * 25)[0]
    image = paper()
    d.lock(image)
    assert not feed(d, [image] * 25, 2)[0]
    d.reset()
    assert len(feed(d, [image] * 25, 4)[0]) == 1
    blurry = cv2.GaussianBlur(image, (61, 61), 20)
    d.reset()
    assert not feed(d, [blurry] * 25)[0]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    lib = TemplateLibrary(tmp_path_factory.mktemp("templates"), FIXTURE)
    lib.load()
    return lib


@pytest.fixture(scope="module")
def examples():
    return (synthesize(FIXTURE, seed=1, angle=0)[0],
            synthesize(FIXTURE, seed=1, angle=8)[0],
            synthesize(FIXTURE, seed=1, angle=0, answers={"2": "A"})[0])


def test_same_page_rotation_but_modified_answer_is_new(library, examples):
    signatures = [answer_fingerprint(image, library.match(image)) for image in examples]
    assert same_answers(signatures[0], signatures[0])
    assert same_answers(signatures[0], signatures[1])
    assert not same_answers(signatures[0], signatures[2])


def test_unknown_page_is_rejected_and_retried_not_locked():
    calls = []
    library = SimpleNamespace(match=lambda image: calls.append(1))
    d = AutoDetector(library, replace(DetectionConfig(), retry_seconds=1))
    image = paper()
    for index in range(60):
        state, captured, match = d.step(image, index / 15)
        assert captured is None and match is None
    assert 1 < len(calls) < 5
    assert state["reason"] in ("unmatched", "settling")


async def test_manual_and_auto_reuse_do_not_add_questions(library, examples):
    settings = Settings(_env_file=None)
    session = Session(settings)
    engine = Engine(settings, session, None, library, SimpleNamespace())
    first = await engine.capture(examples[0], grade=False)
    await asyncio.gather(*list(engine.tasks.values()))
    before = len(session.data["questions"])
    again = await engine.capture(examples[1], grade=False, source="auto")
    assert again["capture_id"] == first["capture_id"] and again["reused"]
    assert len(session.data["questions"]) == before
    assert len(engine.captures) == 1
    changed = await engine.capture(examples[2], grade=False)
    await asyncio.gather(*list(engine.tasks.values()))
    assert changed["capture_id"] != first["capture_id"]
    assert len(engine.captures) == 2
    await engine.close()


async def test_auto_mode_change_resets_detector():
    settings = Settings(_env_file=None)
    engine = Engine(settings, Session(settings), None, None, SimpleNamespace())
    engine.stability.lock(paper())
    await engine.reset_detector()
    assert engine.stability.state == "idle" and engine.stability.reference is None
