import json

import cv2
import numpy as np
from conftest import FIXTURE
from replay_capture import evaluate, replay
from synth_capture import synthesize

from app.vision.matcher import TemplateLibrary


def test_replay_uses_recorded_timestamps_and_reuses_same_page(tmp_path):
    library = TemplateLibrary(tmp_path / "templates", FIXTURE)
    library.load()
    image, _ = synthesize(FIXTURE, seed=1, angle=0)
    # 录制保留原分辨率，不能缩成低清视频后仍要求同样的模板匹配内点数。
    blank = np.full_like(image, 45)
    images = [blank] * 5 + [image] * 20 + [blank] * 7 + [image] * 20
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (1920, 1080))
    assert writer.isOpened()
    for frame in images:
        writer.write(frame)
    writer.release()
    frames = tmp_path / "frames.jsonl"
    frames.write_text("\n".join(json.dumps({"index": i, "t": i / 15}) for i in range(len(images))))
    first = replay(path, library, tmp_path / "first", frames_file=frames)
    second = replay(path, library, tmp_path / "second", frames_file=frames)
    assert first["timestamp_source"] == "recorded"
    assert first["new_captures"] == 1 and first["reused"] == 1
    assert [(h["t"], h["action"]) for h in first["triggers"]] == [(h["t"], h["action"]) for h in second["triggers"]]
    assert first["evaluation"] is None  # 没有标注就不编造准确率
    assert first["frames"] == len(images)


def test_evaluation_distinguishes_reuse_and_false_triggers():
    hits = [{"t": 1.6, "action": "capture"}, {"t": 4.0, "action": "reuse"}, {"t": 6.5, "action": "capture"}]
    report = evaluate(hits, {"events": [{"ready": 1, "until": 2, "action": "capture"},
                                        {"ready": 3.5, "until": 4.5, "action": "reuse"},
                                        {"ready": 9, "until": 10}], "invalid": [{"start": 6, "end": 7}]})
    assert report["matched"] == 2
    assert len(report["misses"]) == 1
    assert report["stable_to_trigger_ms"] == [600, 500]
    assert len(report["invalid_interval_triggers"]) == 1
