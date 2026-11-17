import json
import time
from dataclasses import asdict
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.grading.client import ModelClients
from app.main import create_app
from app.vision.recorder import Recorder, RecordingError
from app.vision.stability import DetectionConfig


class FakeCamera:
    backend = "mock"

    def __init__(self):
        self.calls = 0

    def latest_packet(self):
        self.calls += 1
        frame = np.full((80, 120, 3), 220, np.uint8)
        cv2.putText(frame, str(self.calls), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (10, 10, 10), 1)
        return self.calls, time.monotonic(), frame


def make_recorder(tmp_path, **kwargs):
    s = Settings(_env_file=None, camera_backend="mock", recording_enabled=True,
                 recordings_dir=tmp_path, recording_fps=30, recording_min_free_bytes=0, **kwargs)
    return Recorder(s, FakeCamera(), asdict(DetectionConfig()))


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("wait timeout")
        time.sleep(.01)


def test_record_stop_and_timestamp_metadata(tmp_path):
    r = make_recorder(tmp_path)
    try:
        info = r.start("缓慢放下")
        assert info["active"]
        with pytest.raises(RecordingError):
            r.start()
        r.mark()
        r.note("detector", reason="settling", camera_sequence=1)
        wait_for(lambda: r.status()["frames"] >= 3)
    finally:
        result = r.stop()
    assert not result["active"] and result["reason"] == "user"
    assert r.stop() == r.status()
    assert r.list()[0]["id"] == result["id"]
    video = r.file(result["id"], "video.avi")
    cap = cv2.VideoCapture(str(video))
    try:
        assert cap.isOpened()
        assert cap.read()[0]
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == result["frames"]
    finally:
        cap.release()
    frames = [json.loads(s) for s in r.file(result["id"], "frames.jsonl").read_text().splitlines()]
    assert [f["index"] for f in frames] == list(range(len(frames)))
    assert all(b["t"] >= a["t"] for a, b in pairwise(frames))
    events = r.file(result["id"], "events.jsonl").read_text()
    assert "expected_ready" in events and "settling" in events
    with pytest.raises(RecordingError):
        r.file("../../.env", "manifest.json")
    with pytest.raises(RecordingError):
        r.file(result["id"], "../../.env")


def test_limits_and_disabled(tmp_path):
    r = make_recorder(tmp_path, recording_max_seconds=1)
    r.start()
    wait_for(lambda: not r.status()["active"])
    assert r.status()["reason"] == "duration_limit"
    r.settings.recording_enabled = False
    with pytest.raises(RecordingError) as exc:
        r.start()
    assert exc.value.status == 403
    r.settings.recording_enabled = True
    r.settings.recording_max_bytes = 30_000
    r.start()
    wait_for(lambda: not r.status()["active"])
    assert r.status()["reason"] == "size_limit"
    r.settings.recording_total_bytes = 1
    with pytest.raises(RecordingError) as exc:
        r.start()
    assert exc.value.status == 507


def test_codec_failure_is_reported(tmp_path, monkeypatch):
    r = make_recorder(tmp_path)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: SimpleNamespace(isOpened=lambda: False, release=lambda: None))
    with pytest.raises(RecordingError) as exc:
        r.start()
    assert exc.value.status == 503
    assert not r.status()["active"] and r.status()["status"] == "failed"


def test_camera_failure_and_shutdown(tmp_path):
    r = make_recorder(tmp_path)
    r.start()
    r.camera.backend = "unavailable"
    wait_for(lambda: not r.status()["active"])
    assert r.status()["reason"] == "camera_source_changed"
    r.camera.backend = "mock"
    r.start()
    assert r.stop("shutdown")["reason"] == "shutdown"


async def warmup(self):
    return None


def test_recording_api_origin_and_files(tmp_path):
    settings = Settings(_env_file=None, camera_backend="mock", camera_index=-1, auto_capture=False,
                        recording_enabled=True, recordings_dir=tmp_path / "recordings", recording_min_free_bytes=0,
                        templates_dir=tmp_path / "templates", mock_captures_dir=tmp_path, tts_provider="mock")
    with patch.object(ModelClients, "warmup", warmup), TestClient(create_app(settings)) as c:
        assert c.get("/api/debug/recordings").json()["state"]["enabled"]
        assert c.post("/api/debug/recordings/start", json={}, headers={"Origin": "https://evil.test"}).status_code == 403
        assert c.post("/api/debug/recordings/start", data={"label": "test"}).status_code == 422
        result = c.post("/api/debug/recordings/start", json={})
        assert result.status_code == 200, result.text
        record_id = result.json()["id"]
        assert c.post("/api/debug/recordings/start", json={}).status_code == 409
        assert c.post("/api/debug/recordings/mark", json={}).status_code == 200
        assert c.get(f"/api/debug/recordings/{record_id}/video.avi").status_code == 409
        assert c.post("/api/debug/recordings/stop", json={}).status_code == 200
        assert c.get(f"/api/debug/recordings/{record_id}/manifest.json").status_code == 200
        assert c.get(f"/api/debug/recordings/{record_id}/.env").status_code == 404
