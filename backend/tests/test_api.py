import json
import os
from unittest.mock import patch

import cv2
import pytest
from conftest import FIXTURE
from fastapi.testclient import TestClient
from synth_capture import synthesize

from app.config import Settings
from app.grading.client import ModelClients
from app.main import create_app


async def fake_warmup(self):
    self.health.update(hwr_vllm=True, cloud_llm=True)


async def fake_hwr(self, blocks, qtype):
    return "B", False


async def fake_stream(self, messages, **context):
    value = json.loads(messages[-1]["content"])
    # 固定测试桩结果，不从请求里的模板答案计算，防止回归后又偷偷依赖参考答案。
    assert "参考答案" not in value and "考点" not in value
    answer = "B" if "下列方程是一元二次方程" in value["题干"] else "A"
    if value["未作答"]:
        status = "blank"
    else:
        status = "right" if value["学生作答"] in (answer if isinstance(answer, list) else [answer]) else "wrong"
    yield json.dumps(
        {"model_answer": str(answer), "status": status, "student_display": value["学生作答"], "error_note": None}
    ) + "\n"
    yield "请对照参考答案检查。"


def make_app(tmp_path):
    return create_app(
        Settings(
            camera_backend="mock",
            auto_capture=False,
            tts_provider="mock",
            templates_dir=tmp_path,
            fixture_template=FIXTURE,
        )
    )


def run_flow(app):
    image, _ = synthesize(FIXTURE, seed=1)
    payload = cv2.imencode(".jpg", image)[1].tobytes()
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "session.snapshot"
            response = client.post(
                "/api/debug/inject-image",
                files={"file": ("capture.jpg", payload, "image/jpeg")},
                data={"grade": "true"},
            )
            assert response.status_code == 200
            cid = response.json()["capture_id"]
            seen = []
            for _ in range(500):
                event = ws.receive_json()
                seen.append(event)
                if event["type"] == "page.grade_done":
                    break
            else:
                pytest.fail("未收到批改结束")
            kinds = [e["type"] for e in seen]
            for typ in (
                "capture.taken",
                "page.matched",
                "question.ocr",
                "question.verdict",
                "page.grade_done",
            ):
                assert typ in kinds
            assert (
                kinds.index("capture.taken")
                < kinds.index("page.matched")
                < kinds.index("question.ocr")
                < kinds.index("question.verdict")
                < kinds.index("page.grade_done")
            )
            done = next(e for e in seen if e["type"] == "page.grade_done")
            print("\nWS事件顺序：", " → ".join(dict.fromkeys(kinds)))
            print("批改结果：", json.dumps(done, ensure_ascii=False))
            assert not done["error_uids"]
            assert done["total"] == 3
            assert done["right_count"] >= 1
            assert done["blank_count"] == 1
            assert len(done["wrong_uids"]) >= 1
            image_response = client.get(f"/api/captures/{cid}.jpg")
            assert image_response.status_code == 200
            assert "immutable" in image_response.headers["cache-control"]
            snapshot = client.get("/api/session/state").json()
            assert snapshot["captures"][cid]["grade_result"] == done
            assert len(snapshot["questions"]) == 3
        with client.websocket_connect("/ws") as recovered:
            state = recovered.receive_json()
            assert state["captures"][cid]["grade_result"] == done
        return done


def test_websocket_end_to_end(tmp_path):
    with (
        patch.object(ModelClients, "warmup", fake_warmup),
        patch("app.grading.hwr_client.HWRClient.recognize", fake_hwr),
        patch.object(ModelClients, "stream", fake_stream),
    ):
        run_flow(make_app(tmp_path))


@pytest.mark.cloud
@pytest.mark.skipif(os.environ.get("RUN_CLOUD_TESTS") != "1", reason="需显式启用真实云端测试")
def test_real_cloud_websocket(tmp_path):
    run_flow(make_app(tmp_path))


def test_startup_tts_warmup_and_playback_receipts(tmp_path):
    with patch.object(ModelClients, "warmup", fake_warmup):
        # Explicitly opt in on a mock-only app: never load any GPU/voice model.
        app = create_app(
            Settings(
                _env_file=None,
                camera_backend="mock",
                auto_capture=False,
                tts_provider="mock",
                tts_warmup=True,
                templates_dir=tmp_path,
            )
        )
        with TestClient(app) as client:
            health = client.get("/api/health").json()
            assert health["details"]["tts_warmed"]
            assert health["details"]["tts_warmup_ms"] is not None
            with client.websocket_connect("/ws") as ws:
                seq = ws.receive_json()["turn_seq"]
                ws.send_json({"type": "voice.playback_state", "playing": True, "turn_seq": seq})
                ws.send_json({"type": "question.select", "question_uid": None})
                ws.receive_json()  # barrier: playback receipt was processed
                assert app.state.voice.is_playing
                assert not app.state.voice.playing  # not synthesizing
                ws.send_json({"type": "voice.playback_state", "playing": False, "turn_seq": seq})
                ws.send_json({"type": "question.select", "question_uid": None})
                ws.receive_json()
                assert not app.state.voice.is_playing


def test_invalid_upload_and_health(tmp_path):
    with patch.object(ModelClients, "warmup", fake_warmup), TestClient(make_app(tmp_path)) as client:
        health = client.get("/api/health").json()
        assert health["camera"]["backend"] == "mock"
        assert health["templates_count"] == 1
        assert client.post("/api/debug/inject-image", files={"file": ("bad.png", b"bad")}).status_code == 400
        assert client.get("/api/templates/no_such_paper").status_code == 404
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "session.set_mode", "auto_grade": False, "voice_mode": "off"})
            assert ws.receive_json()["mode"]["auto_grade"] is False
