"""求解不泄漏模板答案；待确认不记错；追问仅使用当前页。"""
import asyncio
import json
import os
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.config import Settings
from app.engine import Engine
from app.grading.client import ModelClients
from app.grading.judge import FirstLineParser, judge_messages
from app.state import Session
from app.voice.controller import VoiceController


class RecordingSession(Session):
    def __init__(self, settings):
        super().__init__(settings)
        self.events = []

    async def emit(self, event):
        self.events.append(event)
        await super().emit(event)


def make_engine(stream):
    settings = Settings(_env_file=None, tts_provider="mock")
    session = RecordingSession(settings)
    engine = Engine(settings, session, None, None, SimpleNamespace(stream=stream))
    item = {"question_uid": "q1", "capture_id": "cap1", "answer_quads": [], "mark_pos": [1, 1]}
    session.data["questions"]["q1"] = dict(item)
    session.data["captures"]["cap1"] = {}
    engine.captures["cap1"] = {
        "items": {"q1": item}, "started": perf_counter(), "timings_ms": {},
        "verdict_count": 0, "question_count": 1,
    }
    engine.hwr.recognize = AsyncMock(return_value=("B", False))
    return engine, session


QUESTION = {
    "id": "q1", "type": "choice", "stem": "1+1=?", "options": ["A. 2", "B. 3"],
    "reference_answer": "B", "answer_display": "模板错误答案", "analysis": "模板错误解析", "tag": "预设考点",
}
CROP = {"is_blank": False, "blocks": [{"jpeg": b"test"}]}


@pytest.mark.parametrize("status", ["right", "wrong", "blank", "uncertain"])
def test_parser_accepts_all_states(status):
    p = FirstLineParser()
    verdict, text = p.feed(json.dumps({"status": status, "model_answer": "2"}) + "\n正文")
    assert verdict["status"] == status and text == "正文"


@pytest.mark.parametrize("low,blank,model_status,expected", [
    (False, False, "wrong", "wrong"),
    (False, False, "uncertain", "uncertain"),
    (True, False, "right", "uncertain"),
    (True, False, "blank", "uncertain"),
    (False, True, "blank", "blank"),
])
async def test_engine_uses_cloud_not_template(low, blank, model_status, expected):
    calls = []

    async def stream(messages, **kwargs):
        body = json.loads(messages[-1]["content"])
        calls.append(body)
        assert not {"参考答案", "考点", "模板解析"} & body.keys()
        yield json.dumps({"status": model_status, "model_answer": "A. 2"}) + "\n实时解释"

    engine, session = make_engine(stream)
    engine.hwr.recognize.return_value = ("" if low else "B", low)
    result = await engine.grade_question("cap1", QUESTION, {**CROP, "is_blank": blank})
    assert result == expected and len(calls) == 1
    q = session.data["questions"]["q1"]
    assert q["source"] == "cloud_llm"
    assert q["reference_display"] == "A. 2" and q["explain_text"] == "实时解释"
    if low:
        assert calls[0]["识别不确定"] is True and calls[0]["未作答"] is False
        assert q["is_blank"] is False
    if expected == "uncertain":
        assert q["error_quad"] is None


async def test_failure_never_falls_back_to_template():
    async def stream(*args, **kwargs):
        raise TimeoutError
        yield  # async generator

    engine, session = make_engine(stream)
    assert await engine.grade_question("cap1", QUESTION, CROP) is None
    q = session.data["questions"]["q1"]
    assert q["status"] == "idle" and q["explain_text"] == ""
    assert not any(e["type"] == "question.verdict" for e in session.events)


async def test_page_summary_keeps_uncertain_separate():
    async def stream(*args, **kwargs):
        yield '{"status":"uncertain","error_note":"缺少条件"}\n请补充条件'

    engine, session = make_engine(stream)
    cap = engine.captures["cap1"]
    cap.update(grade_started=False, image=np.zeros((4, 4, 3), dtype=np.uint8), match=SimpleNamespace(
        template=SimpleNamespace(page={"questions": [QUESTION], "page_no": 1}, paper={"paper_id": "paper"})
    ))
    with patch("app.engine.crop_answers", return_value={"q1": CROP}):
        await engine.grade("cap1")
    result = session.data["captures"]["cap1"]["grade_result"]
    assert result["uncertain_uids"] == ["q1"]
    assert result["wrong_uids"] == [] and result["right_count"] == 0
    assert "待确认" in result["summary_text"]


def make_voice():
    settings = Settings(_env_file=None, tts_provider="mock")
    session = RecordingSession(settings)
    c = VoiceController(settings, session, SimpleNamespace())
    session.data["active_capture_id"] = "new"
    session.data["captures"] = {"old": {"page_no": 1}, "new": {"page_no": 2}}
    session.data["questions"] = {
        "oldq": {"capture_id": "old", "stem": "OLD_PAGE_SENTINEL"},
        "newq": {"capture_id": "new", "no": "2", "stem": "2+2=?", "options": ["A.4", "B.5"],
                 "reference_display": "REFERENCE_SENTINEL", "tag": "TAG_SENTINEL",
                 "explain_text": "ANALYSIS_SENTINEL", "source": "local"},
    }
    return c, session


def test_context_has_current_page_and_no_precomputed_answers():
    c, _ = make_voice()
    for uid in (None, "newq"):
        context = c.context(uid)
        assert "SENTINEL" not in json.dumps(context)
        assert context["当前页"]["page_no"] == 2
    assert c.context("oldq")["当前页"]["page_no"] == 1
    assert c.context("newq")["当前题"]["options"] == ["A.4", "B.5"]
    assert "先询问" in c.context(None)["选题说明"]


async def test_global_history_is_scoped_and_followup_preserved():
    c, s = make_voice()
    seen = []

    async def stream(messages, **kwargs):
        seen.append(messages)
        yield "你想问哪一道题？"

    c.clients.stream = stream
    s.data["threads"]["all"] = [{"role": "user", "text": "OLD_HISTORY_SENTINEL", "capture_id": "old"}]
    await c.chat("我不会做", speak=False)
    await asyncio.wait_for(c.task, 2)
    await c.chat("第2题，先给提示", speak=False)
    await asyncio.wait_for(c.task, 2)
    assert "SENTINEL" not in json.dumps(seen)
    assert [m["content"] for m in seen[-1][1:]] == ["我不会做", "你想问哪一道题？", "第2题，先给提示"]
    assert all(m.get("capture_id") == "new" for m in s.data["threads"]["all"][1:])
    await c.close()


async def test_capture_clears_previous_question_selection():
    _, s = make_voice()
    s.data["selected_question_uid"] = "oldq"
    await s.emit({"type": "capture.taken", "capture_id": "third"})
    assert s.data["selected_question_uid"] is None
    assert s.data["active_capture_id"] == "third"


@pytest.mark.cloud
@pytest.mark.skipif(os.environ.get("RUN_CLOUD_TESTS") != "1", reason="需显式启用真实云端测试")
async def test_cloud_solves_without_reference_and_requests_missing_information():
    clients = ModelClients(Settings())
    try:
        for question, answer, expected in [
            (QUESTION, "A", "right"),
            ({"type": "math", "stem": "求图中三角形ABC的面积（图未提供）"}, "12", "uncertain"),
        ]:
            output = ""
            async for text in clients.stream(judge_messages(question, answer, False)):
                output += text
            parser = FirstLineParser()
            verdict, explanation = parser.feed(output)
            verdict = verdict or parser.finish()
            print("真实云端：", verdict, explanation)
            assert verdict["status"] == expected and explanation.strip()
    finally:
        await clients.http.aclose()


@pytest.mark.cloud
@pytest.mark.skipif(os.environ.get("RUN_CLOUD_TESTS") != "1", reason="需显式启用真实云端测试")
async def test_cloud_followup_uses_question_and_conversation():
    c, session = make_voice()
    clients = ModelClients(Settings())
    c.clients = clients
    # 同页两道不同题，未选题时不得自行挑题。
    session.data["questions"]["another"] = {
        "capture_id": "new", "no": "3", "stem": "解方程x+3=8", "qtype": "math",
    }
    try:
        for text, scope, uid in [
            ("我还是不会做，你再讲一下", "all", None),
            ("先给我一个提示，不要直接告诉我选项", "question", "newq"),
            ("现在请完整解答刚才这道题", "question", "newq"),
        ]:
            await c.chat(text, scope=scope, question_uid=uid, speak=False)
            await asyncio.wait_for(c.task, 120)
            answer = session.data["threads"][uid or "all"][-1]["text"]
            print("真实追问：", text, "=>", answer)
            assert answer.strip()
        assert "4" in answer or "四" in answer
    finally:
        await c.close()
        await clients.http.aclose()
