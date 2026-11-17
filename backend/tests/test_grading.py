import pytest

from app.grading.judge import FirstLineParser, exact_match
from app.voice.normalizer import SentenceSplitter, normalize_tts


@pytest.mark.parametrize("cuts", [[1, 3, 12], [25, 45], [200]])
def test_first_line_stream(cuts):
    text = '{"status":"wrong","student_display":"B","error_note":"注意负根","error_line":1}\n要保留两个根。'
    parser = FirstLineParser()
    position = 0
    verdict = None
    body = ""
    for end in cuts + [len(text)]:
        data, delta = parser.feed(text[position:end])
        verdict = data or verdict
        body += delta
        position = end
    assert verdict["status"] == "wrong"
    assert verdict["error_line"] == 1
    assert body == "要保留两个根。"


@pytest.mark.parametrize(
    "text",
    [
        '{"status":"maybe"}',
        "not JSON",
    ],
)
def test_invalid_header(text):
    with pytest.raises(ValueError):
        parser = FirstLineParser()
        parser.feed(text)
        parser.finish()


@pytest.mark.parametrize(
    "text,status",
    [
        ('```json\n{"status":"right"}\n```\n讲解', "right"),
        ('{"status":"wrong","error_line":-1}\n讲解', "wrong"),
        ('{"status":"wrong","student_display":"C","step_diagnosis":"符号混淆","board_steps":"x"}\n讲解', "wrong"),
    ],
)
def test_lenient_header(text, status):
    parser = FirstLineParser()
    verdict, body = parser.feed(text)
    assert verdict["status"] == status and verdict["error_line"] is None
    assert body.strip() == "讲解"


def test_no_newline_and_blank():
    parser = FirstLineParser()
    assert parser.feed('{"status":"blank"}') == (None, "")
    assert parser.finish()["status"] == "blank"


def test_exact_match():
    q = {"type": "choice", "reference_answer": ["A", "a"]}
    assert exact_match(q, "（Ａ）")
    assert not exact_match(q, "A", True)
    assert not exact_match(q, "B")
    assert not exact_match({"type": "math", "reference_answer": "3"}, "3")


def test_tts_split():
    splitter = SentenceSplitter()
    assert splitter.feed("我们先看条件，") == ["我们先看条件，"]
    assert splitter.feed("这一步不能直接除以x，因为x有可能为零。")
    assert "的平方" in normalize_tts("$x²±x₁≠0$")
    assert "$" not in normalize_tts("$x²$")


def test_passage_and_optional_fields():
    import json

    from app.grading.judge import judge_messages

    q = {"type": "text", "stem": "Why?", "passage": "The answer is in this passage."}
    messages = judge_messages(q, "Because...", False)
    body = json.loads(messages[-1]["content"])
    assert body["阅读材料"] == q["passage"]
    # 模板解析不再喂给模型，由模型实时做题生成讲解；缺省的参考答案不出现。
    assert "模板解析" not in body
    assert "参考答案" not in body
    q2 = {**q, "reference_answer": "ANSWER_SENTINEL", "analysis": "ANALYSIS_SENTINEL", "tag": "TAG_SENTINEL"}
    request = judge_messages(q2, "A", False, low_confidence=True)
    body2 = json.loads(request[-1]["content"])
    assert body2["识别不确定"] is True
    assert not {"参考答案", "模板解析", "考点"} & body2.keys()
    assert "SENTINEL" not in json.dumps(request)


async def test_writing_never_gets_right_or_wrong():
    import json
    from time import perf_counter
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.config import Settings
    from app.engine import Engine
    from app.state import Session

    settings = Settings(_env_file=None)
    session = Session(settings)
    events = []

    async def emit(event):
        events.append(event)

    session.emit = emit
    seen = []

    async def stream(messages, **context):
        seen.append(json.loads(messages[-1]["content"]))
        yield '{"status":"wrong","student_display":"My essay"}\n要点完整，可以增加衔接词。'

    clients = SimpleNamespace(stream=stream)
    engine = Engine(settings, session, None, None, clients)
    engine.hwr.recognize = AsyncMock(return_value=("My essay", False))
    engine.captures["cap1"] = {
        "items": {
            "q1": {"question_uid": "q1", "answer_quads": [], "mark_pos": [10, 10], "passage": "文章全文"}
        },
        "started": perf_counter(),
        "timings_ms": {},
        "verdict_count": 0,
        "question_count": 1,
    }
    result = await engine.grade_question(
        "cap1",
        {"id": "q1", "type": "writing", "stem": "Write an essay"},
        {"is_blank": False, "blocks": [{"jpeg": b"dummy"}]},
    )
    assert result == "blank"
    verdict = next(e for e in events if e["type"] == "question.verdict")
    assert verdict["is_blank"] is False
    assert verdict["error_quad"] is None
    assert seen[0]["阅读材料"] == "文章全文"
    assert events[-1]["full_text"] == "要点完整，可以增加衔接词。"
