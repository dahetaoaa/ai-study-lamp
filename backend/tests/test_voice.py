"""语音测试全部使用纯逻辑/mock，不下载或加载torch、STT、TTS权重。"""

import asyncio
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config import Settings
from app.state import Session
from app.voice.controller import VoiceController
from app.voice.normalizer import clean_transcript, is_backchannel, meaningful_speech, speech_instruct
from app.voice.stt import STT


class RecordingSession(Session):
    def __init__(self, settings):
        super().__init__(settings)
        self.recorded = []
        self.received = {}

    async def emit(self, event):
        self.recorded.append(event)
        await super().emit(event)
        if isinstance(event, dict):
            self.received.setdefault(event["type"], asyncio.Event()).set()

    async def wait(self, typ):
        await asyncio.wait_for(self.received.setdefault(typ, asyncio.Event()).wait(), 2)


def setup_voice():
    settings = Settings(_env_file=None, tts_provider="mock", stt_provider="sensevoice")
    session = RecordingSession(settings)
    controller = VoiceController(settings, session, SimpleNamespace())
    return controller, session


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("嗯，我我我想问这个这个怎么做", "我想问这个怎么做"),
        ("um I I want to know, you know, why", "I want to know, why"),
        ("就是说，为什么不是A", "为什么不是A"),
        ("嗯嗯", ""),
        ("哦", ""),
        ("那个", ""),
        ("那个数为什么不能为零", "那个数为什么不能为零"),
    ],
)
def test_clean_transcript(raw, clean):
    assert clean_transcript(raw) == clean


@pytest.mark.parametrize("raw", ["嗯嗯", "对", "好的", "哦", "啊"])
def test_backchannel(raw):
    assert is_backchannel(raw)
    assert not meaningful_speech(raw)


def test_instruct():
    normal = speech_instruct("我们先看题目的条件。")
    assert speech_instruct("答对了，真棒！") == normal
    assert speech_instruct("注意，这里漏掉负根。") == normal
    assert "自然平实" in normal and "拖长尾音" in normal


async def test_duck_acknowledgement_does_not_interrupt():
    c, s = setup_voice()
    c.playing = True
    c.task = asyncio.create_task(asyncio.Event().wait())
    original = c.task
    v = c.new_input()
    c.stt.transcribe = AsyncMock(side_effect=["嗯嗯", "好的"])
    await v.begin()
    assert any(e == {"type": "voice.duck", "on": True} for e in s.recorded)
    await v.partial(b"\0" * 640, v.utterance)
    await v.end()
    assert not original.done()
    assert s.data["turn_seq"] == 0
    assert {"type": "voice.duck", "on": False} in s.recorded
    assert not any(
        isinstance(e, dict) and e["type"] in ("voice.interrupted", "asr.final", "chat.message_start")
        for e in s.recorded
    )
    await c.close()


async def test_meaningful_partial_interrupts_and_keeps_context():
    c, s = setup_voice()
    s.data["questions"]["q1"] = {"stem": "If it rains", "options": ["A. rains"], "tag": "主将从现"}
    s.data["selected_question_uid"] = "q1"
    c.playing = True
    c.task = asyncio.create_task(asyncio.Event().wait())
    old = c.task
    v = c.new_input()
    c.stt.transcribe = AsyncMock(return_value="嗯，为什么不能选B")
    await v.begin()
    await v.partial(b"\0" * 640, v.utterance)
    assert old.cancelled()
    assert s.data["turn_seq"] == 1
    event = next(e for e in s.recorded if isinstance(e, dict) and e["type"] == "voice.interrupted")
    assert event["old_turn_seq"] == 0 and event["new_turn_seq"] == 1
    assert "主将从现" in c.stt.transcribe.call_args.args[1]
    assert {"type": "voice.duck", "on": False} in s.recorded
    await c.close()


async def test_chat_cancellation_and_audio_header():
    c, s = setup_voice()

    async def stream(*args, **kwargs):
        yield "我们先看题目的条件，"
        await asyncio.Event().wait()

    c.clients.stream = stream
    await c.chat("为什么", speak=True)
    await s.wait("tts.start")
    seq = s.data["turn_seq"]
    frames = [e for e in s.recorded if isinstance(e, bytes)]
    assert frames
    assert struct.unpack("<BBH", frames[0][:4]) == (2, 0, seq)
    assert len(frames[0][4:]) % 2 == 0
    await c.interrupt("stop")
    index = next(
        i for i, e in enumerate(s.recorded) if isinstance(e, dict) and e["type"] == "voice.interrupted"
    )
    assert not any(isinstance(e, bytes) for e in s.recorded[index + 1 :])
    done = next(e for e in s.recorded if isinstance(e, dict) and e["type"] == "chat.message_done")
    assert done["interrupted"]
    s.data["turn_seq"] = 65535
    assert await c.interrupt("stop") == 0
    await c.close()


async def test_qwen_asr_multipart_context():
    seen = []

    async def handler(request):
        body = await request.aread()
        seen.append((request.url.path, body))
        return httpx.Response(200, json={"text": "language Chinese<asr_text>为什么不选B"})

    settings = Settings(_env_file=None, stt_provider="qwen3_asr")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        stt = STT(settings, SimpleNamespace(http=client))
        text = await stt.transcribe(b"\0" * 1920, "主将从现 whether")
    assert text == "为什么不选B"
    assert seen[0][0] == "/v1/audio/transcriptions"
    assert b"RIFF" in seen[0][1]
    assert "主将从现".encode() in seen[0][1]
    assert b'name="prompt"' in seen[0][1]


async def test_continued_utterance_merges_before_first_delta():
    c, s = setup_voice()
    v = c.new_input()

    async def stream(*args, **kwargs):
        await asyncio.Event().wait()
        yield "unused"

    c.clients.stream = stream
    await c.chat("为什么方程两边", utterance_id="u_first", voice_input=v)
    await s.wait("chat.message_start")
    old = c.task
    c.stt.transcribe = AsyncMock(return_value="不能同时除以零")
    await v.begin()
    assert old.cancelled()
    assert v.merge_prefix == "为什么方程两边"
    assert not v.pending_barge and not v.barge
    await v.end()
    users = [m for m in s.data["threads"]["all"] if m["role"] == "user"]
    assert len(users) == 1 and users[0]["text"] == "为什么方程两边不能同时除以零"
    assert not users[0]["is_barge_in"]
    assert not any(isinstance(e, dict) and e["type"] == "voice.interrupted" for e in s.recorded)
    assert not any(isinstance(e, dict) and e.get("interrupted") for e in s.recorded)
    await c.close()


@pytest.mark.parametrize("reason", ["expired", "delta", "other_input"])
async def test_continuation_never_swallows_real_barge_in(reason):
    c, s = setup_voice()
    v = c.new_input()

    async def stream(*args, **kwargs):
        if reason == "delta":
            yield "开始讲解"
        await asyncio.Event().wait()

    c.clients.stream = stream
    await c.chat("为什么", utterance_id="u_first", voice_input=v)
    await s.wait("chat.delta" if reason == "delta" else "chat.message_start")
    if reason == "expired":
        c.pending_voice["at"] -= 2
    if reason == "other_input":
        v = c.new_input()
    await v.begin()
    assert not v.merge_prefix and v.pending_barge
    assert {"type": "voice.duck", "on": True} in s.recorded
    await c.close()


def test_vad_end_silence_default_and_override():
    assert Settings(_env_file=None).vad_end_silence_ms == 700
    assert Settings(_env_file=None, vad_end_silence_ms=400).vad_end_silence_ms == 400


async def test_playback_tail_can_duck_after_synthesis_ends():
    c, s = setup_voice()
    v = c.new_input()
    # 浏览器只在首包/队列播完回执；合成结束时仍缓存了数秒音频。
    v.playing = True
    c.playing = False
    c.stt.transcribe = AsyncMock(return_value="嗯嗯")
    await v.begin()
    await v.end()
    assert {"type": "voice.duck", "on": True} in s.recorded
    assert {"type": "voice.duck", "on": False} in s.recorded
    assert not any(isinstance(e, dict) and e["type"] == "voice.interrupted" for e in s.recorded)
    await c.interrupt("stop")
    assert not v.playing
    await c.close()


async def test_tts_warmup_drains_audio_once():
    c, _ = setup_voice()
    calls = []

    async def stream(text):
        calls.append(text)
        yield b"\0" * 640
        yield b"\0" * 640

    c.tts.stream = stream
    await c.tts.warmup()
    await c.tts.warmup()
    assert len(calls) == 1
    assert c.tts.warmed and c.tts.warmup_ms is not None
    assert c.tts.available


@pytest.mark.parametrize("raises", [True, False])
async def test_tts_warmup_failure_marks_health(raises):
    c, _ = setup_voice()

    async def stream(text):
        if raises:
            raise RuntimeError("missing model")
        if False:
            yield b""

    c.tts.stream = stream
    await c.tts.warmup()
    assert c.tts.failed and not c.tts.available and not c.tts.warmed
