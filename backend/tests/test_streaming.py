"""Text completion must not wait for TTS; tempo changes duration, not pitch."""
import asyncio
import ctypes.util
from contextlib import aclosing
from types import SimpleNamespace

import numpy as np
import pytest
from test_voice import setup_voice

from app.voice.tempo import Tempo


async def test_long_text_finishes_while_audio_is_blocked():
    c, s = setup_voice()
    gate = asyncio.Event()
    native_started = asyncio.Event()
    closed = asyncio.Event()
    chunks = [f"第{i}段这是用于检查长回答队列不会反压文字的内容。" for i in range(80)]

    async def stream(*args, **kwargs):
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0)

    async def tts(text):
        native_started.set()
        try:
            await gate.wait()
            yield b"\0\0" * 120
        finally:
            closed.set()

    c.clients.stream = stream
    c.tts = SimpleNamespace(available=True, stream=tts)
    await c.chat("测试", speak=True)
    try:
        await s.wait("chat.message_done")
        assert native_started.is_set() and not gate.is_set()
        answer = s.data["threads"]["all"][-1]
        assert answer["text"] == "".join(chunks)
        assert not answer["interrupted"]
        assert not c.task.done()  # still owns the audio worker for cancellation
        await c.interrupt("stop")
        assert closed.is_set()
        assert len([e for e in s.recorded if isinstance(e, dict) and e["type"] == "chat.message_done"]) == 1
        assert not answer["interrupted"]  # stopping audio must not mark completed text truncated
    finally:
        await c.close()


async def test_audio_failure_does_not_stop_text():
    c, s = setup_voice()

    async def stream(*args, **kwargs):
        for _ in range(60):
            yield "这个较长的句子用于检查合成失败之后文字还能继续。"
            await asyncio.sleep(0)

    async def tts(text):
        raise RuntimeError("fake TTS failure")
        yield b""

    c.clients.stream = stream
    c.tts = SimpleNamespace(available=True, stream=tts)
    await c.chat("测试", speak=True)
    await asyncio.wait_for(c.task, 2)
    errors = [e for e in s.recorded if isinstance(e, dict) and e["type"] == "error"]
    assert len(errors) == 1
    assert s.data["threads"]["all"][-1]["text"].count("这个较长") == 60
    assert any(isinstance(e, dict) and e["type"] == "tts.end" and e["failed"] for e in s.recorded)
    await c.close()


async def test_output_safety_limit_reports_incomplete_answer():
    c, s = setup_voice()
    c.settings.chat_max_chars = 1000

    async def stream(*args, **kwargs):
        for _ in range(100):
            yield "测" * 100

    c.clients.stream = stream
    await c.chat("测试", speak=False)
    await asyncio.wait_for(c.task, 2)
    assert len(s.data["threads"]["all"][-1]["text"]) == 1000
    done = next(e for e in s.recorded if isinstance(e, dict) and e["type"] == "chat.message_done")
    assert done["failed"]
    await c.close()


@pytest.mark.skipif(not ctypes.util.find_library("sonic"), reason="需系统libsonic")
def test_tempo_1_2_preserves_pitch_and_streams():
    rate = 24000
    pcm = (np.sin(np.arange(rate * 2) / rate * 2 * np.pi * 440) * 16000).astype("<i2").tobytes()
    tempo = Tempo(1.2)
    parts = []
    try:
        for offset in range(0, len(pcm), 1920):
            parts.append(tempo.feed(pcm[offset:offset + 1920]))
        assert any(parts)  # emits before EOF; does not buffer the whole sentence
        parts.append(tempo.finish())
    finally:
        tempo.close()
        tempo.close()  # idempotent
    samples = np.frombuffer(b"".join(parts), dtype="<i2")
    assert abs((len(pcm) / 2 / len(samples)) - 1.2) < 0.02
    peak = np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(abs(np.fft.rfft(samples)))]
    assert abs(peak - 440) < 5


@pytest.mark.skipif(not ctypes.util.find_library("sonic"), reason="需系统libsonic")
async def test_tts_native_generator_is_closed_on_cancel():
    c, _ = setup_voice()
    c.tts.settings.tts_provider = "qwen3_tts"
    closed = asyncio.Event()

    async def native(text):
        try:
            while True:
                yield b"\0\0" * 2400
                await asyncio.sleep(0)
        finally:
            closed.set()

    c.tts._native_stream = native
    async with aclosing(c.tts.stream("测试")) as stream:
        assert len(await anext(stream)) > 0
    assert closed.is_set()
    await c.close()
