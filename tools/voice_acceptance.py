"""真实WebSocket语音验收（不加载模型）。会消耗云端对话调用。

4090: backend/.venv/bin/python tools/voice_acceptance.py --fixtures /tmp/lamp-voice
Mac同样可运行；--url默认wss://172.17.114.10:8000/ws，忽略自签证书，不走代理。
保存完整事件/24kHz PCM WAV，验证实时20ms上行、语音问答、文字TTS、打断及附和。
"""

import argparse
import asyncio
import inspect
import json
import ssl
import struct
import time
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np
from websockets.asyncio.client import connect


def load_pcm(path):
    with wave.open(str(path), "rb") as f:
        assert f.getsampwidth() == 2 and f.getnchannels() == 1, "需要PCM16单声道WAV"
        rate = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    if rate != 16000:
        data = np.interp(
            np.arange(round(len(data) * 16000 / rate)) * rate / 16000, np.arange(len(data)), data
        ).astype("<i2")
    # 去掉合成文件两端长静音，让声学末尾与计时起点一致（保留80ms尾音）。
    loud = np.flatnonzero(np.abs(data.astype(np.int32)) > 250)
    if loud.size:
        data = data[max(0, loud[0] - 1280) : min(len(data), loud[-1] + 1280)]
    return data.astype("<i2").tobytes()


class Probe:
    def __init__(self, ws, out):
        self.ws, self.out = ws, out
        self.events = []
        self.seq = 0
        self.audio = {}
        self.signal = asyncio.Event()
        self.origin = time.perf_counter()
        self.playback_end = 0
        self.playback_seq = None
        self.ended = set()

    async def send(self, **event):
        await self.ws.send(json.dumps(event, ensure_ascii=False))

    async def receive(self):
        async for raw in self.ws:
            now = time.perf_counter()
            if isinstance(raw, bytes):
                typ, flags, seq = struct.unpack("<BBH", raw[:4])
                assert typ == 2 and flags in (0, 1) and len(raw[4:]) % 2 == 0
                event = {"type": "audio", "turn_seq": seq, "flags": flags, "bytes": len(raw) - 4}
                if raw[4:]:
                    assert seq == self.seq, "打断后出现旧轮次音频"
                    self.audio.setdefault(seq, bytearray()).extend(raw[4:])
                    if self.playback_seq != seq:
                        self.playback_seq = seq
                        self.playback_end = now
                        await self.send(type="voice.playback_state", playing=True, turn_seq=seq)
                    self.playback_end = max(now, self.playback_end) + (len(raw) - 4) / 48000
            else:
                event = json.loads(raw)
                if event["type"] in ("session.snapshot", "chat.message_start", "tts.start"):
                    self.seq = event["turn_seq"]
                elif event["type"] == "voice.interrupted":
                    self.seq = event["new_turn_seq"]
                    self.playback_seq = None
                    self.playback_end = now
                elif event["type"] == "tts.end":
                    self.ended.add(event["turn_seq"])
                if event["type"] not in ("chat.delta", "session.snapshot"):
                    print(round(now - self.origin, 3), json.dumps(event, ensure_ascii=False), flush=True)
            self.events.append({"at": now, **event})
            self.signal.set()

    async def playback(self):
        """按音频时长模拟浏览器播放完毕回执，不把生成完误当播完。"""
        while True:
            await asyncio.sleep(0.02)
            if (
                self.playback_seq is not None
                and self.playback_seq in self.ended
                and time.perf_counter() >= self.playback_end
            ):
                await self.send(type="voice.playback_state", playing=False, turn_seq=self.playback_seq)
                self.playback_seq = None

    async def wait(self, kind, start=0, predicate=lambda e: True, timeout=90):
        async with asyncio.timeout(timeout):
            while True:
                self.signal.clear()
                for e in self.events[start:]:
                    if e["type"] == "error":
                        raise AssertionError(e)
                    if e["type"] == kind and predicate(e):
                        return e
                await self.signal.wait()

    async def upload(self, pcm):
        # 固定20ms节拍，前置320ms和后置800ms静音。
        data = b"\0" * 10240 + pcm
        speech_bytes = len(data)
        data += b"\0" * 25600
        started = time.perf_counter()
        input_end = None
        for offset in range(0, len(data), 640):
            await asyncio.sleep(max(0, started + offset / 32000 - time.perf_counter()))
            await self.ws.send(
                struct.pack("<BBH", 1, 0, self.seq) + data[offset : offset + 640].ljust(640, b"\0")
            )
            if input_end is None and offset + 640 >= speech_bytes:
                input_end = time.perf_counter()
        return input_end

    async def chat(self, text):
        await self.send(
            type="chat.send",
            client_msg_id="probe_" + uuid4().hex[:12],
            scope="all",
            question_uid=None,
            text=text,
            speak=True,
        )

    async def settle(self):
        async with asyncio.timeout(120):
            while self.playback_seq is not None:
                await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)

    def save(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "events.json").write_text(json.dumps(self.events, ensure_ascii=False, indent=2))
        for seq, pcm in self.audio.items():
            with wave.open(str(self.out / f"turn_{seq}.wav"), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(pcm)


async def main(args):
    tls = ssl.create_default_context()
    tls.check_hostname = False
    tls.verify_mode = ssl.CERT_NONE
    kwargs = {"ssl": tls} if args.url.startswith("wss:") else {}
    if "proxy" in inspect.signature(connect).parameters:
        kwargs["proxy"] = None
    async with connect(args.url, max_size=8 * 1024 * 1024, **kwargs) as ws:
        p = Probe(ws, args.output)
        receiver = asyncio.create_task(p.receive())
        playback = asyncio.create_task(p.playback())
        result = {}
        snapshot = await p.wait("session.snapshot")
        try:
            await p.send(type="session.set_mode", auto_capture=False, voice_mode="vad")
            await p.send(type="question.select", question_uid=None)
            if args.only in ("all", "voice", "slow"):
                start = len(p.events)
                slow = args.only == "slow"
                end = await p.upload(load_pcm(args.fixtures / ("slow.wav" if slow else "question.wav")))
                final = await p.wait("asr.final", start, lambda e: not slow or "不能同时除以零" in e["text"])
                response_start = p.events.index(final)
                seq = (await p.wait("chat.message_start", response_start))["turn_seq"]
                delta = await p.wait("chat.delta", response_start)
                audio = await p.wait("audio", response_start, lambda e: e["bytes"] > 0)
                vad_end = [
                    e
                    for e in p.events[start:response_start]
                    if e["type"] == "vad.state" and not e["speaking"]
                ][-1]
                finish = await p.wait("tts.end", start, lambda e, seq=seq: e["turn_seq"] == seq)
                assert not finish["interrupted"]
                order = [
                    "vad.state",
                    "asr.partial",
                    "asr.final",
                    "chat.message_start",
                    "chat.delta",
                    "tts.start",
                    "audio",
                    "tts.end",
                ]
                indices = [next(i for i, e in enumerate(p.events[start:]) if e["type"] == k) for k in order]
                assert indices == sorted(indices), indices
                result["slow" if slow else "voice"] = {
                    "text": final["text"],
                    "audio_seconds": len(p.audio[seq]) / 48000,
                    "input_end_ms": {
                        k: round((e["at"] - end) * 1000, 1)
                        for k, e in [("asr_final", final), ("chat_delta", delta), ("audio", audio)]
                    },
                    "vad_end_ms": {
                        k: round((e["at"] - vad_end["at"]) * 1000, 1)
                        for k, e in [("asr_final", final), ("chat_delta", delta), ("audio", audio)]
                    },
                }
                assert any(p.audio[seq]), "TTS只返回了静音"
                finals = [e for e in p.events[start:] if e["type"] == "asr.final"]
                if slow:
                    assert len(finals) >= 2, "未触发续说合并，请检查素材停顿长度"
                    assert "为什么方程两边" in final["text"]
                    assert not any(e["type"] == "voice.interrupted" for e in p.events[start:])
                    marker = len(p.events)
                    await p.send(type="question.select", question_uid=None)
                    current = await p.wait("session.snapshot", marker)
                    old_users = [m for m in snapshot["threads"].get("all", []) if m["role"] == "user"]
                    new_users = [m for m in current["threads"]["all"] if m["role"] == "user"]
                    assert len(new_users) == len(old_users) + 1
                    assert not new_users[-1]["is_barge_in"]
                    result["slow"]["final_events"] = len(finals)
                    result["slow"]["merged_user_turns"] = 1
                else:
                    assert len(finals) == 1, "普通短问句不应拆为多轮"
                    await p.settle()
            if args.only in ("all", "text"):
                start = len(p.events)
                sent = time.perf_counter()
                await p.chat("请用一句简短的话解释什么是分母，最多二十个字。")
                await p.wait("chat.user_message", start)
                seq = (await p.wait("chat.message_start", start))["turn_seq"]
                audio = await p.wait("audio", start, lambda e: e["bytes"] > 0)
                await p.wait("tts.end", start, lambda e, seq=seq: e["turn_seq"] == seq)
                result["text"] = {
                    "first_audio_ms": round((audio["at"] - sent) * 1000, 1),
                    "audio_seconds": len(p.audio[seq]) / 48000,
                }
                await p.settle()
            for scenario in ("ack", "barge", "tail"):
                if args.only not in ("all", scenario):
                    continue
                start = len(p.events)
                await p.chat("请用大约一百字解释为什么除数不能为零，分三句说明。")
                seq = (await p.wait("tts.start", start))["turn_seq"]
                await p.wait("audio", start, lambda e: e["bytes"] > 0)
                if scenario == "tail":
                    await p.wait("tts.end", start, lambda e, seq=seq: e["turn_seq"] == seq)
                    assert p.playback_end - time.perf_counter() > 3, "需要至少3秒尚未播完的缓存"
                marker = len(p.events)
                fixture = "ack" if scenario == "tail" else scenario
                await p.upload(load_pcm(args.fixtures / f"{fixture}.wav"))
                duck = await p.wait("voice.duck", marker, lambda e: e["on"])
                unduck = await p.wait("voice.duck", marker, lambda e: not e["on"])
                if scenario == "barge":
                    interrupted = await p.wait("voice.interrupted", marker)
                    final = await p.wait("asr.final", marker)
                    assert final["is_barge_in"] and interrupted["old_turn_seq"] == seq
                    assert "更简单的例子" in final["text"], "打断提问被截断"
                    await p.wait(
                        "tts.end", marker, lambda e, seq=seq: e["turn_seq"] != seq and not e["interrupted"]
                    )
                    result[scenario] = {
                        "text": final["text"],
                        "duck_to_interrupt_ms": round((interrupted["at"] - duck["at"]) * 1000, 1),
                    }
                else:
                    finish = await p.wait("tts.end", start, lambda e, seq=seq: e["turn_seq"] == seq)
                    assert not finish["interrupted"]
                    assert not any(e["type"] in ("voice.interrupted", "asr.final") for e in p.events[marker:])
                    result[scenario] = {
                        "duck_ms": round((unduck["at"] - duck["at"]) * 1000, 1),
                        "continued": True,
                    }
                await p.settle()
            print("RESULT", json.dumps(result, ensure_ascii=False), flush=True)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            # 只恢复测试改变的模式和焦点，不清空用户批改数据。
            await p.send(type="chat.stop", turn_seq=p.seq)
            await p.send(type="session.set_mode", **snapshot["mode"])
            await p.send(type="question.select", question_uid=snapshot["selected_question_uid"])
            receiver.cancel()
            playback.cancel()
            await asyncio.gather(receiver, playback, return_exceptions=True)
            p.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="wss://172.17.114.10:8000/ws")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/lamp-voice-results"))
    parser.add_argument(
        "--only", choices=("all", "voice", "text", "ack", "barge", "tail", "slow"), default="all"
    )
    asyncio.run(main(parser.parse_args()))
