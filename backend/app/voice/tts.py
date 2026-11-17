"""可中止的TTS适配器；生产者线程持锁，取消后不再产生旧轮次音频。"""

import asyncio
import importlib.util
import logging
import queue
import threading
from contextlib import aclosing
from time import perf_counter

import numpy as np

from app.voice.normalizer import normalize_tts, speech_instruct
from app.voice.tempo import Tempo


class TTS:
    def __init__(self, settings, clients):
        self.settings = settings
        self.clients = clients
        self.model = None
        self.failed = False
        self.warmed = False
        self.warmup_ms = None
        self.lock = threading.Lock()

    async def warmup(self):
        """预热失败保持文字服务可用，但健康检查必须标红。"""
        if self.warmed:
            return
        started = perf_counter()
        try:
            size = 0
            async for pcm in self.stream("你好，我们开始学习吧。"):
                size += len(pcm)
            if not size:
                raise ValueError("TTS预热未产生音频")
            self.warmed = True
            self.warmup_ms = round((perf_counter() - started) * 1000, 2)
            logging.getLogger(__name__).info("TTS warmup complete: %.2f ms", self.warmup_ms)
        except Exception as error:  # noqa: BLE001 — 失败标红，保留文字/批改服务
            self.failed = True
            logging.getLogger(__name__).error("TTS warmup failed (%s)", type(error).__name__)

    @property
    def available(self):
        module = {"kokoro": "kokoro", "qwen3_tts": "faster_qwen3_tts", "mock": None, "vllm_omni": None}.get(
            self.settings.tts_provider, "missing_tts_provider"
        )
        return not self.failed and (module is None or importlib.util.find_spec(module) is not None)

    def generate(self, text):
        provider = self.settings.tts_provider
        if provider == "mock":
            yield np.zeros(2400, np.float32), 24000
        elif provider == "kokoro":
            if self.model is None:
                from kokoro import KPipeline

                self.model = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh")
            for _, _, audio in self.model(text, voice=self.settings.kokoro_voice):
                yield audio, 24000
        elif provider == "qwen3_tts":
            if self.model is None:
                from faster_qwen3_tts import FasterQwen3TTS

                # 库按平台选择CUDA或GGML；模型路径可指向本地离线目录。
                self.model = FasterQwen3TTS.from_pretrained(self.settings.tts_model)
            for audio, sr, _ in self.model.generate_custom_voice_streaming(
                text=text,
                speaker=self.settings.tts_voice,
                language="Auto",
                chunk_size=4,
                instruct=speech_instruct(text),
            ):
                yield audio, sr
        else:
            raise ValueError("未知TTS_PROVIDER")

    async def stream(self, text):
        # 加速在服务端应用，WebAudio保持1倍播放，避免升调和重复加速。
        speed = self.settings.tts_speed if self.settings.tts_provider != "mock" else 1.0
        tempo = Tempo(speed) if speed != 1.0 else None
        try:
            async with aclosing(self._native_stream(text)) as source:
                async for pcm in source:
                    output = tempo.feed(pcm) if tempo else pcm
                    if output:
                        yield output
            if tempo:
                tail = tempo.finish()
                if tail:
                    yield tail
        finally:
            if tempo:
                tempo.close()

    async def _native_stream(self, text):
        text = normalize_tts(text)
        if not text:
            return
        if self.settings.tts_provider == "vllm_omni":
            key = self.settings.tts_api_key.get_secret_value()
            headers = {"Authorization": "Bearer " + key} if key else {}
            async with self.clients.http.stream(
                "POST",
                self.settings.tts_base_url.rstrip("/") + "/audio/speech",
                headers=headers,
                json={
                    "model": self.settings.tts_model,
                    "input": text,
                    "voice": self.settings.tts_voice,
                    "response_format": "pcm",
                    "stream": True,
                    "stream_format": "audio",
                    "sample_rate": 24000,
                    "instructions": speech_instruct(text),
                },
                timeout=30,
            ) as response:
                response.raise_for_status()
                pending = b""
                async for data in response.aiter_bytes(chunk_size=7680):
                    pending += data
                    usable = len(pending) // 2 * 2
                    if usable:
                        yield pending[:usable]
                        pending = pending[usable:]
                if pending:
                    raise ValueError("TTS返回非PCM16对齐数据")
            return
        stop = threading.Event()
        chunks = queue.Queue(maxsize=6)

        def put(value):
            while not stop.is_set():
                try:
                    chunks.put(value, timeout=0.05)
                    return
                except queue.Full:
                    pass

        def produce():
            # 取消等待中的轮次时，不会偷偷加载或继续计算模型。
            while not stop.is_set():
                if self.lock.acquire(timeout=0.05):
                    break
            else:
                return
            generator = None
            try:
                if stop.is_set():
                    return
                generator = self.generate(text)
                for audio, sr in generator:
                    if stop.is_set():
                        break
                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()
                    data = np.asarray(audio, dtype=np.float32).reshape(-1)
                    if sr != 24000:
                        data = np.interp(
                            np.arange(round(len(data) * 24000 / sr)) * sr / 24000, np.arange(len(data)), data
                        )
                    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
                    for offset in range(0, len(pcm), 15360):
                        put(pcm[offset : offset + 15360])
                put(None)
            except Exception as error:
                self.failed = True
                put(error)
            finally:
                if generator is not None:
                    generator.close()
                self.lock.release()

        thread = threading.Thread(target=produce, name="tts-producer", daemon=True)
        thread.start()
        try:
            while True:
                # 非阻塞轮询避免取消协程后线程池永久卡在queue.get。
                try:
                    item = chunks.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.005)
                    continue
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
