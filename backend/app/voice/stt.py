"""SenseVoiceSmall 懒加载；推理锁保证部分识别与终审不并发访问模型。"""

import importlib.util
import re
import threading

import numpy as np


class SenseVoice:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.failed = False
        self.lock = threading.Lock()

    @property
    def available(self):
        return not self.failed and importlib.util.find_spec("funasr") is not None

    def transcribe(self, pcm):
        if not pcm:
            return ""
        with self.lock:
            try:
                if self.model is None:
                    from funasr import AutoModel

                    self.model = AutoModel(
                        model="iic/SenseVoiceSmall",
                        device=self.settings.stt_device,
                        disable_update=True,
                        disable_pbar=True,
                        disable_log=True,
                    )
                audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
                result = self.model.generate(
                    input=audio, cache={}, language="auto", use_itn=True, batch_size_s=30, merge_vad=False
                )
                text = result[0].get("text", "") if result else ""
                return re.sub(r"<\|[^|]+\|>", "", text).strip()
            except Exception:
                self.failed = True
                raise


class STT:
    """Qwen3-ASR原生multipart转写；prompt字段直接传入题目context。"""

    def __init__(self, settings, clients):
        self.settings, self.clients = settings, clients
        self.local = SenseVoice(settings)
        self.remote_ok = False

    @property
    def model(self):
        return self.local.model

    @property
    def available(self):
        if self.settings.stt_provider == "qwen3_asr":
            return self.remote_ok
        return self.settings.stt_provider == "sensevoice" and self.local.available

    def headers(self):
        key = self.settings.stt_api_key.get_secret_value()
        return {"Authorization": "Bearer " + key} if key else {}

    async def warmup(self):
        if self.settings.stt_provider == "qwen3_asr":
            try:
                response = await self.clients.http.get(
                    self.settings.stt_base_url.rstrip("/") + "/models", headers=self.headers(), timeout=3
                )
                self.remote_ok = response.is_success
            except Exception:
                self.remote_ok = False

    async def transcribe(self, pcm, context=""):
        import asyncio
        import io
        import wave

        if self.settings.stt_provider == "sensevoice":
            return await asyncio.to_thread(self.local.transcribe, pcm)
        if self.settings.stt_provider != "qwen3_asr":
            raise ValueError("未知STT_PROVIDER")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm)
        try:
            async with asyncio.timeout(self.settings.stt_timeout):
                response = await self.clients.http.post(
                    self.settings.stt_base_url.rstrip("/") + "/audio/transcriptions",
                    headers=self.headers(),
                    files={"file": ("utterance.wav", buffer.getvalue(), "audio/wav")},
                    data={
                        "model": self.settings.stt_model,
                        "prompt": context[:24000],
                        "response_format": "json",
                        "temperature": "0",
                    },
                    timeout=self.settings.stt_timeout,
                )
                response.raise_for_status()
                text = response.json()["text"]
                self.remote_ok = True
                return re.sub(r"^language\s+\w+<asr_text>", "", text).strip()
        except Exception:
            self.remote_ok = False
            raise
