"""全局复用 OpenAI 兼容 HTTP 连接池，SSE 首包与总时限分开控制。"""

import asyncio
import json
from time import perf_counter

import httpx

from app.telemetry import elapsed, log, timing


class ModelClients:
    def __init__(self, settings):
        self.settings = settings
        self.http = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=24, keepalive_expiry=60),
            timeout=httpx.Timeout(12, connect=4),
            follow_redirects=False,
        )
        self.cloud_slots = asyncio.Semaphore(settings.cloud_concurrency)
        self.hwr_slots = asyncio.Semaphore(8)
        self.health = {"hwr_vllm": False, "ocr_vllm": False, "cloud_llm": False}

    def headers(self, hwr=False):
        key = self.settings.hwr_api_key if hwr else self.settings.cloud_llm_api_key
        return {"Authorization": "Bearer " + key.get_secret_value()} if key.get_secret_value() else {}

    async def warmup(self):
        s = self.settings

        async def probe(name, url, headers):
            try:
                response = await self.http.get(url.rstrip("/") + "/models", headers=headers, timeout=3)
                self.health[name] = response.is_success
            except httpx.HTTPError:
                self.health[name] = False

        await asyncio.gather(
            probe("hwr_vllm", s.hwr_base_url, self.headers(True)),
            probe("cloud_llm", s.cloud_llm_base_url, self.headers()),
            probe("ocr_vllm", s.ocr_base_url, {}),
        )

    async def stream(self, messages, max_tokens=1500, **context):
        """首个正文 token 之前的失败（429/5xx/网络/超时）自动退避重试；出字后失败直接抛出。"""
        attempts = self.settings.cloud_retries + 1
        for attempt in range(attempts):
            produced = False
            try:
                async for text in self._stream_once(messages, max_tokens, **context):
                    produced = True
                    yield text
                return
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                try:
                    body = error.response.text[:300]
                except (httpx.HTTPError, RuntimeError):
                    body = ""
                log.warning("云端LLM HTTP %s（第%d次）：%s", status, attempt + 1, body)
                if produced or attempt + 1 >= attempts or (status < 500 and status not in (408, 409, 429)):
                    raise
            except (httpx.TransportError, TimeoutError, ValueError) as error:
                log.warning("云端LLM %s（第%d次）：%s", type(error).__name__, attempt + 1, error)
                if produced or attempt + 1 >= attempts:
                    raise
            await asyncio.sleep(0.4 * (2**attempt))

    async def _stream_once(self, messages, max_tokens, **context):
        s = self.settings
        payload = {
            "model": s.cloud_llm_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "enable_thinking": False,
        }
        waiting = perf_counter()
        async with self.cloud_slots:
            timing("llm_queue_wait", elapsed(waiting), **context)
            started = perf_counter()
            first = True
            chars = 0
            chunks = 0
            finish_reason = None
            request_id = None
            response_model = None
            outcome = "failed"
            try:
                async with asyncio.timeout(s.cloud_total_timeout):
                    async with self.http.stream(
                        "POST",
                        s.cloud_llm_base_url.rstrip("/") + "/chat/completions",
                        headers=self.headers(),
                        json=payload,
                        timeout=httpx.Timeout(
                            s.cloud_first_token_timeout, connect=s.cloud_first_token_timeout
                        ),
                    ) as response:
                        if response.is_error:
                            await response.aread()  # 关闭前读出错误体，便于定位 4xx/5xx 原因
                        response.raise_for_status()
                        iterator = response.aiter_lines().__aiter__()
                        while True:
                            try:
                                # 心跳包不重置首个有效 token 的截止时间。
                                deadline = (
                                    s.cloud_first_token_timeout - (perf_counter() - started)
                                    if first
                                    else s.cloud_total_timeout
                                )
                                async with asyncio.timeout(max(0.001, deadline)):
                                    line = await anext(iterator)
                            except StopAsyncIteration:
                                break
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            value = json.loads(data)
                            if value.get("error"):
                                raise ValueError("云端返回错误事件")
                            request_id = value.get("id", request_id)
                            response_model = value.get("model", response_model)
                            choices = value.get("choices") or []
                            if choices and choices[0].get("finish_reason"):
                                finish_reason = choices[0]["finish_reason"]
                            text = choices[0].get("delta", {}).get("content") if choices else None
                            if text:
                                chars += len(text)
                                chunks += 1
                                if first:
                                    timing("llm_ttft", elapsed(started), **context)
                                    first = False
                                yield text
                        if first:
                            raise ValueError("云端未返回正文")
                self.health["cloud_llm"] = True
                outcome = "completed"
                if finish_reason == "length":
                    raise ValueError("模型达到输出长度上限，回答可能不完整")
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except (httpx.HTTPError, TimeoutError, ValueError):
                self.health["cloud_llm"] = False
                outcome = "failed"
                raise
            finally:
                timing("llm_done", elapsed(started), **context, chars=chars, chunks=chunks,
                       request_id=request_id, model=response_model or s.cloud_llm_model,
                       finish_reason=finish_reason, outcome=outcome)
