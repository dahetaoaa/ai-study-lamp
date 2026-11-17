"""多图同题、逐题并发识别。仅识别最终保留的手写内容。"""

import asyncio
import base64
import json

import httpx


class HWRClient:
    def __init__(self, clients):
        self.clients = clients

    async def recognize(self, blocks, qtype):
        if not blocks:
            raise ValueError("题目缺少有效答题区")
        prompt = (
            "只读出图片中最终保留的手写内容，忽略所有印刷体、括号、题干和横线，忽略划掉或涂黑的废笔迹。"
            "没有手写输出空字符串。选择题只输出手写选项字母；数学题保留换行和原步骤，不解题、不纠错。"
            '严格只输出JSON：{"student_raw":"...","low_confidence":false}。'
            f"题型：{qtype}。各图为同一题的答题区。"
        )
        content = [{"type": "text", "text": prompt}]
        for block in blocks:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(block["jpeg"]).decode()
                    },
                }
            )
        c, s = self.clients, self.clients.settings
        payload = {
            "model": s.hwr_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 600,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        }
        async with c.hwr_slots:
            try:
                async with asyncio.timeout(s.hwr_timeout):
                    response = await c.http.post(
                        s.hwr_base_url.rstrip("/") + "/chat/completions",
                        headers=c.headers(True),
                        json=payload,
                        timeout=s.hwr_timeout,
                    )
                    response.raise_for_status()
                    raw = response.json()["choices"][0]["message"]["content"].strip()
                    c.health["hwr_vllm"] = True
            except (httpx.HTTPError, TimeoutError):
                c.health["hwr_vllm"] = False
                raise
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        low = False
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                raw, low = str(parsed.get("student_raw", "")), bool(parsed.get("low_confidence", False))
            elif isinstance(parsed, str):
                raw = parsed
        except json.JSONDecodeError:
            # 兼容仅输出字母的本地小模型；长回答必须交云端消歧。
            low = len(raw) > 30 and qtype == "choice"
        return raw, low
