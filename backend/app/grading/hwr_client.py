"""多图同题、逐题并发识别。仅识别最终保留的手写内容。"""

import asyncio
import base64
import json
import re
import unicodedata

import httpx


_FREEFORM_CHOICE_BRACKET_RE = re.compile(r"^[（(]?\s*([A-Ha-h])\s*[）)]?$")
_FREEFORM_FOOTER_RE = re.compile(r"(?:试卷\s*)?第\s*\d+\s*/\s*\d+\s*页|试卷第\s*\d+\s*页")
_FREEFORM_LLM_SOLVE_RE = re.compile(r"故[选A-H]|错误[；;。]|正确[；;。]|判别式")


def parse_recognition(content, qtype, finish_reason=None, *, freeform=False):
    """不把截断JSON、格式错误或印刷题号当作可靠学生答案。"""
    if finish_reason == "length" or not isinstance(content, str):
        return "", True
    text = content.strip()
    if text.startswith("```"):
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if not fenced:
            return "", True
        text = fenced.group(1).strip()
    low = False
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # 仅兼容很短的裸选项；其他非JSON结果不再直接当作高置信度识别。
        value = text
        low = True
        if text.startswith(("{", "[")):
            return "", True
    if isinstance(value, dict):
        raw = value.get("student_raw")
        confidence = value.get("low_confidence", False)
        if not isinstance(raw, str) or not isinstance(confidence, bool):
            return "", True
        low = confidence
    elif isinstance(value, str):
        raw = value
    else:
        return "", True
    raw = unicodedata.normalize("NFKC", raw).strip()
    if len(raw) > 4000 or re.search(r"(.{1,16})\1{12,}", raw):
        return "", True
    if freeform and raw:
        if _FREEFORM_FOOTER_RE.search(raw) or (len(raw) > 10 and _FREEFORM_LLM_SOLVE_RE.search(raw)):
            return "", False
    if qtype == "choice" and raw:
        if freeform:
            bm = _FREEFORM_CHOICE_BRACKET_RE.fullmatch(raw)
            if not bm:
                return "", True
            raw = bm.group(1).upper()
        else:
            if not re.fullmatch(r"[A-Ha-h]", raw):
                return "", True
            raw = raw.upper()
        # 兼容既有只返回字母的本地接口，但不兼容A.单词/题号等印刷内容。
        if text == raw or text.upper() == raw:
            low = False
    return raw, low


def merge_freeform_choice_result(
    hwr_raw: str,
    hwr_low: bool,
    pre_ocr_handwriting: str = "",
    *,
    ink_pixels: int = 0,
    pre_ocr_confident: bool = False,
) -> tuple[str, bool]:
    """融合 Qwen3.5-2B HWR 与 PaddleOCR-VL-1.5 括号独立单字母识别结果。"""
    hwr_norm = unicodedata.normalize("NFKC", hwr_raw or "").strip().upper()
    if not re.fullmatch(r"[A-H]", hwr_norm):
        hwr_norm = ""
    pre_norm = unicodedata.normalize("NFKC", pre_ocr_handwriting or "").strip().strip("$").strip().upper()
    if not re.fullmatch(r"[A-H]", pre_norm):
        pre_norm = ""

    if hwr_norm and pre_norm and hwr_norm == pre_norm:
        return hwr_norm, False
    if hwr_norm and not hwr_low:
        return hwr_norm, False
    if pre_norm and (pre_ocr_confident or ink_pixels >= 35):
        return pre_norm, False
    if hwr_norm:
        return hwr_norm, bool(hwr_low)
    return "", bool(hwr_low)


_TEMPLATE_PROMPT = (
    "只读出图片中最终保留的手写内容，忽略所有印刷体、括号、题干和横线，忽略划掉或涂黑的废笔迹。"
    "没有手写输出空字符串。选择题只输出手写选项字母；数学题保留换行和原步骤，不解题、不纠错。"
    '严格只输出JSON：{"student_raw":"...","low_confidence":false}。'
)

_FREEFORM_PROMPT = (
    "你正在做无模板试卷答题区手写识别。图片是从试卷裁剪的局部区域，可能完全没有手写，"
    "只包含印刷文字、印刷括号（ ）/()、空白横线____、表格线、几何/坐标图或页脚。\n"
    "严格规则：\n"
    '1. 若区域内没有清晰的学生手写笔迹（只有印刷汉字/英文/数字、空括号、空横线、表格线、印刷曲线或页脚），'
    '必须输出：{"student_raw":"","low_confidence":false}。没有手写时 low_confidence 必须为 false。\n'
    "2. 仅当确实看到手写笔迹但字迹潦草难以辨认时，才可将 low_confidence 设为 true。\n"
    "3. 严禁输出印刷题干、选项、页脚或图形标注，严禁自己解题或解释。\n"
    "4. 选择题（choice）只输出单个大写手写选项字母（如 B），不要带括号；数学填空/解答题保留手写原步骤，不纠错。\n"
    '严格只输出JSON：{"student_raw":"...","low_confidence":false}。'
)


class HWRClient:
    def __init__(self, clients):
        self.clients = clients

    async def recognize(self, blocks, qtype, *, freeform=False):
        if not blocks:
            raise ValueError("题目缺少有效答题区")
        if freeform:
            prompt = f"{_FREEFORM_PROMPT}\n题型：{qtype}。各图为同一题的答题区。"
        else:
            prompt = f"{_TEMPLATE_PROMPT}题型：{qtype}。各图为同一题的答题区。"
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
                    choice = response.json()["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason")
                    c.health["hwr_vllm"] = True
            except (httpx.HTTPError, TimeoutError):
                c.health["hwr_vllm"] = False
                raise
        raw, low = parse_recognition(content, qtype, finish_reason, freeform=freeform)
        if any(block.get("alignment_uncertain") for block in blocks):
            low = True
        if not freeform and not raw and any(block.get("ink_pixels", 0) > 0 for block in blocks):
            # 模板差分模式下检测到新增墨迹但模型未读出，不能伪装为确定未作答；无模板模式裁剪框自带印刷体，信任HWR空串判定。
            low = True
        return raw, low
