"""确定性的数学口语化与增量分句，字幕保持原始文本。"""

import re


def normalize_tts(text):
    text = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s*|[-*+]\s+)", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = text.replace("²", "的平方").replace("³", "的立方").replace("±", "正负").replace("≠", "不等于")
    text = text.replace("≤", "小于等于").replace("≥", "大于等于").replace("×", "乘以").replace("÷", "除以")
    for sub, word in zip("₀₁₂₃₄₅₆₇₈₉", "零一二三四五六七八九"):
        text = text.replace(sub, word)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\2分之\1", text)
    text = re.sub(r"\^\{?2\}?", "的平方", text)
    text = text.replace("\\times", "乘以").replace("\\neq", "不等于").replace("\\pm", "正负")
    text = re.sub(r"\$|_{2,}|[{}]|\\[()[\]]", "", text)
    return text.strip()


class SentenceSplitter:
    def __init__(self):
        self.buffer = ""
        self.first = True

    def feed(self, delta):
        self.buffer += delta
        result = []
        while self.buffer:
            minimum = 6 if self.first else 15
            cut = next(
                (
                    i + 1
                    for i, char in enumerate(self.buffer)
                    if i + 1 >= minimum and char in "，。！？；：,.!?;:\n"
                ),
                None,
            )
            if cut is None:
                # 无标点异常长输出也不能无限积压。
                if len(self.buffer) < 180:
                    break
                cut = 180
            result.append(self.buffer[:cut])
            self.buffer = self.buffer[cut:]
            self.first = False
        return result

    def finish(self):
        value, self.buffer = self.buffer, ""
        return value


def clean_transcript(text):
    """只去口头停顿与重复，不把包含这些字的正常词语一并删除。"""
    text = re.sub(r"\b(?:um|uh|er|you\s+know)\b[ ,]*", "", text, flags=re.I)
    text = re.sub(
        r"(^|[\s，。！？,!?])(?:嗯+|啊+|呃+|额+|哦+|唔+|那个|就是说|然后呢)(?=$|[\s，。！？,!?])", r"\1", text
    )
    text = re.sub(r"^(?:嗯+|啊+|呃+|额+|哦+|唔+|就是说|然后呢)[\s，,。]*", "", text)
    text = re.sub(r"[\s，,]*(?:嗯+|啊+|呃+|额+|哦+|唔+|然后呢)$", "", text)
    text = re.sub(r"([一-鿿]{1,4})\1+", r"\1", text)
    text = re.sub(r"\b([a-zA-Z]+)(?:\s+\1\b)+", r"\1", text, flags=re.I)
    return text.strip(" ，,。!?！？；;")


def is_backchannel(text):
    return clean_transcript(text) in ("", "对", "好的", "好", "是的", "嗯", "哦")


def meaningful_speech(text):
    cleaned = clean_transcript(text)
    return not is_backchannel(cleaned) and len(re.sub(r"[^\w一-鿿]", "", cleaned)) >= 2


def speech_instruct(text):
    # 各句保持统一自然风格，避免关键词触发夸张表扬或忽快忽慢。
    return "自然平实地说话，情绪克制，语调连贯，正常语速；不要夸张重音、拖长尾音或播音腔。"
