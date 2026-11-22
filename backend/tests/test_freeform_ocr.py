import json
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.grading.client import ModelClients
from app.main import create_app


async def fake_warmup(self):
    self.health.update(hwr_vllm=True, ocr_vllm=False, cloud_llm=True)


async def fake_vlm_layout(self, base_url, model_name, headers, b64_url, timeout_s):
    return {
        "subject": "math",
        "title": "无模板随堂小测 · 一元二次方程",
        "passage": None,
        "questions": [
            {
                "number": "1",
                "type": "choice",
                "stem": "如图，抛物线 $y=x^2-2x-3$ 与 $x$ 轴的一个交点为 $(3,0)$，则另一个交点为（ ）（图略：顶点在第四象限）",
                "options": ["A. $(-1,0)$", "B. $(1,0)$", "C. $(-3,0)$", "D. $(0,-3)$"],
                "has_figure": True,
                "bbox": [80, 110, 900, 320],
                "answer_bbox": [610, 120, 720, 185],
                "student_handwriting": "A",
                "is_blank": False,
            },
            {
                "number": "2",
                "type": "fill",
                "stem": "用因式分解法解方程 $x(x-2)=0$ 的两个根是 ______。",
                "options": None,
                "has_figure": False,
                "bbox": [80, 360, 900, 520],
                "answer_bbox": [520, 390, 760, 460],
                "student_handwriting": "",
                "is_blank": True,
            },
        ],
    }


async def fake_hwr(self, blocks, qtype, *, freeform=False):
    if qtype == "choice":
        return "A", False
    return "", False


async def fake_judge_stream(self, messages, **context):
    req = json.loads(messages[-1]["content"])
    if req["未作答"]:
        yield json.dumps({"status": "blank", "model_answer": "x_1=0, x_2=2", "student_display": ""}) + "\n"
        yield "本题考查因式分解法，令每个因式为 0 得 $x_1=0, x_2=2$。"
    else:
        yield json.dumps({"status": "right", "model_answer": "A", "student_display": req["学生作答"]}) + "\n"
        yield "由对称轴 $x=1$ 可知另一交点为 $(-1,0)$，选 A。"


def make_untemplated_exam_image():
    # 深色桌面背景 + 倾斜白色试卷 + 几何示意图 + 模拟文字横条
    img = np.full((1080, 1920, 3), 42, dtype=np.uint8)
    paper = np.full((1200, 860, 3), 245, dtype=np.uint8)
    for y in range(90, 980, 48):
        cv2.line(paper, (70, y), (780, y), (45, 45, 45), 2)
    # 模拟几何图框
    cv2.circle(paper, (660, 260), 65, (60, 60, 60), 2)
    src = np.float32([[0, 0], [860, 0], [860, 1200], [0, 1200]])
    dst = np.float32([[520, 60], [1360, 85], [1400, 1010], [490, 995]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(paper, H, (1920, 1080), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
    return warped


def test_freeform_untemplated_paper_end_to_end(tmp_path):
    # templates_dir 为空目录，模拟撤掉所有预设模板！
    app = create_app(
        Settings(
            _env_file=None,
            camera_backend="mock",
            auto_capture=False,
            tts_provider="mock",
            templates_dir=tmp_path,
            freeform_ocr_enabled=True,
        )
    )
    image = make_untemplated_exam_image()
    payload = cv2.imencode(".jpg", image)[1].tobytes()

    with (
        patch.object(ModelClients, "warmup", fake_warmup),
        patch("app.vision.page_ocr.PageOCRClient._call_vlm_layout", fake_vlm_layout),
        patch("app.grading.hwr_client.HWRClient.recognize", fake_hwr),
        patch.object(ModelClients, "stream", fake_judge_stream),
        TestClient(app) as client,
    ):
        assert client.get("/api/templates").json()["papers"] == []
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "session.snapshot"
            r = client.post(
                "/api/debug/inject-image",
                files={"file": ("untemplated.jpg", payload, "image/jpeg")},
                data={"grade": "true"},
            )
            assert r.status_code == 200
            events = []
            for _ in range(200):
                ev = ws.receive_json()
                events.append(ev)
                if ev["type"] == "page.grade_done":
                    break
            types = [e["type"] for e in events]
            assert "page.unmatched" not in types
            assert "page.matched" in types
            matched = next(e for e in events if e["type"] == "page.matched")
            assert matched["mode"] == "freeform"
            assert len(matched["questions"]) == 2
            done = next(e for e in events if e["type"] == "page.grade_done")
            assert done["total"] == 2
            assert done["right_count"] == 1
            assert done["blank_count"] == 1


def test_parallel_spotting_chunking_and_engine_ink_gate_skip(tmp_path):
    import asyncio
    import base64
    from unittest.mock import AsyncMock

    from app.grading.client import ModelClients
    from app.vision.page_ocr import PageOCRClient, merge_chunk_blocks

    # 构造接近真实竖版 A4 的图像，含第 1 题手写 "B"、第 2 题空白横线
    img = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    paper = np.full((1400, 1000, 3), 245, dtype=np.uint8)
    cv2.putText(paper, "1. Equation x^2=1 (   )", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2)
    cv2.putText(paper, "B", (435, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 3)
    cv2.putText(paper, "A. 1   B. -1   C. 0   D. 2", (120, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    cv2.putText(paper, "2. Fill blank:", (100, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2)
    cv2.line(paper, (340, 1070), (520, 1070), (30, 30, 30), 2)
    src = np.float32([[0, 0], [1000, 0], [1000, 1400], [0, 1400]])
    dst = np.float32([[480, 40], [1420, 50], [1440, 1030], [460, 1020]])
    H = cv2.getPerspectiveTransform(src, dst)
    cv2.warpPerspective(paper, H, (1920, 1080), dst=img, borderMode=cv2.BORDER_TRANSPARENT)

    captured_payloads = []

    class FakeResp:
        def __init__(self, text):
            self._text = text

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self._text}}]}

    async def fake_post(url, json=None, timeout=None, **kwargs):
        captured_payloads.append(json)
        idx = len(captured_payloads)
        if idx == 1:
            # 上半块：返回 Q1 题干、手写 B、选项行及一条接近中缝的共享行
            txt = (
                "1. Equation x^2=1 (   )<|LOC_100|><|LOC_220|><|LOC_490|><|LOC_270|><|LOC_490|><|LOC_275|><|LOC_100|><|LOC_225|>\n"
                "B<|LOC_430|><|LOC_218|><|LOC_465|><|LOC_270|><|LOC_465|><|LOC_274|><|LOC_430|><|LOC_222|>\n"
                "A. 1   B. -1   C. 0   D. 2<|LOC_120|><|LOC_300|><|LOC_650|><|LOC_345|><|LOC_650|><|LOC_348|><|LOC_120|><|LOC_303|>\n"
                "Shared seam line<|LOC_120|><|LOC_940|><|LOC_500|><|LOC_980|><|LOC_500|><|LOC_982|><|LOC_120|><|LOC_942|>"
            )
            return FakeResp(txt)
        # 下半块：包含重叠带上的 Shared seam line 与第 2 题
        txt = (
            "Shared seam line<|LOC_120|><|LOC_20|><|LOC_500|><|LOC_60|><|LOC_500|><|LOC_62|><|LOC_120|><|LOC_22|>\n"
            "2. Fill blank:<|LOC_100|><|LOC_540|><|LOC_320|><|LOC_590|><|LOC_320|><|LOC_593|><|LOC_100|><|LOC_543|>"
        )
        return FakeResp(txt)

    settings = Settings(
        _env_file=None,
        templates_dir=tmp_path,
        ocr_model="PaddleOCR-VL-1.5",
        freeform_ocr_enabled=True,
        freeform_ocr_provider="ocr_vllm",
    )
    clients = ModelClients(settings)
    clients.health["ocr_vllm"] = True
    clients.http.post = AsyncMock(side_effect=fake_post)

    async def _run():
        try:
            ocr_client = PageOCRClient(clients)
            match = await ocr_client.extract_page(img)
            return match
        finally:
            await clients.http.aclose()

    match = asyncio.run(_run())
    assert match is not None
    assert match.ocr_source == "ocr_vllm:PaddleOCR-VL-1.5"
    assert len(captured_payloads) == 2
    for pl in captured_payloads:
        assert pl["max_tokens"] == 1600
        b64 = pl["messages"][0]["content"][0]["image_url"]["url"].split(",", 1)[1]
        chunk_mat = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        assert chunk_mat.shape[1] == 784
        assert chunk_mat.shape[0] % 28 == 0
        assert chunk_mat.shape[0] * chunk_mat.shape[1] <= 520_000

    # 验证 merge_chunk_blocks 保留 poly 并对重叠行去重
    merged = merge_chunk_blocks(
        [
            (0, 1000, [{"box": [100, 940, 400, 980], "poly": [[100, 940], [400, 940], [400, 980], [100, 980]], "text": "Shared seam line"}]),
            (900, 1980, [{"box": [100, 40, 400, 80], "poly": [[100, 40], [400, 40], [400, 80], [100, 80]], "text": "Shared seam"}]),
        ],
        1400,
        1980,
    )
    assert len(merged) == 1
    assert merged[0]["text"] == "Shared seam line"
    assert merged[0]["poly"][0][1] == 940.0

    # 验证 Engine.grade_question 在 freeform 下跳过 is_blank=True 的 HWR 调用并融合选择题
    qs = match.template.page["questions"]
    assert len(qs) == 2
    assert match.crops[qs[1]["id"]]["is_blank"] is True

    app = create_app(
        Settings(
            _env_file=None,
            camera_backend="mock",
            auto_capture=False,
            tts_provider="mock",
            templates_dir=tmp_path,
            freeform_ocr_enabled=True,
        )
    )
    hwr_calls = []

    async def counting_hwr(self, blocks, qtype, *, freeform=False):
        hwr_calls.append((qtype, freeform))
        return "", True  # 模拟 HWR 不确定，测试 merge_freeform_choice_result 融合 pre_ocr_handwriting

    async def fake_extract_page(self, _img):
        # 确保第 1 题非空白且有置信手写 B，第 2 题明确空白
        q0, q1 = qs[0], qs[1]
        match.crops[q0["id"]]["is_blank"] = False
        match.crops[q0["id"]]["pre_ocr_handwriting"] = "B"
        match.crops[q0["id"]]["pre_ocr_confident"] = True
        match.crops[q1["id"]]["is_blank"] = True
        return match

    payload = cv2.imencode(".jpg", img)[1].tobytes()
    with (
        patch.object(ModelClients, "warmup", fake_warmup),
        patch("app.vision.page_ocr.PageOCRClient.extract_page", fake_extract_page),
        patch("app.grading.hwr_client.HWRClient.recognize", counting_hwr),
        patch.object(ModelClients, "stream", fake_judge_stream),
        TestClient(app) as client,
    ):
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "session.snapshot"
            r = client.post(
                "/api/debug/inject-image",
                files={"file": ("freeform_parallel.jpg", payload, "image/jpeg")},
                data={"grade": "true"},
            )
            assert r.status_code == 200
            events = []
            for _ in range(200):
                ev = ws.receive_json()
                events.append(ev)
                if ev["type"] == "page.grade_done":
                    break
            assert len(hwr_calls) == 1
            assert hwr_calls[0] == ("choice", True)
            verdicts = [e for e in events if e["type"] == "question.verdict"]
            assert any(v["student_display"] == "B" and v["status"] == "right" for v in verdicts)
            done = next(e for e in events if e["type"] == "page.grade_done")
            assert done["right_count"] == 1
            assert done["blank_count"] == 1


def test_spotting_repetition_guard_and_retry(tmp_path):
    """验证 Spotting 复读检测及单块触发 length/复读后的原图自愈重试。"""
    import asyncio
    from unittest.mock import AsyncMock

    from app.grading.client import ModelClients
    from app.vision.page_ocr import PageOCRClient, has_spotting_repetition

    # 1. 合法的 10 行树状图坐标块（文本相同但各有不同 LOC 坐标）不能误判为复读
    valid_diagram = "\n".join(
        f"\\( \\alpha \\)<|LOC_{100 + i * 30}|><|LOC_400|><|LOC_{120 + i * 30}|><|LOC_400|>"
        f"<|LOC_{120 + i * 30}|><|LOC_420|><|LOC_{100 + i * 30}|><|LOC_420|>"
        for i in range(10)
    )
    assert has_spotting_repetition(valid_diagram) is False

    # 2. 下标自增无坐标循环（x_1 .. x_8）与 finish_reason=length 均应判定为复读
    loop_txt = "8. 表格题<|LOC_100|><|LOC_100|><|LOC_300|><|LOC_100|><|LOC_300|><|LOC_130|><|LOC_100|><|LOC_130|>\n"
    loop_txt += "\n".join(f"\\( x_{i} \\)" for i in range(1, 8))
    assert has_spotting_repetition(loop_txt) is True
    assert has_spotting_repetition("1. 正常行<|LOC_10|><|LOC_10|><|LOC_20|><|LOC_10|><|LOC_20|><|LOC_20|><|LOC_10|><|LOC_20|>", finish_reason="length") is True

    # 3. 构造竖版试卷图（使上下分块高度明显不同），验证 chunk 0 触发复读后仅重试 chunk 0
    warped = np.full((1940, 1400, 3), 245, dtype=np.uint8)
    # 在下半区绘制文本占位条，使 find_blank_split_y 稳定落在 ~820 附近（chunk0 较矮、chunk1 较高）
    cv2.rectangle(warped, (150, 1100), (1200, 1800), (60, 60, 60), -1)
    chunk0_calls = {"n": 0}
    chunk1_calls = {"n": 0}

    class FakeResp:
        def __init__(self, text, finish_reason="stop"):
            self._text = text
            self._finish = finish_reason

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self._text}, "finish_reason": self._finish}]}

    async def fake_post(url, json=None, timeout=None, **kwargs):
        import base64

        b64 = json["messages"][0]["content"][0]["image_url"]["url"].split(",", 1)[1]
        chunk_mat = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        is_top_chunk = chunk_mat.shape[0] < 580
        if is_top_chunk:
            chunk0_calls["n"] += 1
            if chunk0_calls["n"] == 1:
                # 第一次 chunk 0 模拟自回归复读截断
                return FakeResp(loop_txt, finish_reason="length")
            # 第二次为 chunk 0 原图重试，正常返回第 8、9 题
            return FakeResp(
                "8. 选择题（ ）<|LOC_120|><|LOC_100|><|LOC_500|><|LOC_100|><|LOC_500|><|LOC_140|><|LOC_120|><|LOC_140|>\n"
                "A. 1 B. 2 C. 3 D. 4<|LOC_140|><|LOC_160|><|LOC_600|><|LOC_160|><|LOC_600|><|LOC_200|><|LOC_140|><|LOC_200|>\n"
                "9. 第二题（ ）<|LOC_120|><|LOC_300|><|LOC_500|><|LOC_300|><|LOC_500|><|LOC_340|><|LOC_120|><|LOC_340|>\n"
                "A. 1 B. 2 C. 3 D. 4<|LOC_140|><|LOC_360|><|LOC_600|><|LOC_360|><|LOC_600|><|LOC_400|><|LOC_140|><|LOC_400|>",
                finish_reason="stop",
            )
        chunk1_calls["n"] += 1
        return FakeResp(
            "11. 填空题<|LOC_120|><|LOC_120|><|LOC_500|><|LOC_120|><|LOC_500|><|LOC_160|><|LOC_120|><|LOC_160|>",
            finish_reason="stop",
        )

    settings = Settings(_env_file=None, templates_dir=tmp_path, ocr_model="PaddleOCR-VL-1.5", freeform_ocr_enabled=True)
    clients = ModelClients(settings)
    clients.http.post = AsyncMock(side_effect=fake_post)

    async def _run():
        try:
            ocr_client = PageOCRClient(clients)
            return await ocr_client._call_ocr_vllm("", warped, 1400, 1940)
        finally:
            await clients.http.aclose()

    paper, page, crops = asyncio.run(_run())
    assert chunk0_calls["n"] == 2
    assert chunk1_calls["n"] == 1
    assert page is not None
    nums = [str(q["number"]) for q in page["questions"]]
    assert nums == ["8", "9", "11"]

