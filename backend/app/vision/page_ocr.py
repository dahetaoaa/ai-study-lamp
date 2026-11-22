"""无预设模板自由试卷整页识别、纸张几何矫正与题目/答题区动态提取。

设计分层：
1. 纸面几何层（`app.vision.paper_rectify`）：Lab 低频光照归一化 + RANSAC/Huber 四边拟合 + 文本行微倾角精校；
2. 整页 OCR 层（`PageOCRClient`）：支持 PaddleOCR-VL-1.5 上下半页并发分块 Spotting（`batch=2` + 光照归一化对比度增强）、
   RapidOCR 本地行引擎与云端/本地 VLM 兜底；
3. 题目与答题区切分层（`app.vision.freeform_segmenter` + `app.vision.freeform_ink_gate`）：
   全版式自由试卷切题与毫秒级答题区墨迹门控裁剪。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.telemetry import log, timing
from app.vision.freeform_segmenter import (
    ENG_LETTER_RE,
    FOOTER_RE,
    LOC_LINE_RE,
    MATH_HINT_RE,
    PAIRED_BRACKET_RE as BRACKET_RE,
    QNUM_RE,
    SECTION_RE,
    _infer_subject,
    _parse_paddle_vl_blocks,
    _strip_ocr_tags,
    build_freeform_page_from_lines,
    build_freeform_page_from_vlm_json,
    detect_figure_regions,
    find_underlines,
)
from app.vision.matcher import TemplatePage
from app.vision.paper_rectify import (
    NORM_H,
    NORM_W,
    detect_paper_quad,
    order_quad,
    rectify_paper,
    resize_for_ocr,
)

__all__ = [
    "BRACKET_RE",
    "ENG_LETTER_RE",
    "FOOTER_RE",
    "FreeformMatch",
    "LOC_LINE_RE",
    "MATH_HINT_RE",
    "NORM_H",
    "NORM_W",
    "PAGE_LAYOUT_PROMPT",
    "PageOCRClient",
    "QNUM_RE",
    "SECTION_RE",
    "_infer_subject",
    "_parse_paddle_vl_blocks",
    "_strip_ocr_tags",
    "build_freeform_page_from_lines",
    "build_freeform_page_from_vlm_json",
    "detect_figure_regions",
    "detect_paper_quad",
    "enhance_contrast_paper",
    "find_blank_split_y",
    "find_underlines",
    "freeform_fingerprint",
    "has_spotting_repetition",
    "merge_chunk_blocks",
    "order_quad",
    "rectify_paper",
    "resize_for_ocr",
]

SPOTTING_CHUNK_TARGET_W = 784
SPOTTING_CHUNK_MAX_PIXELS = 520_000
SPOTTING_OVERLAP_HALF_PX = 60
SPOTTING_JPEG_QUALITY = 85


def has_spotting_repetition(
    text: str,
    *,
    partial: bool = False,
    finish_reason: str | None = None,
) -> bool:
    """检测 PaddleOCR-VL Spotting 输出是否陷入自回归复读或达到 max_tokens 截断。

    仅在完成行（流式模式下忽略末尾未换行缓冲行）上判断：
    - `finish_reason == "length"`：达到长度上限被截断；
    - 连续 >= 6 条短行或 >= 10 条任意行缺失 `<|LOC_` 坐标标签（如 `x_1, x_2, ...` 下标自增循环）；
    - 连续 >= 4 条完全相同的带坐标原始行；
    - 单块非空输出行数 >= 95（正常半页试卷最多 ~45 行）。
    """
    if finish_reason == "length":
        return True
    if partial and not text.endswith("\n"):
        idx = text.rfind("\n")
        if idx < 0:
            return False
        text = text[:idx]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 95:
        return True
    no_loc_streak = 0
    no_loc_short_streak = 0
    same_raw_streak = 0
    prev_raw: str | None = None
    for ln in lines:
        if "<|LOC_" not in ln:
            no_loc_streak += 1
            if len(ln) <= 48:
                no_loc_short_streak += 1
            else:
                no_loc_short_streak = 0
            if no_loc_short_streak >= 6 or no_loc_streak >= 10:
                return True
        else:
            no_loc_streak = 0
            no_loc_short_streak = 0
        if ln == prev_raw:
            same_raw_streak += 1
            if same_raw_streak >= 4:
                return True
        else:
            same_raw_streak = 1
            prev_raw = ln
    return False


def freeform_fingerprint(image: np.ndarray) -> list[np.ndarray]:
    """无模板页面的墨迹去重指纹：将拉平后的纸面划分为 4×4 网格二值块复用 same_answers 比较。"""
    warped, _, _, _, _ = rectify_paper(image)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (320, 448), interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 13) > 100
    patches = []
    for r in range(4):
        for c in range(4):
            patches.append(ink[r * 112 : (r + 1) * 112, c * 80 : (c + 1) * 80])
    return patches


def _align28(val: float, min_val: int = 28) -> int:
    return max(min_val, int(round(val / 28.0)) * 28)


def _resize_chunk_for_spotting(
    img: np.ndarray,
    target_w: int = SPOTTING_CHUNK_TARGET_W,
    max_pixels: int = SPOTTING_CHUNK_MAX_PIXELS,
) -> tuple[np.ndarray, int, int]:
    """将分块图像按 28px 对齐缩放至目标宽度（784）且总像素不超过 520,000。"""
    h, w = img.shape[:2]
    tw = _align28(target_w)
    th = _align28(h * (tw / float(max(1, w))))
    if tw * th > max_pixels:
        scale = (max_pixels / float(tw * th)) ** 0.5
        tw = _align28(tw * scale)
        th = _align28(th * scale)
        while tw * th > max_pixels and tw > 28 and th > 28:
            if th >= tw:
                th -= 28
            else:
                tw -= 28
    if (tw, th) == (w, h):
        return img, tw, th
    interp = cv2.INTER_AREA if tw < w or th < h else cv2.INTER_CUBIC
    return cv2.resize(img, (tw, th), interpolation=interp), tw, th


def enhance_contrast_paper(img: np.ndarray) -> np.ndarray:
    """光照归一化 + 纸面底噪漂白：仅用于 Spotting OCR 输入，避免阴影梯度和浅色纸纹干扰。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg_f = np.maximum(bg.astype(np.float32), 32.0)
    norm = np.clip((gray.astype(np.float32) / bg_f) * 255.0, 0, 255).astype(np.uint8)
    lut = np.clip((np.arange(256, dtype=np.float32) - 35.0) * (255.0 / (226.0 - 35.0)), 0, 255).astype(np.uint8)
    clean = cv2.LUT(norm, lut)
    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)


def find_blank_split_y(img: np.ndarray) -> int:
    """在页面中部 [0.41*h, 0.59*h] 区间寻找墨迹最稀疏的行间空白带作为上下分块切缝。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 16)
    x_l, x_r = int(w * 0.03), int(w * 0.97)
    row_ink = (bw[:, x_l:x_r] > 0).sum(axis=1).astype(np.float32)
    row_smooth = cv2.GaussianBlur(row_ink.reshape(-1, 1), (1, 15), 0).flatten()

    lo = max(40, int(h * 0.41))
    hi = min(h - 40, int(h * 0.59))
    if hi <= lo:
        return h // 2
    sub = row_smooth[lo:hi]
    min_v = float(sub.min())
    cand = np.where(sub <= min_v + max(2.0, w * 0.002))[0]
    if len(cand) > 0:
        breaks = np.where(np.diff(cand) > 3)[0]
        segs = np.split(cand, breaks + 1)
        best_seg = max(segs, key=lambda s: (len(s), -abs((s.mean() + lo) - h * 0.5)))
        return int(round(lo + best_seg.mean()))
    return int(lo + np.argmin(sub))


def merge_chunk_blocks(
    chunk_results: list[tuple[int, int, list[dict[str, Any]]]],
    page_w: int,
    page_h: int,
    dedup_overlap: bool = True,
) -> list[dict[str, Any]]:
    """将上下分块 Spotting 块还原到整页坐标（保留四点 poly 与 raw_text），并在重叠带去重。"""
    all_items: list[dict[str, Any]] = []
    for ci, (cy0, cy1, blocks) in enumerate(chunk_results):
        for b in blocks:
            if not isinstance(b, dict) or "box" not in b:
                continue
            x0, y0, x1, y1 = b["box"]
            py0, py1 = cy0 + y0, cy0 + y1
            poly = b.get("poly")
            shifted_poly = (
                [[float(p[0]), float(p[1]) + cy0] for p in poly]
                if isinstance(poly, list) and len(poly) == 4
                else None
            )
            dist_to_cut = min(
                (py0 - cy0) if ci > 0 else 1e9,
                (cy1 - py1) if ci < len(chunk_results) - 1 else 1e9,
            )
            item: dict[str, Any] = {
                "label": b.get("label", "line"),
                "box": [int(x0), int(py0), int(x1), int(py1)],
                "text": b["text"],
                "raw_text": b.get("raw_text", b["text"]),
                "chunk_idx": ci,
                "dist_to_cut": float(dist_to_cut),
            }
            if shifted_poly is not None:
                item["poly"] = shifted_poly
            all_items.append(item)

    def _clean(it: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": it["label"],
            "box": it["box"],
            "text": it["text"],
            "raw_text": it["raw_text"],
        }
        if "poly" in it:
            out["poly"] = it["poly"]
        return out

    if not dedup_overlap or len(chunk_results) <= 1:
        return [_clean(it) for it in all_items]

    kept: list[dict[str, Any]] = []
    dropped = [False] * len(all_items)
    for i in range(len(all_items)):
        if dropped[i]:
            continue
        ai = all_items[i]
        ax0, ay0, ax1, ay1 = ai["box"]
        ah = max(1, ay1 - ay0)
        aw = max(1, ax1 - ax0)
        for j in range(i + 1, len(all_items)):
            if dropped[j]:
                continue
            aj = all_items[j]
            if ai["chunk_idx"] == aj["chunk_idx"]:
                continue
            bx0, by0, bx1, by1 = aj["box"]
            bh = max(1, by1 - by0)
            bw = max(1, bx1 - bx0)
            iy = max(0, min(ay1, by1) - max(ay0, by0))
            ix = max(0, min(ax1, bx1) - max(ax0, bx0))
            y_ov = iy / min(ah, bh)
            x_ov = ix / min(aw, bw)
            if y_ov >= 0.45 and x_ov >= 0.35:
                score_i = (ai["dist_to_cut"] > 4.0, len(ai["text"]), ai["dist_to_cut"])
                score_j = (aj["dist_to_cut"] > 4.0, len(aj["text"]), aj["dist_to_cut"])
                if score_j > score_i:
                    dropped[i] = True
                    break
                dropped[j] = True
        if not dropped[i]:
            kept.append(_clean(ai))
    return kept


PAGE_LAYOUT_PROMPT = (
    "请对这张试卷页面做整页题目OCR与区域定位。忽略页眉页脚，按从上到下题号顺序提取本页每一道题。\n"
    "要求：\n"
    "1. 准确转录印刷题干（数学公式用标准 $...$ LaTeX，选择题提取 options 列表；若题目配有几何图/函数图象/插图，在题干末尾简要客观描述图形关键已知标记，并设 has_figure=true）。\n"
    "2. 坐标统一用 [x1, y1, x2, y2] 归一化整数（范围 0~1000，左上角为 [0,0]，右下角为 [1000,1000]）：\n"
    "   - bbox：整道题（含题干、选项、配图及作答留白）的外接框；\n"
    "   - answer_bbox：该题学生填写答案的括号、横线或下方空白作答区框；\n"
    "3. 不要把学生的手写答案混进印刷 stem 里！若能看到该题括号/横线/空白处有学生手写作答，填入 student_handwriting（无手写填空字符串 \"\"，并设 is_blank=true）。\n"
    "严格只输出如下 JSON 对象，不要输出任何 Markdown 代码块解释：\n"
    '{"subject":"math|en","title":"试卷标题或章节","passage":null,"questions":['
    '{"number":"1","type":"choice|fill|solution|writing","stem":"...","options":["A. ..."],"has_figure":false,'
    '"bbox":[60,120,920,260],"answer_bbox":[580,125,680,175],"student_handwriting":"","is_blank":true}]}'
)


@dataclass
class FreeformMatch:
    template: TemplatePage
    homography: np.ndarray
    crops: dict
    ocr_source: str


def _encode_full_page_b64(warped: np.ndarray, page_w: int, page_h: int) -> str | None:
    scale = min(1.0, 1540.0 / max(page_w, page_h))
    ocr_img = (
        cv2.resize(warped, (round(page_w * scale), round(page_h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else warped
    )
    ok, encoded = cv2.imencode(".jpg", ocr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()


class PageOCRClient:
    """整页无模板 OCR 与切题客户端：优先走本地专用 OCR 服务（若启用且在线），自动回退本地/云端多模态视觉模型。"""

    def __init__(self, clients):
        self.clients = clients
        self._rapid_engine = None

    def _try_load_rapidocr(self):
        if self._rapid_engine is not None:
            return self._rapid_engine
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._rapid_engine = RapidOCR()
            return self._rapid_engine
        except ImportError:
            self._rapid_engine = False
            return False

    async def extract_page(self, image: np.ndarray) -> FreeformMatch | None:
        started = __import__("time").perf_counter()
        warped, h_page_to_cam, _, page_w, page_h = await asyncio.to_thread(rectify_paper, image)

        s = self.clients.settings
        ocr_source = "none"
        paper = page = crops = None
        b64_url: str | None = None

        async def _get_b64_url() -> str | None:
            nonlocal b64_url
            if b64_url is None:
                b64_url = await asyncio.to_thread(_encode_full_page_b64, warped, page_w, page_h)
            return b64_url

        # 路径 A：若本地专用 OCR vLLM 服务（8002，如 PaddleOCR-VL-1.5）已就绪，优先调用本地专用 OCR
        if self.clients.health.get("ocr_vllm") or getattr(s, "freeform_ocr_provider", "auto") == "ocr_vllm":
            try:
                is_paddle_vl = "paddleocr" in s.ocr_model.lower()
                use_parallel = is_paddle_vl and page_h >= 900 and int(getattr(s, "ocr_parallel_chunks", 2) or 1) >= 2
                url_for_ocr = "" if use_parallel else (await _get_b64_url() or "")
                if use_parallel or url_for_ocr:
                    paper, page, crops = await self._call_ocr_vllm(url_for_ocr, warped, page_w, page_h)
                    if page and page["questions"]:
                        ocr_source = f"ocr_vllm:{s.ocr_model}"
            except Exception as exc:  # noqa: BLE001
                log.warning("专用OCR服务调用失败，转入下一候选：%s", type(exc).__name__)

        # 路径 B：若安装了本地 RapidOCR 行级引擎，零显存极速提取行坐标并结构化切题
        if not (page and page["questions"]):
            rapid = self._try_load_rapidocr()
            if rapid:
                try:
                    res, _ = await asyncio.to_thread(rapid, warped)
                    if res:
                        lines = []
                        for box, txt, score in res:
                            xs = [p[0] for p in box]
                            ys = [p[1] for p in box]
                            lines.append(
                                {
                                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                                    "text": txt,
                                    "score": float(score),
                                }
                            )
                        paper, page, crops = await asyncio.to_thread(
                            build_freeform_page_from_lines, lines, warped, page_w, page_h
                        )
                        if page and page["questions"]:
                            ocr_source = "rapidocr_local"
                except Exception as exc:  # noqa: BLE001
                    log.warning("RapidOCR 切题异常：%s", type(exc).__name__)

        # 路径 C：调用配置的多模态整页解析模型（支持本机 8001 Qwen3.5-2B 或云端视觉模型如 qwen3-vl-plus）
        if not (page and page["questions"]):
            full_b64 = await _get_b64_url()
            if not full_b64:
                return None
            providers = self._vlm_candidates()
            for name, base_url, model_name, headers, timeout_s in providers:
                try:
                    parsed = await self._call_vlm_layout(base_url, model_name, headers, full_b64, timeout_s)
                    if parsed and parsed.get("questions"):
                        paper, page, crops = await asyncio.to_thread(
                            build_freeform_page_from_vlm_json, parsed, warped, page_w, page_h
                        )
                        if page and page["questions"]:
                            ocr_source = f"{name}:{model_name}"
                            break
                except Exception as exc:  # noqa: BLE001
                    log.warning("整页VLM OCR (%s/%s) 未命中：%s", name, model_name, type(exc).__name__)

        timing("freeform_ocr", round((__import__("time").perf_counter() - started) * 1000, 1), source=ocr_source)
        if not (paper and page and page["questions"]):
            return None

        gray_page = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        tpl = TemplatePage(
            paper=paper,
            page=page,
            image=gray_page,
            points=np.empty((0, 2), np.float32),
            descriptors=np.empty((0, 61), np.uint8),
        )
        return FreeformMatch(template=tpl, homography=h_page_to_cam, crops=crops, ocr_source=ocr_source)

    def _vlm_candidates(self) -> list[tuple[str, str, str, dict, float]]:
        s = self.clients.settings
        pref = getattr(s, "freeform_vlm_model", "").strip()
        out = []
        if s.hwr_base_url:
            out.append(("local_vlm", s.hwr_base_url, s.hwr_model, self.clients.headers(True), max(30.0, s.hwr_timeout)))
        if s.cloud_llm_api_key.get_secret_value():
            cloud_vl = pref or "qwen3-vl-plus"
            if getattr(s, "freeform_prefer_cloud_vl", False):
                out.insert(0, ("cloud_vl", s.cloud_llm_base_url, cloud_vl, self.clients.headers(False), 38.0))
            else:
                out.append(("cloud_vl", s.cloud_llm_base_url, cloud_vl, self.clients.headers(False), 38.0))
        return out

    def _can_stream_spotting(self) -> bool:
        http = self.clients.http
        if not callable(getattr(http, "stream", None)):
            return False
        cls_post = getattr(type(http), "post", None)
        inst_post = getattr(http, "post", None)
        return cls_post is not None and getattr(inst_post, "__func__", None) is cls_post

    async def _post_spotting_chunk_with_meta(
        self,
        chunk_img: np.ndarray,
        max_tokens: int = 1600,
        *,
        target_w: int = SPOTTING_CHUNK_TARGET_W,
        max_pixels: int = SPOTTING_CHUNK_MAX_PIXELS,
        repetition_penalty: float | None = None,
    ) -> tuple[str, str | None, bool]:
        s = self.clients.settings

        def _prepare() -> str:
            resized, _, _ = _resize_chunk_for_spotting(chunk_img, target_w=target_w, max_pixels=max_pixels)
            ok, enc = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, SPOTTING_JPEG_QUALITY])
            if not ok:
                raise RuntimeError("cv2.imencode failed for spotting chunk")
            return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()

        chunk_b64 = await asyncio.to_thread(_prepare)
        url = s.ocr_base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": s.ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": chunk_b64}},
                        {"type": "text", "text": "Spotting:"},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty

        if self._can_stream_spotting():
            payload["stream"] = True
            buf: list[str] = []
            finish_reason: str | None = None
            aborted = False
            async with self.clients.http.stream("POST", url, json=payload, timeout=35.0) as response:
                response.raise_for_status()
                self.clients.health["ocr_vllm"] = True
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    ch = obj["choices"][0]
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    delta = ch.get("delta", {}).get("content") or ""
                    if delta:
                        buf.append(delta)
                        if "\n" in delta and has_spotting_repetition("".join(buf), partial=True):
                            aborted = True
                            break
            return "".join(buf), finish_reason, aborted

        response = await self.clients.http.post(url, json=payload, timeout=35.0)
        response.raise_for_status()
        self.clients.health["ocr_vllm"] = True
        ch = response.json()["choices"][0]
        return ch["message"]["content"], ch.get("finish_reason"), False

    async def _post_spotting_chunk(
        self,
        chunk_img: np.ndarray,
        max_tokens: int = 1600,
        *,
        target_w: int = SPOTTING_CHUNK_TARGET_W,
        max_pixels: int = SPOTTING_CHUNK_MAX_PIXELS,
        repetition_penalty: float | None = None,
    ) -> str:
        content, _, _ = await self._post_spotting_chunk_with_meta(
            chunk_img,
            max_tokens=max_tokens,
            target_w=target_w,
            max_pixels=max_pixels,
            repetition_penalty=repetition_penalty,
        )
        return content

    async def _call_ocr_vllm(
        self, b64_url: str, warped: np.ndarray, page_w: int, page_h: int
    ) -> tuple[dict | None, dict | None, dict | None]:
        s = self.clients.settings
        is_paddle_vl = "paddleocr" in s.ocr_model.lower()
        parallel_chunks = int(getattr(s, "ocr_parallel_chunks", 2) or 1)

        if is_paddle_vl and page_h >= 900 and parallel_chunks >= 2:
            def _make_chunks() -> list[tuple[int, int, np.ndarray, np.ndarray]]:
                split_y = find_blank_split_y(warped)
                enhanced = enhance_contrast_paper(warped)
                bounds = [
                    (0, min(page_h, split_y + SPOTTING_OVERLAP_HALF_PX)),
                    (max(0, split_y - SPOTTING_OVERLAP_HALF_PX), page_h),
                ]
                return [(cy0, cy1, enhanced[cy0:cy1, :], warped[cy0:cy1, :]) for cy0, cy1 in bounds]

            chunks = await asyncio.to_thread(_make_chunks)

            async def _run_chunk(ci: int, enh_sub: np.ndarray, raw_sub: np.ndarray) -> str:
                txt, fin, aborted = await self._post_spotting_chunk_with_meta(enh_sub, max_tokens=1600)
                if not (aborted or has_spotting_repetition(txt, finish_reason=fin)):
                    return txt
                log.warning(
                    "Spotting 分块 %d 触发防复读重试 (finish=%s, aborted=%s, lines=%d)",
                    ci,
                    fin,
                    aborted,
                    len(txt.splitlines()),
                )
                txt_raw, fin_raw, aborted_raw = await self._post_spotting_chunk_with_meta(raw_sub, max_tokens=1600)
                if not (aborted_raw or has_spotting_repetition(txt_raw, finish_reason=fin_raw)):
                    return txt_raw
                txt_alt, fin_alt, aborted_alt = await self._post_spotting_chunk_with_meta(
                    raw_sub,
                    max_tokens=1600,
                    target_w=840,
                    max_pixels=600_000,
                    repetition_penalty=1.02,
                )
                if not (aborted_alt or has_spotting_repetition(txt_alt, finish_reason=fin_alt)):
                    return txt_alt
                return max(
                    (txt, txt_raw, txt_alt),
                    key=lambda t: sum(1 for ln in t.splitlines() if "<|LOC_" in ln),
                )

            raw_texts = await asyncio.gather(
                *(_run_chunk(ci, enh_sub, raw_sub) for ci, (_, _, enh_sub, raw_sub) in enumerate(chunks))
            )
            parsed_chunks: list[tuple[int, int, list[dict[str, Any]]]] = []
            for (cy0, cy1, _, _), raw_txt in zip(chunks, raw_texts):
                c_blocks = _parse_paddle_vl_blocks(raw_txt, page_w, cy1 - cy0)
                if c_blocks and isinstance(c_blocks[0], dict) and "stem" in c_blocks[0]:
                    return await asyncio.to_thread(
                        build_freeform_page_from_vlm_json, {"questions": c_blocks}, warped, page_w, page_h
                    )
                parsed_chunks.append((cy0, cy1, c_blocks))

            merged_blocks = merge_chunk_blocks(parsed_chunks, page_w, page_h, dedup_overlap=True)
            if merged_blocks:
                return await asyncio.to_thread(build_freeform_page_from_lines, merged_blocks, warped, page_w, page_h)
            raw_text = "\n".join(raw_texts)
        else:
            if not b64_url:
                b64_url = await asyncio.to_thread(_encode_full_page_b64, warped, page_w, page_h) or ""
            prompt = "Spotting:" if is_paddle_vl else PAGE_LAYOUT_PROMPT
            payload = {
                "model": s.ocr_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": b64_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 2800,
                "temperature": 0.0,
                "stream": False,
            }
            if not is_paddle_vl:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
                payload["enable_thinking"] = False

            response = await self.clients.http.post(
                s.ocr_base_url.rstrip("/") + "/chat/completions",
                json=payload,
                timeout=35.0,
            )
            response.raise_for_status()
            self.clients.health["ocr_vllm"] = True
            choice = response.json()["choices"][0]
            raw_text = choice["message"]["content"]
            if is_paddle_vl and has_spotting_repetition(raw_text, finish_reason=choice.get("finish_reason")):
                log.warning("整页 Spotting 触发防复读重试 (finish=%s)", choice.get("finish_reason"))
                payload["repetition_penalty"] = 1.02
                retry_resp = await self.clients.http.post(
                    s.ocr_base_url.rstrip("/") + "/chat/completions",
                    json=payload,
                    timeout=35.0,
                )
                retry_resp.raise_for_status()
                retry_txt = retry_resp.json()["choices"][0]["message"]["content"]
                if sum(1 for ln in retry_txt.splitlines() if "<|LOC_" in ln) >= sum(
                    1 for ln in raw_text.splitlines() if "<|LOC_" in ln
                ):
                    raw_text = retry_txt

            blocks = _parse_paddle_vl_blocks(raw_text, page_w, page_h)
            if blocks and isinstance(blocks[0], dict) and "stem" in blocks[0]:
                return await asyncio.to_thread(
                    build_freeform_page_from_vlm_json, {"questions": blocks}, warped, page_w, page_h
                )
            if blocks:
                return await asyncio.to_thread(build_freeform_page_from_lines, blocks, warped, page_w, page_h)

        # 若返回纯文本无坐标块，按文本段落均匀分配行带兜底切题
        para_lines = [ln.strip() for ln in _strip_ocr_tags(raw_text).splitlines() if ln.strip()]
        if not para_lines:
            return None, None, None
        synth_lines = []
        step_h = max(28, int(page_h * 0.84 / max(1, len(para_lines))))
        for idx, txt in enumerate(para_lines):
            y0 = int(page_h * 0.06) + idx * step_h
            synth_lines.append({"box": [int(page_w * 0.07), y0, int(page_w * 0.92), y0 + step_h - 4], "text": txt})
        return await asyncio.to_thread(build_freeform_page_from_lines, synth_lines, warped, page_w, page_h)

    async def _call_vlm_layout(
        self, base_url: str, model_name: str, headers: dict, b64_url: str, timeout_s: float
    ) -> dict | None:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": b64_url}},
                        {"type": "text", "text": PAGE_LAYOUT_PROMPT},
                    ],
                }
            ],
            "max_tokens": 2800,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        }
        response = await self.clients.http.post(
            base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        text = content.strip()
        if text.startswith("```"):
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            text = text[start : end + 1]
        return json.loads(text)
