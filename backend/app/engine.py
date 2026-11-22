"""视觉到批改的异步编排；CPU重活均进入线程池。"""

import asyncio
import time
from collections import OrderedDict
from time import perf_counter
from uuid import uuid4

import cv2
import httpx
import numpy as np

from app.grading.hwr_client import HWRClient, merge_freeform_choice_result
from app.grading.judge import FirstLineParser, judge_messages
from app.telemetry import elapsed, log, timing
from app.vision.annotations import annotation_geometry
from app.vision.autocapture import AutoDetector, answer_fingerprint, page_key, same_answers
from app.vision.cropper import crop_answers
from app.vision.matcher import project, quad
from app.vision.page_ocr import PageOCRClient, freeform_fingerprint
from app.vision.stability import DetectionConfig


class Engine:
    def __init__(self, settings, session, camera, library, clients):
        self.settings, self.session, self.camera = settings, session, camera
        self.library, self.clients = library, clients
        self.hwr = HWRClient(clients)
        self.page_ocr = PageOCRClient(clients)
        self.captures = OrderedDict()
        self.tasks: dict[str, asyncio.Task] = {}
        self.detector = AutoDetector(
            library,
            DetectionConfig.from_settings(settings),
            allow_freeform=getattr(settings, "freeform_ocr_enabled", True),
        )
        self.stability = self.detector.stability
        self.detector_lock = asyncio.Lock()
        self.recorder = None
        self.last_detection_at = 0.0
        self.detection_state = {"reason": "starting"}
        self.monitor_task = None
        self.capture_lock = asyncio.Lock()

    async def start(self):
        self.monitor_task = asyncio.create_task(self.monitor())

    async def close(self):
        tasks = list(self.tasks.values()) + ([self.monitor_task] if self.monitor_task else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def spawn(self, capture_id, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks[capture_id] = task

        def finish(done):
            if self.tasks.get(capture_id) is done:
                self.tasks.pop(capture_id, None)
            if not done.cancelled() and done.exception():
                log.error("批改任务异常：%s", type(done.exception()).__name__)

        task.add_done_callback(finish)

    async def reset_detector(self):
        async with self.detector_lock:
            self.stability.reset()

    async def monitor(self):
        last_emit, last_seq = 0, -1
        last_reason = None
        last_recording = None
        while True:
            start = perf_counter()
            try:
                enabled = self.session.data["mode"]["auto_capture"]
                seq, frame_ts, frame = await asyncio.to_thread(self.camera.latest_packet)
                capture, match = None, None
                if not enabled:
                    state = {"stability": "idle", "reason": "paused", "progress": 0}
                elif self.camera.backend == "mock" and self.settings.camera_backend != "mock":
                    state = {"stability": "idle", "reason": "no_camera", "progress": 0}
                elif not frame_ts or start - frame_ts > 1:
                    state = {"stability": "idle", "reason": "camera_stalled", "progress": 0}
                elif seq != last_seq:
                    async with self.detector_lock:
                        state, capture, match = await asyncio.to_thread(self.detector.step, frame, frame_ts)
                    if self.recorder:
                        self.recorder.note("detector", camera_sequence=seq, frame_t=frame_ts, **state)
                else:
                    state = self.detection_state
                last_seq = seq
                self.detection_state = state
                self.last_detection_at = perf_counter()
                if state.get("reason") != last_reason:
                    last_reason = state.get("reason")
                    log.info("自动抓拍状态：%s", state)
                if capture is not None:
                    await self.capture(capture, self.session.data["mode"]["auto_grade"],
                                       source="auto", match=match, match_checked=True)
                if self.recorder:
                    recording = self.recorder.status()
                    signature = (recording.get("id"), recording["status"], recording.get("duration_ms", 0) // 1000)
                    if signature != last_recording:
                        last_recording = signature
                        await self.session.emit({"type": "recording.state", **recording})
                if start - last_emit >= 0.2:
                    await self.session.emit({
                        "type": "camera.state", "status": "mock" if self.camera.backend == "mock" else "streaming",
                        **self.detection_state,
                    })
                    last_emit = start
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — 监控异常必须可见且不能永久停掉自动检测
                log.exception("自动抓拍循环异常")
                self.detection_state = {"stability": "idle", "reason": "detector_error", "progress": 0}
                await self.session.emit({"type": "camera.state", "status": "error", **self.detection_state})
                await asyncio.sleep(0.5)
            await asyncio.sleep(max(0.005, 1 / 15 - (perf_counter() - start)))

    async def capture(self, image=None, grade=True, *, source="manual", match=None, match_checked=False):
        async with self.capture_lock:
            started = perf_counter()
            if image is None:
                image = await asyncio.to_thread(self.camera.latest)
            if image is None or image.size == 0:
                raise ValueError("没有可抓拍图像")
            match_start = perf_counter()
            if not match_checked:
                match = await asyncio.to_thread(self.library.match, image) if self.library else None
            match_ms = self.detection_state.get("match_ms", 0) if match_checked else elapsed(match_start)
            if match:
                fingerprint = await asyncio.to_thread(answer_fingerprint, image, match)
            elif getattr(self.settings, "freeform_ocr_enabled", True):
                fingerprint = await asyncio.to_thread(freeform_fingerprint, image)
            else:
                fingerprint = None
            async with self.detector_lock:
                self.stability.lock(image)
            if match:
                for old_id, old in reversed(self.captures.items()):
                    if not old.get("match") or old.get("freeform") or page_key(old["match"]) != page_key(match):
                        continue
                    if not await asyncio.to_thread(same_answers, fingerprint, old.get("fingerprint", [])):
                        continue
                    self.captures.move_to_end(old_id)
                    self.detection_state = {**self.detection_state, "stability": "locked", "reason": "duplicate"}
                    await self.session.emit({"type": "capture.reused", "capture_id": old_id, "source": source})
                    if self.recorder:
                        self.recorder.note("capture.reused", capture_id=old_id, source=source,
                                           paper_id=page_key(match)[0], page_no=page_key(match)[1])
                    result = self.session.data["captures"].get(old_id, {}).get("grade_result", {})
                    if grade and old_id not in self.tasks and (not old["grade_started"] or result.get("error_uids")):
                        old["grade_started"] = False
                        self.spawn(old_id, self.grade(old_id))
                    h, w = old["image"].shape[:2]
                    return {"capture_id": old_id, "image_url": f"/api/captures/{old_id}.jpg",
                            "width": w, "height": h, "reused": True}
            elif fingerprint:
                for old_id, old in reversed(self.captures.items()):
                    if not old.get("freeform") or not old.get("match"):
                        continue
                    if not await asyncio.to_thread(same_answers, fingerprint, old.get("fingerprint", [])):
                        continue
                    self.captures.move_to_end(old_id)
                    self.detection_state = {**self.detection_state, "stability": "locked", "reason": "duplicate"}
                    await self.session.emit({"type": "capture.reused", "capture_id": old_id, "source": source})
                    h, w = old["image"].shape[:2]
                    return {"capture_id": old_id, "image_url": f"/api/captures/{old_id}.jpg",
                            "width": w, "height": h, "reused": True}
            ok, encoded = await asyncio.to_thread(cv2.imencode, ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if not ok:
                raise ValueError("图像编码失败")
            capture_id = "cap_" + uuid4().hex[:16]
            height, width = image.shape[:2]
            url = f"/api/captures/{capture_id}.jpg"
            cap = {
                "image": image,
                "jpeg": encoded.tobytes(),
                "match": match,
                "fingerprint": fingerprint,
                "timings_ms": {"capture": elapsed(started), "match": match_ms},
                "started": started,
                "grade_started": False,
            }
            self.captures[capture_id] = cap
            while len(self.captures) > 20:
                old_id, _ = self.captures.popitem(last=False)
                task = self.tasks.get(old_id)
                if task:
                    task.cancel()
                self.session.data["captures"].pop(old_id, None)
                old_questions = [
                    k for k, v in self.session.data["questions"].items() if v["capture_id"] == old_id
                ]
                for key in old_questions:
                    self.session.data["questions"].pop(key, None)
            timing("capture", cap["timings_ms"]["capture"], capture_id=capture_id)
            await self.session.emit(
                {
                    "type": "capture.taken",
                    "capture_id": capture_id,
                    "image_url": url,
                    "image_width": width,
                    "image_height": height,
                    "ts": time.strftime("%H:%M:%S"),
                    "source": source,
                }
            )
            if self.recorder:
                self.recorder.note("capture.taken", capture_id=capture_id, source=source,
                                   paper_id=page_key(match)[0] if match else None,
                                   page_no=page_key(match)[1] if match else None)
            self.spawn(capture_id, self.match_and_grade(capture_id, grade))
            return {"capture_id": capture_id, "image_url": url, "width": width, "height": height}

    async def match_and_grade(self, capture_id, grade):
        cap = self.captures[capture_id]
        try:
            match = cap["match"]
            timing("match", cap["timings_ms"].get("match", 0), capture_id=capture_id)
            if match is None and getattr(self.settings, "freeform_ocr_enabled", True):
                await self.session.timeline("detect", "未命中入库模板，正在执行整页 OCR 与图题切分…")
                free = await self.page_ocr.extract_page(cap["image"])
                if free is not None:
                    match = free
                    cap["freeform"] = True
                    cap["pre_crops"] = free.crops
                    cap["timings_ms"]["page_ocr_source"] = free.ocr_source
            if match is None:
                await self.session.emit(
                    {"type": "page.unmatched", "capture_id": capture_id, "reason": "no_page"}
                )
                await self.session.timeline("alert", "未能从画面识别到有效试卷题目，请调整纸张位置或光线后重拍")
                return
            cap["match"] = match
            mode_name = "freeform" if cap.get("freeform") else "template"
            page, paper = match.template.page, match.template.paper
            height, width = cap["image"].shape[:2]
            items = []
            for q in page["questions"]:
                geometry = annotation_geometry(q, match.homography, page["width_px"], page["height_px"])
                q_quad = project(quad(q["bbox"]), match.homography)
                pts = np.array(q_quad)
                low, high = pts.min(axis=0), pts.max(axis=0)
                uid = q["id"]
                # 同一道题重复抓拍保留历史，不覆盖历史页的识别与讲解。
                if uid in self.session.data["questions"]:
                    uid = f"{uid}@{capture_id}"
                items.append(
                    {
                        "question_uid": uid,
                        "template_question_id": q["id"],
                        "capture_id": capture_id,
                        "no": str(q["number"]),
                        "qtype": q["type"],
                        "short_stem": q["stem"][:80],
                        "stem": q["stem"],
                        "options": q.get("options"),
                        "tag": q.get("tag", ""),
                        "passage_id": q.get("passage_id"),
                        "passage": paper.get("passages", {}).get(q.get("passage_id"), ""),
                        "status": "judging" if grade and q["type"] != "listening" else "idle",
                        "quad": q_quad,
                        "bbox": [
                            float(low[0]),
                            float(low[1]),
                            float(high[0] - low[0]),
                            float(high[1] - low[1]),
                        ],
                        **geometry,
                        "answer_quads": [
                            project(quad(a["bbox"]), match.homography) for a in q.get("answer_areas", [])
                        ],
                    }
                )
            cap["items"] = {item["template_question_id"]: item for item in items}
            await self.session.emit(
                {
                    "type": "page.matched",
                    "capture_id": capture_id,
                    "mode": mode_name,
                    "paper_id": paper["paper_id"],
                    "paper_title": paper["title"],
                    "subject": paper["subject"],
                    "page_no": page["page_no"],
                    "image_width": width,
                    "image_height": height,
                    "homography": match.homography.tolist(),
                    "page_quad": project(quad([0, 0, page["width_px"], page["height_px"]]), match.homography),
                    "questions": items,
                }
            )
            await self.session.timeline("detect", f"检测到 {paper['title']} · 第{page['page_no']}页")
            if grade:
                await self.grade(capture_id)
        except (cv2.error, ValueError, KeyError) as error:
            await self.session.error("VISION_FAILED", "图像处理失败，请重新抓拍", capture_id=capture_id)
            log.warning("视觉处理失败：%s", type(error).__name__)

    async def grade_current(self):
        cid = self.session.data["active_capture_id"]
        if not cid or cid not in self.captures:
            await self.session.error("NO_CAPTURE", "请先抓拍试卷")
            return
        if cid in self.tasks:
            await self.session.error("GRADE_BUSY", "页面仍在处理，请稍后再试")
            return
        cap = self.captures[cid]
        if cap["match"] is None:
            self.spawn(cid, self.match_and_grade(cid, True))
        else:
            self.spawn(cid, self.grade(cid))

    async def grade(self, capture_id):
        cap = self.captures[capture_id]
        if cap["grade_started"]:
            return
        cap["grade_started"] = True
        cap["verdict_count"] = 0
        match = cap["match"]
        start = perf_counter()
        if cap.get("freeform") and cap.get("pre_crops"):
            crops = cap["pre_crops"]
        else:
            crops = await asyncio.to_thread(crop_answers, cap["image"], match)
        cap["timings_ms"]["crop"] = elapsed(start)
        timing("crop", elapsed(start), capture_id=capture_id)
        questions = [q for q in match.template.page["questions"] if q["type"] != "listening"]
        for q in questions:
            row = self.session.data["questions"][cap["items"][q["id"]]["question_uid"]]
            row.update(status="judging", explain_text="", board_steps=[])
        await self.session.emit(self.session.snapshot())
        cap["question_count"] = len(questions)
        results = await asyncio.gather(
            *(self.grade_question(capture_id, q, crops[q["id"]]) for q in questions)
        )
        cap["timings_ms"]["all_explanations"] = elapsed(cap["started"])
        scored = [(q, r) for q, r in zip(questions, results) if q["type"] != "writing"]
        right = sum(r == "right" for _, r in scored)
        blank = sum(r == "blank" for _, r in scored)
        errors = [cap["items"][q["id"]]["question_uid"] for q, r in zip(questions, results) if r is None]
        wrong = [cap["items"][q["id"]]["question_uid"] for q, r in scored if r == "wrong"]
        uncertain = [cap["items"][q["id"]]["question_uid"] for q, r in zip(questions, results) if r == "uncertain"]
        summary = f"{len(scored)}道计分题，对了{right}道，{len(wrong)}道需订正，{blank}道未作答。"
        if uncertain:
            summary += f"{len(uncertain)}道待确认，暂不判对错。"
        if len(questions) > len(scored):
            summary += f"另有{len(questions) - len(scored)}道写作题只作点评，不计对错。"
        if errors:
            summary += f"{len(errors)}道识别或判题失败，可重新抓拍重试。"
        await self.session.emit(
            {
                "type": "page.grade_done",
                "capture_id": capture_id,
                "paper_id": match.template.paper["paper_id"],
                "page_no": match.template.page["page_no"],
                "total": len(scored),
                "right_count": right,
                "wrong_uids": wrong,
                "blank_count": blank,
                "summary_text": summary,
                "timings_ms": dict(cap["timings_ms"]),
                "error_uids": errors,
                "uncertain_uids": uncertain,
            }
        )
        await self.session.timeline("alert" if errors or uncertain else "ok", summary)
        # 释放高清数组与校正结果占用，JPEG供历史浏览；保留当前20张原图支持重批。

    async def grade_question(self, cid, question, crop):
        cap = self.captures[cid]
        item = cap["items"][question["id"]]
        if cap.get("match") and "primary_area_index" in crop:
            match = cap["match"]
            page = match.template.page
            item.update(annotation_geometry(
                question, match.homography, page["width_px"], page["height_px"],
                crop.get("primary_area_index", 0), crop.get("ink_bbox"),
            ))
        question = {**question, "passage": item.get("passage", "")}
        uid = item["question_uid"]
        context = {"capture_id": cid, "question_uid": uid}
        raw, low, blank = "", False, crop["is_blank"]
        verdict_status = None
        explained = ""
        verdict_data = None

        async def verdict(data, source):
            nonlocal verdict_status, verdict_data
            if verdict_status is not None:
                return
            if low:
                data = {**data, "status": "uncertain", "error_note": "手写识别不确定，请确认作答内容。"}
            elif blank:
                data = {**data, "status": "blank", "student_display": ""}
            elif question["type"] == "writing" and data["status"] != "uncertain":
                data = {**data, "status": "blank"}
            verdict_data, verdict_status = data, data["status"]
            elapsed_ms = elapsed(cap["started"])
            if "first_verdict" not in cap["timings_ms"]:
                cap["timings_ms"]["first_verdict"] = elapsed_ms
                timing("first_verdict", elapsed_ms, capture_id=cid)
            cap["timings_ms"]["all_verdicts"] = elapsed_ms
            cap["verdict_count"] += 1
            if cap["verdict_count"] == cap["question_count"]:
                timing("all_verdicts", elapsed_ms, capture_id=cid)
            timing("verdict", elapsed_ms, **context)
            answer_quads = item["answer_quads"]
            index = item.get("primary_area_index", 0)
            area_quad = answer_quads[min(index, len(answer_quads) - 1)] if answer_quads else None
            error_quad = (item.get("ink_quad") or area_quad) if verdict_status == "wrong" else None
            # 展示模型实时求出的答案，不再将模板答案作为正确答案发送给前端。
            reference = data.get("model_answer") or ""
            await self.session.emit(
                {
                    "type": "question.verdict",
                    **context,
                    "status": verdict_status,
                    "is_blank": blank,
                    "source": source,
                    "student_display": data.get("student_display") or raw,
                    "model_answer": data.get("model_answer"),
                    "reference_display": reference,
                    "tag": question.get("tag", ""),
                    "mark_pos": item["mark_pos"],
                    "mark_quad": item.get("mark_quad"),
                    "primary_area_index": item.get("primary_area_index", 0),
                    "error_quad": error_quad,
                    "error_note": data.get("error_note"),
                    "error_step_desc": data.get("error_step_desc"),
                    "step_diagnosis": data.get("step_diagnosis"),
                }
            )

        try:
            start = perf_counter()
            if cap.get("freeform"):
                # 无模板模式下：经 ink gate 确认空白时直接 0ms 跳过 HWR；非空白则跑 HWR 并在选择题下融合 Spotting 预提取结果
                if crop.get("is_blank"):
                    raw, low, blank = "", False, True
                else:
                    pre_hw = (crop.get("pre_ocr_handwriting") or "").strip()
                    if crop.get("blocks"):
                        raw, low = await self.hwr.recognize(crop["blocks"], question["type"], freeform=True)
                        if question["type"] == "choice":
                            raw, low = merge_freeform_choice_result(
                                raw,
                                low,
                                pre_hw,
                                ink_pixels=crop.get("clean_ink_pixels", 0),
                                pre_ocr_confident=crop.get("pre_ocr_confident", False),
                            )
                        elif not raw.strip() and pre_hw:
                            raw, low = pre_hw, False
                        blank = not raw.strip() and not low
                    else:
                        raw, low = pre_hw, False
                        blank = not raw
            elif not blank:
                raw, low = await self.hwr.recognize(crop["blocks"], question["type"])
                blank = not raw.strip() and not low
            hwr_ms = elapsed(start)
            cap["timings_ms"].setdefault("hwr", {})[uid] = hwr_ms
            timing("hwr", hwr_ms, **context)
            await self.session.emit(
                {
                    "type": "question.ocr",
                    **context,
                    "is_blank": blank,
                    "student_raw": raw,
                    "student_lines": [
                        {"text": line, "latex": None, "low_confidence": low, "quad": None}
                        for line in raw.splitlines()
                    ],
                }
            )
            parser = FirstLineParser()
            messages = judge_messages(
                question,
                raw,
                blank,
                low_confidence=low,
                question_jpeg=crop.get("question_jpeg") if question.get("has_diagram") else None,
            )
            async for chunk in self.clients.stream(messages, **context):
                data, text = parser.feed(chunk)
                if data:
                    await verdict(data, "cloud_llm")
                if text and not explained:
                    text = text.lstrip()
                if text:
                    explained += text
                    await self.session.emit({"type": "question.explain_delta", **context, "delta": text})
            data = parser.finish()
            if data:
                await verdict(data, "cloud_llm")
        except (TimeoutError, httpx.TimeoutException):
            await self.session.error(
                "CLOUD_LLM_TIMEOUT" if raw or blank else "HWR_TIMEOUT", "模型响应超时，请重试", **context
            )
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
            log.warning("题目处理失败 %s：%s", uid, type(error).__name__)
            await self.session.error("MODEL_RESPONSE_ERROR", "识别或判题服务暂不可用，请重试", **context)
        if verdict_status is None:
            # 空白是真实可确定的状态；网络失败不能把其他学生答案伪判为错误。
            if blank:
                await verdict({"status": "blank", "student_display": ""}, "local")
            else:
                self.session.data["questions"][uid]["status"] = "idle"
        await self.session.emit(
            {
                "type": "question.explain_done",
                **context,
                "full_text": explained,
                "board_steps": (verdict_data or {}).get("board_steps", []),
            }
        )
        return verdict_status
