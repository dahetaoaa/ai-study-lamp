"""FastAPI单进程入口：HTTP、单条多路WS、静态前端。"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, Settings
from app.engine import Engine
from app.grading.client import ModelClients
from app.knowledge.service import get_overview_payload, get_subject_graph
from app.state import Session
from app.vision.camera import Camera
from app.vision.matcher import TemplateLibrary
from app.vision.recorder import Recorder, RecordingError
from app.voice.controller import VoiceController
from app.voice.vad import SileroVAD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# HTTP请求日志可能包含代理参数，禁用第三方逐请求INFO输出。
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class RecordingRequest(BaseModel):
    label: str = Field(default="", max_length=100)


class SnapshotRequest(BaseModel):
    grade: bool = True


def decode_image(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] * image.shape[1] > 24_000_000:
        raise ValueError("图像无效或像素过大")
    return image


def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        cv2.setNumThreads(settings.opencv_threads)
        app.state.session = Session(settings)
        app.state.library = TemplateLibrary(settings.templates_dir, settings.fixture_template)
        await asyncio.to_thread(app.state.library.load)
        app.state.clients = ModelClients(settings)
        app.state.camera = Camera(settings)
        app.state.engine = Engine(
            settings, app.state.session, app.state.camera, app.state.library, app.state.clients
        )
        app.state.recorder = Recorder(settings, app.state.camera, app.state.engine.stability.parameters())
        app.state.engine.recorder = app.state.recorder
        app.state.voice = VoiceController(settings, app.state.session, app.state.clients)
        app.state.stream_count = 0
        app.state.resized_cache = {}
        app.state.camera.start()
        await asyncio.gather(app.state.clients.warmup(), app.state.voice.stt.warmup())
        if settings.tts_warmup:
            # 完整短句走真实流式路径，模型加载/CUDA graph捕获不落到首次对话。
            await app.state.voice.tts.warmup()
        await app.state.engine.start()
        try:
            yield
        finally:
            await app.state.engine.close()
            try:
                await asyncio.to_thread(app.state.recorder.stop, "shutdown")
            except RecordingError:
                logging.getLogger(__name__).exception("录制退出未完成")
            await app.state.voice.close()
            await asyncio.to_thread(app.state.camera.stop)
            await app.state.clients.http.aclose()

    app = FastAPI(title="AI学习台灯", lifespan=lifespan)

    @app.get("/api/health")
    async def health():
        state = app.state
        frame = state.camera.frame
        models = {
            **state.clients.health,
            "stt": state.voice.stt.available,
            "tts": state.voice.tts.available,
            "vad": SileroVAD().available and all(not v.vad.failed for v in state.voice.inputs),
        }
        return {
            "ok": all(models[k] for k in ("hwr_vllm", "cloud_llm", "stt", "tts", "vad"))
            and bool(state.engine.monitor_task and not state.engine.monitor_task.done())
            and state.engine.detection_state.get("reason") not in ("detector_error", "no_camera", "camera_stalled"),
            "platform": sys.platform,
            "camera": {
                "ok": state.camera.thread.is_alive(),
                "backend": state.camera.backend,
                "res": [frame.shape[1], frame.shape[0]],
            },
            "auto_detection": {
                "running": bool(state.engine.monitor_task and not state.engine.monitor_task.done()),
                "enabled": state.session.data["mode"]["auto_capture"],
                "state": state.engine.detection_state,
            },
            "models": models,
            "templates_count": len(state.library.papers),
            "details": {
                "ocr_vllm": "PaddleOCR-VL-1.5 (Spotting:) 线上常驻整页无模板OCR与题框提取" if models.get("ocr_vllm") else "未开启独立OCR服务时自动回退多模态VLM",
                "freeform_ocr_enabled": settings.freeform_ocr_enabled,
                "voice": "TTS可由TTS_WARMUP启动预热；失败标红但保留文字服务。远程STT不在后端加载。",
                "tts_warmed": state.voice.tts.warmed,
                "tts_warmup_ms": state.voice.tts.warmup_ms,
                "voice_loaded": {
                    "stt": state.voice.stt.model is not None,
                    "tts": state.voice.tts.model is not None,
                },
            },
        }

    @app.get("/api/templates")
    async def templates():
        return {
            "papers": [
                {
                    "paper_id": p["paper_id"],
                    "title": p["title"],
                    "subject": p["subject"],
                    "page_count": len(p["pages"]),
                    "question_count": sum(len(pg["questions"]) for pg in p["pages"]),
                }
                for p in app.state.library.papers.values()
            ]
        }

    @app.get("/api/templates/{paper_id}")
    async def template(paper_id: str):
        if paper_id not in app.state.library.papers:
            raise HTTPException(404, "模板不存在")
        return app.state.library.papers[paper_id]

    @app.get("/api/session/state")
    async def session_state():
        app.state.session.data["recording"] = app.state.recorder.status()
        return app.state.session.snapshot()

    @app.get("/api/knowledge/overview")
    async def knowledge_overview():
        return get_overview_payload(app.state.session.data)

    @app.get("/api/knowledge/graph/{subject}")
    async def knowledge_graph(subject: str):
        if subject not in ("math", "en"):
            raise HTTPException(404, "不支持的学科图谱")
        return get_subject_graph(subject, app.state.session.data)

    async def recording_call(fn, *args):
        try:
            return await asyncio.to_thread(fn, *args)
        except RecordingError as error:
            raise HTTPException(error.status, str(error)) from error

    def recording_origin(request):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "录制操作只允许来自当前页面")

    @app.get("/api/debug/recordings")
    async def recordings():
        return {"state": app.state.recorder.status(), "clips": await recording_call(app.state.recorder.list)}

    @app.post("/api/debug/recordings/start")
    async def recording_start(body: RecordingRequest, request: Request):
        recording_origin(request)
        result = await recording_call(app.state.recorder.start, body.label)
        app.state.recorder.note("initial_state", mode=dict(app.state.session.data["mode"]),
                                detection=dict(app.state.engine.detection_state))
        await app.state.session.emit({"type": "recording.state", **result})
        return result

    @app.post("/api/debug/recordings/stop")
    async def recording_stop(body: RecordingRequest, request: Request):
        recording_origin(request)
        result = await recording_call(app.state.recorder.stop)
        await app.state.session.emit({"type": "recording.state", **result})
        return result

    @app.post("/api/debug/recordings/mark")
    async def recording_mark(body: RecordingRequest, request: Request):
        recording_origin(request)
        return await recording_call(app.state.recorder.mark)

    @app.get("/api/debug/recordings/{record_id}/{name}")
    async def recording_file(record_id: str, name: str):
        path = await recording_call(app.state.recorder.file, record_id, name)
        return FileResponse(path, filename=f"{record_id}-{name}", headers={"Cache-Control": "no-store"})

    @app.post("/api/camera/snapshot")
    async def snapshot(body: SnapshotRequest):
        return await app.state.engine.capture(grade=body.grade)

    @app.post("/api/debug/inject-image")
    async def inject(file: UploadFile = File(...), grade: bool = Form(True)):
        data = await file.read(settings.max_upload_bytes + 1)
        await file.close()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "图片不能超过20MB")
        try:
            image = await asyncio.to_thread(decode_image, data)
        except (ValueError, cv2.error):
            raise HTTPException(400, "请上传有效的JPG或PNG图片")
        if app.state.camera.backend == "mock":
            await asyncio.to_thread(app.state.camera.inject, image)
        return await app.state.engine.capture(image, grade)

    @app.get("/api/captures/{capture_id}.jpg")
    async def capture_image(capture_id: str):
        cap = app.state.engine.captures.get(capture_id)
        if cap is None:
            raise HTTPException(404, "抓拍已过期或不存在")
        return Response(
            cap["jpeg"],
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    def resize_jpeg(jpeg, width):
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        resized = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
        return cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()

    @app.get("/api/camera/stream")
    async def stream(
        request: Request, fps: int = Query(20, ge=1, le=20), width: int = Query(1280, ge=320, le=1920)
    ):
        if app.state.stream_count >= 2:
            raise HTTPException(429, "最多同时开启两路相机预览")
        app.state.stream_count += 1

        async def frames():
            last = -1
            try:
                while not await request.is_disconnected():
                    seq, jpeg = app.state.camera.preview()
                    if seq != last and jpeg:
                        if width != 1280:
                            key = (seq, width)
                            task = app.state.resized_cache.get(key)
                            if task is None:
                                task = asyncio.create_task(asyncio.to_thread(resize_jpeg, jpeg, width))
                                app.state.resized_cache[key] = task
                            jpeg = await asyncio.shield(task)
                            app.state.resized_cache = {
                                k: v for k, v in app.state.resized_cache.items() if k[0] >= seq - 2
                            }
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                        last = seq
                    await asyncio.sleep(1 / fps)
            finally:
                app.state.stream_count -= 1

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def handle_json(ws, data, voice_input):
        session, engine, voice = app.state.session, app.state.engine, app.state.voice
        typ = data.get("type")
        if typ == "session.set_mode":
            mode = session.data["mode"]
            was_auto = mode["auto_capture"]
            for key in ("auto_capture", "auto_grade"):
                if key in data and isinstance(data[key], bool):
                    mode[key] = data[key]
            if mode["auto_capture"] != was_auto:
                await engine.reset_detector()
            if data.get("voice_mode") in ("vad", "ptt", "off"):
                mode["voice_mode"] = data["voice_mode"]
            await session.emit(session.snapshot())
        elif typ == "camera.capture":
            await engine.capture(grade=bool(data.get("grade", True)))
        elif typ == "session.grade_current":
            await engine.grade_current()
        elif typ == "session.reset":
            for task in tuple(engine.tasks.values()):
                task.cancel()
            await asyncio.gather(*tuple(engine.tasks.values()), return_exceptions=True)
            await voice.interrupt("reset")
            seq = session.data["turn_seq"]
            session.reset()
            session.data["turn_seq"] = seq
            engine.captures.clear()
            await engine.reset_detector()
            session.data["recording"] = app.state.recorder.status()
            await session.emit(session.snapshot())
        elif typ == "question.select":
            uid = data.get("question_uid")
            if uid is not None and uid not in session.data["questions"]:
                await session.error("UNKNOWN_QUESTION", "题目不存在")
                return
            session.data["selected_question_uid"] = uid
            if uid:
                session.data["active_capture_id"] = session.data["questions"][uid]["capture_id"]
            await session.emit(session.snapshot())
        elif typ == "chat.send":
            client_id = data.get("client_msg_id")
            if client_id and any(
                m.get("client_msg_id") == client_id
                for thread in session.data["threads"].values()
                for m in thread
            ):
                return
            await voice.chat(
                data.get("text", ""),
                data.get("scope", "all"),
                data.get("question_uid"),
                bool(data.get("speak", True)),
                client_msg_id=client_id,
            )
        elif typ == "chat.stop":
            if data.get("turn_seq") == session.data["turn_seq"]:
                await voice.interrupt("stop")
        elif typ == "voice.ptt":
            if isinstance(data.get("active"), bool):
                # 打断在接收协程立即执行，不排在音频推理之后。
                if data["active"] and session.data["mode"]["voice_mode"] == "ptt" and voice.is_playing:
                    await voice.interrupt("barge_in")
                    voice_input.ptt_barged = True
                await voice_input.submit(data["active"])
        elif typ == "voice.playback_state":
            if data.get("turn_seq") == session.data["turn_seq"]:
                voice_input.playing = bool(data.get("playing"))
                if not voice_input.playing:
                    from time import perf_counter

                    voice.playback_until = perf_counter() + 0.25
        else:
            await session.error("UNKNOWN_EVENT", "未知控制事件")

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        peer = app.state.session.connect(ws)
        sender = asyncio.create_task(app.state.session.sender(peer))
        voice_input = app.state.voice.new_input()

        async def receive():
            while True:
                packet = await ws.receive()
                if packet["type"] == "websocket.disconnect":
                    break
                if packet.get("bytes") is not None:
                    await voice_input.submit(packet["bytes"])
                elif packet.get("text") is not None:
                    try:
                        data = json.loads(packet["text"])
                        if not isinstance(data, dict):
                            raise ValueError("不是对象")
                        await handle_json(ws, data, voice_input)
                    except (ValueError, TypeError):
                        await app.state.session.error("INVALID_EVENT", "控制消息格式无效")

        receiver = asyncio.create_task(receive())
        try:
            done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, RuntimeError, TimeoutError, asyncio.CancelledError):
            pass
        finally:
            # TestClient/ASGI服务器可能取消整个连接作用域，清理必须屏蔽外层取消。
            import anyio

            with anyio.CancelScope(shield=True):
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                app.state.session.peers.discard(peer)
                await voice_input.close()

    # /api与/ws必须先注册，最后挂载静态根目录；无frontend也能独立启动。
    template_assets = ROOT / "data/templates"
    if template_assets.is_dir():
        app.mount("/data/templates", StaticFiles(directory=template_assets), name="template-assets")
    frontend = ROOT / "frontend"
    if frontend.is_dir():

        class FrontendFiles(StaticFiles):
            """前端走 ETag 协商缓存：每次都校验，代码更新后浏览器不会跑旧模块；未改动时仍是 304。"""

            def file_response(self, *args, **kwargs):
                response = super().file_response(*args, **kwargs)
                response.headers["Cache-Control"] = "no-cache"
                return response

        app.mount("/", FrontendFiles(directory=frontend, html=True), name="frontend")
    else:

        @app.get("/")
        async def empty_frontend():
            return HTMLResponse(
                '<!doctype html><html lang="zh"><meta charset="utf-8"><title>AI学习台灯</title><p>后端已启动，前端目录尚未就绪。</p></html>'
            )

    return app


app = create_app()
