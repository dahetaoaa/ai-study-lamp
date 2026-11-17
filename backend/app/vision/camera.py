"""相机线程独占读取与预览 JPEG 编码，多路订阅复用同一帧。"""

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.telemetry import log


class Camera:
    def __init__(self, settings):
        self.settings = settings
        self.backend = "mock"
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = np.full((1080, 1920, 3), 65, np.uint8)
        self.jpeg = b""
        self.sequence = 0
        self.frame_sequence = 0
        self.frame_timestamp = 0.0
        self.thread = None
        self.mock_images = []
        self.injected = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def latest(self):
        with self.lock:
            return self.frame.copy()

    def latest_packet(self):
        """视频采样和检测共用真实采集序号，不用预览刷新序号冒充新帧。"""
        with self.lock:
            return self.frame_sequence, self.frame_timestamp, self.frame.copy()

    def preview(self):
        with self.lock:
            return self.sequence, self.jpeg

    def inject(self, frame):
        with self.lock:
            self.injected = frame.copy()
            self.frame = frame.copy()
            self.frame_sequence += 1
            self.frame_timestamp = time.monotonic()

    def _run(self):
        s = self.settings
        cap = None
        if s.camera_backend != "mock" and s.camera_index >= 0:
            self.backend = (
                s.camera_backend
                if s.camera_backend != "auto"
                else ("avfoundation" if sys.platform == "darwin" else "v4l2")
            )
            api = cv2.CAP_AVFOUNDATION if self.backend == "avfoundation" else cv2.CAP_V4L2
            try:
                cap = cv2.VideoCapture(s.camera_index, api)
                if not cap.isOpened():
                    raise RuntimeError("相机不可用")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.camera_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except (cv2.error, RuntimeError):
                log.warning("相机打开失败，自动降级 mock 图片模式")
                if cap:
                    cap.release()
                cap, self.backend = None, "mock"
        files = sorted(Path(s.mock_captures_dir).glob("*.jpg")) + sorted(
            Path(s.mock_captures_dir).glob("*.png")
        )
        self.mock_images = [img for p in files if (img := cv2.imread(str(p))) is not None]
        start, last_encode, failures = time.monotonic(), 0.0, 0
        placeholder = np.full((1080, 1920, 3), 65, np.uint8)
        last_source = None
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        failures += 1
                        if failures > 20:
                            log.warning("相机读取中断，自动降级 mock")
                            cap.release()
                            cap, self.backend = None, "mock"
                        self.stop_event.wait(0.03)
                        continue
                    failures = 0
                    # 在采集源统一转正，保证预览、抓拍和识别坐标使用同一方向。
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                else:
                    with self.lock:
                        injected = self.injected
                    if injected is not None:
                        frame = injected
                    elif self.mock_images:
                        phase = (now - start) % 9
                        image = self.mock_images[int((now - start) // 9) % len(self.mock_images)]
                        if phase < 1 or phase > 8:
                            # 运动阶段持续产生大帧差，便于 LOCKED 重武装。
                            shade = 20 + int(now * 750) % 180
                            frame = np.full((1080, 1920, 3), shade, np.uint8)
                        else:
                            frame = image
                    else:
                        frame = placeholder
                with self.lock:
                    self.frame = frame
                    self.frame_sequence += 1
                    self.frame_timestamp = now
                if now - last_encode >= 0.05:
                    # mock稳定帧内容不变，直接复用JPEG，避免空闲时重复耗用CPU。
                    if frame is not last_source:
                        h, w = frame.shape[:2]
                        preview = cv2.resize(frame, (1280, max(1, round(h * 1280 / w))))
                        ok, data = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            with self.lock:
                                self.jpeg = data.tobytes()
                            last_source = frame
                    with self.lock:
                        self.sequence += 1
                    last_encode = now
                if cap is None:
                    self.stop_event.wait(1 / 30)
        finally:
            if cap is not None:
                cap.release()
