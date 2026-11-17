"""临时本地录像：独立线程采样现有Camera，不额外打开设备，不录音。"""
import json
import queue
import re
import shutil
import threading
import time
from pathlib import Path
from uuid import uuid4

import cv2

from app.telemetry import log

ID = re.compile(r"rec_[0-9a-f]{12}\Z")
FILES = {"video.avi", "frames.jsonl", "events.jsonl", "manifest.json"}


class RecordingError(ValueError):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


class Recorder:
    def __init__(self, settings, camera, parameters):
        self.settings, self.camera, self.parameters = settings, camera, parameters
        self.root = Path(settings.recordings_dir)
        self.lock = threading.RLock()
        self.thread = None
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.events = queue.Queue(maxsize=4096)
        self.meta = {"active": False, "status": "idle"}
        self.started = 0.0
        self.stop_reason = "user"

    def status(self):
        with self.lock:
            result = {**self.meta, "enabled": self.settings.recording_enabled}
            if result["active"]:
                result["duration_ms"] = round((time.monotonic() - self.started) * 1000)
            return result

    def list(self):
        if not self.root.exists():
            return []
        result = []
        for path in sorted(self.root.glob("rec_*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not ID.fullmatch(path.parent.name) or path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                meta = json.loads(path.read_text())
                if meta.get("id") != path.parent.name:
                    continue
                if meta.get("active") and meta["id"] != self.meta.get("id"):
                    meta = {**meta, "active": False, "status": "interrupted", "reason": "unclean_shutdown"}
                result.append(meta)
            except (OSError, ValueError):
                continue
            if len(result) >= 40:
                break
        return result

    def file(self, record_id, name):
        if not ID.fullmatch(record_id) or name not in FILES:
            raise RecordingError("录制文件不存在", 404)
        parent = self.root / record_id
        path = parent / name
        if parent.is_symlink() or path.is_symlink() or not (parent / "manifest.json").is_file() or not path.is_file():
            raise RecordingError("录制文件不存在", 404)
        if self.status().get("active") and self.meta.get("id") == record_id:
            raise RecordingError("请停止录制后再下载")
        return path

    def start(self, label=""):
        with self.lock:
            if not self.settings.recording_enabled:
                raise RecordingError("临时录制功能未启用", 403)
            if self.camera.backend == "mock" and self.settings.camera_backend != "mock":
                raise RecordingError("真实相机不可用，不能将演示图片录作真实样本", 503)
            if self.thread and self.thread.is_alive():
                raise RecordingError("已有录制任务，请先停止")
            self.root.mkdir(parents=True, exist_ok=True)
            used = sum(p.stat().st_size for p in self.root.glob("rec_*/*") if p.is_file() and not p.is_symlink())
            remaining = self.settings.recording_total_bytes - used
            if remaining < min(10 * 1024 * 1024, self.settings.recording_max_bytes):
                raise RecordingError("录制总空间已达上限，请先整理样本", 507)
            if shutil.disk_usage(self.root).free < self.settings.recording_min_free_bytes:
                raise RecordingError("磁盘剩余空间不足", 507)
            self.limit = min(remaining, self.settings.recording_max_bytes)
            self.started = time.monotonic()
            record_id = "rec_" + uuid4().hex[:12]
            self.folder = self.root / record_id
            self.folder.mkdir()
            self.meta = {
                "id": record_id, "active": True, "status": "starting", "label": label[:100],
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "fps": self.settings.recording_fps,
                "frames": 0, "duplicate_polls": 0, "dropped_samples": 0, "dropped_events": 0,
                "duration_ms": 0, "parameters": self.parameters, "audio": False,
                "source_backend": self.camera.backend, "max_seconds": self.settings.recording_max_seconds,
                "max_bytes": self.limit,
            }
            self.stop_reason = "user"
            self.stop_event.clear()
            self.ready.clear()
            self.events = queue.Queue(maxsize=4096)
            self._save_manifest()
            self.thread = threading.Thread(target=self._run, name="capture-recorder", daemon=True)
            self.thread.start()
        if not self.ready.wait(5):
            self.stop("startup_timeout")
            raise RecordingError("录像编码器启动超时", 503)
        status = self.status()
        if status["status"] == "failed":
            raise RecordingError("录制启动失败：" + status.get("reason", ""), 503)
        return status

    def _save_manifest(self):
        tmp = self.folder / "manifest.tmp"
        tmp.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2))
        tmp.replace(self.folder / "manifest.json")

    def note(self, kind, **data):
        with self.lock:
            if not self.meta["active"]:
                return
            event = {"type": kind, "t": round(time.monotonic() - self.started, 6), **data}
            try:
                self.events.put_nowait(event)
            except queue.Full:
                self.meta["dropped_events"] += 1

    def mark(self):
        if not self.status()["active"]:
            raise RecordingError("当前没有录制")
        self.note("expected_ready")
        return self.status()

    def stop(self, reason="user"):
        with self.lock:
            thread = self.thread
            if not thread or not thread.is_alive():
                return self.status()
            self.stop_reason = reason
            self.stop_event.set()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RecordingError("正在结束录制，请稍后查询状态", 503)
        return self.status()

    def _run(self):
        writer = None
        reason, failed = "user", False
        last_seq, last_fresh = -1, time.monotonic()
        next_sample = self.started
        interval = 1 / self.settings.recording_fps
        shape = None
        try:
            with (self.folder / "frames.jsonl").open("w") as frame_log, (self.folder / "events.jsonl").open("w") as event_log:
                while not self.stop_event.is_set():
                    now = time.monotonic()
                    if now - self.started >= self.settings.recording_max_seconds:
                        reason = "duration_limit"
                        break
                    if self.camera.backend != self.meta["source_backend"]:
                        reason, failed = "camera_source_changed", True
                        break
                    seq, frame_ts, frame = self.camera.latest_packet()
                    if seq == last_seq or frame_ts <= 0:
                        with self.lock:
                            self.meta["duplicate_polls"] += 1
                        if now - last_fresh > 3:
                            reason, failed = "camera_stalled", True
                            break
                    else:
                        last_seq, last_fresh = seq, now
                        if shape is None:
                            shape = frame.shape[:2]
                            height, width = shape
                            writer = cv2.VideoWriter(str(self.folder / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                                     self.settings.recording_fps, (width, height))
                            if not writer.isOpened():
                                raise RuntimeError("codec_unavailable")
                            with self.lock:
                                self.meta.update(status="recording", width=width, height=height)
                                self._save_manifest()
                            self.ready.set()
                        if frame.shape[:2] != shape:
                            reason, failed = "resolution_changed", True
                            break
                        writer.write(frame)
                        frame_log.write(json.dumps({"index": self.meta["frames"], "t": max(0, frame_ts - self.started),
                                                    "camera_sequence": seq, "sample_t": now - self.started}) + "\n")
                        with self.lock:
                            self.meta["frames"] += 1
                        size = (self.folder / "video.avi").stat().st_size
                        # VideoWriter有内部IO缓冲；预留一帧原始大小和512KiB，不能只看尚未刷盘的stat。
                        if size + frame.nbytes + 512 * 1024 >= self.limit:
                            reason = "size_limit"
                            break
                        if shutil.disk_usage(self.root).free < self.settings.recording_min_free_bytes:
                            reason = "disk_limit"
                            break
                    self._flush_events(event_log)
                    frame_log.flush()
                    next_sample += interval
                    now = time.monotonic()
                    if now > next_sample + interval:
                        skipped = int((now - next_sample) / interval)
                        with self.lock:
                            self.meta["dropped_samples"] += skipped
                        next_sample += skipped * interval
                    self.stop_event.wait(max(0, next_sample - now))
                if self.stop_event.is_set():
                    reason = self.stop_reason
                self._flush_events(event_log)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            reason, failed = type(error).__name__ + ": " + str(error)[:120], True
            log.warning("录像失败：%s", reason)
        finally:
            if writer:
                writer.release()
            with self.lock:
                self.meta.update(active=False, status="failed" if failed else "stopped", reason=reason,
                                 duration_ms=round((time.monotonic() - self.started) * 1000))
                self.meta["download_base"] = f"/api/debug/recordings/{self.meta['id']}/"
                try:
                    self._save_manifest()
                except OSError:
                    self.meta.update(status="failed", reason="manifest_write_failed")
                self.ready.set()

    def _flush_events(self, target):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            target.write(json.dumps(event, ensure_ascii=False) + "\n")
        target.flush()
