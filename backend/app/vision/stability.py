"""自动稳帧、累计场景变化与去重；线上和录制回放共享此状态机。"""
from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    version: str = "placement-v2"
    stable_seconds: float = 0.55
    motion_threshold: float = 2.5
    change_ratio: float = 0.045
    change_delta: float = 18.0
    min_sharpness: float = 25.0
    rearm_seconds: float = 0.18
    retry_seconds: float = 2.0

    @classmethod
    def from_settings(cls, settings):
        return cls(stable_seconds=settings.auto_stable_seconds,
                   change_ratio=settings.auto_change_ratio,
                   min_sharpness=settings.auto_min_sharpness)


def small_gray(frame):
    if frame is None or not frame.size:
        raise ValueError("没有有效相机帧")
    h, w = frame.shape[:2]
    scale = min(1, 320 / max(h, w))
    return cv2.cvtColor(cv2.resize(frame, (max(8, round(w * scale)), max(8, round(h * scale)))), cv2.COLOR_BGR2GRAY)


def difference(a, b):
    if b is None or a.shape != b.shape:
        return None
    # 降低传感器噪声/一像素边缘变化，抵消整体曝光偏移，不抹掉局部纸面变化。
    diff = cv2.GaussianBlur(a, (3, 3), 0).astype(np.float32) - cv2.GaussianBlur(b, (3, 3), 0)
    return np.abs(diff - np.median(diff))


class Stability:
    def __init__(self, config=None):
        self.config = config or DetectionConfig()
        self.reset()

    def reset(self):
        self.state = "idle"
        self.previous = None
        self.reference = None
        self.settling_at = None
        self.settle_reference = None
        self.moving_at = None
        self.absent_at = None
        self.force_cycle = False
        self.retry_at = 0.0
        self.blocked_reason = ""
        self.frames = deque(maxlen=3)

    def lock(self, frame):
        self.reference = small_gray(frame)
        self.state = "locked"
        self.settling_at = self.moving_at = self.absent_at = None
        self.force_cycle = False
        self.frames.clear()

    def reject(self, reason, now):
        self.state = "rejected"
        self.blocked_reason = reason
        self.retry_at = now + self.config.retry_seconds
        self.settling_at = None
        self.frames.clear()

    def update(self, frame, now):
        c = self.config
        gray = small_gray(frame)
        diff = difference(gray, self.previous)
        motion = float(diff.mean()) if diff is not None else 100.0
        raw_motion = float(cv2.absdiff(gray, self.previous).mean()) if self.previous is not None and gray.shape == self.previous.shape else 100.0
        self.previous = gray
        change = difference(gray, self.reference)
        changed = float((change > c.change_delta).mean()) if change is not None else 1.0
        h, w = gray.shape
        center = gray[h // 8: h * 7 // 8, w // 8: w * 7 // 8]
        sharpness = float(cv2.Laplacian(center, cv2.CV_64F).var())
        _, _, stats, _ = cv2.connectedComponentsWithStats((gray > 145).astype(np.uint8))
        paper_ratio = float(max(stats[1:, cv2.CC_STAT_AREA], default=0) / gray.size)
        paper = paper_ratio > 0.18
        textured = sharpness >= c.min_sharpness
        captured, progress, reason = None, 0.0, "waiting_paper"

        if self.state == "rejected":
            if changed >= c.change_ratio or now >= self.retry_at:
                self.state = "idle"
                self.force_cycle = True  # 前次候选未成功抓拍，允许同帧重新匹配
            else:
                reason = self.blocked_reason

        if self.state in ("locked", "captured"):
            self.state = "locked"
            reason = "already_captured"
            self.absent_at = (now if self.absent_at is None else self.absent_at) if not paper else None
            departed = self.absent_at is not None and now - self.absent_at >= c.rearm_seconds
            # reference=None 兼容启动前手工设置的锁定状态；正常使用累计变化判据。
            altered = changed >= c.change_ratio if self.reference is not None else raw_motion > 12
            self.moving_at = (now if self.moving_at is None else self.moving_at) if altered else None
            moved = self.moving_at is not None and now - self.moving_at >= c.rearm_seconds
            if departed or moved:
                self.force_cycle = departed
                self.state = "idle"
                self.settling_at = None
                self.frames.clear()
                reason = "scene_changed"

        if self.state not in ("locked", "rejected"):
            if not paper:
                reason = "waiting_paper"
                self.settling_at = None
                self.frames.clear()
                self.state = "idle"
                # 短暂离开后再出现相同试卷，也算新一次放置。
                if self.reference is not None:
                    self.absent_at = now if self.absent_at is None else self.absent_at
                    if now - self.absent_at >= c.rearm_seconds:
                        self.force_cycle = True
            elif motion >= c.motion_threshold:
                reason = "moving"
                self.settling_at = None
                self.frames.clear()
                self.state = "idle"
            elif not textured:
                reason = "blurry"
                self.settling_at = None
                self.frames.clear()
                self.state = "idle"
            else:
                if self.settling_at is None:
                    self.settling_at = now
                    self.settle_reference = gray
                drift = difference(gray, self.settle_reference)
                drifting = drift is not None and float((drift > 8).mean()) >= c.change_ratio
                if drifting:
                    self.settling_at, self.settle_reference = now, gray
                    self.frames.clear()
                self.state = "settling"
                reason = "moving" if drifting else "settling"
                self.frames.append((sharpness, frame))
                progress = min(1.0, (now - self.settling_at) / c.stable_seconds)
                if progress >= 1:
                    if self.reference is not None and changed < c.change_ratio and not self.force_cycle:
                        self.state, reason = "locked", "already_captured"
                    else:
                        captured = max(self.frames, key=lambda item: item[0])[1].copy()
                        self.lock(captured)
                        self.state, reason = "captured", "candidate"
        return {
            "stability": self.state, "reason": reason, "progress": round(progress, 3),
            "motion_score": round(motion, 2), "scene_change_ratio": round(changed, 4),
            "sharpness": round(sharpness, 2), "paper_ratio": round(paper_ratio, 3),
            "detector_version": c.version,
        }, captured

    def parameters(self):
        return asdict(self.config)
