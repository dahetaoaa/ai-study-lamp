from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.vision.camera import Camera


@pytest.mark.parametrize("source", ["camera", "mock", "injected"])
def test_camera_orientation_and_preview(monkeypatch, tmp_path, source):
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    camera = Camera(SimpleNamespace(
        camera_backend="v4l2" if source == "camera" else "mock",
        camera_index=0,
        camera_width=6,
        camera_height=4,
        mock_captures_dir=tmp_path,
    ))

    class Capture:
        released = False

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            return True, frame.copy()

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: capture)
    if source == "injected":
        camera.inject(frame)
    elif source == "mock":
        assert cv2.imwrite(str(tmp_path / "frame.png"), frame)
        # 固定在 mock 稳定阶段，避免循环生成的运动占位帧。
        times = iter([100.0, 102.0])
        monkeypatch.setattr("app.vision.camera.time.monotonic", lambda: next(times))

    previews = []

    def encode(_, image, __):
        previews.append(image.copy())
        camera.stop_event.set()  # 同步运行一帧，不启动线程或访问真实设备。
        return True, np.array([1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", encode)
    camera._run()

    # 真实相机顺时针旋转；已有方向的 mock/注入图片保持原样。
    expected = np.rot90(frame, k=-1) if source == "camera" else frame
    np.testing.assert_array_equal(camera.latest(), expected)
    h, w = expected.shape[:2]
    np.testing.assert_array_equal(
        previews[0], cv2.resize(expected, (1280, round(h * 1280 / w)))
    )
    assert camera.preview() == (1, b"\x01\x02\x03")
    assert capture.released is (source == "camera")
