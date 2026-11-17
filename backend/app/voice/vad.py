"""Silero CPU ONNX懒加载，20ms输入累积到模型所需的32ms窗口。"""

import importlib.util

import numpy as np


class SileroVAD:
    def __init__(self):
        self.model = None
        self.pending = np.empty(0, np.float32)
        self.failed = False

    @property
    def available(self):
        return not self.failed and all(
            importlib.util.find_spec(name) for name in ("silero_vad", "onnxruntime", "torch")
        )

    def probabilities(self, pcm):
        try:
            if self.model is None:
                from silero_vad import load_silero_vad

                self.model = load_silero_vad(onnx=True)
            import torch

            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            self.pending = np.concatenate((self.pending, samples))
            result = []
            while len(self.pending) >= 512:
                chunk, self.pending = self.pending[:512], self.pending[512:]
                result.append(float(self.model(torch.from_numpy(chunk.copy()), 16000).item()))
            return result
        except Exception:
            self.failed = True
            raise

    def reset(self):
        self.pending = np.empty(0, np.float32)
        if self.model is not None:
            self.model.reset_states()
