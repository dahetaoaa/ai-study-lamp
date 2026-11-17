"""libsonic 的流式保音调变速：改变语速，不提高音高，不等待整句音频。"""
import ctypes
import ctypes.util
from functools import lru_cache


@lru_cache(maxsize=1)
def library():
    name = ctypes.util.find_library("sonic")
    if not name:
        raise RuntimeError("TTS_SPEED需要libsonic；请安装libsonic0或将TTS_SPEED设为1")
    lib = ctypes.CDLL(name)
    signatures = {
        "sonicCreateStream": ([ctypes.c_int, ctypes.c_int], ctypes.c_void_p),
        "sonicDestroyStream": ([ctypes.c_void_p], None),
        "sonicSetSpeed": ([ctypes.c_void_p, ctypes.c_float], None),
        "sonicSamplesAvailable": ([ctypes.c_void_p], ctypes.c_int),
        "sonicFlushStream": ([ctypes.c_void_p], ctypes.c_int),
        "sonicWriteShortToStream": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.c_int], ctypes.c_int),
        "sonicReadShortFromStream": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.c_int], ctypes.c_int),
    }
    for fn, (args, result) in signatures.items():
        getattr(lib, fn).argtypes = args
        getattr(lib, fn).restype = result
    return lib


class Tempo:
    def __init__(self, speed, sample_rate=24000):
        self.lib = library()
        self.handle = self.lib.sonicCreateStream(sample_rate, 1)
        if not self.handle:
            raise MemoryError("无法创建语速处理器")
        self.lib.sonicSetSpeed(self.handle, float(speed))

    def feed(self, pcm):
        if len(pcm) % 2:
            raise ValueError("PCM16数据未对齐")
        size = len(pcm) // 2
        buf = (ctypes.c_short * size).from_buffer_copy(pcm)
        if not self.lib.sonicWriteShortToStream(self.handle, buf, size):
            raise RuntimeError("语速处理失败")
        return self.read()

    def read(self):
        size = self.lib.sonicSamplesAvailable(self.handle)
        if not size:
            return b""
        buf = (ctypes.c_short * size)()
        count = self.lib.sonicReadShortFromStream(self.handle, buf, size)
        if count < 0:
            raise RuntimeError("读取变速音频失败")
        return ctypes.string_at(buf, count * 2)

    def finish(self):
        if not self.lib.sonicFlushStream(self.handle):
            raise RuntimeError("刷新变速音频失败")
        return self.read()

    def close(self):
        if self.handle:
            self.lib.sonicDestroyStream(self.handle)
            self.handle = None
