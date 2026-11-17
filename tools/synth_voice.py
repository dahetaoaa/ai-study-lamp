"""仅在4090生成语音验收素材；不启动第二个常驻服务。

用法（先关闭后端TTS或尚未加载时，避免重复占显存）：
  backend/.venv/bin/python tools/synth_voice.py --output /tmp/lamp-voice
"""

import argparse
import asyncio
import sys
import wave
from pathlib import Path


async def main(output):
    if sys.platform != "linux":
        raise SystemExit("仅允许在Linux GPU目标机运行，不在Mac加载模型")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("需要CUDA")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "backend"))
    from app.config import Settings
    from app.voice.tts import TTS

    settings = Settings(
        _env_file=None,
        tts_provider="qwen3_tts",
        tts_model=str(root / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    )
    tts = TTS(settings, None)
    output.mkdir(parents=True, exist_ok=True)
    for name, text in {
        "question": "什么是分母？",
        "barge": "等一下，请用更简单的例子解释。",
        "ack": "嗯嗯。",
        "slow_part1": "为什么方程两边",
        "slow_part2": "不能同时除以零？",
    }.items():
        if (output / f"{name}.wav").exists():
            continue  # 保留已验收素材，便于比较阈值修改前后
        chunks = [pcm async for pcm in tts.stream(text)]
        with wave.open(str(output / f"{name}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(24000)
            f.writeframes(b"".join(chunks))
        print(name, text, "seconds", sum(map(len, chunks)) / 48000, flush=True)
    import numpy as np

    parts = []
    for name in ("slow_part1", "slow_part2"):
        with wave.open(str(output / f"{name}.wav"), "rb") as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        loud = np.flatnonzero(np.abs(data.astype(np.int32)) > 250)
        data = data[max(0, loud[0] - 960) : min(len(data), loud[-1] + 960)]
        parts.append(data.tobytes())
    # 两端各40ms，加820ms静音，制造约900ms的自然思考停顿。
    pcm = parts[0] + b"\0" * 39360 + parts[1]
    with wave.open(str(output / "slow.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(pcm)
    print("slow", "为什么方程两边……不能同时除以零？", "seconds", len(pcm) / 48000, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args().output))
