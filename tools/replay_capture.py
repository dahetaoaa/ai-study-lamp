"""Offline placement replay. No camera access and no model/API requests.

Example: backend/.venv/bin/python tools/replay_capture.py --recording rec_<id>
Annotations: {"events":[{"ready":1.2,"until":3.0,"action":"capture"}],
              "invalid":[{"start":0,"end":1.0}]}
"""
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import Settings
from app.vision.autocapture import (
    AutoDetector,
    answer_fingerprint,
    page_key,
    same_answers,
)
from app.vision.matcher import TemplateLibrary
from app.vision.recorder import ID
from app.vision.stability import DetectionConfig


def evaluate(triggers, labels):
    expected = labels.get("events", [])
    used, latencies, misses = set(), [], []
    for event in expected:
        found = next((i for i, hit in enumerate(triggers) if i not in used
                      and event["ready"] - 0.2 <= hit["t"] <= event["until"]
                      and (not event.get("action") or hit["action"] == event["action"])), None)
        if found is None:
            misses.append(event)
        else:
            used.add(found)
            latencies.append(round((triggers[found]["t"] - event["ready"]) * 1000, 2))
    false = [hit for hit in triggers if any(region["start"] <= hit["t"] <= region["end"]
                                           for region in labels.get("invalid", []))]
    return {"expected": len(expected), "matched": len(used), "misses": misses,
            "unassigned_triggers": [hit for i, hit in enumerate(triggers) if i not in used],
            "invalid_interval_triggers": false, "stable_to_trigger_ms": latencies}


def replay(video, library, output, config=None, frames_file=None, annotations=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录必须为空，避免覆盖已有结果")
    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    if frames_file and Path(frames_file).exists():
        metadata = [json.loads(line) for line in Path(frames_file).read_text().splitlines() if line.strip()]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("视频无法读取")
    detector = AutoDetector(library, config)
    history, triggers, costs = [], [], []
    index, previous_time = 0, -1.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 and not metadata:
        cap.release()
        raise ValueError("视频没有有效时间信息，请提供frames.jsonl")
    try:
        with (output / "decisions.jsonl").open("w") as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if metadata:
                    if index >= len(metadata) or metadata[index]["index"] != index:
                        raise ValueError("视频帧与时间戳不匹配")
                    now = float(metadata[index]["t"])
                else:
                    now = index / fps
                if now < previous_time:
                    raise ValueError("视频时间戳倒退")
                previous_time = now
                start = perf_counter()
                state, candidate, match = detector.step(frame, now)
                if candidate is not None:
                    key = page_key(match)
                    fingerprint = answer_fingerprint(candidate, match)
                    same = next((i for i in range(len(history) - 1, -1, -1)
                                 if history[i][0] == key and same_answers(fingerprint, history[i][1])), None)
                    action = "reuse" if same is not None else "capture"
                    hit = {"t": round(now, 6), "frame": index, "action": action,
                           "paper_id": key[0], "page_no": key[1], "match_ms": state.get("match_ms")}
                    if same is None:
                        name = f"capture_{index:06d}.jpg"
                        if not cv2.imwrite(str(output / name), candidate, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                            raise OSError("无法保存候选图")
                        hit["image"] = name
                        history.append((key, fingerprint))
                        history = history[-20:]
                    else:
                        history.append(history.pop(same))
                    triggers.append(hit)
                cost = (perf_counter() - start) * 1000
                costs.append(cost)
                log.write(json.dumps({"t": now, "frame": index, **state, "compute_ms": round(cost, 2)}, ensure_ascii=False) + "\n")
                index += 1
    finally:
        cap.release()
    if metadata and len(metadata) != index:
        raise ValueError("实际解码帧数与录制元数据不一致，样本可能被截断")
    if not index:
        raise ValueError("视频没有可解码帧")
    report = {"video": str(video), "frames": index, "timestamp_source": "recorded" if metadata else "nominal_fps",
              "parameters": asdict(detector.stability.config), "triggers": triggers,
              "new_captures": sum(t["action"] == "capture" for t in triggers),
              "reused": sum(t["action"] == "reuse" for t in triggers),
              "compute_ms": {"p50": round(float(np.percentile(costs, 50)), 2),
                             "p95": round(float(np.percentile(costs, 95)), 2), "max": round(max(costs), 2)},
              "evaluation": evaluate(triggers, annotations) if annotations else None}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--recording")
    source.add_argument("--video", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--stable-seconds", type=float)
    p.add_argument("--change-ratio", type=float)
    args = p.parse_args()
    settings = Settings()
    config = DetectionConfig.from_settings(settings)
    if args.recording:
        if not ID.fullmatch(args.recording):
            p.error("录制ID格式无效")
        folder = settings.recordings_dir / args.recording
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("active"):
            p.error("请先停止录制")
        config = DetectionConfig(**manifest["parameters"])
        video = folder / "video.avi"
    else:
        video = args.video
        folder = video.parent
    if args.stable_seconds is not None:
        if not 0.2 <= args.stable_seconds <= 3:
            p.error("stable-seconds范围0.2至3")
        config = replace(config, stable_seconds=args.stable_seconds)
    if args.change_ratio is not None:
        if not 0.005 <= args.change_ratio <= 0.5:
            p.error("change-ratio范围0.005至0.5")
        config = replace(config, change_ratio=args.change_ratio)
    library = TemplateLibrary(settings.templates_dir, settings.fixture_template)
    library.load()
    output = args.output or ROOT / "data/capture_replays" / ("replay_" + uuid4().hex[:10])
    labels = json.loads(args.annotations.read_text()) if args.annotations else None
    report = replay(video, library, output, config, folder / "frames.jsonl", labels)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("保存至：", output)


if __name__ == "__main__":
    main()
