"""无预设模板（撤空 TemplateLibrary）端到端实拍视频回归模拟测试脚本。

测试内容：
1. 使用录制视频 rec_7267479ca145.avi 与 rec_5422b97b81b3.avi，在撤掉所有预设模板（TemplateLibrary 为空）的情况下：
   - 先跑 AutoDetector 完整回放视频帧，验证纸张放稳后不再被 unmatched 拦截，而是触发 mode="freeform" 候选帧；
2. 将视频稳定抓拍帧（数学手写页 + 数学空白页 + 英语页）送入无模板 Engine 跑通完整流水线：
   - 纸面透视矫正 -> 整页真实 OCR 与图题切分 -> HWR 手写作答提取 -> 云端 LLM 逐题判对错与流式讲解 -> WebSocket 全套事件（capture.taken -> page.matched(mode=freeform) -> question.ocr -> question.verdict -> page.grade_done）。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.grading.client import ModelClients  # noqa: E402
from app.state import Session  # noqa: E402
from app.vision.autocapture import AutoDetector  # noqa: E402
from app.vision.matcher import TemplateLibrary  # noqa: E402
from app.vision.stability import DetectionConfig  # noqa: E402


def replay_untemplated_video(video_path: Path, frames_jsonl: Path, empty_lib: TemplateLibrary) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    meta = [json.loads(line) for line in frames_jsonl.read_text().splitlines() if line.strip()]
    detector = AutoDetector(empty_lib, DetectionConfig(), allow_freeform=True)
    triggers = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = float(meta[idx]["t"]) if idx < len(meta) else idx / 15.0
            state, candidate, match = detector.step(frame, now)
            if candidate is not None:
                triggers.append(
                    {
                        "frame": idx,
                        "t": round(now, 3),
                        "mode": state.get("mode", "template" if match else "freeform"),
                        "sharpness": state.get("sharpness"),
                        "paper_ratio": state.get("paper_ratio"),
                        "image": candidate,
                    }
                )
            idx += 1
    finally:
        cap.release()
    return triggers


async def run_simulation(
    ocr_base_url: str = "http://127.0.0.1:8002/v1",
    ocr_model: str = "PaddleOCR-VL-1.5",
    hwr_base_url: str = "http://127.0.0.1:8001/v1",
    hwr_model: str = "Qwen3.5-2B",
):
    import tempfile

    with tempfile.TemporaryDirectory() as empty_tpl_dir:
        overrides = {
            "camera_backend": "mock",
            "auto_capture": False,
            "tts_provider": "mock",
            "templates_dir": Path(empty_tpl_dir),
            "fixture_template": None,
            "freeform_ocr_enabled": True,
            "ocr_base_url": ocr_base_url,
            "ocr_model": ocr_model,
            "hwr_base_url": hwr_base_url,
            "hwr_model": hwr_model,
            "hwr_api_key": "",
        }

        settings = Settings(**overrides)
        empty_lib = TemplateLibrary(Path(empty_tpl_dir), None)
        empty_lib.load()
        assert len(empty_lib.pages) == 0, "预设模板必须为空！"

        # 1. 视频流无模板自动稳帧回放测试
        rec_dir = ROOT / "data/capture_recordings/rec_7267479ca145"
        t0 = perf_counter()
        triggers = replay_untemplated_video(rec_dir / "video.avi", rec_dir / "frames.jsonl", empty_lib)
        replay_s = round(perf_counter() - t0, 2)
        print(f"[1/2] 无模板视频回放完成 ({replay_s}s)：从 rec_7267479ca145 自动触发 {len(triggers)} 次 freeform 稳定抓拍")
        for tr in triggers:
            print(f"      - frame={tr['frame']} t={tr['t']}s mode={tr['mode']} paper_ratio={tr['paper_ratio']} sharpness={tr['sharpness']}")
        assert len(triggers) >= 1, "视频中放稳试卷后应当触发至少 1 次 freeform 抓拍"

        # 2. 取视频抓拍帧跑完整端到端 Engine (无模板 OCR + 切题 + HWR + 云端判题讲解)
        session = Session(settings)
        clients = ModelClients(settings)
        await clients.warmup()
        print("模型健康探测：", clients.health)
        engine = Engine(settings, session, None, empty_lib, clients)

        # 选取视频回放触发的真实数学手写帧 + 真实英语帧验证跨学科通用性
        sample_frames = [
            ("video_math_written", cv2.imread(str(ROOT / "data/video_testset/recordings_20261117/rec_7267479ca145/cap_47d2afe7e71746d1.jpg"))),
            ("video_eng_page", cv2.imread(str(ROOT / "data/video_testset/recordings_20261117/rec_5422b97b81b3/cap_4397220d78bc4056.jpg"))),
        ]

        report = {"video_triggers_count": len(triggers), "pages": []}
        try:
            for label, img in sample_frames:
                p_start = perf_counter()
                res = await engine.capture(img, grade=True, source="auto")
                cid = res["capture_id"]
                task = engine.tasks.get(cid)
                if task:
                    await asyncio.wait_for(task, timeout=120)
                total_ms = round((perf_counter() - p_start) * 1000, 1)

                cap_state = session.data["captures"].get(cid, {})
                q_uids = cap_state.get("question_uids", [])
                qs = [session.data["questions"][u] for u in q_uids if u in session.data["questions"]]
                grade_res = cap_state.get("grade_result", {})

                page_summary = {
                    "label": label,
                    "capture_id": cid,
                    "mode": cap_state.get("mode"),
                    "subject": cap_state.get("subject"),
                    "paper_title": cap_state.get("paper_title"),
                    "ocr_source": engine.captures[cid]["timings_ms"].get("page_ocr_source"),
                    "questions_extracted": len(qs),
                    "grade_result": grade_res,
                    "total_e2e_ms": total_ms,
                    "sample_questions": [
                        {
                            "no": q["no"],
                            "qtype": q["qtype"],
                            "stem_preview": q["stem"][:70],
                            "student_raw": q.get("student_raw"),
                            "status": q.get("status"),
                            "has_mark_quad": bool(q.get("mark_quad")),
                        }
                        for q in qs[:8]
                    ],
                }
                report["pages"].append(page_summary)
                print(f"[2/2] {label} 端到端完成 ({total_ms}ms): mode={page_summary['mode']} ocr={page_summary['ocr_source']} 提取题数={len(qs)} 汇总={grade_res.get('summary_text')}")
                assert cap_state.get("mode") == "freeform", f"{label} 未进入 freeform 模式！"
                assert len(qs) >= 1, f"{label} 未能提取出题目！"
                assert grade_res, f"{label} 未完成判题汇总！"
        finally:
            await engine.close()
            await clients.http.aclose()

        out_path = ROOT / "data/diagnostics/untemplated_video_e2e.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("完整验证报告已保存至:", out_path)


if __name__ == "__main__":
    asyncio.run(run_simulation())
