"""Task 3 端到端实测验证脚本：freeform_ink_gate + 8001 Qwen3.5-2B HWRClient(freeform=True) + 融合。

验证 9 张实拍图（7 张空白页 75 题 + 2 张手写数学页 14 题）：
1. 真值答题框（基于 baseline rectify_paper + spotting_raw_cache.json）
2. 真值答题框（基于 paper_rectify.rectify_paper v2 + 8002 PaddleOCR-VL-1.5 实测或投影）
3. 当前 page_ocr.py 预测答题框（审计空白页门控与 HWR 表现）

输出完整指标并保存至 data/diagnostics/verify_freeform_ink_gate_hwr.json。
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.grading.client import ModelClients  # noqa: E402
from app.grading.hwr_client import HWRClient, merge_freeform_choice_result  # noqa: E402
from app.vision.freeform_ink_gate import prepare_freeform_answer_blocks  # noqa: E402
from app.vision.matcher import TemplateLibrary, project, quad  # noqa: E402
from app.vision.page_ocr import (  # noqa: E402
    _parse_paddle_vl_blocks,
    build_freeform_page_from_lines,
    rectify_paper as rectify_base,
)
from app.vision.paper_rectify import rectify_paper as rectify_v2, resize_for_ocr  # noqa: E402

CACHE_PATH = ROOT / "data/diagnostics/spotting_raw_cache.json"
OUT_PATH = ROOT / "data/diagnostics/verify_freeform_ink_gate_hwr.json"
WRITTEN_MATH_P1 = {"1": "B", "2": "C", "3": "A", "4": "A", "5": "B", "6": "B", "7": "B"}


async def _eval_question(
    hwr: HWRClient,
    warped: np.ndarray,
    gray: np.ndarray,
    areas: list[dict],
    qtype: str,
    ocr_blocks: list[dict],
    pre_hw: str = "",
) -> dict:
    t_gate0 = perf_counter()
    gate = prepare_freeform_answer_blocks(
        warped,
        gray,
        areas,
        qtype,
        ocr_blocks,
        pre_ocr_handwriting=pre_hw,
    )
    gate_ms = (perf_counter() - t_gate0) * 1000.0

    if gate["is_blank"]:
        return {
            "status": "blank",
            "pred_ans": "",
            "low_conf": False,
            "gate_skipped": True,
            "clean_ink_pixels": gate["clean_ink_pixels"],
            "gate_ms": round(gate_ms, 3),
            "hwr_ms": 0.0,
        }

    t_hwr0 = perf_counter()
    raw, low = await hwr.recognize(gate["blocks"], qtype, freeform=True)
    if qtype == "choice":
        raw, low = merge_freeform_choice_result(
            raw,
            low,
            pre_hw,
            ink_pixels=gate["clean_ink_pixels"],
        )
    hwr_ms = (perf_counter() - t_hwr0) * 1000.0

    if not raw.strip():
        status = "uncertain" if low else "blank"
    else:
        status = "uncertain" if low else "recognized"
    return {
        "status": status,
        "pred_ans": raw,
        "low_conf": low,
        "gate_skipped": False,
        "clean_ink_pixels": gate["clean_ink_pixels"],
        "gate_ms": round(gate_ms, 3),
        "hwr_ms": round(hwr_ms, 1),
    }


def _infer_pre_hw_near_areas(areas: list[dict], ocr_blocks: list[dict], page_w: int) -> str:
    import re

    for a in areas:
        ax, ay, aw, ah = a["bbox"]
        for b in ocr_blocks:
            t = str(b.get("text") or "").strip().strip("$").strip()
            bx0, by0, bx1, by1 = b["box"]
            if re.fullmatch(r"[A-Ha-h]", t) and (bx1 - bx0) < page_w * 0.08:
                cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                if (ax - 10) <= cx <= (ax + aw + 10) and (ay - 10) <= cy <= (ay + ah + 10):
                    return t.upper()
    return ""


async def run_gt_eval(
    name: str,
    cache_items: list[dict],
    gt_lib: TemplateLibrary,
    clients: ModelClients,
    hwr: HWRClient,
    *,
    use_rectify_v2: bool,
) -> dict:
    blank_pages = []
    written_pages = []
    all_gate_ms = []

    for item in cache_items:
        img = cv2.imread(item["img_path"])
        gt_m = gt_lib.match(img)
        if use_rectify_v2:
            warped, h_free, _, pw, ph = rectify_v2(img)
            ocr_img, _ = resize_for_ocr(warped)
            _, enc = cv2.imencode(".jpg", ocr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()
            resp = await clients.http.post(
                clients.settings.ocr_base_url.rstrip("/") + "/chat/completions",
                json={
                    "model": clients.settings.ocr_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": b64}},
                                {"type": "text", "text": "Spotting:"},
                            ],
                        }
                    ],
                    "max_tokens": 2800,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=45.0,
            )
            resp.raise_for_status()
            ocr_blocks = _parse_paddle_vl_blocks(resp.json()["choices"][0]["message"]["content"], pw, ph)
        else:
            warped = cv2.imread(item["warped_path"])
            _, h_free, _, pw, ph = rectify_base(img)
            ocr_blocks = item["blocks"]

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        h_tpl_to_warped = np.linalg.inv(h_free) @ gt_m.homography
        is_written = item["paper_id"] == "math_xmyz_2025_oct" and item["page_no"] == 1

        q_tasks = []
        q_meta = []
        t_page0 = perf_counter()
        for q in gt_m.template.page["questions"]:
            areas = []
            for a in q.get("answer_areas", []):
                pts = np.float32(project(quad(a["bbox"]), h_tpl_to_warped))
                x0, y0 = int(round(pts[:, 0].min())), int(round(pts[:, 1].min()))
                x1, y1 = int(round(pts[:, 0].max())), int(round(pts[:, 1].max()))
                areas.append(
                    {"id": a.get("id", "a0"), "bbox": [x0, y0, x1 - x0, y1 - y0], "hint": a.get("hint", "")}
                )
            pre_hw = _infer_pre_hw_near_areas(areas, ocr_blocks, pw) if is_written else ""
            q_meta.append((str(q["number"]), q["type"], pre_hw))
            q_tasks.append(_eval_question(hwr, warped, gray, areas, q["type"], ocr_blocks, pre_hw))

        results = await asyncio.gather(*q_tasks)
        page_wall_ms = round((perf_counter() - t_page0) * 1000, 2)
        for r in results:
            all_gate_ms.append(r["gate_ms"])

        if not is_written:
            blank_ok = sum(1 for r in results if r["status"] == "blank")
            uncertain = sum(1 for r in results if r["status"] == "uncertain")
            fp = sum(1 for r in results if r["status"] == "recognized")
            skipped = sum(1 for r in results if r["gate_skipped"])
            blank_pages.append(
                {
                    "case": item["case"],
                    "paper_id": item["paper_id"],
                    "page_no": item["page_no"],
                    "q_count": len(results),
                    "blank_ok": blank_ok,
                    "uncertain": uncertain,
                    "false_positive": fp,
                    "gate_skipped": skipped,
                    "gate_plus_hwr_wall_ms": page_wall_ms,
                    "hwr_only_wall_ms": round(sum(r["hwr_ms"] for r in results), 2),
                }
            )
        else:
            details = []
            correct = 0
            for (qno, _, pre_hw), r in zip(q_meta, results):
                gt_ans = WRITTEN_MATH_P1[qno]
                ok = r["status"] == "recognized" and r["pred_ans"] == gt_ans
                correct += int(ok)
                details.append({"qno": qno, "gt": gt_ans, "pre_hw": pre_hw, **r, "exact_match": ok})
            written_pages.append(
                {
                    "case": item["case"],
                    "paper_id": item["paper_id"],
                    "page_no": item["page_no"],
                    "correct": correct,
                    "total": len(details),
                    "gate_plus_hwr_wall_ms": page_wall_ms,
                    "questions": details,
                }
            )

    summary = {
        "mode": name,
        "blank_questions": sum(p["q_count"] for p in blank_pages),
        "blank_exact": sum(p["blank_ok"] for p in blank_pages),
        "blank_uncertain": sum(p["uncertain"] for p in blank_pages),
        "blank_false_positive": sum(p["false_positive"] for p in blank_pages),
        "blank_gate_skipped": sum(p["gate_skipped"] for p in blank_pages),
        "blank_avg_gate_plus_hwr_ms": round(float(np.mean([p["gate_plus_hwr_wall_ms"] for p in blank_pages])), 2),
        "written_questions": sum(p["total"] for p in written_pages),
        "written_correct": sum(p["correct"] for p in written_pages),
        "avg_gate_ms_per_q": round(float(np.mean(all_gate_ms)), 3),
    }
    return {"summary": summary, "blank_pages": blank_pages, "written_pages": written_pages}


async def run_pred_eval(cache_items: list[dict], hwr: HWRClient) -> dict:
    blank_pages = []
    written_pages = []
    all_gate_ms = []

    for item in cache_items:
        warped = cv2.imread(item["warped_path"])
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        pw, ph = item["page_w"], item["page_h"]
        _, pred_page, crops = build_freeform_page_from_lines(item["blocks"], warped, pw, ph)
        is_written = item["paper_id"] == "math_xmyz_2025_oct" and item["page_no"] == 1

        q_tasks = []
        q_meta = []
        t_page0 = perf_counter()
        for pq in pred_page["questions"]:
            pre_hw = str(crops[pq["id"]].get("pre_ocr_handwriting") or "").strip()
            q_meta.append((str(pq["number"]), pq["type"], pre_hw))
            q_tasks.append(
                _eval_question(hwr, warped, gray, pq["answer_areas"], pq["type"], item["blocks"], pre_hw)
            )
        results = await asyncio.gather(*q_tasks)
        page_wall_ms = round((perf_counter() - t_page0) * 1000, 2)
        for r in results:
            all_gate_ms.append(r["gate_ms"])

        if not is_written:
            blank_pages.append(
                {
                    "case": item["case"],
                    "q_count": len(results),
                    "blank_ok": sum(1 for r in results if r["status"] == "blank"),
                    "uncertain": sum(1 for r in results if r["status"] == "uncertain"),
                    "false_positive": sum(1 for r in results if r["status"] == "recognized"),
                    "gate_skipped": sum(1 for r in results if r["gate_skipped"]),
                    "gate_plus_hwr_wall_ms": page_wall_ms,
                    "non_blank_details": [
                        {"qno": qno, **r}
                        for (qno, _, _), r in zip(q_meta, results)
                        if r["status"] != "blank"
                    ],
                }
            )
        else:
            details = []
            correct = 0
            for (qno, _, pre_hw), r in zip(q_meta, results):
                gt_ans = WRITTEN_MATH_P1.get(qno, "")
                ok = bool(gt_ans and r["status"] == "recognized" and r["pred_ans"] == gt_ans)
                correct += int(ok)
                details.append({"qno": qno, "gt": gt_ans, "pre_hw": pre_hw, **r, "exact_match": ok})
            written_pages.append(
                {
                    "case": item["case"],
                    "correct": correct,
                    "total": len(details),
                    "gate_plus_hwr_wall_ms": page_wall_ms,
                    "questions": details,
                }
            )

    summary = {
        "mode": "current_page_ocr_predicted_boxes",
        "blank_questions": sum(p["q_count"] for p in blank_pages),
        "blank_exact": sum(p["blank_ok"] for p in blank_pages),
        "blank_uncertain": sum(p["uncertain"] for p in blank_pages),
        "blank_false_positive": sum(p["false_positive"] for p in blank_pages),
        "blank_gate_skipped": sum(p["gate_skipped"] for p in blank_pages),
        "blank_avg_gate_plus_hwr_ms": round(float(np.mean([p["gate_plus_hwr_wall_ms"] for p in blank_pages])), 2),
        "written_questions": sum(p["total"] for p in written_pages),
        "written_correct": sum(p["correct"] for p in written_pages),
        "avg_gate_ms_per_q": round(float(np.mean(all_gate_ms)), 3),
    }
    return {"summary": summary, "blank_pages": blank_pages, "written_pages": written_pages}


async def main() -> None:
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    gt_lib = TemplateLibrary(ROOT / "data/templates")
    gt_lib.load()

    settings = Settings(
        _env_file=None,
        ocr_base_url="http://127.0.0.1:8002/v1",
        ocr_model="PaddleOCR-VL-1.5",
        hwr_base_url="http://127.0.0.1:8001/v1",
        hwr_model="Qwen3.5-2B",
        hwr_api_key="",
    )
    clients = ModelClients(settings)
    hwr = HWRClient(clients)
    await clients.warmup()

    try:
        # Warmup 1 call to 8001
        dummy = np.full((60, 100, 3), 245, np.uint8)
        _, enc = cv2.imencode(".jpg", dummy)
        await hwr.recognize([{"jpeg": enc.tobytes(), "hint": "choice_letter", "area_index": 0}], "choice", freeform=True)

        gt_base = await run_gt_eval("gt_boxes_rectify_base", cache["items"], gt_lib, clients, hwr, use_rectify_v2=False)
        gt_v2 = await run_gt_eval("gt_boxes_rectify_v2", cache["items"], gt_lib, clients, hwr, use_rectify_v2=True)
        pred_base = await run_pred_eval(cache["items"], hwr)
    finally:
        await clients.http.aclose()

    report = {
        "gt_boxes_rectify_base": gt_base,
        "gt_boxes_rectify_v2": gt_v2,
        "pred_boxes_current_page_ocr": pred_base,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for k, v in report.items():
        print(f"=== {k} ===")
        print(json.dumps(v["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved full verification report to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
