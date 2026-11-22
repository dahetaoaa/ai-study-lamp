"""对 baseline_rec_*/capture_*.jpg（两段视频自动稳帧触发存下来的 9 张原始视频帧）
跑同样的无模板整页 OCR + 切题 + HWR 检验。
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))

from app.config import Settings  # noqa: E402
from app.grading.client import ModelClients  # noqa: E402
from app.grading.hwr_client import HWRClient, merge_freeform_choice_result  # noqa: E402
from app.vision.matcher import TemplateLibrary, project, quad  # noqa: E402
from app.vision.page_ocr import PageOCRClient  # noqa: E402

from eval_freeform_videos import (  # noqa: E402
    DATA_DIR,
    WRITTEN_MATH_P1,
    box_iou,
    normalize_text,
    quad_aabb,
)


async def evaluate_baseline_frame(
    rec_dir_name: str,
    img_path: Path,
    paper_id: str,
    page_no: int,
    gt_lib: TemplateLibrary,
    page_ocr: PageOCRClient,
    hwr: HWRClient,
) -> dict:
    image = cv2.imread(str(img_path))
    gt_match = gt_lib.match(image)
    gt_paper = gt_lib.papers[paper_id]
    gt_page = next(pg for pg in gt_paper["pages"] if pg["page_no"] == page_no)
    gt_qs = {str(q["number"]): q for q in gt_page["questions"]}
    is_written_math = paper_id == "math_xmyz_2025_oct" and page_no == 1

    gt_cam_boxes = {}
    if gt_match is not None:
        for qno, gq in gt_qs.items():
            gt_cam_boxes[qno] = quad_aabb(project(quad(gq["bbox"]), gt_match.homography))

    t_start = perf_counter()
    free_match = await page_ocr.extract_page(image)
    ocr_ms = round((perf_counter() - t_start) * 1000, 1)

    if free_match is None:
        return {
            "case": f"{rec_dir_name}/{img_path.name}",
            "paper_id": paper_id,
            "page_no": page_no,
            "error": "extract_page_returned_none",
            "ocr_ms": ocr_ms,
        }

    pred_paper = free_match.template.paper
    pred_page = free_match.template.page
    pred_qs_list = pred_page["questions"]
    pred_qs = {str(q["number"]): q for q in pred_qs_list}

    t_hwr0 = perf_counter()

    async def run_one_hwr(pq: dict):
        qid = pq["id"]
        crop = free_match.crops.get(qid, {})
        if pq["type"] == "listening":
            return qid, "", False, "listening", 0.0
        if crop.get("is_blank"):
            return qid, "", False, "blank", 0.0
        q_start = perf_counter()
        blocks = crop.get("blocks") or []
        pre_hw = str(crop.get("pre_ocr_handwriting") or "").strip()
        if blocks:
            raw, low = await hwr.recognize(blocks, pq["type"], freeform=True)
            if pq["type"] == "choice":
                raw, low = merge_freeform_choice_result(
                    raw,
                    low,
                    pre_hw,
                    ink_pixels=crop.get("clean_ink_pixels", 0),
                    pre_ocr_confident=crop.get("pre_ocr_confident", False),
                )
            elif not raw.strip() and pre_hw:
                raw, low = pre_hw, False
        else:
            raw, low = pre_hw, False
        q_ms = round((perf_counter() - q_start) * 1000, 1)
        if not raw.strip():
            status = "uncertain" if low else "blank"
        elif low:
            status = "uncertain"
        else:
            status = "recognized"
        return qid, raw, low, status, q_ms

    hwr_results_list = await asyncio.gather(*(run_one_hwr(pq) for pq in pred_qs_list))
    hwr_wall_ms = round((perf_counter() - t_hwr0) * 1000, 1)
    total_ms = round((perf_counter() - t_start) * 1000, 1)
    hwr_by_id = {qid: (raw, low, st, qms) for qid, raw, low, st, qms in hwr_results_list}

    gt_nums = list(gt_qs.keys())
    pred_nums = [str(q["number"]) for q in pred_qs_list]
    matched_nums = [n for n in gt_nums if n in pred_qs]
    missing_nums = [n for n in gt_nums if n not in pred_qs]
    extra_nums = [n for n in pred_nums if n not in gt_qs]

    expected_subject = gt_paper["subject"]
    subject_ok = pred_paper["subject"] == expected_subject
    expected_has_passage = any(gq.get("passage_id") for gq in gt_page["questions"])
    pred_has_passage = bool(pred_paper.get("passages"))

    per_question = []
    for qno in gt_nums:
        gq = gt_qs[qno]
        expected_ans = WRITTEN_MATH_P1[qno] if is_written_math else ""
        if qno not in pred_qs:
            per_question.append(
                {
                    "no": qno,
                    "detected": False,
                    "gt_type": gq["type"],
                    "expected_ans": expected_ans,
                    "ans_exact": False,
                }
            )
            continue

        pq = pred_qs[qno]
        raw, low, st, qms = hwr_by_id[pq["id"]]
        pred_cam_box = quad_aabb(project(quad(pq["bbox"]), free_match.homography))
        iou = round(box_iou(gt_cam_boxes[qno], pred_cam_box), 3) if qno in gt_cam_boxes else None

        gt_full = gq["stem"] + (" " + " ".join(gq.get("options") or []) if gq.get("options") else "")
        pred_full = pq["stem"] + (" " + " ".join(pq.get("options") or []) if pq.get("options") else "")
        stem_sim = round(difflib.SequenceMatcher(None, normalize_text(gt_full), normalize_text(pred_full)).ratio(), 3)

        gt_opt_cnt = len(gq.get("options") or [])
        pred_opt_cnt = len(pq.get("options") or [])

        if expected_ans == "":
            ans_exact = st == "blank" and raw.strip() == "" and not low
        else:
            ans_exact = st == "recognized" and not low and raw.strip().upper() == expected_ans.upper()

        per_question.append(
            {
                "no": qno,
                "detected": True,
                "gt_type": gq["type"],
                "pred_type": pq["type"],
                "type_ok": gq["type"] == pq["type"],
                "gt_opts": gt_opt_cnt,
                "pred_opts": pred_opt_cnt,
                "stem_sim": stem_sim,
                "cam_iou": iou,
                "gt_cam_box": [round(v, 1) for v in gt_cam_boxes[qno]] if qno in gt_cam_boxes else None,
                "pred_cam_box": [round(v, 1) for v in pred_cam_box],
                "expected_ans": expected_ans,
                "pred_ans": raw,
                "low_conf": low,
                "hwr_status": st,
                "ans_exact": ans_exact,
                "hwr_ms": qms,
                "stem_preview": pq["stem"][:90],
            }
        )

    recall = len(matched_nums) / max(1, len(gt_nums))
    precision = len(matched_nums) / max(1, len(pred_nums))
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    detected_rows = [r for r in per_question if r["detected"]]

    return {
        "case": f"{rec_dir_name}/{img_path.name}",
        "paper_id": paper_id,
        "page_no": page_no,
        "written_page": is_written_math,
        "gt_match_ok": gt_match is not None
        and gt_match.template.paper["paper_id"] == paper_id
        and gt_match.template.page["page_no"] == page_no,
        "ocr_source": free_match.ocr_source,
        "ocr_ms": ocr_ms,
        "hwr_wall_ms": hwr_wall_ms,
        "total_local_ms": total_ms,
        "subject_expected": expected_subject,
        "subject_pred": pred_paper["subject"],
        "subject_ok": subject_ok,
        "passage_expected": expected_has_passage,
        "passage_extracted": pred_has_passage,
        "gt_q_count": len(gt_nums),
        "pred_q_count": len(pred_nums),
        "matched_q_count": len(matched_nums),
        "missing_nums": missing_nums,
        "extra_nums": extra_nums,
        "q_recall": round(recall, 4),
        "q_precision": round(precision, 4),
        "q_f1": round(f1, 4),
        "type_acc": round(sum(r["type_ok"] for r in detected_rows) / max(1, len(detected_rows)), 4),
        "mean_stem_sim": round(float(np.mean([r["stem_sim"] for r in detected_rows])) if detected_rows else 0.0, 4),
        "mean_cam_iou": round(
            float(np.mean([r["cam_iou"] for r in detected_rows if r["cam_iou"] is not None])) if detected_rows else 0.0,
            4,
        ),
        "ans_exact_count": sum(r["ans_exact"] for r in per_question),
        "ans_exact_rate": round(sum(r["ans_exact"] for r in per_question) / max(1, len(per_question)), 4),
        "questions": per_question,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/diagnostics/freeform_round1_video_frames.json",
    )
    args = parser.parse_args()

    gt_settings = Settings()
    gt_lib = TemplateLibrary(gt_settings.templates_dir)
    gt_lib.load()

    with tempfile.TemporaryDirectory() as empty_dir:
        free_settings = Settings(
            templates_dir=Path(empty_dir),
            fixture_template=None,
            freeform_ocr_enabled=True,
            ocr_base_url="http://127.0.0.1:8002/v1",
            ocr_model="PaddleOCR-VL-1.5",
            hwr_base_url="http://127.0.0.1:8001/v1",
            hwr_model="Qwen3.5-2B",
            hwr_api_key="",
        )
        clients = ModelClients(free_settings)
        await clients.warmup()
        page_ocr = PageOCRClient(clients)
        hwr = HWRClient(clients)

        report_files = sorted(DATA_DIR.glob("baseline_rec_*/report.json"))
        cases = []
        try:
            for rpath in report_files:
                rep = json.loads(rpath.read_text())
                rec_dir = rpath.parent
                for trig in rep["triggers"]:
                    ipath = rec_dir / trig["image"]
                    res = await evaluate_baseline_frame(
                        rec_dir.name,
                        ipath,
                        trig["paper_id"],
                        int(trig["page_no"]),
                        gt_lib,
                        page_ocr,
                        hwr,
                    )
                    cases.append(res)
                    print(
                        f"[{res['case']}] {res['paper_id']} p{res['page_no']} | "
                        f"Q: {res.get('matched_q_count')}/{res.get('gt_q_count')} (pred={res.get('pred_q_count')}, F1={res.get('q_f1')}) | "
                        f"type={res.get('type_acc')} stem={res.get('mean_stem_sim')} IoU={res.get('mean_cam_iou')} | "
                        f"Ans={res.get('ans_exact_count')}/{res.get('gt_q_count')} | "
                        f"OCR={res.get('ocr_ms')}ms HWR={res.get('hwr_wall_ms')}ms Total={res.get('total_local_ms')}ms"
                    )
        finally:
            await clients.http.aclose()

    all_qs = [q for c in cases for q in c.get("questions", [])]
    det_qs = [q for q in all_qs if q.get("detected")]
    written_qs = [q for q in all_qs if q["expected_ans"] != ""]
    blank_qs = [q for q in all_qs if q["expected_ans"] == ""]

    total_gt = sum(c.get("gt_q_count", 0) for c in cases)
    total_pred = sum(c.get("pred_q_count", 0) for c in cases)
    total_matched = sum(c.get("matched_q_count", 0) for c in cases)
    g_rec = total_matched / max(1, total_gt)
    g_prec = total_matched / max(1, total_pred)
    g_f1 = (2 * g_rec * g_prec / (g_rec + g_prec)) if (g_rec + g_prec) > 0 else 0.0

    summary = {
        "cases_count": len(cases),
        "total_gt_questions": total_gt,
        "total_pred_questions": total_pred,
        "total_matched_questions": total_matched,
        "question_recall": round(g_rec, 4),
        "question_precision": round(g_prec, 4),
        "question_f1": round(g_f1, 4),
        "subject_accuracy": round(sum(c.get("subject_ok", False) for c in cases) / max(1, len(cases)), 4),
        "qtype_accuracy": round(sum(q.get("type_ok", False) for q in det_qs) / max(1, len(det_qs)), 4),
        "mean_stem_similarity": round(float(np.mean([q["stem_sim"] for q in det_qs])) if det_qs else 0.0, 4),
        "mean_camera_iou": round(
            float(np.mean([q["cam_iou"] for q in det_qs if q.get("cam_iou") is not None])) if det_qs else 0.0, 4
        ),
        "written_questions_total": len(written_qs),
        "written_exact_correct": sum(q["ans_exact"] for q in written_qs),
        "written_accuracy": round(sum(q["ans_exact"] for q in written_qs) / max(1, len(written_qs)), 4),
        "blank_questions_total": len(blank_qs),
        "blank_exact_correct": sum(q["ans_exact"] for q in blank_qs),
        "blank_false_positive_or_uncertain": sum(not q["ans_exact"] for q in blank_qs),
        "blank_accuracy": round(sum(q["ans_exact"] for q in blank_qs) / max(1, len(blank_qs)), 4),
        "overall_answer_accuracy": round(sum(q["ans_exact"] for q in all_qs) / max(1, len(all_qs)), 4),
        "avg_ocr_ms": round(float(np.mean([c["ocr_ms"] for c in cases])), 1),
        "avg_hwr_wall_ms": round(float(np.mean([c["hwr_wall_ms"] for c in cases])), 1),
        "avg_total_local_ms": round(float(np.mean([c["total_local_ms"] for c in cases])), 1),
        "p95_total_local_ms": round(float(np.percentile([c["total_local_ms"] for c in cases], 95)), 1),
    }

    report = {"summary": summary, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== VIDEO FRAMES OVERALL SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
