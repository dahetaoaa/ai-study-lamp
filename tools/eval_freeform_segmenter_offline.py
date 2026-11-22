"""离线快速评测 freeform_segmenter：直接读取 spotting_raw_cache.json + warped_round0/ 对照 template.json 评测全部 9 张图 89 题。"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.vision.freeform_segmenter import (  # noqa: E402
    _parse_paddle_vl_blocks,
    build_freeform_page_from_lines,
)
from app.vision.matcher import TemplateLibrary  # noqa: E402

CACHE_PATH = ROOT / "data/diagnostics/spotting_raw_cache.json"
WRITTEN_MATH_P1 = {"1": "B", "2": "C", "3": "A", "4": "A", "5": "B", "6": "B", "7": "B"}


def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", "", s or "")
    s = s.replace("（", "(").replace("）", ")").replace("．", ".").replace("，", ",").replace("：", ":")
    s = re.sub(r"[_＿]{2,}", "_", s)
    s = s.replace("$", "")
    return s.lower()


def main() -> None:
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    gt_lib = TemplateLibrary(Settings().templates_dir)
    gt_lib.load()

    all_gt = 0
    all_pred = 0
    all_matched = 0
    all_type_ok = 0
    all_det = 0
    all_stem_sims = []
    written_hw_ok = 0
    written_total = 0

    for idx, item in enumerate(cache["items"]):
        case = item["case"]
        paper_id = item["paper_id"]
        page_no = item["page_no"]
        page_w = item["page_w"]
        page_h = item["page_h"]
        warped = cv2.imread(item["warped_path"])
        if warped is None:
            raise FileNotFoundError(item["warped_path"])

        t0 = perf_counter()
        blocks = _parse_paddle_vl_blocks(item["raw_spotting_text"], page_w, page_h)
        pred_paper, pred_page, crops = build_freeform_page_from_lines(blocks, warped, page_w, page_h)
        ms = round((perf_counter() - t0) * 1000, 1)

        gt_paper = gt_lib.papers[paper_id]
        gt_page = next(pg for pg in gt_paper["pages"] if pg["page_no"] == page_no)
        gt_qs = {str(q["number"]): q for q in gt_page["questions"]}
        pred_qs_list = pred_page["questions"]
        pred_qs = {str(q["number"]): q for q in pred_qs_list}

        gt_nums = list(gt_qs.keys())
        pred_nums = [str(q["number"]) for q in pred_qs_list]
        matched = [n for n in gt_nums if n in pred_qs]
        missing = [n for n in gt_nums if n not in pred_qs]
        extra = [n for n in pred_nums if n not in gt_qs]

        rec = len(matched) / max(1, len(gt_nums))
        prec = len(matched) / max(1, len(pred_nums))
        f1 = (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0

        type_ok_cnt = 0
        stem_sims = []
        for qno in matched:
            gq = gt_qs[qno]
            pq = pred_qs[qno]
            if gq["type"] == pq["type"]:
                type_ok_cnt += 1
            gt_full = gq["stem"] + (" " + " ".join(gq.get("options") or []) if gq.get("options") else "")
            pred_full = pq["stem"] + (" " + " ".join(pq.get("options") or []) if pq.get("options") else "")
            sim = difflib.SequenceMatcher(None, normalize_text(gt_full), normalize_text(pred_full)).ratio()
            stem_sims.append(sim)

        type_acc = type_ok_cnt / max(1, len(matched))
        mean_sim = float(np.mean(stem_sims)) if stem_sims else 0.0
        passage_ok = bool(pred_paper.get("passages"))
        passage_keys = list((pred_paper.get("passages") or {}).keys())

        all_gt += len(gt_nums)
        all_pred += len(pred_nums)
        all_matched += len(matched)
        all_type_ok += type_ok_cnt
        all_det += len(matched)
        all_stem_sims.extend(stem_sims)

        print(
            f"[{idx}] {case} ({paper_id} p{page_no}) | {ms}ms | "
            f"gt={len(gt_nums)} pred={len(pred_nums)} match={len(matched)} F1={f1:.4f} | "
            f"type_acc={type_acc:.4f} stem={mean_sim:.4f} | "
            f"subj={pred_paper['subject']}=={gt_paper['subject']} | "
            f"passages={passage_keys}"
        )
        if missing or extra:
            print(f"    missing={missing} extra={extra}")

        for qno in matched:
            gq = gt_qs[qno]
            pq = pred_qs[qno]
            crop = crops[pq["id"]]
            if gq["type"] != pq["type"]:
                print(f"    [TYPE MISMATCH] Q{qno}: gt={gq['type']} pred={pq['type']} stem={pq['stem'][:70]!r}")
            if paper_id == "math_xmyz_2025_oct" and page_no == 1:
                written_total += 1
                exp_hw = WRITTEN_MATH_P1[qno]
                pre_hw = crop.get("pre_ocr_handwriting", "")
                ok = pre_hw == exp_hw
                if ok:
                    written_hw_ok += 1
                ab = pq["answer_areas"][0]["bbox"] if pq["answer_areas"] else None
                print(f"    [MATH_P1] Q{qno}: exp={exp_hw} pre_hw={pre_hw!r} ok={ok} ans_bbox={ab} q_bbox={pq['bbox']}")
            elif paper_id == "math_xmyz_2025_oct" and page_no == 2 and qno in ("10", "11", "12", "13", "14", "15", "16"):
                ab = pq["answer_areas"][0]["bbox"] if pq["answer_areas"] else None
                print(f"    [MATH_P2] Q{qno} ({pq['type']}): pre_hw={crop.get('pre_ocr_handwriting')!r} ans_bbox={ab} has_fig={pq['has_diagram']}")
            elif paper_id == "eng_xmyz_2023_m1" and page_no == 8 and qno in ("64", "65", "66", "67", "68", "69", "70", "71"):
                ab = pq["answer_areas"][0]["bbox"] if pq["answer_areas"] else None
                print(f"    [ENG_P8] Q{qno} ({pq['type']}): opts={len(pq.get('options') or [])} pid={pq.get('passage_id')} bbox={pq['bbox']} ans_bbox={ab}")

    g_rec = all_matched / max(1, all_gt)
    g_prec = all_matched / max(1, all_pred)
    g_f1 = (2 * g_rec * g_prec / (g_rec + g_prec)) if (g_rec + g_prec) > 0 else 0.0
    print("\n=== SUMMARY ===")
    print(
        json.dumps(
            {
                "total_gt": all_gt,
                "total_pred": all_pred,
                "total_matched": all_matched,
                "question_recall": round(g_rec, 4),
                "question_precision": round(g_prec, 4),
                "question_f1": round(g_f1, 4),
                "qtype_accuracy": round(all_type_ok / max(1, all_det), 4),
                "mean_stem_similarity": round(float(np.mean(all_stem_sims)) if all_stem_sims else 0.0, 4),
                "written_pre_hw_exact": f"{written_hw_ok}/{written_total}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
