"""离线 OCR：对模板页跑 PaddleOCR-VL（版面块+内容，含公式 LaTeX）与 PP-OCRv5（行级坐标）。

用法: .venv-ocr/bin/python tools/ocr_pages.py [paper_id ...] [--only-lines] [--pages 1,2]
输出: data/templates/{paper_id}/ocr/page_XX.vl.json / page_XX.lines.json
"""
import argparse, glob, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "data", "templates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("papers", nargs="*")
    ap.add_argument("--only-lines", action="store_true")
    ap.add_argument("--only-vl", action="store_true")
    ap.add_argument("--pages", default="")
    a = ap.parse_args()
    papers = a.papers or sorted(os.listdir(TPL))
    want = {int(x) for x in a.pages.split(",") if x}

    lines_ocr = vl = None
    if not a.only_vl:
        from paddleocr import PaddleOCR
        lines_ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                              use_textline_orientation=False)
    if not a.only_lines:
        from paddleocr import PaddleOCRVL
        vl = PaddleOCRVL()

    for pid in papers:
        out = os.path.join(TPL, pid, "ocr"); os.makedirs(out, exist_ok=True)
        for img in sorted(glob.glob(os.path.join(TPL, pid, "page_*.png"))):
            n = int(os.path.basename(img)[5:7])
            if want and n not in want:
                continue
            if lines_ocr:
                t = time.time(); r = lines_ocr.predict(img)[0]
                lines = [{"text": s, "score": round(float(c), 3), "box": [int(v) for v in b]}
                         for s, c, b in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"])]
                json.dump(lines, open(f"{out}/page_{n:02d}.lines.json", "w"), ensure_ascii=False, indent=0)
                print(pid, n, "lines", len(lines), f"{time.time()-t:.1f}s", flush=True)
            if vl:
                t = time.time(); r = vl.predict(img)[0]
                blocks = [{"label": b["block_label"], "bbox": [int(v) for v in b["block_bbox"]],
                           "content": b["block_content"]} for b in r.json["res"]["parsing_res_list"]]
                json.dump(blocks, open(f"{out}/page_{n:02d}.vl.json", "w"), ensure_ascii=False, indent=1)
                print(pid, n, "vl blocks", len(blocks), f"{time.time()-t:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
