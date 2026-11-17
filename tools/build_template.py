"""离线建模板（几何部分）：根据 PP-OCRv5 行坐标 + 图像横线检测，切出每道题的 bbox / answer_areas / mark_anchor，
生成草稿 data/templates/{paper_id}/draft.json，供 flash 修正内容、补答案后写成 template.json。

用法: .venv/bin/python tools/build_template.py [paper_id ...]
      .venv/bin/python tools/build_template.py --preview paper_id   # 画出 template.json(或 draft.json) 的框，输出 preview/page_NN.jpg
"""
import argparse, glob, json, os, re

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "data", "templates")
PAGE_W, PAGE_H = 1654, 2339
FOOTER_Y = 2240          # 底部“第 x/y 页”页脚以下忽略
QNUM = re.compile(r"^\s*(\d{1,3})\s*[.．、,，]\s*\S")
SECTION = re.compile(r"^\s*([一二三四五六七八九十]+、|[IVX]+[.、．]|第[一二三四五六七八九十]+[部节]分?)")

META = {
    "math_xmyz_2025_oct": ("厦门一中 2025-2026 九年级上 10月月考 · 数学", "math"),
    "eng_xmyz_2023_m1": ("厦门一中 2023-2024 九年级上 第一次月考 · 英语", "en"),
    "eng_xmwgy_2024_mid": ("厦门外国语 2024-2025 九年级上 期中 · 英语", "en"),
}


def box_xywh(b):
    return [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]


def find_underlines(gray, roi):
    """在 roi=[x,y,w,h] 内找横线（填空横线），返回 [x1,y,x2] 列表（页面坐标）。"""
    x, y, w, h = roi
    sub = gray[y:y + h, x:x + w]
    if sub.size == 0:
        return []
    bw = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (55, 1))
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(lines)
    out = []
    for i in range(1, n):
        sx, sy, sw, sh, _ = stats[i]
        if 55 <= sw <= 700 and sh <= 8:
            out.append([int(x + sx), int(y + sy + sh // 2), int(x + sx + sw)])
    return sorted(out, key=lambda s: (s[1] // 20, s[0]))


def bracket_area(line):
    """首行里的中文/英文括号（选择题作答括号）位置估计。"""
    t = line["text"]
    m = None
    for m in re.finditer(r"[（(]\s*[）)]", t):
        pass
    if not m:
        return None
    x0, y0, x1, y1 = line["box"]
    cx = x0 + (x1 - x0) * (m.start() + 1) / max(len(t), 1)
    return [int(cx - 45), int(y0 - 14), 110, int(y1 - y0 + 28)]


def segment_page(pid, n, gray, lines, subject):
    lines = [l for l in lines if l["box"][1] < FOOTER_Y]
    starts = [i for i, l in enumerate(lines) if QNUM.match(l["text"]) and l["box"][0] < 420]
    qs = []
    for k, i in enumerate(starts):
        j_end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        seg = []
        for l in lines[i:j_end]:
            if seg and SECTION.match(l["text"]):
                break
            seg.append(l)
        first = seg[0]
        num = QNUM.match(first["text"]).group(1)
        x0 = min(l["box"][0] for l in seg); y0 = min(l["box"][1] for l in seg)
        x1 = max(l["box"][2] for l in seg); y1 = max(l["box"][3] for l in seg)
        # 下一题顶部（含留白，数学解答题的作答区）
        next_top = lines[j_end]["box"][1] if j_end < len(lines) else FOOTER_Y
        text = "\n".join(l["text"] for l in seg)
        areas = []
        br = None
        for l in seg:  # 题干可能跨行，取选项行之前最后一个括号
            if re.match(r"^\s*A[.．、]", l["text"]):
                break
            br = bracket_area(l) or br
        if br:
            areas.append({"bbox": br, "hint": "choice_letter"})
        ul = find_underlines(gray, [max(0, x0 - 20), y0 - 10, min(PAGE_W - x0 + 20, x1 - x0 + 120), y1 - y0 + 20])
        ul = [u for u in ul if u[2] - u[0] <= 420 and any(
            abs(u[1] - l["box"][3]) <= 14 and l["box"][0] - 300 <= u[0] <= l["box"][2] + 300 for l in seg)]
        for sx, sy, ex in ul:
            areas.append({"bbox": [sx - 8, sy - 58, ex - sx + 16, 66],
                          "hint": "math_text" if subject == "math" else "english_text"})
        has_options = bool(re.search(r"(^|\s)A[.．、]", text)) and bool(re.search(r"(^|\s)B[.．、]", text))
        if not br and has_options:
            # 无括号的选择题（英语卷）：学生常把字母写在题号前或横线上
            fx0, fy0, _, fy1 = first["box"]
            areas.insert(0, {"bbox": [max(0, fx0 - 140), fy0 - 14, 150, fy1 - fy0 + 28], "hint": "choice_letter"})
        gap = next_top - y1
        qtype = "choice" if has_options else ("fill" if ul else ("math" if subject == "math" else "text"))
        if qtype in ("math", "text") and gap > 120:
            areas.append({"bbox": [x0, y1 + 6, PAGE_W - 120 - x0, gap - 16],
                          "hint": "math_steps" if subject == "math" else "english_text"})
        fcy = (first["box"][1] + first["box"][3]) / 2
        row = [l["box"][2] for l in lines if abs((l["box"][1] + l["box"][3]) / 2 - fcy) < 18]
        row += [a["bbox"][0] + a["bbox"][2] for a in areas
                if abs(a["bbox"][1] + a["bbox"][3] / 2 - fcy) < 40 and a["bbox"][0] > first["box"][0]]
        fx1 = max(row)
        mark = [int(min(fx1 + 60, PAGE_W - 90)), int(fcy)]
        bottom = max(y1, next_top - 10) if qtype in ("math", "text") else y1
        qs.append({
            "id": f"{pid}_p{n}_q{num}", "number": num, "type": qtype,
            "ocr_text": text,
            "bbox": [int(x0 - 10), int(y0 - 10), int(min(PAGE_W - 60, max(x1, mark[0] + 60)) - x0 + 20), int(bottom - y0 + 20)],
            "mark_anchor": mark,
            "answer_areas": [{"id": f"a{i+1}", **a} for i, a in enumerate(areas)],
        })
    return qs


def build(pid):
    title, subject = META.get(pid, (pid, "math"))
    pages = []
    for img in sorted(glob.glob(os.path.join(TPL, pid, "page_*.png"))):
        n = int(os.path.basename(img)[5:7])
        gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        lp = os.path.join(TPL, pid, "ocr", f"page_{n:02d}.lines.json")
        lines = json.load(open(lp)) if os.path.exists(lp) else []
        pages.append({"page_no": n, "image_file": os.path.basename(img),
                      "features_file": f"page_{n:02d}_akaze.npz",
                      "width_px": gray.shape[1], "height_px": gray.shape[0],
                      "questions": segment_page(pid, n, gray, lines, subject)})
    draft = {"paper_id": pid, "title": title, "subject": subject, "grade": "9",
             "source_pdf": f"{pid}.pdf", "render_dpi": 200, "pages": pages}
    json.dump(draft, open(os.path.join(TPL, pid, "draft.json"), "w"), ensure_ascii=False, indent=1)
    print(pid, [(p["page_no"], len(p["questions"])) for p in pages])


def preview(pid):
    src = os.path.join(TPL, pid, "template.json")
    if not os.path.exists(src):
        src = os.path.join(TPL, pid, "draft.json")
    t = json.load(open(src))
    out = os.path.join(TPL, pid, "preview"); os.makedirs(out, exist_ok=True)
    for p in t["pages"]:
        im = cv2.cvtColor(cv2.imread(os.path.join(TPL, pid, p["image_file"]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        for q in p["questions"]:
            x, y, w, h = q["bbox"]
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 120, 0), 2)
            cv2.putText(im, f"{q['number']}:{q['type']}", (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            for a in q.get("answer_areas", []):
                ax, ay, aw, ah = a["bbox"]
                cv2.rectangle(im, (ax, ay), (ax + aw, ay + ah), (0, 160, 0), 2)
            mx, my = q["mark_anchor"]
            cv2.circle(im, (mx, my), 14, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(out, f"page_{p['page_no']:02d}.jpg"), cv2.resize(im, None, fx=0.6, fy=0.6),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("preview ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("papers", nargs="*")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    for pid in a.papers or sorted(META):
        preview(pid) if a.preview else build(pid)
