"""模板校验：字段完整性、坐标合法性、id 唯一。用法: .venv/bin/python tools/check_template.py [paper_id ...]"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "data", "templates")
TYPES = {"choice", "fill", "math", "text", "listening", "writing"}
HINTS = {"choice_letter", "english_text", "math_text", "math_steps"}


def check(pid):
    t = json.load(open(os.path.join(TPL, pid, "template.json")))
    errs, ids, n = [], set(), 0
    for k in ("paper_id", "title", "subject", "render_dpi", "pages"):
        if k not in t:
            errs.append(f"缺字段 {k}")
    for p in t["pages"]:
        W, H = p["width_px"], p["height_px"]
        if not os.path.exists(os.path.join(TPL, pid, p["image_file"])):
            errs.append(f"p{p['page_no']} 缺图片")
        for q in p["questions"]:
            n += 1
            tag = f"p{p['page_no']} q{q.get('number')}"
            if q["id"] in ids:
                errs.append(f"{tag} id 重复")
            ids.add(q["id"])
            if q.get("type") not in TYPES:
                errs.append(f"{tag} type 非法: {q.get('type')}")
            for k in ("stem", "bbox", "mark_anchor", "answer_areas"):
                if q.get(k) in (None, ""):
                    errs.append(f"{tag} 缺 {k}")
            if q.get("type") == "choice" and not q.get("options"):
                errs.append(f"{tag} 选择题缺 options")
            if q.get("type") not in ("listening",) and not q.get("reference_answer"):
                errs.append(f"{tag} 缺 reference_answer")
            boxes = [("bbox", q.get("bbox"))] + [("area", a.get("bbox")) for a in q.get("answer_areas", [])]
            for name, b in boxes:
                if not b or len(b) != 4 or b[2] <= 0 or b[3] <= 0 or b[0] < -20 or b[1] < -20 \
                        or b[0] + b[2] > W + 20 or b[1] + b[3] > H + 20:
                    errs.append(f"{tag} {name} 越界/非法: {b}")
            for a in q.get("answer_areas", []):
                if a.get("hint") not in HINTS:
                    errs.append(f"{tag} answer_area hint 非法: {a.get('hint')}")
            mx, my = q.get("mark_anchor") or (-1, -1)
            if not (0 <= mx <= W and 0 <= my <= H):
                errs.append(f"{tag} mark_anchor 越界")
    print(f"{pid}: {len(t['pages'])} 页 {n} 题, {len(errs)} 个问题")
    for e in errs:
        print("  -", e)
    return not errs


if __name__ == "__main__":
    pids = sys.argv[1:] or sorted(d for d in os.listdir(TPL) if os.path.exists(os.path.join(TPL, d, "template.json")))
    ok = all([check(p) for p in pids])
    sys.exit(0 if ok else 1)
