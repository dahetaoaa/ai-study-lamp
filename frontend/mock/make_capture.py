"""生成前端 ?mock=1 回放用的抓拍图与题目坐标。

用法（仓库根目录）：
    .venv/bin/python frontend/mock/make_capture.py

做的事：把 data/templates/math_xmyz_2025_oct/page_01.png 写上手写答案后，
用单应矩阵 H 透视贴到 1920x1080 深色书桌背景上，输出 frontend/mock/capture.jpg；
再用同一个 H 把 template.json 里的 bbox / mark_anchor / answer_areas 投影到抓拍原图像素空间，
按后端 engine.py 的 page.matched 结构写进 frontend/mock/capture-data.js。
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TPL_DIR = ROOT / "data/templates/math_xmyz_2025_oct"
OUT_DIR = Path(__file__).resolve().parent
W, H = 1920, 1080
PAGE_NO = 1
CAPTURE_ID = "cap_mock01"
# 学生手写（空字符串 = 未作答）。参考答案：B A A A B C B
ANSWERS = {"1": "B", "2": "C", "3": "A", "4": "", "5": "B", "6": "A", "7": "D"}


def quad(b):
    x, y, w, h = b
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def project(points, homography):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), homography).reshape(-1, 2).tolist()


def desk_background(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (24, 32, 44)  # BGR 深棕
    # 木纹：沿 x 方向的低频条纹
    grain = np.zeros((H, W), np.float32)
    for _ in range(140):
        y0 = rng.uniform(0, H)
        amp = rng.uniform(1.5, 6)
        f = rng.uniform(0.002, 0.008)
        ph = rng.uniform(0, 6.3)
        y = y0 + np.sin(xx[0] * f + ph) * amp
        width = rng.uniform(0.6, 2.4)
        grain += np.exp(-((yy - y[None, :]) ** 2) / (2 * width**2)) * rng.uniform(-9, 7)
    base += grain[..., None] * np.array([0.6, 0.8, 1.0], np.float32)
    # 台灯光池
    d = np.sqrt(((xx - 980) / 1150) ** 2 + ((yy - 470) / 820) ** 2)
    light = np.clip(1.35 - d, 0.35, 1.25)
    base *= light[..., None]
    base += rng.normal(0, 2.2, (H, W, 1)).astype(np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(7)
    template = json.loads((TPL_DIR / "template.json").read_text(encoding="utf-8"))
    paper = template
    page = next(p for p in template["pages"] if p["page_no"] == PAGE_NO)
    pw, ph = page["width_px"], page["height_px"]
    gray = cv2.imread(str(TPL_DIR / page["image_file"]), cv2.IMREAD_GRAYSCALE)
    sheet = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    # 纸张偏暖白
    sheet = sheet / 255.0 * np.array([226, 240, 246], np.float32)
    sheet = np.clip(sheet, 0, 255).astype(np.uint8)

    # 在答题区写手写字母（模板坐标系）
    pen = (70, 38, 28)
    for q in page["questions"]:
        ans = ANSWERS.get(str(q["number"]), "")
        if not ans or not q.get("answer_areas"):
            continue
        x, y, w, h = q["answer_areas"][0]["bbox"]
        scale = h / 26.0
        (tw, th), _ = cv2.getTextSize(ans, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, 3)
        org = (int(x + (w - tw) / 2), int(y + (h + th) / 2) - 2)
        cv2.putText(sheet, ans, org, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, pen, 4, cv2.LINE_AA)

    # 目标四角：轻微旋转 + 透视（俯拍略有梯形），底部超出画面（相机裁切）
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dst = np.float32([[478, 34], [1462, 6], [1508, 1426], [452, 1452]])
    homography = cv2.getPerspectiveTransform(src, dst)

    desk = desk_background(rng)
    warped = cv2.warpPerspective(sheet, homography, (W, H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), homography, (W, H))
    # 纸张投影
    shadow = cv2.GaussianBlur(cv2.warpPerspective(np.full((ph, pw), 255, np.uint8),
                                                  np.float32([[1, 0, 14], [0, 1, 18], [0, 0, 1]]) @ homography,
                                                  (W, H)), (0, 0), 16)
    out = desk.astype(np.float32) * (1 - 0.55 * shadow[..., None] / 255.0)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    out = out * (1 - m) + warped.astype(np.float32) * m
    # 光照渐变（左上亮、右下暗）
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = 1.04 - 0.14 * (xx / W) - 0.10 * (yy / H)
    out *= grad[..., None]
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), 0.7)
    cv2.imwrite(str(OUT_DIR / "capture.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])

    passages = paper.get("passages", {})
    items = []
    for q in page["questions"]:
        q_quad = project(quad(q["bbox"]), homography)
        pts = np.array(q_quad)
        low, high = pts.min(axis=0), pts.max(axis=0)
        items.append(
            {
                "question_uid": q["id"],
                "template_question_id": q["id"],
                "capture_id": CAPTURE_ID,
                "no": str(q["number"]),
                "qtype": q["type"],
                "short_stem": q["stem"][:80],
                "stem": q["stem"],
                "options": q.get("options"),
                "tag": q.get("tag", ""),
                "passage_id": q.get("passage_id"),
                "passage": passages.get(q.get("passage_id"), ""),
                "status": "judging" if q["type"] != "listening" else "idle",
                "quad": [[round(a, 1), round(b, 1)] for a, b in q_quad],
                "bbox": [round(float(low[0]), 1), round(float(low[1]), 1),
                         round(float(high[0] - low[0]), 1), round(float(high[1] - low[1]), 1)],
                "mark_pos": [round(v, 1) for v in project([q.get("mark_anchor", q["bbox"][:2])], homography)[0]],
                "answer_quads": [[[round(a, 1), round(b, 1)] for a, b in project(quad(a["bbox"]), homography)]
                                 for a in q.get("answer_areas", [])],
                # 回放脚本用：参考答案、模板解析、学生手写
                "_reference": q.get("answer_display") or q.get("reference_answer"),
                "_reference_letter": q.get("reference_answer"),
                "_analysis": q.get("analysis", ""),
                "_student": ANSWERS.get(str(q["number"]), ""),
            }
        )
    matched = {
        "type": "page.matched",
        "capture_id": CAPTURE_ID,
        "mode": "template",
        "paper_id": paper["paper_id"],
        "paper_title": paper["title"],
        "subject": paper["subject"],
        "page_no": page["page_no"],
        "image_width": W,
        "image_height": H,
        "homography": [[round(v, 8) for v in row] for row in homography.tolist()],
        "page_quad": [[round(a, 1), round(b, 1)] for a, b in project(quad([0, 0, pw, ph]), homography)],
        "questions": items,
    }
    js = (
        "// 由 frontend/mock/make_capture.py 生成，请勿手改。坐标为 capture.jpg 原图像素。\n"
        "export default " + json.dumps(matched, ensure_ascii=False, indent=1) + ";\n"
    )
    (OUT_DIR / "capture-data.js").write_text(js, encoding="utf-8")

    # 校验图：画出投影框，供人工核对（不提交给前端使用）
    check = out.copy()
    for it in items:
        cv2.polylines(check, [np.int32(it["quad"])], True, (0, 200, 0), 2)
        for aq in it["answer_quads"]:
            cv2.polylines(check, [np.int32(aq)], True, (0, 0, 255), 2)
        cv2.circle(check, tuple(int(v) for v in it["mark_pos"]), 7, (255, 0, 255), -1)
    cv2.imwrite("/tmp/fe_work/capture_check.jpg", check)
    print("ok", len(items), "questions")


if __name__ == "__main__":
    main()
