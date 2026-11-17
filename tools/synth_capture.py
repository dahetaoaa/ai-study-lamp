#!/usr/bin/env python3
"""把模板透视贴入书桌，写入指定答案，并保存真实单应矩阵用于误差验收。"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def handwriting_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Kaiti.ttc",
        "/System/Library/Fonts/Supplemental/STKaiti.ttf",
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def synthesize(template_path, page_no=1, mode="mixed", seed=0, answers=None, angle=None):
    rng = np.random.default_rng(seed)
    template_path = Path(template_path)
    paper = json.loads(template_path.read_text())
    page = next(p for p in paper["pages"] if p["page_no"] == page_no)
    image = Image.open(template_path.parent / page["image_file"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    written = {}
    for index, q in enumerate(page["questions"]):
        if q["type"] == "listening" or not q.get("answer_areas"):
            continue
        kind = ("right", "wrong", "blank")[index % 3] if mode == "mixed" else mode
        ref = q.get("reference_answer", "A")
        ref = ref[0] if isinstance(ref, list) else ref
        text = (
            ""
            if kind == "blank"
            else str(ref)
            if kind == "right"
            else ("B" if str(ref).upper() != "B" else "C")
            if q["type"] == "choice"
            else "x = 3"
        )
        if answers and (q["id"] in answers or str(q["number"]) in answers):
            text = str(answers.get(q["id"], answers.get(str(q["number"]))))
        written[q["id"]] = text
        if not text:
            continue
        x, y, w, h = q["answer_areas"][0]["bbox"]
        font = handwriting_font(max(18, min(48, int(h * 0.72))))
        bounds = draw.textbbox((0, 0), text, font=font)
        while (bounds[2] - bounds[0] > w * 0.85) and font.size > 14:
            font = handwriting_font(font.size - 2)
            bounds = draw.textbbox((0, 0), text, font=font)
        tx = x + (w - (bounds[2] - bounds[0])) / 2 + rng.uniform(-2, 2)
        ty = y + (h - (bounds[3] - bounds[1])) / 2 - bounds[1] + rng.uniform(-2, 2)
        draw.text((tx, ty), text, font=font, fill=(23, 33, 70), stroke_width=0)
    source = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    ph, pw = source.shape[:2]
    # 保留上下纸边；±15°旋转在1920x1080内完整可见。
    scale = min(880 / ph, 760 / pw)
    theta = np.deg2rad(float(angle) if angle is not None else rng.uniform(-15, 15))
    corners = np.float32([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]]) * scale
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = corners @ rotation.T + [960 + rng.uniform(-130, 130), 540]
    target += rng.uniform(-12, 12, (4, 2))
    h = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), np.float32(target))
    yy, xx = np.mgrid[:1080, :1920]
    wood = 73 + 7 * np.sin(xx * 0.023) + rng.normal(0, 1.8, (1080, 1920))
    background = np.stack([wood * 0.72, wood * 0.88, wood], axis=-1).clip(0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(source, h, (1920, 1080), borderValue=(255, 255, 255))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), h, (1920, 1080))
    composed = np.where((mask > 128)[..., None], warped, background)
    gradient = (
        rng.uniform(0.78, 0.92)
        + rng.uniform(0.05, 0.2) * (xx / 1920)
        + rng.uniform(-0.06, 0.06) * (yy / 1080)
    )
    composed = (composed * gradient[..., None]).clip(0, 255).astype(np.uint8)
    composed = cv2.GaussianBlur(composed, (3, 3), float(rng.uniform(0.2, 0.6)))
    metadata = {
        "paper_id": paper["paper_id"],
        "page_no": page_no,
        "seed": seed,
        "angle": float(np.rad2deg(theta)),
        "homography": h.tolist(),
        "page_quad": target.tolist(),
        "answers": written,
        "template_path": str(template_path.resolve()),
    }
    return composed, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--mode", choices=["right", "wrong", "blank", "mixed"], default="mixed")
    parser.add_argument("--answers", help='题号/ID到答案的JSON对象，例如 {"1":"B","2":""}')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--angle", type=float)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/mock_captures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        image, meta = synthesize(
            args.template,
            args.page,
            args.mode,
            args.seed + i,
            json.loads(args.answers) if args.answers else None,
            args.angle,
        )
        name = f"{meta['paper_id']}_p{args.page:02d}_{args.mode}_{args.seed + i:03d}"
        cv2.imwrite(str(args.output_dir / (name + ".jpg")), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        (args.output_dir / (name + ".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"已生成 {name}.jpg，旋转 {meta['angle']:.1f}°")


if __name__ == "__main__":
    main()
