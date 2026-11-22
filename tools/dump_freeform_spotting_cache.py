"""留存 9 张实拍图在无模板拉平与 PaddleOCR-VL-1.5 Spotting 下的原始输出与拉平图像。"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.vision.page_ocr import _parse_paddle_vl_blocks, rectify_paper  # noqa: E402

DATA_DIR = ROOT / "data/video_testset/recordings_20261117"
DEFAULT_OUT_JSON = ROOT / "data/diagnostics/spotting_raw_cache.json"
DEFAULT_WARPED_DIR = ROOT / "data/diagnostics/warped_round0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--ocr-model", default="PaddleOCR-VL-1.5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--warped-dir", type=Path, default=DEFAULT_WARPED_DIR)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.warped_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(DATA_DIR.glob("rec_*/cap_*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No cap_*.jpg found under {DATA_DIR}")

    results = []
    with httpx.Client(timeout=60.0) as client:
        for ipath in img_paths:
            rec_name = ipath.parent.name
            case_name = f"{rec_name}/{ipath.name}"
            jpath = ipath.with_suffix(".json")
            paper_id = None
            page_no = None
            if jpath.exists():
                obs = json.loads(jpath.read_text(encoding="utf-8"))
                cap_meta = obs.get("capture") or {}
                paper_id = cap_meta.get("paper_id")
                page_no = cap_meta.get("page_no")

            image = cv2.imread(str(ipath))
            if image is None:
                raise RuntimeError(f"Failed to read image: {ipath}")

            t0 = perf_counter()
            warped, h_page_to_cam, page_quad, page_w, page_h = rectify_paper(image)
            rectify_ms = round((perf_counter() - t0) * 1000, 2)

            warped_filename = f"{rec_name}__{ipath.name}"
            warped_path = args.warped_dir / warped_filename
            cv2.imwrite(str(warped_path), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])

            scale = min(1.0, 1540.0 / max(page_w, page_h))
            ocr_img = (
                cv2.resize(warped, (round(page_w * scale), round(page_h * scale)), interpolation=cv2.INTER_AREA)
                if scale < 1.0
                else warped
            )
            ok, encoded = cv2.imencode(".jpg", ocr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError(f"Failed to encode image: {ipath}")
            b64_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()

            payload = {
                "model": args.ocr_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": b64_url}},
                            {"type": "text", "text": "Spotting:"},
                        ],
                    }
                ],
                "max_tokens": 2800,
                "temperature": 0.0,
                "stream": False,
            }

            t1 = perf_counter()
            resp = client.post(args.ocr_base_url.rstrip("/") + "/chat/completions", json=payload)
            resp.raise_for_status()
            ocr_vllm_ms = round((perf_counter() - t1) * 1000, 2)

            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            blocks = _parse_paddle_vl_blocks(raw_text, page_w, page_h)

            entry = {
                "case": case_name,
                "paper_id": paper_id,
                "page_no": page_no,
                "img_path": str(ipath),
                "warped_path": str(warped_path),
                "rectify_ms": rectify_ms,
                "ocr_vllm_ms": ocr_vllm_ms,
                "orig_shape": list(image.shape),
                "warped_shape": list(warped.shape),
                "ocr_input_shape": list(ocr_img.shape),
                "page_w": page_w,
                "page_h": page_h,
                "page_quad": page_quad.tolist(),
                "homography": h_page_to_cam.tolist(),
                "raw_spotting_text": raw_text,
                "blocks_count": len(blocks),
                "blocks": blocks,
            }
            results.append(entry)
            print(
                f"[{case_name}] ({paper_id} p{page_no}) "
                f"rectify={rectify_ms}ms ocr_vllm={ocr_vllm_ms}ms "
                f"warped={list(warped.shape)} blocks={len(blocks)}",
                flush=True,
            )

    out_payload = {
        "ocr_base_url": args.ocr_base_url,
        "ocr_model": args.ocr_model,
        "count": len(results),
        "items": results,
    }
    args.output.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} spotting cache items to {args.output}", flush=True)


if __name__ == "__main__":
    main()
