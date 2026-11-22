"""Evaluate recorded capture cases against manually reviewed blank/written labels.
Only --hwr calls the existing local handwriting model; never calls the cloud judge.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.config import Settings
from app.grading.client import ModelClients
from app.grading.hwr_client import HWRClient
from app.vision.annotations import annotation_geometry
from app.vision.cropper import crop_answers
from app.vision.matcher import TemplateLibrary

DATA = ROOT / 'data/video_testset/recordings_20261117'
# Full capture images were visually reviewed: all English pages and math p2 are blank.
# The two math p1 captures have these seven handwritten option letters.
WRITTEN = {'1': 'B', '2': 'C', '3': 'A', '4': 'A', '5': 'B', '6': 'B', '7': 'B'}


async def evaluate(use_hwr=False, video_frames=False):
    settings = Settings(hwr_base_url='http://127.0.0.1:8001/v1', hwr_model='Qwen3.5-2B', hwr_api_key='')
    library = TemplateLibrary(settings.templates_dir)
    library.load()
    clients = ModelClients(settings)
    hwr = HWRClient(clients)
    report = {'local_hwr_used': use_hwr, 'label_basis': 'manual inspection of all saved capture images', 'cases': []}
    observed_files = sorted(DATA.glob('rec_*/cap_*.json'))
    sources = [(path, path.with_suffix('.jpg'), 'capture') for path in observed_files]
    if video_frames:
        index = {}
        for path in observed_files:
            capture = json.loads(path.read_text())['capture']
            index[(path.parent.name, capture['paper_id'], capture['page_no'])] = path
        for directory in sorted(DATA.glob('baseline_rec_*')):
            source_report = json.loads((directory / 'report.json').read_text())
            clip = directory.name.removeprefix('baseline_')
            for hit in source_report['triggers']:
                if hit.get('image'):
                    observed_path = index[(clip, hit['paper_id'], hit['page_no'])]
                    sources.append((observed_path, directory / hit['image'], 'video_frame'))
    try:
        for path, image_path, source_kind in sources:
            observed = json.loads(path.read_text())
            image = cv2.imread(str(image_path))
            started = perf_counter()
            match = library.match(image)
            if match is None:
                report['cases'].append({'file': str(path.relative_to(ROOT)), 'error': 'unmatched'})
                continue
            crops = crop_answers(image, match)
            vision_ms = round((perf_counter() - started) * 1000, 2)
            old = {q['template_question_id']: q for q in observed['questions']}
            written_page = match.template.paper['paper_id'] == 'math_xmyz_2025_oct' and match.template.page['page_no'] == 1
            case = {'file': str(image_path.relative_to(ROOT)), 'source_kind': source_kind, 'paper_id': match.template.paper['paper_id'],
                    'page_no': match.template.page['page_no'], 'vision_ms': vision_ms, 'questions': []}
            for q in match.template.page['questions']:
                crop = crops[q['id']]
                expected = WRITTEN[str(q['number'])] if written_page else ''
                raw, low = '', False
                if not crop['is_blank']:
                    if use_hwr:
                        raw, low = await hwr.recognize(crop['blocks'], q['type'])
                    else:
                        raw, low = '(not executed)', True
                blank = crop['is_blank'] or (not raw.strip() and not low)
                previous = old[q['id']]
                geometry = annotation_geometry(q, match.homography, match.template.page['width_px'], match.template.page['height_px'],
                                               crop.get('primary_area_index', 0), crop.get('ink_bbox'))
                case['questions'].append({
                    'id': q['id'], 'expected': expected, 'before_status': previous.get('status'),
                    'before_raw': previous.get('student_raw'), 'after_blank': blank, 'after_raw': raw,
                    'after_low': low, 'match_label': (blank if not expected else not blank and not low and raw.strip().upper() == expected),
                    'primary_area_index': crop.get('primary_area_index'), 'mark_quad': geometry['mark_quad'],
                    'area_diagnostics': crop.get('area_diagnostics'),
                })
            report['cases'].append(case)
    finally:
        await clients.http.aclose()
    rows = [q for c in report['cases'] for q in c.get('questions', [])]
    blank = [q for q in rows if not q['expected']]
    written = [q for q in rows if q['expected']]
    report['summary'] = {
        'cases': len(report['cases']), 'questions': len(rows), 'blank_expected': len(blank),
        'blank_before': sum(q['before_status'] == 'blank' for q in blank),
        'false_verdicts_before': sum(q['before_status'] in ('right', 'wrong') for q in blank),
        'blank_after': sum(q['after_blank'] for q in blank),
        'written_expected': len(written), 'written_correct_after': sum(q['match_label'] for q in written),
        'written_erased': sum(q['after_blank'] for q in written),
        'remaining': [q['id'] for q in rows if not q['match_label']],
    }
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hwr', action='store_true')
    p.add_argument('--video-frames', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output filename to preserve prior results')
    result = asyncio.run(evaluate(args.hwr, args.video_frames))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    print('Saved:', args.output)
