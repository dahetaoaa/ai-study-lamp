"""Small local-HWR regression over the already-labelled written video regions."""
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
from app.vision.cropper import crop_answers
from app.vision.matcher import TemplateLibrary
from diagnose_blank_video import CORPUS, OUT, cases


async def main():
    clients = ModelClients(Settings(_env_file=None))
    hwr = HWRClient(clients)
    jobs = []
    for cid, image, match, observed in cases():
        if match.template.paper['paper_id'] != 'math_xmyz_2025_oct' or match.template.page['page_no'] != 1:
            continue
        crops = crop_answers(image, match)
        for q in match.template.page['questions']:
            old = next(v for v in observed['questions'] if v['template_question_id'] == q['id'])
            jobs.append((cid, q['id'], old['student_raw'], crops[q['id']]))
    library = TemplateLibrary(ROOT / 'data/templates')
    library.load()
    expected_answers = ['B', 'C', 'A', 'A', 'B', 'B', 'B']
    for path in sorted(CORPUS.glob('baseline_rec_*/report.json')):
        for hit in json.loads(path.read_text())['triggers']:
            if hit['paper_id'] != 'math_xmyz_2025_oct' or hit['page_no'] != 1:
                continue
            image = cv2.imread(str(path.parent / hit['image']))
            match = library.match(image)
            assert match is not None
            crops = crop_answers(image, match)
            for q in match.template.page['questions']:
                jobs.append((f'{path.parent.name}_{hit["frame"]}', q['id'],
                             expected_answers[int(q['number'])-1], crops[q['id']]))
    async def one(item):
        cid, qid, expected, crop = item
        start = perf_counter()
        raw, low = await hwr.recognize(crop['blocks'], 'choice')
        return {'capture': cid, 'qid': qid, 'expected': expected, 'raw': raw, 'low': low,
                'ms': round((perf_counter()-start)*1000, 1)}
    try:
        rows = await asyncio.gather(*(one(job) for job in jobs))
    finally:
        await clients.http.aclose()
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print('PASS', sum(r['raw'] == r['expected'] and not r['low'] for r in rows), '/', len(rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'hwr_written_results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
