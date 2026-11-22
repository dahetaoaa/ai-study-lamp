"""Check decoded replay candidates, independently rematching each video frame."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.vision.cropper import crop_answers
from app.vision.matcher import TemplateLibrary

CORPUS = Path('/home/ubur/dahetao/data/video_testset/recordings_20261117')
OUT = ROOT / 'data/diagnostics/blank_review/replay'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lib = TemplateLibrary(ROOT / 'data/templates')
    lib.load()
    results = []
    for path in sorted(CORPUS.glob('baseline_rec_*/report.json')):
        report = json.loads(path.read_text())
        for hit in report['triggers']:
            if 'image' not in hit:
                continue
            image = cv2.imread(str(path.parent / hit['image']))
            match = lib.match(image)
            assert match and match.template.paper['paper_id'] == hit['paper_id'] and match.template.page['page_no'] == hit['page_no']
            crops = crop_answers(image, match)
            expected_blank = not (hit['paper_id'] == 'math_xmyz_2025_oct' and hit['page_no'] == 1)
            height, width = match.template.image.shape
            warped = cv2.warpPerspective(image, np.linalg.inv(match.homography), (width,height),borderValue=(255,255,255))
            for q in match.template.page['questions']:
                row = crops[q['id']]
                results.append({'video':path.parent.name,'frame':hit['frame'],'qid':q['id'],
                                'expected_blank':expected_blank,'is_blank':row['is_blank'],
                                'area_diagnostics':row['area_diagnostics']})
                if row['is_blank'] != expected_blank:
                    for index, area in enumerate(q.get('answer_areas',[])):
                        x,y,w,h=area['bbox'];pad=45
                        observed=warped[max(0,y-pad):min(height,y+h+pad),max(0,x-pad):min(width,x+w+pad)]
                        ref=cv2.cvtColor(match.template.image[max(0,y-pad):min(height,y+h+pad),max(0,x-pad):min(width,x+w+pad)],cv2.COLOR_GRAY2BGR)
                        cv2.imwrite(str(OUT/f'{path.parent.name}_{hit["frame"]}_{q["id"]}_{index}.jpg'),np.concatenate([ref,observed],axis=1))
    errors=[r for r in results if r['expected_blank']!=r['is_blank']]
    print('Replay candidates: all',len(results),'mismatches',len(errors))
    for row in errors: print(json.dumps(row,ensure_ascii=False))
    (OUT/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
