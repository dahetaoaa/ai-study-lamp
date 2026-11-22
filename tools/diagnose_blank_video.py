"""Real-video blank diagnostics; reads parent's immutable test corpus."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.vision.cropper import crop_answers

CORPUS = Path('/home/ubur/dahetao/data/video_testset/recordings_20261117')
OUT = ROOT / 'data/diagnostics/blank_review'


def cases():
    for path in sorted(CORPUS.glob('rec_*/cap_*.json')):
        observed = json.loads(path.read_text())
        cap = observed['capture']
        paper = json.loads((ROOT / 'data/templates' / cap['paper_id'] / 'template.json').read_text())
        page = next(p for p in paper['pages'] if p['page_no'] == cap['page_no'])
        template_image = cv2.imread(str(ROOT / 'data/templates' / cap['paper_id'] / page['image_file']), 0)
        match = SimpleNamespace(template=SimpleNamespace(paper=paper, page=page, image=template_image), homography=np.array(cap['homography']))
        image = cv2.imread(str(path.with_suffix('.jpg')))
        yield path.stem, image, match, observed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for cid, image, match, observed in cases():
        found = crop_answers(image, match)
        expected_blank = not (match.template.paper['paper_id'] == 'math_xmyz_2025_oct' and match.template.page['page_no'] == 1)
        h,w=match.template.image.shape
        warped=cv2.warpPerspective(image,np.linalg.inv(match.homography),(w,h),borderValue=(255,255,255))
        for question in match.template.page['questions']:
            row=found[question['id']]
            results.append({'capture':cid,'qid':question['id'],'expected_blank':expected_blank,'is_blank':row['is_blank'],
                            'primary_area_index':row.get('primary_area_index'),'area_diagnostics':row.get('area_diagnostics')})
            if row['is_blank'] != expected_blank:
                for i,area in enumerate(question.get('answer_areas',[])):
                    x,y,aw,ah=map(int,area['bbox']);pad=45
                    obs=warped[max(0,y-pad):min(h,y+ah+pad),max(0,x-pad):min(w,x+aw+pad)]
                    ref=cv2.cvtColor(match.template.image[max(0,y-pad):min(h,y+ah+pad),max(0,x-pad):min(w,x+aw+pad)],cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(str(OUT/f'{cid}_{question["id"]}_{i}.jpg'),np.concatenate([ref,obs],axis=1))
    print('all',len(results),'mismatches',sum(x['is_blank']!=x['expected_blank'] for x in results))
    for x in results:
        if x['is_blank']!=x['expected_blank']: print(json.dumps(x,ensure_ascii=False))
    (OUT/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
