"""Engine must never draw right/wrong marks on a confidently blank recorded answer."""
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cv2
import pytest

from app.config import Settings
from app.engine import Engine
from app.state import Session
from app.vision.cropper import crop_answers
from app.vision.matcher import TemplateLibrary, project, quad

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/video_testset/recordings_20261117'


@pytest.mark.skipif(not DATA.exists(), reason='本机用户视频测试集未随Git提交')
async def test_real_blank_cases_remain_blank_even_if_cloud_says_right():
    settings = Settings(_env_file=None)
    library = TemplateLibrary(ROOT / 'data/templates')
    library.load()
    count = 0
    for source in sorted(DATA.glob('rec_*/cap_*.json')):
        old = json.loads(source.read_text())
        if old['capture']['paper_id'] == 'math_xmyz_2025_oct' and old['capture']['page_no'] == 1:
            continue  # 真实手写正例另有HWR与cropper测试
        image = cv2.imread(str(source.with_suffix('.jpg')))
        match = library.match(image)
        assert match is not None
        crops = crop_answers(image, match)
        session = Session(settings)
        events = []

        async def emit(event, log=events):
            log.append(event)

        async def stream(*args, **kwargs):
            yield '{"status":"right","student_display":"fabricated"}\n独立解题参考。'

        session.emit = emit
        engine = Engine(settings, session, None, library, SimpleNamespace(stream=stream))
        engine.hwr.recognize = AsyncMock(side_effect=AssertionError('可靠空白不应进入HWR'))
        items = {}
        for q in match.template.page['questions']:
            items[q['id']] = {'question_uid': q['id'], 'capture_id': 'test',
                              'mark_pos': [0, 0], 'answer_quads': [project(quad(a['bbox']), match.homography) for a in q['answer_areas']]}
        engine.captures['test'] = {'items': items, 'match': match, 'started': perf_counter(),
                                   'timings_ms': {}, 'verdict_count': 0, 'question_count': len(items)}
        for question in match.template.page['questions']:
            assert crops[question['id']]['is_blank'], question['id']
            result = await engine.grade_question('test', question, crops[question['id']])
            assert result == 'blank'
            count += 1
        for event in events:
            if event['type'] == 'question.verdict':
                assert event['status'] == 'blank' and event['is_blank']
                assert event['student_display'] == '' and event['error_quad'] is None
                assert len(event['mark_quad']) == 4
    assert count == 75
