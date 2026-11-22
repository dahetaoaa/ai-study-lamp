"""Live regression using existing user-recorded frames; temporarily pauses auto capture and restores it."""
import asyncio
import json
from pathlib import Path
import ssl

import httpx
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/video_testset/recordings_20261117'
SAMPLES = [
    ('blank', DATA/'rec_7267479ca145/cap_02231e8004844303.jpg'),
    ('written', DATA/'rec_7267479ca145/cap_47d2afe7e71746d1.jpg'),
]
ANSWERS = {'1': 'B', '2': 'C', '3': 'A', '4': 'A', '5': 'B', '6': 'B', '7': 'B'}


async def main():
    output = DATA / 'live_checked'
    output.mkdir(exist_ok=True)
    summary, geometry_cases = [], []
    async with httpx.AsyncClient(base_url='https://127.0.0.1:8000', verify=False, timeout=60) as client:
        async with connect('wss://127.0.0.1:8000/ws', ssl=ssl._create_unverified_context()) as ws:
            original = json.loads(await ws.recv())
            mode = original['mode']['auto_capture']
            try:
                await ws.send(json.dumps({'type': 'session.set_mode', 'auto_capture': False}))
                async with asyncio.timeout(10):
                    while True:
                        packet = await ws.recv()
                        if isinstance(packet, str):
                            event = json.loads(packet)
                            if event['type'] == 'session.snapshot' and not event['mode']['auto_capture']:
                                break
                for kind, image in SAMPLES:
                    response = await client.post('/api/debug/inject-image',
                        files={'file': (image.name, image.read_bytes(), 'image/jpeg')}, data={'grade': 'true'})
                    response.raise_for_status()
                    cid = response.json()['capture_id']
                    current = (await client.get('/api/session/state')).json()
                    result = current['captures'][cid].get('grade_result')
                    if not result:
                        async with asyncio.timeout(90):
                            while True:
                                packet = await ws.recv()
                                if isinstance(packet, bytes):
                                    continue
                                event = json.loads(packet)
                                if event['type'] == 'page.grade_done' and event['capture_id'] == cid:
                                    result = event
                                    break
                        current = (await client.get('/api/session/state')).json()
                    rows = [q for q in current['questions'].values() if q['capture_id'] == cid]
                    assert not result['error_uids'], result
                    assert all(len(q.get('mark_quad', [])) == 4 for q in rows)
                    if kind == 'blank':
                        assert len(rows) == 9
                        assert all(q['status'] == 'blank' and q['is_blank'] and q['error_quad'] is None for q in rows), rows
                    else:
                        assert len(rows) == 7
                        assert all(not q['is_blank'] and q['student_raw'] == ANSWERS[q['no']] for q in rows), rows
                    destination = output / (cid + '.json')
                    destination.write_text(json.dumps({'capture': current['captures'][cid], 'questions': rows}, ensure_ascii=False, indent=2))
                    geometry_cases.append({'observed': str(destination.relative_to(ROOT))})
                    summary.append({'kind': kind, 'capture_id': cid, 'questions': len(rows),
                                    'statuses': {q['no']: q['status'] for q in rows},
                                    'timings_ms': result.get('timings_ms')})
            finally:
                await ws.send(json.dumps({'type': 'session.set_mode', 'auto_capture': mode}))
                async with asyncio.timeout(10):
                    while True:
                        packet = await ws.recv()
                        if isinstance(packet, str):
                            event = json.loads(packet)
                            if event['type'] == 'session.snapshot' and event['mode']['auto_capture'] == mode:
                                break
    (output/'cases.json').write_text(json.dumps(geometry_cases, ensure_ascii=False, indent=2))
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Auto-capture restored; saved:', output)


if __name__ == '__main__':
    asyncio.run(main())
