"""Extract representative frames and event-aligned cases without changing recordings."""
import json
from pathlib import Path
import ssl
import urllib.request

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/video_testset/recordings_20261117'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context()
    snapshot = json.load(urllib.request.urlopen('https://127.0.0.1:8000/api/session/state', context=context))
    cases = []
    for folder in sorted((ROOT / 'data/capture_recordings').glob('rec_*')):
        target = OUT / folder.name
        target.mkdir(exist_ok=True)
        frames = [json.loads(s) for s in (folder / 'frames.jsonl').read_text().splitlines()]
        events = [json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()]
        selected = {}
        for t in range(0, int(frames[-1]['t']) + 1, 3):
            i = min(range(len(frames)), key=lambda i: abs(frames[i]['t'] - t))
            selected[i] = {'kind': 'overview'}
        for event in events:
            if event['type'] in ('capture.taken', 'capture.reused', 'expected_ready'):
                i = min(range(len(frames)), key=lambda i: abs(frames[i]['t'] - event['t']))
                selected[i] = {'kind': event['type'], 'event': event}
        cap = cv2.VideoCapture(str(folder / 'video.avi'))
        thumbnails = []
        try:
            for i, info in sorted(selected.items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode {folder.name} frame {i}')
                path = target / f'frame_{i:06d}.png'
                if not path.exists():
                    cv2.imwrite(str(path), frame)
                item = {'clip': folder.name, 'frame': i, 't': frames[i]['t'], 'image': str(path.relative_to(ROOT)), **info}
                event = info.get('event', {})
                cid = event.get('capture_id')
                if cid and cid in snapshot.get('captures', {}):
                    capture_path = target / f'{cid}.jpg'
                    if not capture_path.exists():
                        capture_path.write_bytes(urllib.request.urlopen(f'https://127.0.0.1:8000/api/captures/{cid}.jpg', context=context).read())
                    observed = {'capture': snapshot['captures'][cid], 'questions': [q for q in snapshot['questions'].values() if q['capture_id'] == cid]}
                    (target / f'{cid}.json').write_text(json.dumps(observed, ensure_ascii=False, indent=2))
                    item['capture_image'] = str(capture_path.relative_to(ROOT))
                    item['observed'] = str((target / f'{cid}.json').relative_to(ROOT))
                cases.append(item)
                thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                thumb.thumbnail((216, 384))
                card = Image.new('RGB', (224, 416), 'white')
                card.paste(thumb, ((224-thumb.width)//2, 26))
                ImageDraw.Draw(card).text((5, 5), f'{frames[i]["t"]:.2f}s / {i} / {info["kind"][:6]}', fill='black')
                thumbnails.append(card)
        finally:
            cap.release()
        sheet = Image.new('RGB', (224*5, 416*((len(thumbnails)+4)//5)), '#cccccc')
        for i, thumb in enumerate(thumbnails):
            sheet.paste(thumb, ((i%5)*224, (i//5)*416))
        sheet.save(target / 'contact.jpg', quality=92)
        print(folder.name, len(selected), 'cases', target/'contact.jpg')
    (OUT/'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print('Saved',len(cases),'cases to',OUT)


if __name__ == '__main__':
    main()
