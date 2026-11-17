"""One explicitly labelled live text/TTS smoke test; may play in connected browsers."""
import asyncio
import json
from pathlib import Path
import ssl
import struct
from time import perf_counter
from uuid import uuid4
import wave

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]


async def main():
    output = ROOT / 'data/diagnostics' / ('voice-' + uuid4().hex[:10])
    output.mkdir(parents=True)
    report, pcm = {}, bytearray()
    async with connect('wss://127.0.0.1:8000/ws', ssl=ssl._create_unverified_context()) as ws:
        await ws.recv()
        started = perf_counter()
        await ws.send(json.dumps({
            'type': 'chat.send', 'scope': 'all', 'speak': True,
            'client_msg_id': 'diagnostic-' + uuid4().hex,
            'text': '这是语音链路测试。请简短解释：为什么x的平方等于1时，x既可以是1也可以是负1？',
        }, ensure_ascii=False))
        seq = None
        async with asyncio.timeout(120):
            while True:
                raw = await ws.recv()
                at = round((perf_counter() - started) * 1000, 2)
                if isinstance(raw, bytes):
                    if len(raw) >= 4 and struct.unpack('<BBH', raw[:4])[2] == seq:
                        if len(raw) > 4:
                            report.setdefault('audio_first_ms', at)
                        pcm.extend(raw[4:])
                    continue
                event = json.loads(raw)
                typ = event['type']
                if typ == 'chat.message_start':
                    seq = event['turn_seq']
                    report['msg_id'] = event['msg_id']
                elif typ == 'chat.delta' and event.get('turn_seq') == seq:
                    report.setdefault('text_first_ms', at)
                elif typ == 'chat.message_done' and event.get('turn_seq') == seq:
                    report.update(text_done_ms=at, text=event['full_text'], failed=event.get('failed', False))
                elif typ == 'tts.start' and event.get('turn_seq') == seq:
                    report['speed'] = event.get('speed')
                elif typ == 'tts.end' and event.get('turn_seq') == seq:
                    report.update(tts_done_ms=at, tts_failed=event.get('failed', False))
                    break
                elif typ == 'error':
                    raise RuntimeError(event.get('message'))
                elif typ == 'voice.interrupted' and event.get('old_turn_seq') == seq:
                    raise RuntimeError('Live test interrupted')
    report['audio_ms'] = round(len(pcm) / 48, 2)
    assert report.get('text') and not report['failed'] and not report['tts_failed'], report
    assert pcm and report.get('speed') == 1.2, report
    assert report['text_done_ms'] < report['tts_done_ms'], report
    with wave.open(str(output / 'speech.wav'), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    (output / 'timings.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Saved:', output)


if __name__ == '__main__':
    asyncio.run(main())
