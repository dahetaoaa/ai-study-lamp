"""单进程内存会话；每个 WS 只有一个写入任务，慢客户端不会拖慢批改。"""

import asyncio
import copy
import time
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import WebSocket


@dataclass(eq=False)
class Peer:
    ws: WebSocket
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=512))


class Session:
    def __init__(self, settings):
        self.settings = settings
        self.peers: set[Peer] = set()
        self.reset()

    def reset(self):
        self.data = {
            "type": "session.snapshot",
            "mode": {
                "auto_capture": self.settings.auto_capture,
                "auto_grade": self.settings.auto_grade,
                "voice_mode": self.settings.voice_mode,
            },
            "turn_seq": 0,
            "active_capture_id": None,
            "selected_question_uid": None,
            "recording": {"enabled": self.settings.recording_enabled, "active": False, "status": "idle"},
            "captures": {},
            "questions": {},
            "threads": {"all": []},
            "events": [],
            "study_started_at": time.time(),
        }

    def snapshot(self):
        return copy.deepcopy(self.data)

    def connect(self, ws):
        peer = Peer(ws)
        peer.queue.put_nowait(self.snapshot())
        self.peers.add(peer)
        return peer

    async def sender(self, peer):
        try:
            while True:
                data = await peer.queue.get()
                async with asyncio.timeout(3):
                    if isinstance(data, bytes):
                        await peer.ws.send_bytes(data)
                    else:
                        await peer.ws.send_json(data)
        finally:
            self.peers.discard(peer)

    async def emit(self, event):
        if isinstance(event, dict):
            self._apply(event)
        for peer in tuple(self.peers):
            try:
                peer.queue.put_nowait(copy.deepcopy(event) if isinstance(event, dict) else event)
            except asyncio.QueueFull:
                self.peers.discard(peer)
                asyncio.create_task(peer.ws.close(code=1013, reason="客户端接收过慢，请重新连接"))

    def _apply(self, event):
        d, typ = self.data, event["type"]
        if typ == "recording.state":
            d["recording"] = {k: v for k, v in event.items() if k != "type"}
        elif typ == "capture.reused":
            d["active_capture_id"] = event["capture_id"]
            d["selected_question_uid"] = None
        elif typ == "capture.taken":
            d["active_capture_id"] = event["capture_id"]
            d["selected_question_uid"] = None
            d["captures"][event["capture_id"]] = {k: v for k, v in event.items() if k != "type"}
        elif typ == "page.matched":
            cap = d["captures"][event["capture_id"]]
            cap.update({k: v for k, v in event.items() if k not in ("type", "questions")})
            cap["question_uids"] = [q["question_uid"] for q in event["questions"]]
            for q in event["questions"]:
                d["questions"][q["question_uid"]] = {**q, "explain_text": "", "board_steps": []}
        elif typ.startswith("question.") and "question_uid" in event:
            q = d["questions"].get(event["question_uid"])
            if q is None or q["capture_id"] != event.get("capture_id"):
                return
            if typ == "question.explain_delta":
                q["explain_text"] = q.get("explain_text", "") + event["delta"]
            elif typ == "question.explain_done":
                q.update(explain_text=event["full_text"], board_steps=event["board_steps"])
            else:
                q.update({k: v for k, v in event.items() if k != "type"})
        elif typ == "page.grade_done":
            if event["capture_id"] in d["captures"]:
                d["captures"][event["capture_id"]]["grade_result"] = event
        elif typ == "timeline.event":
            d["events"].append({k: v for k, v in event.items() if k != "type"})
            d["events"] = d["events"][-200:]
        elif typ in ("chat.user_message", "asr.final", "chat.message_start"):
            key = event.get("question_uid") if event.get("scope") == "question" else "all"
            msg = {
                **event,
                "role": event.get("role", "user"),
                "text": event.get("text", ""),
                "interrupted": False,
                "is_barge_in": event.get("is_barge_in", False),
            }
            d["threads"].setdefault(key, []).append(msg)
            d["threads"][key] = d["threads"][key][-100:]
        elif typ in ("chat.delta", "chat.message_done"):
            for messages in d["threads"].values():
                for msg in reversed(messages):
                    if msg["msg_id"] == event["msg_id"]:
                        if typ == "chat.delta":
                            msg["text"] += event["delta"]
                        else:
                            msg.update(
                                text=event["full_text"],
                                interrupted=event["interrupted"],
                                board_steps=event.get("board_steps"),
                            )
                        return
        elif typ == "voice.interrupted":
            d["turn_seq"] = event["new_turn_seq"]

    async def timeline(self, kind, text):
        await self.emit(
            {
                "type": "timeline.event",
                "id": "ev_" + uuid4().hex[:10],
                "time": time.strftime("%H:%M"),
                "kind": kind,
                "text": text,
            }
        )

    async def error(self, code, message, **context):
        await self.emit({"type": "error", "code": code, "message": message, "recoverable": True, **context})
