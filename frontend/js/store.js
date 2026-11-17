// 会话状态：S.d 与后端 session.snapshot 结构完全一致（由 applyServerEvent 增量维护，逐行对齐 backend/app/state.py::_apply）；
// S.ui / S.cam / S.voice 为纯前端状态。

export function emptyData() {
  return {
    type: 'session.snapshot',
    mode: { auto_capture: true, auto_grade: true, voice_mode: 'vad' },
    turn_seq: 0,
    active_capture_id: null,
    selected_question_uid: null,
    captures: {},
    questions: {},
    threads: { all: [] },
    events: [],
    study_started_at: Date.now() / 1000,
  };
}

const omit = (obj, ...keys) => {
  const o = {};
  for (const k in obj) if (!keys.includes(k)) o[k] = obj[k];
  return o;
};

/** 把一条服务端事件合并进会话数据 d（与后端 Session._apply 同语义；mock 服务端也复用它）。 */
export function applyServerEvent(d, ev) {
  const typ = ev.type;
  if (typ === 'recording.state') {
    d.recording = omit(ev, 'type');
  } else if (typ === 'capture.reused') {
    d.active_capture_id = ev.capture_id;
    d.selected_question_uid = null;
  } else if (typ === 'capture.taken') {
    d.active_capture_id = ev.capture_id;
    d.selected_question_uid = null;
    d.captures[ev.capture_id] = omit(ev, 'type');
  } else if (typ === 'page.matched') {
    const cap = d.captures[ev.capture_id] || (d.captures[ev.capture_id] = { capture_id: ev.capture_id });
    Object.assign(cap, omit(ev, 'type', 'questions'));
    cap.question_uids = ev.questions.map((q) => q.question_uid);
    for (const q of ev.questions) d.questions[q.question_uid] = { ...q, explain_text: '', board_steps: [] };
  } else if (typ.startsWith('question.') && 'question_uid' in ev) {
    const q = d.questions[ev.question_uid];
    if (!q || q.capture_id !== ev.capture_id) return;
    if (typ === 'question.explain_delta') q.explain_text = (q.explain_text || '') + ev.delta;
    else if (typ === 'question.explain_done') Object.assign(q, { explain_text: ev.full_text, board_steps: ev.board_steps });
    else Object.assign(q, omit(ev, 'type'));
  } else if (typ === 'page.grade_done') {
    if (d.captures[ev.capture_id]) d.captures[ev.capture_id].grade_result = ev;
  } else if (typ === 'timeline.event') {
    d.events.push(omit(ev, 'type'));
    d.events = d.events.slice(-200);
  } else if (typ === 'chat.user_message' || typ === 'asr.final' || typ === 'chat.message_start') {
    const key = ev.scope === 'question' ? ev.question_uid : 'all';
    const msg = { ...ev, role: ev.role || 'user', text: ev.text || '', interrupted: false, is_barge_in: ev.is_barge_in || false };
    (d.threads[key] || (d.threads[key] = [])).push(msg);
    d.threads[key] = d.threads[key].slice(-100);
  } else if (typ === 'chat.delta' || typ === 'chat.message_done') {
    for (const messages of Object.values(d.threads)) {
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.msg_id === ev.msg_id) {
          if (typ === 'chat.delta') msg.text += ev.delta;
          else Object.assign(msg, { text: ev.full_text, interrupted: ev.interrupted, board_steps: ev.board_steps ?? null });
          return;
        }
      }
    }
  } else if (typ === 'voice.interrupted') {
    d.turn_seq = ev.new_turn_seq;
  }
}

export const S = {
  d: emptyData(),
  gotSnapshot: false,
  conn: 'connecting', // connecting | open | closed | mock
  mock: false,
  ui: {
    view: null, // 中栏显示的抓拍 capture_id；null = 实时画面
    nav: 'hw',
    pane: 'cam',
    failed: new Set(), // 云端/识别失败而保持 idle 的题
    explainDone: new Set(),
    explainAt: new Map(), // uid -> 最近一次 explain_delta 时间
    streamingMsgs: new Set(),
    pending: [], // 乐观显示的文字提问 {client_msg_id, key, text}
    animate: new Set(), // 下一次绘制需要播放笔迹动画的题
    fresh: new Set(), // 左栏新出现的题
    det: null, // 刚匹配的 capture_id（画角框动画）
    unmatched: new Map(), // capture_id -> reason
  },
  cam: { status: null, stability: 'idle', progress: 0, motion_score: 0, at: 0 },
  voice: {
    speaking: false, partial: '', partialFinal: false, captionAt: 0,
    duck: false, playing: false, activeSeq: 0,
    micOn: false, holding: false,
  },
};

export const thread = (key) => S.d.threads[key] || [];
export const capOf = (uid) => S.d.captures[S.d.questions[uid]?.capture_id];

/** 计分口径与后端 engine.grade 一致：写作与听力不计。 */
export function scored(q) {
  return q && q.qtype !== 'writing' && q.qtype !== 'listening';
}

/** 题目在界面上的显示状态（后端 status + qtype + 失败标记）。 */
export function viewStatus(q) {
  if (!q) return 'none';
  if (q.qtype === 'listening') return 'listen';
  if (q.qtype === 'writing' && q.status === 'blank') return q.is_blank ? 'blank' : 'review';
  if (q.status === 'idle' && S.ui.failed.has(q.question_uid)) return 'failed';
  return q.status || 'idle';
}

export function captureStats(cap) {
  const qs = (cap?.question_uids || []).map((u) => S.d.questions[u]).filter(Boolean);
  const sc = qs.filter(scored);
  const st = { total: sc.length, right: 0, wrong: 0, blank: 0, uncertain: 0, judging: 0, idle: 0, failed: 0, review: 0, all: qs.length };
  for (const q of qs) {
    const v = viewStatus(q);
    if (v === 'review') st.review++;
    else if (!scored(q)) continue;
    else if (v in st) st[v]++;
  }
  st.done = st.right + st.wrong + st.blank + st.uncertain;
  return st;
}

export function capturesInOrder() {
  return Object.values(S.d.captures);
}
