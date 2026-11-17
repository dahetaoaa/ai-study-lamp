// ?mock=1 回放：不连后端，按虚拟时间轴推送一套契约事件（与 backend/app/engine.py、voice/controller.py 的推送顺序一致）：
// 稳帧 → 抓拍 → 页面匹配 → 逐题识别/判定/讲解流式（含本地秒判、云端判错、未作答、云端超时）→ 汇总 →
// 选中错题 → 语音追问一轮 → 第二轮中途一次附和（duck 不打断）+ 一次真打断 → 打断后的新一轮。
// 也响应界面操作：question.select / session.set_mode / chat.send / chat.stop。
// 控制台可用 __mock.skip(ms) 快进、__mock.time() 查看虚拟时间。

import CAP from '../mock/capture-data.js';
import { applyServerEvent, emptyData } from './store.js';

const clone = (o) => JSON.parse(JSON.stringify(o));
const hm = (d = new Date()) => String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
const hms = (d = new Date()) => hm(d) + ':' + String(d.getSeconds()).padStart(2, '0');

/* ---------- 脚本内容 ---------- */
const Q = Object.fromEntries(CAP.questions.map((q) => [q.no, q]));
const publicQ = (q) => Object.fromEntries(Object.entries(q).filter(([k]) => !k.startsWith('_')));

const SCRIPT = {
  2: {
    student_display: 'C. $x_1=x_2=1$',
    error_note: '漏了负根',
    error_step_desc: '开平方只取了正根，丢了 $x=-1$',
    step_diagnosis: { orig_expr: 'x^2=1', orig_roots: ['1', '-1'], operation: '只取正的平方根', next_expr: 'x=1', next_roots: ['1'], lost_roots: ['-1'], summary: '解集变小 → 丢根' },
    explain: '移项得 $x^2=1$。开平方时要注意：平方等于 1 的数有两个，$1$ 和 $-1$。你选的 C 是 $x_1=x_2=1$，把 $x=-1$ 这个根漏掉了，而且两个根并不相等。正确答案是 A：$x_1=1,\\ x_2=-1$。',
    board: [
      { latex: 'x^2-1=0', label: '原方程' },
      { latex: 'x^2=1', label: '移项' },
      { latex: 'x=\\pm 1', label: '直接开平方' },
      { latex: 'x_1=1,\\; x_2=-1', label: '写全两个根' },
    ],
  },
  4: {
    explain: '这道题还没作答，我们一起看：13 个班成绩各不相同，从高到低排好后，第 7 名正好在中间，也就是这组数据的**中位数**。前 6 名都比中位数高，所以小林只要把自己班的成绩和中位数比一比，就知道能不能进决赛。答案选 A。',
  },
  6: {
    student_display: 'A. $x=-1$',
    error_note: '代入 x=1 试试',
    explain: '把 $x=1$ 代入方程左边，得到 $a\\cdot1^2+b\\cdot1+c=a+b+c$。题目告诉我们 $a+b+c=0$，所以 $x=1$ 一定满足方程，答案是 C。你选的 A 是 $x=-1$，代入得到的是 $a-b+c$，题目并没有说它等于 0。',
    board: [
      { latex: 'x=1', label: '代入' },
      { latex: 'a\\cdot 1^2+b\\cdot 1+c=a+b+c', label: '化简' },
      { latex: 'a+b+c=0', label: '已知条件' },
      { latex: '\\therefore\\ x=1\\ \\text{是方程的根}', label: '结论' },
    ],
  },
};

const REPLY1 = '$x=1$ 确实是方程的一个根，但不是全部。$x^2=1$ 问的是“哪个数的平方等于 1”：$1^2=1$，$(-1)^2=1$，所以 $x=-1$ 也是根。C 写成 $x_1=x_2=1$，等于说两个根都是 1，漏掉了 $-1$。';
const REPLY2 = '不一定，要看等号右边的数。右边是正数，比如 $x^2=4$，有两个互为相反数的根 $x=\\pm 2$；右边是 $0$，比如 $x^2=0$，两个根相等，都是 $0$；右边是负数的时候，情况就不一样了，因为任何实数的平方都不会小于零，所以这时候方程在实数范围内……';
const REPLY3 = '对！任何实数的平方都不会是负数，所以 $x^2=a$ 在 $a<0$ 时没有实数根。第 5 题的 B 选项 $1+x^2=0$ 就是这种情况，那道题你判断对了。';

/* 可复现的伪随机 */
function rng(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function chunks(text, seed) {
  const r = rng(seed), out = [];
  for (let i = 0; i < text.length;) {
    const n = 2 + Math.floor(r() * 4);
    out.push(text.slice(i, i + n));
    i += n;
  }
  return out;
}

/** 合成一段很轻的“说话感”音频（24kHz PCM16），用来跑通下行播放链路。 */
function ttsFrame(seq, ms, phase, flags = 0) {
  const n = Math.round((24000 * ms) / 1000);
  const buf = new ArrayBuffer(4 + n * 2);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x02);
  dv.setUint8(1, flags);
  dv.setUint16(2, seq, true);
  const pcm = new Int16Array(buf, 4, n);
  for (let i = 0; i < n; i++) {
    const t = phase + i / 24000;
    const env = Math.max(0, Math.sin(2 * Math.PI * 3.6 * t)) ** 1.5;
    const f = 165 + 25 * Math.sin(2 * Math.PI * 0.8 * t);
    pcm[i] = Math.round(1100 * env * (Math.sin(2 * Math.PI * f * t) + 0.35 * Math.sin(4 * Math.PI * f * t)));
  }
  return buf;
}
function endFrame(seq) {
  const buf = new ArrayBuffer(4);
  const dv = new DataView(buf);
  dv.setUint8(0, 0x02);
  dv.setUint8(1, 0x01);
  dv.setUint16(2, seq, true);
  return buf;
}

export class MockTransport {
  constructor({ onJSON, onBinary, onState, speed = 1 }) {
    this.onJSON = onJSON;
    this.onBinary = onBinary;
    this.onState = onState;
    this.speed = Math.max(0.1, speed);
    this.queue = [];
    this.timer = 0;
    this.t0 = 0;
    this.offset = 0;
    this.turn = null; // 当前流式回答 {seq, msgId, cancelled, text}
    this.msgN = 0;
    this.d = emptyData();
    window.__mock = { skip: (ms) => this.skip(ms), time: () => Math.round(this.vt()), transport: this };
  }

  get open() { return true; }

  /* ---------- 虚拟时间调度 ---------- */
  vt() { return (performance.now() - this.t0) * this.speed + this.offset; }
  at(t, fn) {
    let i = this.queue.length;
    while (i > 0 && this.queue[i - 1].t > t) i--;
    this.queue.splice(i, 0, { t, fn });
    this.pump();
  }
  after(dt, fn) { this.at(this.vt() + dt, fn); }
  pump() {
    clearTimeout(this.timer);
    const now = this.vt();
    while (this.queue.length && this.queue[0].t <= now) {
      const { fn } = this.queue.shift();
      try { fn(); } catch (e) { console.error('[mock]', e); }
    }
    if (this.queue.length) this.timer = setTimeout(() => this.pump(), Math.max(0, (this.queue[0].t - this.vt()) / this.speed));
  }
  skip(ms) { this.offset += ms; this.pump(); }

  /* ---------- 服务端推送 ---------- */
  emit(ev) {
    if (ev.type !== 'session.snapshot') applyServerEvent(this.d, ev);
    this.onJSON(clone(ev));
  }
  snapshot() { this.emit(clone({ ...this.d, type: 'session.snapshot' })); }
  bin(buf) { this.onBinary(buf); }
  timeline(kind, text) {
    this.emit({ type: 'timeline.event', id: 'ev_' + Math.random().toString(36).slice(2, 10), time: hm(), kind, text });
  }

  /* ---------- 连接 ---------- */
  connect() {
    this.onState('mock');
    this.t0 = performance.now();
    this.d.study_started_at = Date.now() / 1000 - 32 * 60;
    this.d.events.push(
      { id: 'ev_m1', time: hm(new Date(Date.now() - 32 * 60000)), kind: 'detect', text: '台灯已开启 · 回放模式（契约事件脚本）' },
    );
    this.at(0, () => this.snapshot());
    this.script();
  }

  /* ---------- 客户端事件 ---------- */
  send(obj) {
    const d = this.d;
    switch (obj.type) {
      case 'question.select': {
        const uid = obj.question_uid;
        if (uid != null && !d.questions[uid]) {
          this.emit({ type: 'error', code: 'UNKNOWN_QUESTION', message: '题目不存在', recoverable: true });
          break;
        }
        d.selected_question_uid = uid;
        if (uid) d.active_capture_id = d.questions[uid].capture_id;
        this.after(30, () => this.snapshot());
        break;
      }
      case 'session.set_mode':
        for (const k of ['auto_capture', 'auto_grade']) if (typeof obj[k] === 'boolean') d.mode[k] = obj[k];
        if (['vad', 'ptt', 'off'].includes(obj.voice_mode)) d.mode.voice_mode = obj.voice_mode;
        this.after(30, () => this.snapshot());
        break;
      case 'chat.send': {
        const all = Object.values(d.threads).flat();
        if (obj.client_msg_id && all.some((m) => m.client_msg_id === obj.client_msg_id)) break;
        const scope = obj.scope === 'question' && d.questions[obj.question_uid] ? 'question' : 'all';
        const uid = scope === 'question' ? obj.question_uid : null;
        this.after(60, () => {
          const seq = this.beginTurn();
          this.emit({ type: 'chat.user_message', msg_id: this.mid(), client_msg_id: obj.client_msg_id, scope, question_uid: uid, text: String(obj.text || '').slice(0, 8000) });
          this.answer(this.vt() + 300, seq, scope, uid, this.freeReply(uid), 30);
        });
        break;
      }
      case 'chat.stop':
        if (obj.turn_seq === d.turn_seq && this.turn && !this.turn.cancelled) this.interrupt('stop');
        break;
      case 'camera.capture':
      case 'session.grade_current':
      case 'session.reset':
        this.after(50, () => this.emit({ type: 'error', code: 'MOCK_MODE', message: '回放模式不连接相机与后端，此操作仅在真实环境可用', recoverable: true }));
        break;
      default:
        break; // voice.ptt / voice.playback_state 等：回放模式忽略
    }
    return true;
  }
  sendBinary() { return true; }

  mid() { return 'm_mock' + ++this.msgN; }

  /** 与 controller._chat 一致：有进行中的回答先打断；否则 turn_seq + 1。 */
  beginTurn() {
    if (this.turn && !this.turn.cancelled && !this.turn.done) this.interrupt('barge_in');
    else this.d.turn_seq = (this.d.turn_seq + 1) % 65536;
    return this.d.turn_seq;
  }

  interrupt(reason) {
    const old = this.d.turn_seq, nw = (old + 1) % 65536;
    const turn = this.turn;
    this.emit({ type: 'voice.interrupted', old_turn_seq: old, new_turn_seq: nw, reason });
    if (turn && !turn.cancelled && !turn.done) {
      turn.cancelled = true;
      this.emit({ type: 'chat.message_done', msg_id: turn.msgId, turn_seq: turn.seq, full_text: turn.text, interrupted: true, board_steps: null });
      if (turn.ttsStarted) this.emit({ type: 'tts.end', turn_seq: turn.seq, interrupted: true });
    }
    return nw;
  }

  /** 一轮流式回答：message_start → tts.start → delta + 音频帧 → message_done → 终止帧 → tts.end */
  answer(t0, seq, scope, uid, text, cps = 26) {
    const turn = { seq, msgId: this.mid(), cancelled: false, done: false, text: '', ttsStarted: false };
    this.at(t0, () => {
      this.turn = turn;
      this.emit({ type: 'chat.message_start', msg_id: turn.msgId, turn_seq: seq, scope, question_uid: uid, role: 'assistant' });
    });
    const parts = chunks(text, seq * 97 + text.length);
    let t = t0 + 350;
    this.at(t0 + 300, () => {
      if (turn.cancelled || this.d.turn_seq !== seq) return;
      turn.ttsStarted = true;
      this.emit({ type: 'tts.start', turn_seq: seq, msg_id: turn.msgId, sample_rate: 24000, format: 'pcm_s16le' });
    });
    for (const p of parts) {
      this.at(t, () => {
        if (turn.cancelled || this.d.turn_seq !== seq) return;
        turn.text += p;
        this.emit({ type: 'chat.delta', msg_id: turn.msgId, turn_seq: seq, delta: p });
      });
      t += (p.length / cps) * 1000 * (/[，。；！？]/.test(p) ? 1.6 : 1);
    }
    // 音频：每 200ms 一块
    let phase = 0;
    for (let a = t0 + 420; a < t + 400; a += 200) {
      const ph = phase;
      phase += 0.2;
      this.at(a, () => {
        if (turn.cancelled || this.d.turn_seq !== seq) return;
        this.bin(ttsFrame(seq, 200, ph));
        if (Math.round(ph * 5) % 9 === 8) this.bin(endFrame(seq)); // 句末终止帧（flags=1，负载为空）
      });
    }
    this.at(t + 450, () => {
      if (turn.cancelled || this.d.turn_seq !== seq) return;
      turn.done = true;
      this.bin(endFrame(seq));
      this.emit({ type: 'chat.message_done', msg_id: turn.msgId, turn_seq: seq, full_text: turn.text, interrupted: false, board_steps: null });
      this.emit({ type: 'tts.end', turn_seq: seq, interrupted: false });
    });
    return t + 450;
  }

  freeReply(uid) {
    const q = uid && this.d.questions[uid];
    if (q) {
      const src = CAP.questions.find((x) => x.question_uid === uid);
      return '（回放模式的脚本回复）我们回到这道题本身：' + (src?._analysis || '先读题，找出已知条件和要求的量。') + ' 你觉得哪一步最容易出错？';
    }
    return '（回放模式的脚本回复）今晚数学第 1 页 7 道题，对了 3 道，第 2、6 题需要订正，第 4 题还没作答，第 7 题需要重拍再判。建议先把第 2 题的开平方再练一遍。';
  }

  /* ---------- 主时间轴 ---------- */
  script() {
    const cap = CAP, cidv = cap.capture_id;
    // 1) 稳帧
    this.at(200, () => this.emit({ type: 'camera.state', status: 'mock', stability: 'idle', motion_score: 6.2, progress: 0 }));
    for (let i = 1; i <= 8; i++) {
      this.at(300 + i * 200, () => this.emit({ type: 'camera.state', status: 'mock', stability: 'settling', motion_score: +(2.4 - i * 0.2).toFixed(2), progress: +(i / 8).toFixed(3) }));
    }
    this.at(2050, () => this.emit({ type: 'camera.state', status: 'mock', stability: 'captured', motion_score: 0.6, progress: 1 }));
    // 2) 抓拍 + 匹配
    this.at(2100, () => this.emit({ type: 'capture.taken', capture_id: cidv, image_url: 'mock/capture.jpg', image_width: cap.image_width, image_height: cap.image_height, ts: hms() }));
    this.at(2350, () => {
      this.emit({ ...clone(cap), questions: cap.questions.map(publicQ) });
      this.timeline('detect', `检测到 ${cap.paper_title} · 第${cap.page_no}页`);
    });
    this.at(2450, () => this.emit({ type: 'camera.state', status: 'mock', stability: 'locked', motion_score: 0.4, progress: 1 }));

    // 3) 逐题识别与判定
    const ctx = (q) => ({ capture_id: cidv, question_uid: q.question_uid });
    const ocr = (t, q, raw, low = false) => this.at(t, () => this.emit({
      type: 'question.ocr', ...ctx(q), is_blank: !raw, student_raw: raw,
      student_lines: raw ? raw.split('\n').map((l) => ({ text: l, latex: null, low_confidence: low, quad: null })) : [],
    }));
    const verdict = (t, q, status, source, extra = {}) => this.at(t, () => this.emit({
      type: 'question.verdict', ...ctx(q), status, is_blank: status === 'blank', source,
      student_display: extra.student_display ?? q._student, reference_display: q._reference, tag: q.tag, mark_pos: q.mark_pos,
      error_quad: status === 'wrong' ? q.answer_quads[0] || null : null,
      error_note: extra.error_note ?? null, error_step_desc: extra.error_step_desc ?? null, step_diagnosis: extra.step_diagnosis ?? null,
    }));
    const streamExplain = (t, q, text, cps = 34) => {
      for (const p of chunks(text, q.no.charCodeAt(0) * 31)) {
        this.at(t, () => this.emit({ type: 'question.explain_delta', ...ctx(q), delta: p }));
        t += (p.length / cps) * 1000;
      }
      return t;
    };
    const done = (t, q, full, board = []) => this.at(t, () => this.emit({ type: 'question.explain_done', ...ctx(q), full_text: full, board_steps: board }));

    // 本地秒判（客观题手写与参考答案精确匹配）：verdict + 模板解析一次性下发
    for (const [no, t] of [['1', 2700], ['3', 2860], ['5', 3020]]) {
      const q = Q[no];
      ocr(t, q, q._student);
      verdict(t + 40, q, 'right', 'local', { student_display: q._student });
      this.at(t + 60, () => this.emit({ type: 'question.explain_delta', ...ctx(q), delta: q._analysis }));
      done(t + 80, q, q._analysis);
    }
    // 云端判错 + 流式讲解
    {
      const q = Q['2'], s = SCRIPT[2];
      ocr(2780, q, 'C');
      verdict(3650, q, 'wrong', 'cloud_llm', s);
      const end = streamExplain(3700, q, s.explain);
      done(end + 60, q, s.explain, s.board);
    }
    {
      const q = Q['6'], s = SCRIPT[6];
      ocr(2940, q, 'A', true);
      verdict(3900, q, 'wrong', 'cloud_llm', s);
      const end = streamExplain(3950, q, s.explain, 30);
      done(end + 60, q, s.explain, s.board);
    }
    // 未作答：空白预判 → 云端给参考答案与解析
    {
      const q = Q['4'], s = SCRIPT[4];
      ocr(2600, q, '');
      verdict(4300, q, 'blank', 'cloud_llm', { student_display: '' });
      const end = streamExplain(4350, q, s.explain, 32);
      done(end + 60, q, s.explain);
    }
    // 云端超时：非空题保持 idle（后端静默置回，只推 error），讲解回落为模板解析
    {
      const q = Q['7'];
      ocr(3100, q, 'D');
      this.at(7600, () => {
        this.d.questions[q.question_uid].status = 'idle';
        this.emit({ type: 'error', code: 'CLOUD_LLM_TIMEOUT', message: '模型响应超时，请重试', recoverable: true, ...ctx(q) });
        this.emit({ type: 'question.explain_done', ...ctx(q), full_text: q._analysis, board_steps: [] });
      });
    }
    // 4) 汇总
    this.at(8600, () => {
      const summary = '7道计分题，对了3道，2道需订正，1道未作答。1道识别或判题失败，可重新抓拍重试。';
      this.emit({
        type: 'page.grade_done', capture_id: cidv, paper_id: cap.paper_id, page_no: cap.page_no, total: 7, right_count: 3,
        wrong_uids: [Q['2'].question_uid, Q['6'].question_uid], blank_count: 1, summary_text: summary,
        timings_ms: { capture: 9, match: 21, first_verdict: 640, all_verdicts: 2210 }, error_uids: [Q['7'].question_uid],
      });
      this.timeline('alert', summary);
    });

    // 5) 选中错题（模拟另一端/家长端点选，服务端广播 snapshot）
    const q2 = Q['2'].question_uid;
    this.at(9800, () => {
      this.d.selected_question_uid = q2;
      this.d.active_capture_id = cidv;
      this.snapshot();
    });

    // 6) 语音追问第一轮
    let t = 11000;
    const say = (t0, partials, final, msgId, uttId, barge = false) => {
      this.at(t0, () => this.emit({ type: 'vad.state', speaking: true }));
      partials.forEach((p, i) => this.at(t0 + 500 + i * 550, () => this.emit({ type: 'asr.partial', utterance_id: uttId, text: p })));
      const tEnd = t0 + 500 + partials.length * 550 + 200;
      this.at(tEnd, () => this.emit({ type: 'vad.state', speaking: false }));
      return tEnd;
    };
    let tEnd = say(t, ['为什么', '为什么 C 不对', '为什么 C 不对呢，x 等于 1'], null, null, 'u_mock1');
    let seq1;
    this.at(tEnd + 250, () => {
      seq1 = this.beginTurn();
      this.emit({ type: 'asr.final', utterance_id: 'u_mock1', msg_id: this.mid(), scope: 'question', question_uid: q2, text: '为什么 C 不对呢？x 等于 1 不是对的吗？', is_barge_in: false });
    });
    // seq 在回调里才确定；按后端规则此时为 1
    const end1 = this.answer(tEnd + 600, 1, 'question', q2, REPLY1, 24);

    // 7) 第二轮：中途一次附和（duck 不打断）+ 一次真打断
    t = end1 + 1400;
    tEnd = say(t, ['那开平方', '那开平方的时候，是不是'], null, null, 'u_mock2');
    this.at(tEnd + 250, () => {
      this.beginTurn();
      this.emit({ type: 'asr.final', utterance_id: 'u_mock2', msg_id: this.mid(), scope: 'question', question_uid: q2, text: '那开平方的时候，是不是一定有两个根？', is_barge_in: false });
    });
    const t2 = tEnd + 600;
    this.answer(t2, 2, 'question', q2, REPLY2, 18);
    // 附和：嗯嗯
    this.at(t2 + 1800, () => { this.emit({ type: 'voice.duck', on: true }); this.emit({ type: 'vad.state', speaking: true }); });
    this.at(t2 + 2200, () => this.emit({ type: 'asr.partial', utterance_id: 'u_mock3', text: '嗯嗯' }));
    this.at(t2 + 2600, () => this.emit({ type: 'vad.state', speaking: false }));
    this.at(t2 + 2800, () => this.emit({ type: 'voice.duck', on: false }));
    // 真打断
    const tb = t2 + 5200;
    this.at(tb, () => { this.emit({ type: 'voice.duck', on: true }); this.emit({ type: 'vad.state', speaking: true }); });
    this.at(tb + 450, () => this.emit({ type: 'asr.partial', utterance_id: 'u_mock4', text: '那如果' }));
    this.at(tb + 900, () => {
      this.emit({ type: 'asr.partial', utterance_id: 'u_mock4', text: '那如果右边是负数呢' });
      this.interrupt('barge_in'); // partial 出现 ≥2 个非语气词字符 → 真打断
      this.emit({ type: 'voice.duck', on: false });
      // 打断后迟到的旧轮次音频帧，前端必须丢弃
      this.bin(ttsFrame(2, 200, 0));
      this.bin(ttsFrame(2, 200, 0.2));
    });
    this.at(tb + 1500, () => this.emit({ type: 'vad.state', speaking: false }));
    this.at(tb + 1800, () => {
      this.beginTurn(); // 旧任务已结束 → turn_seq 3 → 4
      this.emit({ type: 'asr.final', utterance_id: 'u_mock4', msg_id: this.mid(), scope: 'question', question_uid: q2, text: '那如果右边是负数，是不是就没有实数根了？', is_barge_in: true });
    });
    const end3 = this.answer(tb + 2150, 4, 'question', q2, REPLY3, 24);
    this.at(end3 + 800, () => this.timeline('talk', '数学第2题答疑 · 小宇追问了开平方后根的个数'));
    void seq1;
  }
}
