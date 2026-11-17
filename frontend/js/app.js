// 入口：连接（WS 或 ?mock=1 回放）→ 事件进 store → 三栏增量重绘；用户操作 → 契约 JSON 事件。

import { S, applyServerEvent } from './store.js';
import { $, $$, NARROW, store, clamp, hms, cid, batcher } from './util.js';
import { WSTransport } from './ws.js';
import { AudioEngine, micFrame, micSupport } from './audio.js';
import { createStage } from './stage.js';
import { createSidebar } from './sidebar.js';
import { createPanel } from './panel.js';
import { createCaptureControls } from './capture-controls.js';

const params = new URLSearchParams(location.search);
S.mock = params.get('mock') === '1';

/* ---------- toast ---------- */
let toastTimer = 0;
function toast(msg, kind) {
  const t = $('#toast');
  $('#toastT').textContent = msg;
  t.className = 'toast' + (kind ? ' ' + kind : '');
  t.hidden = false;
  t.style.animation = 'none';
  void t.offsetWidth;
  t.style.animation = '';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 3800);
}

/* ---------- app 上下文（传给各栏模块） ---------- */
const app = {
  liveSrc: S.mock ? 'mock/capture.jpg' : 'api/camera/stream',
  micError: '',
  toast,
  narrow: () => NARROW.matches,
  micSupport,
  send(obj) {
    const ok = transport.send(obj);
    if (!ok && S.conn !== 'mock') toast('与台灯的连接已断开，正在重连…', 'alert');
    return ok;
  },
  select(uid, o = {}) {
    if (uid && !S.d.questions[uid]) return;
    // 乐观更新：本地立即切换，服务端随后广播 snapshot 同步多端
    S.d.selected_question_uid = uid || null;
    if (uid) {
      S.ui.view = S.d.questions[uid].capture_id;
      S.d.active_capture_id = S.ui.view;
    }
    renderAll();
    transport.send({ type: 'question.select', question_uid: uid || null });
    if (uid && o.fromList && NARROW.matches) goPane('panel');
  },
  setView(capId, o = {}) {
    S.ui.view = capId && S.d.captures[capId] ? capId : null;
    const q = S.d.questions[S.d.selected_question_uid];
    if (q && q.capture_id !== S.ui.view) {
      // 看另一页时，选中态回到全局
      S.d.selected_question_uid = null;
      transport.send({ type: 'question.select', question_uid: null });
    }
    renderAll();
    if (o.fromList && NARROW.matches) goPane('cam');
  },
  goPane: (p) => goPane(p),
  sendChat(text) {
    text = String(text || '').trim();
    if (!text) return false;
    const q = S.d.questions[S.d.selected_question_uid];
    const id = cid('c');
    const msg = { type: 'chat.send', client_msg_id: id, scope: q ? 'question' : 'all', question_uid: q ? q.question_uid : null, text, speak: true };
    if (!transport.send(msg)) { toast('与台灯的连接已断开，消息未发送', 'alert'); return false; }
    S.ui.pending.push({ client_msg_id: id, key: q ? q.question_uid : 'all', text });
    audio.unlock();
    panel.chatAdded(q ? q.question_uid : 'all');
    return true;
  },
  stopChat() {
    transport.send({ type: 'chat.stop', turn_seq: S.d.turn_seq });
    audio.clearQueue();
  },
  setVoiceMode(mode) {
    if (!['vad', 'ptt', 'off'].includes(mode)) return;
    S.d.mode = { ...S.d.mode, voice_mode: mode };
    transport.send({ type: 'session.set_mode', voice_mode: mode });
    if (mode === 'off') stopMic();
    panel.composer();
  },
  async toggleMic() {
    if (S.voice.micOn) stopMic();
    else await startMic();
  },
  async pttDown() {
    if (!(await startMic())) return;
    S.voice.holding = true;
    audio.unlock();
    transport.send({ type: 'voice.ptt', active: true });
    panel.composer();
  },
  pttUp() {
    S.voice.holding = false;
    transport.send({ type: 'voice.ptt', active: false });
    panel.composer();
  },
};

/* ---------- 语音 ---------- */
const audio = new AudioEngine({
  onPlayback(playing, seq) {
    S.voice.playing = playing;
    transport.send({ type: 'voice.playback_state', playing, turn_seq: seq });
    panel.composer();
  },
  onMicFrame(int16) {
    const mode = S.d.mode?.voice_mode;
    const live = (mode === 'vad' && S.voice.micOn) || (mode === 'ptt' && S.voice.holding);
    if (live) transport.sendBinary(micFrame(int16, S.d.turn_seq));
  },
  onNeedGesture() {
    toast('点击页面任意位置即可开启语音播放');
  },
});

async function startMic() {
  const sup = micSupport();
  if (sup !== 'ok') {
    toast(sup === 'insecure' ? '麦克风需要 HTTPS（或 localhost）才能使用' : '当前浏览器不支持麦克风采集', 'alert');
    panel.composer();
    return false;
  }
  try {
    await audio.startMic();
    S.voice.micOn = true;
    app.micError = '';
  } catch (e) {
    app.micError = e && e.name === 'NotAllowedError' ? '麦克风权限被拒绝，请在浏览器地址栏允许后重试' : '无法打开麦克风：' + (e && (e.message || e.name));
    toast(app.micError, 'alert');
    S.voice.micOn = false;
  }
  panel.composer();
  return S.voice.micOn;
}
function stopMic() {
  audio.stopMic();
  S.voice.micOn = false;
  S.voice.holding = false;
  panel.composer();
}

/* ---------- 各栏 ---------- */
const stage = createStage(app);
const sidebar = createSidebar(app);
const panel = createPanel(app);
const captureControls = createCaptureControls(app);

const renderAll = batcher(() => { sidebar.render(); stage.render(); panel.render(); captureControls.render(); });
const renderSide = batcher(() => sidebar.render());
const renderStage = batcher(() => stage.render());
const renderPanel = batcher(() => panel.render());

/* ---------- 连接状态 ---------- */
function setConn(state) {
  S.conn = state;
  const text = { connecting: '连接中', open: '已连接', closed: '重连中…', mock: '回放模式' }[state];
  $$('[data-conn-dot]').forEach((d) => { d.className = 'kdot ' + (state === 'open' ? 'ok' : state === 'mock' ? 'mock' : state === 'closed' ? 'alert' : ''); });
  $$('[data-conn-text]').forEach((t) => (t.textContent = text));
  $('#footNote').textContent = S.mock ? '回放模式 · 契约事件脚本（?mock=1）' : state === 'open' ? '已连接台灯 · 画面在本地识别' : '未连接台灯，自动重连中';
  if (state !== 'open' && state !== 'mock') {
    audio.clearQueue();
    S.voice.playing = false;
    S.voice.speaking = false;
  }
  stage.status();
  captureControls.render();
  if (state === 'open') captureControls.refresh();
  panel.head();
  panel.composer();
}

/* ---------- 服务端事件 ---------- */
function handleEvent(ev) {
  const t = ev.type;
  const q = ev.question_uid && S.d.questions[ev.question_uid];
  switch (t) {
    case 'session.snapshot': {
      const prevSel = S.d.selected_question_uid;
      const first = !S.gotSnapshot;
      S.d = ev;
      S.d.threads = S.d.threads || { all: [] };
      S.gotSnapshot = true;
      // 去掉已落库的乐观消息
      const ids = new Set(Object.values(S.d.threads).flat().map((m) => m.client_msg_id).filter(Boolean));
      S.ui.pending = S.ui.pending.filter((p) => !ids.has(p.client_msg_id));
      // 快照里没有“流式中”标记：重连后不再显示光标
      if (first) S.ui.streamingMsgs.clear();
      for (const id of [...S.ui.streamingMsgs]) {
        const m = Object.values(S.d.threads).flat().find((x) => x.msg_id === id);
        if (!m) S.ui.streamingMsgs.delete(id);
      }
      audio.setActive(ev.turn_seq);
      const sel = S.d.selected_question_uid;
      if (sel && sel !== prevSel && S.d.questions[sel]) S.ui.view = S.d.questions[sel].capture_id;
      else if (first) S.ui.view = S.d.active_capture_id && S.d.captures[S.d.active_capture_id] ? S.d.active_capture_id : null;
      if (S.ui.view && !S.d.captures[S.ui.view]) S.ui.view = null;
      if (S.d.recording?.active) S.ui.view = null;
      renderAll();
      return;
    }
    case 'camera.state':
      S.cam = { ...S.cam, ...ev, at: Date.now() };
      stage.status();
      captureControls.render();
      return;
    case 'recording.state':
      captureControls.event(ev);
      return;
    case 'capture.reused':
      applyServerEvent(S.d, ev);
      S.ui.view = S.d.recording?.active ? null : ev.capture_id;
      if (ev.source === 'manual') toast('相同页面与作答，已复用已有结果');
      renderAll();
      return;
    case 'capture.taken':
      applyServerEvent(S.d, ev);
      S.ui.view = S.d.recording?.active ? null : ev.capture_id; // 采样期间保持实时预览
      renderAll();
      setTimeout(() => sidebar.scrollToCapture(ev.capture_id), 50);
      return;
    case 'page.matched':
      applyServerEvent(S.d, ev);
      ev.questions.forEach((x) => S.ui.fresh.add(x.question_uid));
      if (S.ui.view === ev.capture_id) S.ui.det = ev.capture_id;
      renderAll();
      return;
    case 'page.unmatched':
      S.ui.unmatched.set(ev.capture_id, ev.reason);
      toast(ev.reason === 'blurry' ? '画面不够清晰，请稳住纸张后重拍' : '未匹配到已入库试卷，请调整纸张位置后重拍', 'alert');
      renderAll();
      return;
    case 'question.ocr':
      applyServerEvent(S.d, ev);
      if (S.d.selected_question_uid === ev.question_uid) renderPanel();
      return;
    case 'question.verdict':
      applyServerEvent(S.d, ev);
      if (S.d.questions[ev.question_uid]) {
        S.ui.animate.add(ev.question_uid);
        S.ui.failed.delete(ev.question_uid);
      }
      renderAll();
      return;
    case 'question.explain_delta':
      applyServerEvent(S.d, ev);
      S.ui.explainAt.set(ev.question_uid, Date.now());
      S.ui.explainDone.delete(ev.question_uid);
      panel.explainDelta(ev.question_uid);
      return;
    case 'question.explain_done':
      applyServerEvent(S.d, ev);
      S.ui.explainDone.add(ev.question_uid);
      if (S.d.selected_question_uid === ev.question_uid) renderPanel();
      return;
    case 'page.grade_done':
      applyServerEvent(S.d, ev);
      renderSide();
      if (!S.d.selected_question_uid) renderPanel();
      return;
    case 'timeline.event':
      applyServerEvent(S.d, ev);
      S.ui.freshEvent = ev.id;
      if (!S.d.selected_question_uid) renderPanel();
      if (S.ui.nav === 'report') renderSide();
      return;
    case 'vad.state':
      S.voice.speaking = !!ev.speaking;
      if (ev.speaking) { S.voice.partial = ''; S.voice.partialFinal = false; }
      else S.voice.captionAt = Date.now();
      stage.caption();
      panel.composer();
      return;
    case 'asr.partial':
      S.voice.partial = ev.text || '';
      S.voice.captionAt = Date.now();
      stage.caption();
      panel.composer();
      return;
    case 'asr.final':
    case 'chat.user_message': {
      applyServerEvent(S.d, ev);
      if (t === 'asr.final') { S.voice.partial = ev.text || ''; S.voice.partialFinal = true; S.voice.captionAt = Date.now(); stage.caption(); }
      if (ev.client_msg_id) S.ui.pending = S.ui.pending.filter((p) => p.client_msg_id !== ev.client_msg_id);
      panel.chatAdded(ev.scope === 'question' ? ev.question_uid : 'all');
      return;
    }
    case 'chat.message_start':
      S.voice.pendingTTS = !!ev.speak;
      applyServerEvent(S.d, ev);
      S.ui.streamingMsgs.add(ev.msg_id);
      if (typeof ev.turn_seq === 'number') { S.d.turn_seq = ev.turn_seq; audio.setActive(ev.turn_seq); }
      panel.chatAdded(ev.scope === 'question' ? ev.question_uid : 'all');
      return;
    case 'chat.delta':
      applyServerEvent(S.d, ev);
      panel.chatUpdate(ev.msg_id);
      return;
    case 'chat.message_done':
      applyServerEvent(S.d, ev);
      S.ui.streamingMsgs.delete(ev.msg_id);
      panel.chatUpdate(ev.msg_id);
      return;
    case 'tts.start':
      if (typeof ev.turn_seq === 'number' && ev.turn_seq === S.d.turn_seq) audio.setActive(ev.turn_seq);
      return;
    case 'tts.end':
      if (ev.turn_seq === S.d.turn_seq) S.voice.pendingTTS = false;
      audio.ttsEnd(ev.turn_seq);
      panel.composer();
      return;
    case 'voice.duck':
      S.voice.duck = !!ev.on;
      audio.duck(!!ev.on);
      panel.composer();
      return;
    case 'voice.interrupted':
      applyServerEvent(S.d, ev);
      audio.interrupt(ev.new_turn_seq);
      S.voice.playing = false;
      S.voice.pendingTTS = false;
      S.voice.duck = false;
      panel.composer();
      return;
    case 'error': {
      let msg = ev.message || '出错了';
      if (ev.question_uid && q) {
        // 后端在云端失败时把非空题静默置回 idle（不单独推事件），这里同步本地状态
        if (q.status === 'judging') q.status = 'idle';
        S.ui.failed.add(ev.question_uid);
        msg = `第${q.no}题：${msg.replace(/[，,]?\s*请(稍后)?重试[。.]?$/, '')}，可重拍重试`;
        renderAll();
      }
      toast(msg, 'alert');
      return;
    }
    default:
      // 未知事件忽略（向前兼容）
      if (t && S.d) applyServerEvent(S.d, ev);
  }
}

/* ---------- 传输层 ---------- */
let transport;
if (S.mock) {
  const { MockTransport } = await import('./mock.js');
  transport = new MockTransport({ onJSON: handleEvent, onBinary: (b) => audio.handleBinary(b), onState: setConn, speed: Number(params.get('speed')) || 1 });
} else {
  transport = new WSTransport({ onJSON: handleEvent, onBinary: (b) => audio.handleBinary(b), onState: setConn });
}

/* ---------- 窄屏标签页 ---------- */
function goPane(p) {
  S.ui.pane = p;
  $('#app').dataset.pane = p;
  $$('.tabbar button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.pane === p)));
  store.set('aixb-pane', p);
  if (p === 'cam') setTimeout(() => stage.reframe(true), 0);
}
$$('.tabbar button').forEach((b) => b.addEventListener('click', () => goPane(b.dataset.pane)));

/* ---------- 可拖动分栏（同原型） ---------- */
const LW = { min: 220, max: 420, def: 300 }, RW = { min: 360, max: 720, def: 440 }, MIDMIN = 480;
let lw = +store.get('aixb-lw') || LW.def, rw = +store.get('aixb-rw') || RW.def, effL = lw, effR = rw;
function applyWidths() {
  const el = $('#app');
  if (NARROW.matches) return;
  const W = el.clientWidth;
  let l = clamp(lw, LW.min, LW.max), r = clamp(rw, RW.min, RW.max);
  let over = l + r + MIDMIN - W;
  if (over > 0) { const d = Math.min(over, r - RW.min); r -= d; over -= d; }
  if (over > 0) l -= Math.min(over, l - LW.min);
  effL = l; effR = r;
  el.style.setProperty('--lw', l + 'px');
  el.style.setProperty('--rw', r + 'px');
}
function saveWidths() { store.set('aixb-lw', String(lw)); store.set('aixb-rw', String(rw)); }
function bindHandle(el, side) {
  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    try { el.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    el.classList.add('drag');
    document.body.classList.add('resizing');
    const box = $('#app').getBoundingClientRect();
    const move = (ev) => {
      if (side === 'l') lw = clamp(Math.round(ev.clientX - box.left), LW.min, Math.max(LW.min, Math.min(LW.max, box.width - effR - MIDMIN)));
      else rw = clamp(Math.round(box.right - ev.clientX), RW.min, Math.max(RW.min, Math.min(RW.max, box.width - effL - MIDMIN)));
      applyWidths();
    };
    const up = () => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
      el.classList.remove('drag');
      document.body.classList.remove('resizing');
      saveWidths();
      stage.reframe(true);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  });
  el.addEventListener('dblclick', () => { if (side === 'l') lw = LW.def; else rw = RW.def; saveWidths(); applyWidths(); });
  el.addEventListener('keydown', (e) => {
    const d = e.key === 'ArrowLeft' ? -16 : e.key === 'ArrowRight' ? 16 : 0;
    if (!d) return;
    e.preventDefault();
    if (side === 'l') lw = clamp(effL + d, LW.min, LW.max); else rw = clamp(effR - d, RW.min, RW.max);
    applyWidths();
    saveWidths();
  });
}
bindHandle($('#rzL'), 'l');
bindHandle($('#rzR'), 'r');
window.addEventListener('resize', applyWidths);
NARROW.addEventListener?.('change', () => { if (NARROW.matches) goPane(S.ui.pane || 'cam'); applyWidths(); setTimeout(() => stage.reframe(true), 0); });

/* ---------- 全局手势：解锁音频 ---------- */
const unlock = () => audio.unlock();
document.addEventListener('pointerdown', unlock, { capture: true });
document.addEventListener('keydown', (e) => {
  unlock();
  if (e.key === 'Escape' && S.d.selected_question_uid && !e.target.closest?.('textarea,input')) app.select(null);
}, { capture: true });

/* ---------- 时钟 ---------- */
function tick() {
  const now = new Date();
  $$('[data-clock]').forEach((c) => (c.textContent = hms(now)));
  $('#studyMin').textContent = Math.max(0, Math.floor((now / 1000 - (S.d.study_started_at || now / 1000)) / 60));
  if (S.cam.at && Date.now() - S.cam.at > 3000 && S.cam.stability !== 'idle') stage.status();
}
setInterval(tick, 1000);

/* ---------- 启动 ---------- */
applyWidths();
if (NARROW.matches) goPane(store.get('aixb-pane') || 'cam');
tick();
renderAll();
transport.connect();
if (document.fonts?.ready) document.fonts.ready.then(() => stage.reframe(true));

// 调试/验收入口
window.__app = { S, handleEvent, transport, audio, app };
