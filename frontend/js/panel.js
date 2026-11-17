// 右栏：未选题 = 批改进度 + 台灯动态 + 全局对话；选中题 = 原文卡片 / 题干 / 识别原文 / 判定 / 流式讲解 / 板书 / 追问对话。
// 底部输入框与语音状态条。流式文本（讲解、对话气泡）走增量更新，不整栏重绘。

import { S, thread, viewStatus, captureStats, capturesInOrder } from './store.js';
import { $, $$, esc, rich, tex, plain, subjectName, RM } from './util.js';

const IC = {
  ok: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8.5" fill="none" stroke="currentColor" stroke-width="1.5"/><path class="ln" d="M6.4 10.3l2.4 2.4 4.8-5.2" style="stroke-width:1.8"/></svg>',
  no: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8.5" fill="none" stroke="currentColor" stroke-width="1.5"/><path class="ln" d="M7.2 7.2l5.6 5.6M12.8 7.2l-5.6 5.6" style="stroke-width:1.8"/></svg>',
  blank: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8.5" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="2.5 2.5"/><path class="ln" d="M6.5 10h7" style="stroke-width:1.8"/></svg>',
  review: '<svg viewBox="0 0 20 20"><path class="ln" d="M4 15.5l1-3.5 7.8-7.8a1.8 1.8 0 0 1 2.5 2.5L7.5 14.5z M11.5 5.5l3 3" style="stroke-width:1.6"/></svg>',
  send: '<svg viewBox="0 0 20 20"><path class="ln" d="M10 15.5v-11M5.5 9 10 4.5 14.5 9" style="stroke-width:1.8"/></svg>',
  stop: '<svg viewBox="0 0 12 12"><rect x="1" y="1" width="10" height="10" rx="2" fill="currentColor"/></svg>',
};

const bars = (n = 22) => Array.from({ length: n }, (_, i) =>
  `<i style="animation-delay:${(-(i * 137 % 1000) / 1000).toFixed(2)}s;animation-duration:${(0.7 + (i * 53 % 40) / 100).toFixed(2)}s"></i>`).join('');

export function createPanel(app) {
  const body = $('#pBody'), ta = $('#ask'), sendBtn = $('#send'), mic = $('#mic'), cVoice = $('#cVoice'), hint = $('#micHint');
  let shownSel = undefined;
  let lastExplain = 0, explainTimer = 0;

  const sel = () => S.d.questions[S.d.selected_question_uid] || null;

  function streamingExplain(q) {
    if (!q || S.ui.explainDone.has(q.question_uid)) return false;
    return q.status === 'judging' || Date.now() - (S.ui.explainAt.get(q.question_uid) || 0) < 1500;
  }

  /* ---------- 头部 ---------- */
  function head() {
    const q = sel();
    $('#back').disabled = !q;
    ta.placeholder = q ? '针对这道题继续追问…' : '问问今晚整体的学习情况…';
    if (!q) {
      $('#pTitle').textContent = '台灯动态';
      $('#pMeta').textContent = S.mock ? '回放 · 契约事件' : S.conn === 'open' ? '台灯 · 实时' : '台灯 · 离线';
      return;
    }
    const c = S.d.captures[q.capture_id];
    $('#pTitle').textContent = `${subjectName(c?.subject)} 第${q.no}题`;
    const v = viewStatus(q);
    const meta = {
      judging: '判定中', idle: '待批改', failed: '判题失败 · 可重拍重试', blank: '未作答 · 参考答案',
      review: '作文点评', uncertain: '待确认 · 暂不判对错', listen: '听力 · 不批改',
      right: q.source === 'local' ? '本地秒判' : '云端判题', wrong: q.source === 'local' ? '本地秒判' : '云端判题',
    }[v] || '';
    $('#pMeta').textContent = meta;
  }

  /* ---------- 未选题：进度 + 动态 ---------- */
  function progHTML() {
    const caps = capturesInOrder().filter((c) => (c.question_uids || []).length);
    if (!caps.length) return '';
    const rows = caps.map((c) => {
      const qs = c.question_uids.map((u) => S.d.questions[u]).filter((q) => q && q.qtype !== 'listening');
      const st = captureStats(c);
      const sc = st.judging ? '判定中' : st.uncertain ? `${st.uncertain}题待确认` : st.done === st.total ? `${st.right}/${st.total}` : st.failed ? `${st.failed}题失败` : '—';
      return `<div class="prog-r"><span class="nm">${esc(subjectName(c.subject))} P${esc(c.page_no)}</span><span class="dots">` +
        qs.map((q) => `<span class="sd st-${viewStatus(q)}" title="第${esc(q.no)}题"></span>`).join('') + `</span><span class="sc">${sc}</span></div>`;
    }).join('');
    const last = [...caps].reverse().find((c) => c.grade_result);
    return `<section class="blk" id="progBlk"><div class="blk-h"><span>批改进度</span><span class="r">逐题实时</span></div><div class="prog">${rows}</div>` +
      (last ? `<p class="sum">${esc(last.grade_result.summary_text || '')}</p>` : '') + '</section>';
  }
  function evHTML(e, fresh) {
    return `<li class="k-${esc(e.kind)}${fresh ? ' fresh' : ''}"><time>${esc(e.time || '')}</time><span class="nd"></span><span class="tx">${esc(e.text)}</span></li>`;
  }

  /* ---------- 选中题 ---------- */
  function passageHTML(q) {
    if (!q.passage) return '';
    const open = S.ui.passageOpen?.has(q.passage_id) ? ' open' : '';
    const first = plain(q.passage).slice(0, 36);
    return `<section class="blk"><details class="passage" data-pid="${esc(q.passage_id || '')}"${open}><summary>原文<span class="muted">${esc(first)}…</span><span class="r">${q.passage.length}字</span></summary>` +
      `<div class="ptx">${rich(q.passage)}</div></details></section>`;
  }
  function stemHTML(q) {
    const opts = Array.isArray(q.options) && q.options.length ? `<div class="opts">${q.options.map((o) => `<span>${rich(o)}</span>`).join('')}</div>` : '';
    const tag = q.qtype === 'listening' ? '听力 · 不批改' : q.qtype === 'writing' ? '书面表达' : q.tag || '';
    return `<section class="blk"><div class="blk-h"><span>题目</span><span class="r">${esc(tag)}</span></div><div class="stem">${rich(q.stem || q.short_stem || '')}</div>${opts}</section>`;
  }
  function ocrHTML(q) {
    if (q.qtype === 'listening') return '';
    let inner;
    if (q.student_lines || q.student_raw !== undefined) {
      if (q.is_blank) inner = '<p class="muted">答题区是空白的，没有识别到手写内容。</p>';
      else {
        const lines = q.student_lines && q.student_lines.length ? q.student_lines : [{ text: q.student_raw || '', latex: null, low_confidence: false }];
        const low = lines.some((l) => l.low_confidence);
        inner = '<div class="ocr">' + lines.map((l) => {
          const v = l.latex ? tex(l.latex) : rich(l.text);
          return `<div class="o-row"><span class="o-k">手写</span><span class="o-v h"><span class="hwv">${l.low_confidence ? `<mark class="lc">${v}</mark>` : v}</span></span></div>`;
        }).join('') + (low ? '<div class="o-row"><span></span><span class="lc-note">黄色部分识别不确定，请孩子确认</span></div>' : '') + '</div>';
      }
    } else if (q.status === 'judging') inner = '<p class="vd wait"><span class="sd st-judging"></span>正在识别手写…</p>';
    else inner = '<p class="muted">还没有识别这道题的手写内容。</p>';
    return `<section class="blk"><div class="blk-h"><span>识别原文</span><span class="r">浅底为手写</span></div>${inner}</section>`;
  }
  function kv(rows) {
    return '<dl class="kv">' + rows.filter(Boolean).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('') + '</dl>';
  }
  function verdictHTML(q) {
    const v = viewStatus(q);
    const tag = q.tag ? `<span class="tag">${esc(q.tag)}</span>` : '';
    const ref = !q.model_answer && q.reference_display ? rich(q.reference_display) : '';
    const ai = q.model_answer ? ['AI解答', rich(q.model_answer)] : null;
    let h;
    if (v === 'listen') h = '<p class="vd wait">听力题不批改，也不在图上画标记。</p>';
    else if (v === 'judging') h = '<p class="vd wait"><span class="sd st-judging"></span>判定中…</p>';
    else if (v === 'failed') h = '<p class="vd fail">识别或判题服务暂时失败，可以重拍这一页重试</p>' + kv([tag && ['考点', tag]]);
    else if (v === 'uncertain') h = '<p class="vd wait">待确认 · 暂不判对错</p>' + kv([
      ['识别内容', rich(q.student_display || q.student_raw || '')],
      q.error_note && ['需确认', rich(q.error_note)], ai]);
    else if (v === 'idle') h = '<p class="vd wait"><span class="sd st-idle"></span>已识别，等待批改</p>';
    else if (v === 'right') h = `<p class="vd ok">${IC.ok}正确</p>` + kv([['学生答案', rich(q.student_display || q.student_raw || '')], ref && ['正确答案', ref], ai, tag && ['考点', tag]]);
    else if (v === 'wrong') h = `<p class="vd no">${IC.no}错误</p>` + kv([
      ['学生答案', `<s>${rich(q.student_display || q.student_raw || '')}</s>`], ref && ['正确答案', ref], ai,
      q.error_step_desc && ['出错步骤', rich(q.error_step_desc)], q.error_note && ['批注', `<span style="color:var(--red)">${rich(q.error_note)}</span>`], tag && ['考点', tag]]);
    else if (v === 'blank') h = `<p class="vd blank">${IC.blank}未作答</p>` + kv([ref && ['参考答案', ref], ai, tag && ['考点', tag]]);
    else if (v === 'review') h = `<p class="vd review">${IC.review}作文点评</p>` + kv([['说明', '<span class="muted">作文按要点点评，不判对错，不计入统计</span>'], tag && ['考点', tag]]);
    else h = '<p class="vd wait">等待识别</p>';
    return `<section class="blk" id="vdBlk"><div class="blk-h"><span>判定</span><span class="r">${q.source === 'local' ? '本地规则' : q.source === 'cloud_llm' ? '云端模型' : ''}</span></div>${h}</section>`;
  }
  function diagHTML(q) {
    const d = q.step_diagnosis;
    if (!d || viewStatus(q) !== 'wrong') return '';
    const set = (els, lost) => '<span class="set"><span class="bk">{</span>' +
      [...(lost || []).map((e) => `<span class="el lost">${esc(e)}</span>`), ...(els || []).map((e) => `<span class="el">${esc(e)}</span>`)].join('<span class="cm">,</span>') + '<span class="bk">}</span></span>';
    return '<section class="blk"><div class="blk-h"><span>步骤诊断</span><span class="r">前后解集对比</span></div><div class="dg">' +
      (d.orig_expr ? `<div class="dg-row"><span class="dg-l">原式</span><span class="dg-e">${tex(d.orig_expr)}</span>${set(d.orig_roots)}</div>` : '') +
      (d.operation ? `<div class="dg-op"><span class="rail"></span><span>${rich(d.operation)}</span></div>` : '') +
      (d.next_expr ? `<div class="dg-row bad"><span class="dg-l">下一步</span><span class="dg-e">${tex(d.next_expr)}</span>${set(d.next_roots, d.lost_roots)}</div>` : '') +
      (d.summary ? `<div class="dg-res"><b>${rich(d.summary)}</b></div>` : '') + '</div></section>';
  }
  function explainInner(q) {
    const streaming = streamingExplain(q);
    const t = q.explain_text || '';
    if (!t) return streaming ? '<span class="muted">正在生成讲解</span><span class="caret"></span>' : '<span class="muted">判定完成后，这里会给出讲解。</span>';
    return rich(t) + (streaming ? '<span class="caret"></span>' : '');
  }
  function explainHTML(q) {
    if (q.qtype === 'listening') return '';
    const v = viewStatus(q);
    const title = { wrong: 'AI讲解', right: '解析', blank: '参考答案与解析', review: '作文点评', failed: '参考解析' }[v] || 'AI讲解';
    return `<section class="blk" id="exBlk"><div class="blk-h"><span>${title}</span><span class="r">${v === 'wrong' ? '云端讲解 · 流式' : ''}</span></div><div class="explain" id="explain">${explainInner(q)}</div></section>`;
  }
  function boardHTML(steps) {
    if (!Array.isArray(steps) || !steps.length) return '';
    return '<section class="blk"><div class="blk-h"><span>分步板书</span></div><div class="board">' +
      steps.map((b, i) => `<div class="bd"><span class="n">${i + 1}</span><span class="m">${b.latex ? tex(b.latex) : rich(b.text || '')}</span><span class="l">${esc(b.label || '')}</span></div>`).join('') + '</div></section>';
  }

  /* ---------- 对话 ---------- */
  function msgHTML(m) {
    if (m.role === 'user') {
      const viaVoice = !!m.utterance_id;
      return `<div class="m-kid${m.pending ? ' pending' : ''}" data-mid="${esc(m.msg_id || m.client_msg_id)}"><span class="tx">${esc(m.text)}</span>` +
        (m.is_barge_in ? '<span class="chip red">打断</span>' : '') + (viaVoice && !m.is_barge_in ? '<span class="chip">语音</span>' : '') + '</div>';
    }
    const streaming = S.ui.streamingMsgs.has(m.msg_id);
    return `<div class="m-ai${!m.text && streaming ? ' pending' : ''}" data-mid="${esc(m.msg_id)}">${aiInner(m, streaming)}</div>`;
  }
  function aiInner(m, streaming) {
    const t = m.text ? rich(m.text) : streaming ? '思考中…' : m.interrupted ? '……' : '';
    return `<div class="tx">${t}</div>${streaming ? '<span class="caret"></span>' : ''}${m.interrupted ? '<span class="chip">被打断</span>' : ''}` +
      (Array.isArray(m.board_steps) && m.board_steps.length ? `<div class="board">${m.board_steps.map((b, i) => `<div class="bd"><span class="n">${i + 1}</span><span class="m">${tex(b.latex || '')}</span><span class="l">${esc(b.label || '')}</span></div>`).join('')}</div>` : '');
  }
  function threadHTML(key) {
    const msgs = thread(key).concat(S.ui.pending.filter((p) => p.key === key).map((p) => ({ role: 'user', text: p.text, client_msg_id: p.client_msg_id, pending: true })));
    const ph = key === 'all' ? '可以问问今晚整体的学习情况' : '对这道题有疑问？直接说或在下面输入';
    return `<section class="blk" id="chatBlk"><div class="blk-h"><span>对话</span><span class="r">${key === 'all' ? '今晚的学习情况' : '追问 · 可随时打断'}</span></div>` +
      `<div class="chat" id="thread" data-key="${esc(key)}" data-ph="${ph}">${msgs.map(msgHTML).join('')}</div></section>`;
  }

  /* ---------- 渲染 ---------- */
  function render() {
    head();
    const q = sel();
    const same = shownSel === (q ? q.question_uid : null);
    const top = body.scrollTop;
    const nearBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 60;
    let h = '';
    if (!q) {
      h += progHTML();
      const evs = S.d.events.slice().reverse();
      h += `<section class="blk"><div class="blk-h"><span>今晚</span><span class="r">最新在上</span></div>` +
        (evs.length ? `<ol class="tl" id="tl">${evs.map((e, i) => evHTML(e, i === 0 && S.ui.freshEvent === e.id)).join('')}</ol>` : '<p class="muted">台灯已就绪。把作业纸放到台灯下就会自动开始。</p>') + '</section>';
      if (thread('all').length || S.ui.pending.some((p) => p.key === 'all')) h += threadHTML('all');
    } else {
      h += passageHTML(q) + stemHTML(q) + ocrHTML(q) + verdictHTML(q) + diagHTML(q) + explainHTML(q) + boardHTML(q.board_steps) + threadHTML(q.question_uid);
    }
    S.ui.freshEvent = null;
    body.innerHTML = h;
    if (!same) body.scrollTop = 0;
    else body.scrollTop = nearBottom && streamingAny() ? body.scrollHeight : top;
    shownSel = q ? q.question_uid : null;
    composer();
  }

  function streamingAny() {
    return S.ui.streamingMsgs.size > 0 || (sel() && streamingExplain(sel()));
  }

  function followBottom(el, force) {
    if (!el) return;
    const b = body.getBoundingClientRect(), r = el.getBoundingClientRect();
    if (r.bottom > b.bottom - 16 && (force || r.top < b.bottom + 200)) body.scrollTo({ top: body.scrollTop + r.bottom - b.bottom + 32, behavior: RM ? 'auto' : 'smooth' });
  }

  /** question.explain_delta：只更新讲解正文（节流约 30fps）。 */
  function explainDelta(uid) {
    const q = sel();
    if (!q || q.question_uid !== uid) return;
    const now = Date.now();
    const run = () => {
      lastExplain = Date.now();
      explainTimer = 0;
      const el = $('#explain');
      if (!el) return render();
      el.innerHTML = explainInner(sel());
      followBottom(el);
    };
    if (now - lastExplain > 33) run();
    else if (!explainTimer) explainTimer = setTimeout(run, 33);
  }

  /** chat.delta / message_done：只更新对应气泡。 */
  function chatUpdate(msgId) {
    const el = body.querySelector(`[data-mid="${CSS.escape(msgId)}"]`);
    const m = Object.values(S.d.threads).flat().find((x) => x.msg_id === msgId);
    if (!el || !m) return render();
    el.classList.toggle('pending', !m.text && S.ui.streamingMsgs.has(msgId));
    el.innerHTML = aiInner(m, S.ui.streamingMsgs.has(msgId));
    followBottom(el);
    composer();
  }

  /** 新消息加入当前可见对话：整栏重绘后滚到对话末尾。 */
  function chatAdded(key) {
    const q = sel();
    const visibleKey = q ? q.question_uid : 'all';
    if (key !== visibleKey) return;
    render();
    followBottom($('#thread')?.lastElementChild, true);
  }

  /* ---------- 输入框与语音状态 ---------- */
  function composer() {
    const v = S.voice;
    const busy = S.ui.streamingMsgs.size > 0 || v.pendingTTS || v.playing;
    sendBtn.classList.toggle('stop', busy);
    sendBtn.innerHTML = busy ? IC.stop : IC.send;
    sendBtn.setAttribute('aria-label', busy ? '停止回答' : '发送');
    const mode = S.d.mode?.voice_mode || 'vad';
    $$('#vmode button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    const sup = app.micSupport();
    mic.disabled = mode === 'off';
    mic.classList.toggle('on', (mode === 'vad' && v.micOn) || (mode === 'ptt' && v.holding));
    mic.classList.toggle('warn', sup !== 'ok' && mode !== 'off');
    mic.setAttribute('aria-label', mode === 'ptt' ? '按住说话' : mode === 'vad' ? (v.micOn ? '关闭麦克风' : '打开麦克风（免提）') : '语音输入已关闭');
    mic.title = mic.getAttribute('aria-label');
    // 语音状态条（覆盖在输入框上）
    let who = null, label = '';
    if (v.speaking || v.holding) { who = 'kid'; label = v.partial && v.speaking ? v.partial : v.holding ? '正在听，松开发送' : '小宇正在说…'; }
    else if (v.playing) { who = 'ai'; label = v.duck ? 'AI学伴正在回答…（已降低音量）' : 'AI学伴正在回答…'; }
    else if (S.ui.streamingMsgs.size) { who = 'ai'; label = 'AI学伴正在回答…'; }
    else if (v.pendingTTS) { who = 'ai'; label = '正在合成语音…'; }
    if (!who) { cVoice.hidden = true; cVoice.dataset.who = ''; }
    else {
      cVoice.hidden = false;
      if (cVoice.dataset.who !== who) {
        cVoice.dataset.who = who;
        cVoice.innerHTML = `<span class="wave${who === 'ai' ? ' ai' : ''}">${bars()}</span><span class="vlabel"></span>`;
      }
      cVoice.querySelector('.wave').classList.toggle('duck', who === 'ai' && v.duck);
      cVoice.querySelector('.vlabel').textContent = label;
    }
    // 麦克风提示
    let hintText = '', warn = false;
    if (sup === 'insecure' && mode !== 'off') { hintText = '麦克风需要 HTTPS：请用 localhost 打开，或按 deploy/gen_lan_cert.sh 配置证书。文字提问不受影响。'; warn = true; }
    else if (sup === 'unsupported' && mode !== 'off') { hintText = '当前浏览器不支持麦克风采集，可用文字提问。'; warn = true; }
    else if (app.micError) { hintText = app.micError; warn = true; }
    else if (S.mock) hintText = '回放模式：语音与对话为脚本事件，麦克风不会上传。';
    hint.hidden = !hintText;
    hint.textContent = hintText;
    hint.classList.toggle('warn', warn);
  }

  /* ---------- 交互 ---------- */
  function autoGrow() { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'; }
  ta.addEventListener('input', autoGrow);
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      if (app.sendChat(ta.value)) { ta.value = ''; autoGrow(); }
    }
  });
  sendBtn.addEventListener('click', () => {
    if (sendBtn.classList.contains('stop')) app.stopChat();
    else if (app.sendChat(ta.value)) { ta.value = ''; autoGrow(); }
  });
  $('#back').addEventListener('click', () => { app.select(null); if (app.narrow()) app.goPane('list'); });
  $$('#vmode button').forEach((b) => b.addEventListener('click', () => app.setVoiceMode(b.dataset.mode)));
  body.addEventListener('toggle', (e) => {
    const d = e.target;
    if (!d.classList?.contains('passage')) return;
    S.ui.passageOpen = S.ui.passageOpen || new Set();
    if (d.open) S.ui.passageOpen.add(d.dataset.pid); else S.ui.passageOpen.delete(d.dataset.pid);
  }, true);

  // 麦克风：免提模式点按开关；按住说模式按下/松开
  const down = (e) => {
    if (S.d.mode?.voice_mode !== 'ptt' || (e.button !== undefined && e.button > 0)) return;
    e.preventDefault();
    try { mic.setPointerCapture?.(e.pointerId); } catch { /* 键盘触发 */ }
    app.pttDown();
  };
  const up = () => { if (S.voice.holding) app.pttUp(); };
  mic.addEventListener('pointerdown', down);
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach((ev) => mic.addEventListener(ev, up));
  mic.addEventListener('keydown', (e) => { if ((e.key === ' ' || e.key === 'Enter') && !e.repeat && S.d.mode?.voice_mode === 'ptt') down(e); });
  mic.addEventListener('keyup', (e) => { if (e.key === ' ' || e.key === 'Enter') up(); });
  mic.addEventListener('click', () => { if (S.d.mode?.voice_mode === 'vad') app.toggleMic(); else if (app.micSupport() !== 'ok') app.toast(hint.textContent || '麦克风不可用', 'alert'); });

  return { render, head, explainDelta, chatUpdate, chatAdded, composer };
}
