// 中栏：实时 MJPEG 与定格抓拍图切换；定格图上用 <svg viewBox="0 0 W H"> 按抓拍原图像素坐标画批改层。
// 图片与 SVG 放在同一个 W×H 像素的 .sheet 容器里，整体用 transform 缩放平移（聚光灯缩放），因此任何缩放下都严格对齐。

import { S, viewStatus } from './store.js';
import { $, esc, clamp, rel, plain, subjectName } from './util.js';
import { focusGeometry, markQuad, quadSize, quadProjector, projectPath, ellipsePoints, checkPoints, validQuad, pathOf } from './overlay-geometry.js';

const NS = 'http://www.w3.org/2000/svg';
const deg = (r) => (r * 180) / Math.PI;
const f1 = (v) => (Math.round(v * 10) / 10).toString();
const ptsStr = (pts) => pts.map((p) => f1(p[0]) + ',' + f1(p[1])).join(' ');

function pageGeo(cap) {
  const W = cap.image_width, H = cap.image_height, pq = cap.page_quad;
  if (pq && pq.length === 4) {
    const [tl, tr] = pq;
    const ang = Math.atan2(tr[1] - tl[1], tr[0] - tl[0]);
    const pw = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
    return { ang, pw, s: clamp(pw * 0.055, 22, 150), W, H, pq };
  }
  return { ang: 0, pw: W, s: clamp(W * 0.03, 22, 150), W, H, pq: null };
}

/** 遮罩严格沿题目quad开孔；镜头框可另含标记，不改变遮罩四边。 */
function spotRect(q, g) {
  return focusGeometry(q, g.s, hasMark(q));
}

function hasMark(q) {
  const v = viewStatus(q);
  return ['right', 'wrong', 'blank', 'uncertain', 'review', 'judging', 'failed'].includes(v);
}
function markSVG(q, g, anim) {
  if (!hasMark(q)) return '';
  const v = viewStatus(q), box = markQuad(q, g.s);
  if (!box) return '';
  const size = quadSize(box), project = quadProjector(box);
  if (!project) return '';
  const stroke = clamp(Math.min(size.w, size.h) * 0.075, 1.2, 12);
  if (v === 'right' || v === 'wrong') {
    const paths = v === 'right'
      ? `<path class="mk-p" pathLength="1" d="${projectPath(box, checkPoints())}"/>`
      : `<path class="mk-p" pathLength="1" d="${projectPath(box, [[0.18, 0.14], [0.82, 0.86]])}"/><path class="mk-p p2" pathLength="1" d="${projectPath(box, [[0.82, 0.14], [0.18, 0.86]])}"/>`;
    return `<g class="mk${anim ? ' anim' : ''}" stroke-width="${f1(stroke)}">${paths}</g>`;
  }
  if (v === 'judging') {
    const outline = projectPath(box, ellipsePoints(), true);
    return `<g class="jdg"><path d="${outline}" fill="none" stroke="rgba(30,30,30,.16)" stroke-width="${f1(stroke)}"/>` +
      `<path class="trace" pathLength="1" d="${outline}" fill="none" stroke="rgba(30,30,30,.6)" stroke-width="${f1(stroke)}" stroke-linecap="round" stroke-dasharray=".28 .72"/></g>`;
  }
  const label = v === 'uncertain' ? '待确认' : v === 'blank' ? '未作答' : v === 'review' ? '点评' : '待重拍';
  const cls = v === 'blank' || v === 'uncertain' ? 'blank' : v === 'review' ? 'review' : 'failed';
  const center = project([0.5, 0.5]), left = project([0, 0.5]), right = project([1, 0.5]);
  const angle = deg(Math.atan2(right[1] - left[1], right[0] - left[0]));
  const fs = Math.min(size.h * 0.5, size.w / (label.length + 0.45));
  const shape = projectPath(box, [[0.02, 0.12], [0.98, 0.12], [0.98, 0.88], [0.02, 0.88]], true);
  return `<g class="pill ${cls}${anim ? ' anim' : ''}"><path class="pill-shape" d="${shape}" stroke-width="${f1(stroke * 0.65)}"/>` +
    `<text x="${f1(center[0])}" y="${f1(center[1])}" font-size="${f1(fs)}" transform="rotate(${f1(angle)} ${f1(center[0])} ${f1(center[1])})">${label}</text></g>`;
}

function errorSVG(q, g, anim) {
  if (viewStatus(q) !== 'wrong' || !validQuad(q.error_quad)) return '';
  const size = quadSize(q.error_quad);
  // Project a closed outline onto the actual answer quadrilateral; no page-wide rotation or hand-drawn drift.
  // error_quad may be a tight ink bbox: a radius >= sqrt(0.5) encloses its corners instead of cutting strokes.
  const outline = projectPath(q.error_quad, ellipsePoints(72, -0.22), true);
  const stroke = clamp(Math.min(size.w, size.h) * 0.07, 1.2, 8);
  // Detailed feedback stays in the right panel. A tooltip preserves it without covering neighboring answers.
  const title = q.error_note ? `<title>${esc(plain(q.error_note))}</title>` : '';
  return `<g class="circ${anim ? ' anim' : ''}">${title}<path pathLength="1" stroke-width="${f1(stroke)}" d="${outline}"/></g>`;
}

function detSVG(cap, g) {
  const pq = cap.page_quad;
  if (!pq || pq.length !== 4) return '';
  const L = g.pw * 0.07, sw = Math.max(3, g.pw * 0.007);
  const W = cap.image_width, H = cap.image_height;
  const inside = (p) => [clamp(p[0], sw, W - sw), clamp(p[1], sw, H - sw)];
  let d = '';
  pq.forEach((p, i) => {
    const prev = pq[(i + 3) % 4], next = pq[(i + 1) % 4];
    const u = [next[0] - p[0], next[1] - p[1]], v = [prev[0] - p[0], prev[1] - p[1]];
    const nu = Math.hypot(...u), nv = Math.hypot(...v);
    const c = inside(p);
    const a = inside([p[0] + (u[0] / nu) * L, p[1] + (u[1] / nu) * L]);
    const b = inside([p[0] + (v[0] / nv) * L, p[1] + (v[1] / nv) * L]);
    d += `M${f1(a[0])} ${f1(a[1])}L${f1(c[0])} ${f1(c[1])}L${f1(b[0])} ${f1(b[1])}`;
  });
  const label = '作业纸 · ' + subjectName(cap.subject) + (cap.page_no ? ' 第' + cap.page_no + '页' : '');
  const fs = g.pw * 0.03;
  const tl = inside([pq[0][0], pq[0][1] + fs * 1.4]);
  return `<g class="det"><path d="${d}" stroke-width="${f1(sw)}"/>` +
    `<g transform="translate(${f1(tl[0] + sw * 2)},${f1(tl[1])}) rotate(${f1(deg(g.ang))})"><rect x="0" y="${f1(-fs * 0.9)}" width="${f1(fs * (label.length * 0.95 + 1))}" height="${f1(fs * 1.8)}" rx="${f1(fs * 0.3)}"/>` +
    `<text x="${f1(fs * 0.5)}" y="0" font-size="${f1(fs)}" dominant-baseline="central">${esc(label)}</text></g></g>`;
}

export function createStage(app) {
  const cam = $('#cam'), live = $('#live'), sheet = $('#sheet'), still = $('#still'), ovl = $('#ovl');
  const camMsg = $('#camMsg'), badge = $('#camBadge'), badgeT = $('#camBadgeT');
  const cap = $('#cap'), capT = $('#capT'), capWho = $('#capWho');
  const liveDot = $('#liveDot'), camTitle = $('#camTitle'), btnView = $('#btnView'), btnGrade = $('#btnGrade'), btnCapture = $('#btnCapture');
  const stab = $('#stab'), stabT = $('#stabT');

  ovl.innerHTML = '';
  const gHits = document.createElementNS(NS, 'g');
  const dim = document.createElementNS(NS, 'path');
  const qhl = document.createElementNS(NS, 'path');
  const gMarks = document.createElementNS(NS, 'g');
  const gFx = document.createElementNS(NS, 'g');
  dim.setAttribute('class', 'dim');
  dim.setAttribute('fill-rule', 'evenodd');
  qhl.setAttribute('class', 'qhl');
  ovl.append(dim, qhl, gMarks, gFx, gHits);

  let stillErr = false, liveErr = false, shownCap = null, detTimer = 0, capTimer = 0;
  still.addEventListener('error', () => { if (still.getAttribute('src')) { stillErr = true; status(); } });
  still.addEventListener('load', () => { stillErr = false; status(); });
  live.addEventListener('error', () => { if (live.getAttribute('src')) { liveErr = true; status(); } });
  live.addEventListener('load', () => { liveErr = false; status(); });

  const viewed = () => (S.ui.view && S.d.captures[S.ui.view]) || null;

  function setLive(on) {
    if (on) {
      sheet.hidden = true;
      live.hidden = false;
      if (live.getAttribute('src') !== app.liveSrc) {
        liveErr = false;
        live.src = app.liveSrc;
      }
    } else {
      // 切到定格图或离开工作台时断开 MJPEG 连接，释放后端名额与浏览器同源连接池。
      if (live.getAttribute('src')) live.setAttribute('src', '');
      live.hidden = true;
      sheet.hidden = !viewed();
    }
  }

  function render() {
    const c = viewed();
    const activeWorkbench = app.isWorkbenchActive ? app.isWorkbenchActive() : true;
    setLive(!c && activeWorkbench);
    if (c) {
      const W = c.image_width || 1920, H = c.image_height || 1080;
      sheet.style.width = W + 'px';
      sheet.style.height = H + 'px';
      ovl.setAttribute('viewBox', `0 0 ${W} ${H}`);
      const url = rel(c.image_url);
      if (still.getAttribute('src') !== url) {
        stillErr = false;
        still.src = url;
      }
      drawOverlay(c);
      if (shownCap !== c.capture_id) {
        shownCap = c.capture_id;
        reframe(true);
      } else reframe(false);
    } else {
      shownCap = null;
    }
    status();
  }

  function drawOverlay(c) {
    const g = pageGeo(c);
    const qs = (c.question_uids || []).map((u) => S.d.questions[u]).filter(Boolean);
    let marks = '', hits = '';
    for (const q of qs) {
      const anim = S.ui.animate.has(q.question_uid);
      marks += errorSVG(q, g, anim) + markSVG(q, g, anim);
      if (q.quad) hits += `<polygon class="hit" data-uid="${esc(q.question_uid)}" points="${ptsStr(q.quad)}"><title>${esc('第' + q.no + '题')}</title></polygon>`;
    }
    qs.forEach((q) => S.ui.animate.delete(q.question_uid));
    gMarks.innerHTML = marks;
    gHits.innerHTML = hits;
    // 聚光灯开孔与选中题quad逐点一致，禁止按整页角度重新构造矩形。
    const sel = S.d.questions[S.d.selected_question_uid];
    const r = sel && sel.capture_id === c.capture_id ? spotRect(sel, g) : null;
    if (r) {
      dim.setAttribute('d', `M0 0H${c.image_width}V${c.image_height}H0Z` + pathOf(r.pts));
      qhl.setAttribute('d', pathOf(r.pts));
      qhl.setAttribute('stroke-width', f1(g.s * 0.035));
    }
    dim.classList.toggle('on', !!r);
    qhl.classList.toggle('on', !!r);
    if (S.ui.det === c.capture_id) {
      S.ui.det = null;
      gFx.innerHTML = detSVG(c, g);
      clearTimeout(detTimer);
      detTimer = setTimeout(() => (gFx.innerHTML = ''), 2000);
    }
  }

  function reframe(instant) {
    const c = viewed();
    if (!c) return;
    const cw = cam.clientWidth, ch = cam.clientHeight;
    if (!cw || !ch) return;
    const W = c.image_width || 1920, H = c.image_height || 1080;
    const g = pageGeo(c);
    const sel = S.d.questions[S.d.selected_question_uid];
    const sr = sel && sel.capture_id === c.capture_id ? spotRect(sel, g) : null;
    let r, o;
    if (sr) {
      const m = g.s * 0.5;
      r = { x: sr.box.x - m, y: sr.box.y - m, w: sr.box.w + 2 * m, h: sr.box.h + 2 * m };
      o = { fill: 0.9, fillY: 0.62, max: 2.6 };
    } else if (c.page_quad && c.page_quad.length === 4) {
      const xs = c.page_quad.map((p) => clamp(p[0], 0, W)), ys = c.page_quad.map((p) => clamp(p[1], 0, H));
      const x0 = Math.min(...xs), y0 = Math.min(...ys), x1 = Math.max(...xs), y1 = Math.max(...ys);
      r = { x: x0 - 24, y: y0 - 24, w: x1 - x0 + 48, h: y1 - y0 + 48 };
      o = { fill: 0.98, fillY: 0.98, max: 3 };
    } else {
      r = { x: 0, y: 0, w: W, h: H };
      o = { fill: 1, fillY: 1, max: 1 };
    }
    let s = Math.min((cw * o.fill) / r.w, (ch * o.fillY) / r.h);
    s = Math.min(s, o.max);
    s = Math.max(s, Math.min(cw / W, ch / H)); // 不小于整图 contain
    const fit = (box, img, center) => (img * s > box ? clamp(box / 2 - center * s, box - img * s, 0) : (box - img * s) / 2);
    const tx = fit(cw, W, r.x + r.w / 2), ty = fit(ch, H, r.y + r.h / 2);
    if (instant) sheet.classList.add('nt');
    sheet.style.transform = `translate(${tx.toFixed(1)}px,${ty.toFixed(1)}px) scale(${s.toFixed(5)})`;
    if (instant) {
      void sheet.offsetWidth;
      setTimeout(() => sheet.classList.remove('nt'), 30);
    }
  }

  function status() {
    const c = viewed();
    // 顶栏
    liveDot.classList.toggle('still', !!c);
    if (c) {
      const where = c.paper_id ? `${subjectName(c.subject)} 第${c.page_no}页` : '未匹配';
      camTitle.innerHTML = `定格画面 <span>· ${esc(c.ts || '')} · ${esc(where)}</span>`;
      btnView.hidden = false;
      btnView.textContent = '回到实时画面';
    } else {
      camTitle.innerHTML = `书桌画面 <span>· ${S.mock ? '回放' : 'LIVE'}</span>`;
      const act = S.d.captures[S.d.active_capture_id];
      btnView.hidden = !act;
      btnView.textContent = '查看定格';
    }
    const act = S.d.captures[S.d.active_capture_id];
    const idle = c && c.capture_id === S.d.active_capture_id && (c.question_uids || []).length > 0 &&
      (c.question_uids || []).every((u) => { const q = S.d.questions[u]; return !q || q.status === 'idle'; });
    btnGrade.hidden = !idle;
    btnCapture.disabled = S.conn !== 'open' && S.conn !== 'mock';
    void act;
    // 稳帧进度
    const cs = S.cam;
    const fresh = Date.now() - cs.at < 3000;
    const settling = fresh && cs.stability === 'settling';
    stab.hidden = !settling;
    if (settling) {
      stab.querySelector('.ring').style.setProperty('--p', clamp(cs.progress || 0, 0, 1));
      stabT.textContent = '稳帧 ' + Math.round((cs.progress || 0) * 100) + '%';
    }
    // 画面左上角徽标
    let b = '';
    if (!c && settling) b = '保持纸张静止 · ' + Math.round((cs.progress || 0) * 100) + '%';
    else if (!c && fresh && cs.stability === 'captured') b = '已抓拍，正在识别…';
    else if (!c && S.mock) b = '回放模式 · 模拟相机画面';
    badge.hidden = !b;
    if (b) {
      badgeT.textContent = b;
      badge.querySelector('.ring').hidden = !settling;
      badge.querySelector('.ring').style.setProperty('--p', clamp(cs.progress || 0, 0, 1));
    }
    // 画面提示
    let msg = '';
    if (c) {
      if (stillErr) msg = '这张抓拍图已过期（后端只保留最近 20 张），请重新抓拍';
      else if (S.ui.unmatched.has(c.capture_id)) msg = '未匹配到已入库试卷，请调整纸张位置后重拍';
    } else if (cs.status === 'no_camera') msg = '未检测到相机';
    else if (liveErr) msg = S.conn === 'open' ? '实时画面暂不可用（预览最多两路，或相机未就绪）' : '实时画面暂不可用，正在等待后端…';
    camMsg.hidden = !msg;
    camMsg.textContent = msg;
    caption();
  }

  function caption() {
    const v = S.voice;
    const show = !!v.partial && (v.speaking || Date.now() - v.captionAt < 3000);
    cap.hidden = !show;
    if (show) {
      capWho.textContent = '小宇';
      if (capT.textContent !== v.partial) capT.textContent = v.partial;
      cap.classList.toggle('partial', v.speaking && !v.partialFinal);
      clearTimeout(capTimer);
      if (!v.speaking) capTimer = setTimeout(caption, 3050 - (Date.now() - v.captionAt));
    }
  }

  // 交互
  ovl.addEventListener('click', (e) => {
    const hit = e.target.closest && e.target.closest('.hit');
    if (hit) {
      const uid = hit.dataset.uid;
      app.select(S.d.selected_question_uid === uid ? null : uid);
      e.stopPropagation();
    }
  });
  sheet.addEventListener('click', (e) => {
    if (!(e.target.closest && e.target.closest('.hit')) && S.d.selected_question_uid) app.select(null);
  });
  btnView.addEventListener('click', () => app.setView(viewed() ? null : S.d.active_capture_id));
  btnGrade.addEventListener('click', () => app.send({ type: 'session.grade_current' }));
  btnCapture.addEventListener('click', () => app.send({ type: 'camera.capture', grade: true }));
  const btnUpload = $('#btnUpload'), fileInject = $('#fileInject');
  if (btnUpload && fileInject) {
    btnUpload.addEventListener('click', () => fileInject.click());
    fileInject.addEventListener('change', async () => {
      const f = fileInject.files && fileInject.files[0];
      fileInject.value = '';
      if (!f) return;
      const fd = new FormData();
      fd.append('file', f);
      fd.append('grade', 'true');
      btnUpload.disabled = true;
      app.toast('正在上传并执行整页 OCR 与切题…');
      try {
        const r = await fetch('api/debug/inject-image', { method: 'POST', body: fd });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || '上传识别失败');
        }
      } catch (e) {
        app.toast(e.message || '上传识别失败', 'alert');
      } finally {
        btnUpload.disabled = false;
      }
    });
  }

  if (window.ResizeObserver) new ResizeObserver(() => reframe(true)).observe(cam);

  return { render, reframe, status, caption };
}
