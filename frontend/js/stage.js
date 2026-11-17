// 中栏：实时 MJPEG 与定格抓拍图切换；定格图上用 <svg viewBox="0 0 W H"> 按抓拍原图像素坐标画批改层。
// 图片与 SVG 放在同一个 W×H 像素的 .sheet 容器里，整体用 transform 缩放平移（聚光灯缩放），因此任何缩放下都严格对齐。

import { S, viewStatus } from './store.js';
import { $, esc, clamp, rel, plain, subjectName } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const deg = (r) => (r * 180) / Math.PI;
const f1 = (v) => (Math.round(v * 10) / 10).toString();
const ptsStr = (pts) => pts.map((p) => f1(p[0]) + ',' + f1(p[1])).join(' ');
const pathOf = (pts) => 'M' + pts.map((p) => f1(p[0]) + ' ' + f1(p[1])).join('L') + 'Z';

/* 原型里的手绘圈：略不闭合、带一点抖动的椭圆 */
function loopPath(w, h) {
  const cx = w / 2, cy = h / 2, rx = w / 2 - 3, ry = h / 2 - 3, pts = [], n = 46, a0 = Math.PI * 1.1, turn = Math.PI * 2 * 1.13;
  for (let i = 0; i <= n; i++) {
    const t = i / n, a = a0 + turn * t, k = 1 - 0.06 * t;
    pts.push(f1(cx + Math.cos(a) * rx * k + (t > 0.88 ? (t - 0.88) * 30 : 0)) + ' ' + f1(cy + Math.sin(a) * ry * (k + 0.04 * Math.sin(t * 7)) - 2 * t));
  }
  return 'M' + pts.join(' L');
}

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

const rot = (p, a) => [p[0] * Math.cos(a) - p[1] * Math.sin(a), p[0] * Math.sin(a) + p[1] * Math.cos(a)];
const add = (p, q) => [p[0] + q[0], p[1] + q[1]];

/** 题目在纸面方向上的外接矩形（含批改标记），用于聚光灯与镜头定帧。 */
function spotRect(q, g) {
  const pts = [];
  (q.quad || []).forEach((p) => pts.push(p));
  if (!pts.length && q.bbox) {
    const [x, y, w, h] = q.bbox;
    pts.push([x, y], [x + w, y], [x + w, y + h], [x, y + h]);
  }
  if (q.mark_pos && hasMark(q)) {
    const c = markCenter(q, g);
    const r = g.s * 0.62;
    pts.push(add(c, rot([-r, -r], g.ang)), add(c, rot([r, r], g.ang)), add(c, rot([r, -r], g.ang)), add(c, rot([-r, r], g.ang)));
  }
  if (!pts.length) return null;
  const local = pts.map((p) => rot(p, -g.ang));
  const m = g.s * 0.3;
  const u0 = Math.min(...local.map((p) => p[0])) - m, u1 = Math.max(...local.map((p) => p[0])) + m;
  const v0 = Math.min(...local.map((p) => p[1])) - m, v1 = Math.max(...local.map((p) => p[1])) + m;
  const rect = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]].map((p) => rot(p, g.ang));
  const xs = rect.map((p) => p[0]), ys = rect.map((p) => p[1]);
  return { pts: rect, box: { x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) } };
}

function hasMark(q) {
  const v = viewStatus(q);
  return ['right', 'wrong', 'blank', 'uncertain', 'review', 'judging', 'failed'].includes(v);
}
/** mark_pos 视为标记左缘中点（对齐后端：标记画在 mark_pos 起向右的位置）。 */
function markCenter(q, g) {
  return add(q.mark_pos, rot([g.s * 0.55, 0], g.ang));
}

function markSVG(q, g, anim) {
  const v = viewStatus(q);
  if (!q.mark_pos || !hasMark(q)) return '';
  const s = g.s, a = deg(g.ang);
  const c = markCenter(q, g);
  const tr = `translate(${f1(c[0])},${f1(c[1])}) rotate(${f1(a)})`;
  const k = s / 64;
  if (v === 'right' || v === 'wrong') {
    const paths = v === 'right'
      ? '<path class="mk-p" pathLength="1" d="M7 27 C12 31 16 37 21 46 C28 30 40 15 59 4"/>'
      : '<path class="mk-p" pathLength="1" d="M12 7 C24 19 37 33 53 48"/><path class="mk-p p2" pathLength="1" d="M53 8 C40 19 27 33 11 48"/>';
    return `<g class="mk${anim ? ' anim' : ''}" transform="${tr} scale(${f1(k * 100) / 100}) translate(-32,-27)" stroke-width="4.4">${paths}</g>`;
  }
  if (v === 'judging') {
    const r = s * 0.26, circ = 2 * Math.PI * r;
    return `<g class="jdg" transform="${tr}"><circle r="${f1(r)}" stroke="rgba(30,30,30,.16)" stroke-width="${f1(s * 0.07)}"/>` +
      `<circle class="spin" r="${f1(r)}" stroke="rgba(30,30,30,.6)" stroke-width="${f1(s * 0.07)}" stroke-linecap="round" stroke-dasharray="${f1(circ * 0.28)} ${f1(circ)}"/></g>`;
  }
  const label = v === 'uncertain' ? '待确认' : v === 'blank' ? '未作答' : v === 'review' ? '点评' : '待重拍';
  const cls = v === 'blank' || v === 'uncertain' ? 'blank' : v === 'review' ? 'review' : 'failed';
  const fs = s * 0.36, w = fs * (label.length + 1.2), h = fs * 1.75;
  // 标签左缘贴 mark_pos
  const lc = add(q.mark_pos, rot([w / 2, 0], g.ang));
  return `<g class="pill ${cls}${anim ? ' anim' : ''}" transform="translate(${f1(lc[0])},${f1(lc[1])}) rotate(${f1(a)})">` +
    `<rect x="${f1(-w / 2)}" y="${f1(-h / 2)}" width="${f1(w)}" height="${f1(h)}" rx="${f1(h / 2)}" stroke-width="${f1(s * 0.045)}"/>` +
    `<text font-size="${f1(fs)}">${label}</text></g>`;
}

function errorSVG(q, g, anim) {
  if (viewStatus(q) !== 'wrong' || !q.error_quad || q.error_quad.length !== 4) return '';
  const [tl, tr, br, bl] = q.error_quad;
  const ang = Math.atan2(tr[1] - tl[1], tr[0] - tl[0]);
  const w = (Math.hypot(tr[0] - tl[0], tr[1] - tl[1]) + Math.hypot(br[0] - bl[0], br[1] - bl[1])) / 2;
  const h = (Math.hypot(bl[0] - tl[0], bl[1] - tl[1]) + Math.hypot(br[0] - tr[0], br[1] - tr[1])) / 2;
  const cx = (tl[0] + tr[0] + br[0] + bl[0]) / 4, cy = (tl[1] + tr[1] + br[1] + bl[1]) / 4;
  const W2 = w + g.s * 0.5, H2 = h + g.s * 0.36;
  let out = `<g class="circ${anim ? ' anim' : ''}" transform="translate(${f1(cx)},${f1(cy)}) rotate(${f1(deg(ang))}) translate(${f1(-W2 / 2)},${f1(-H2 / 2)})">` +
    `<path pathLength="1" stroke-width="${f1(g.s * 0.06)}" d="${loopPath(W2, H2)}"/></g>`;
  // 图上批注只写短句（完整内容在右栏），避免放大后超出纸面
  let note = plain(q.error_note || '');
  if ([...note].length > 14) note = [...note].slice(0, 13).join('') + '…';
  if (note) {
    // 批注写在圈的右边；若对错标记在同一行右侧，则写在标记之后，避免压字。
    const cL = rot([cx, cy], -g.ang);
    let x = cL[0] + W2 / 2 + g.s * 0.25;
    let y = cL[1];
    if (q.mark_pos) {
      const mL = rot(markCenter(q, g), -g.ang);
      if (Math.abs(mL[1] - y) < g.s * 1.2 && mL[0] + g.s * 0.6 > x - g.s * 0.2) x = Math.max(x, mL[0] + g.s * 0.75);
    }
    // 右侧放不下（超出纸面右缘）时，改写到圈的右下方、贴纸面右缘
    const fs = g.s * 0.62;
    const noteW = [...note].reduce((w, ch) => w + (/[\x00-\xff]/.test(ch) ? 0.55 : 1), 0) * fs;
    const pq = g.pq;
    if (pq) {
      const right = Math.min(rot(pq[1], -g.ang)[0], rot(pq[2], -g.ang)[0], ...[g.W, g.W].map((wx, i) => rot([wx, i ? g.H : 0], -g.ang)[0]));
      if (x + noteW > right - g.s * 0.2) {
        x = Math.max(cL[0] - W2 / 2, right - g.s * 0.3 - noteW);
        y = cL[1] + H2 / 2 + fs * 0.55;
      }
    }
    const p = rot([x, y], g.ang);
    out += `<text class="rnote${anim ? ' anim' : ''}" font-size="${f1(fs)}" dominant-baseline="central" ` +
      `transform="translate(${f1(p[0])},${f1(p[1])}) rotate(${f1(deg(g.ang) - 4)})">${esc(note)}</text>`;
  }
  return out;
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
      // 切到定格图时断开 MJPEG 连接，释放后端名额与浏览器同源连接池。
      if (live.getAttribute('src')) live.setAttribute('src', '');
      live.hidden = true;
      sheet.hidden = false;
    }
  }

  function render() {
    const c = viewed();
    setLive(!c);
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
    // 聚光灯：选中题所在纸面方向的矩形之外压暗
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

  if (window.ResizeObserver) new ResizeObserver(() => reframe(true)).observe(cam);

  return { render, reframe, status, caption };
}
