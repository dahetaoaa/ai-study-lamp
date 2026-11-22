// 双科总览开屏主页 + 人教版初中数学/英语全套知识图谱交互视图 + 二级功能页路由控制器。

import { S, scored, viewStatus, capturesInOrder } from './store.js';
import { $, $$, esc, tex } from './util.js';

const WEEKDAYS = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];
const STATUS_LABEL = {
  mastered: '已掌握',
  learning: '今天正在学',
  next: '即将解锁',
  locked: '待学习',
};

function greetingByHour(h) {
  if (h < 6) return '夜深了';
  if (h < 11) return '早上好';
  if (h < 14) return '中午好';
  if (h < 18) return '下午好';
  return '晚上好';
}

function formatNum(n) {
  return Number(n || 0).toLocaleString('en-US');
}

export function createHomeAndGraph(app) {
  const state = {
    route: 'overview', // overview | math | en | workbench | vocab | notes
    wbTab: 'hw',
    overview: null,
    graphs: { math: null, en: null },
    kgSubj: 'math',
    kgMode: 'map', // map | syllabus
    kgBook: 'all',
    kgCluster: null,
    kgQuery: '',
    selectedNodeKey: null,
    pillDismissed: false,
  };

  const shell = $('#shell');
  const viewOverview = $('#viewOverview');
  const viewGraph = $('#viewGraph');
  const viewFeature = $('#viewFeature');
  const viewWorkbench = $('#app');
  const kgCanvas = $('#kgCanvas');
  const kgCanvasWrap = $('#kgCanvasWrap');
  const kgInspector = $('#kgInspector');

  /* ---------- 数据拉取 ---------- */
  async function loadOverview() {
    try {
      const r = await fetch('api/knowledge/overview');
      if (r.ok) {
        state.overview = await r.json();
        renderOverview();
        renderRailStats();
      }
    } catch {
      /* 离线或 mock 模式使用默认静态降级 */
    }
  }

  async function ensureGraph(subject, force = false) {
    if (state.graphs[subject] && !force) return state.graphs[subject];
    try {
      const r = await fetch(`api/knowledge/graph/${encodeURIComponent(subject)}`);
      if (r.ok) {
        state.graphs[subject] = await r.json();
      }
    } catch {
      /* ignore */
    }
    return state.graphs[subject];
  }

  /* ---------- 路由切换 ---------- */
  async function navigate(route, opts = {}) {
    if (route === 'math' || route === 'en') {
      state.kgSubj = route;
      state.route = route;
    } else {
      state.route = route;
    }
    if (opts.wbTab) state.wbTab = opts.wbTab;

    shell.dataset.route = state.route;
    viewOverview.hidden = state.route !== 'overview';
    viewGraph.hidden = !(state.route === 'math' || state.route === 'en');
    viewFeature.hidden = !(state.route === 'vocab' || state.route === 'notes');
    viewWorkbench.hidden = state.route !== 'workbench';

    // 更新左侧导航栏 aria-current 高亮
    $$('[data-route-go]').forEach((btn) => {
      const target = btn.dataset.routeGo;
      const tab = btn.dataset.wbTab;
      let active = false;
      if (target === 'overview') active = state.route === 'overview' || state.route === 'math' || state.route === 'en';
      else if (target === 'workbench') active = state.route === 'workbench' && (!tab || tab === state.wbTab);
      else active = state.route === target;
      if (btn.classList.contains('overview-nav') || btn.classList.contains('fn-card')) {
        btn.setAttribute('aria-current', active ? 'page' : 'false');
      }
    });

    // 通知工作台相机层：仅在 workbench 激活时保持实时 MJPEG 流，释放连接池
    app.onRouteChange?.(state.route, state.wbTab);

    if (state.route === 'overview') {
      loadOverview();
      renderOverview();
    } else if (state.route === 'math' || state.route === 'en') {
      state.kgBook = opts.bookId || 'all';
      state.kgCluster = null;
      await ensureGraph(state.kgSubj, opts.refresh);
      const g = state.graphs[state.kgSubj];
      if (g) {
        state.selectedNodeKey = opts.nodeKey || (state.kgSubj === 'math' ? g.current.section_code : g.current.unit_id);
      }
      renderGraphView(true);
    } else if (state.route === 'vocab' || state.route === 'notes') {
      await ensureGraph('en');
      renderFeatureView();
    }

    if (!opts.fromPop) {
      const hash = state.route === 'overview' ? '#overview'
        : state.route === 'workbench' ? `#workbench/${state.wbTab}`
        : `#${state.route}`;
      if (location.hash !== hash) history.replaceState(null, '', hash);
    }
  }

  /* ---------- 左侧边栏与开屏主页渲染 ---------- */
  function updateClockAndGreeting() {
    const now = new Date();
    const m = now.getMonth() + 1;
    const d = now.getDate();
    const wd = WEEKDAYS[now.getDay()];
    const term = state.overview?.student?.term_label || '九年级上学期 · 第 10 周';
    const name = state.overview?.student?.name || '小宇';
    $('#ovDateLine').textContent = `${m}月${d}日 ${wd} · ${term}`;
    $('#ovGreeting').textContent = `${greetingByHour(now.getHours())}，${name}`;

    // 在座计时器 mm:ss
    const elapsedSec = Math.max(4, Math.floor(now / 1000 - (S.d.study_started_at || now / 1000)) + 4);
    const mm = Math.floor(elapsedSec / 60);
    const ss = String(elapsedSec % 60).padStart(2, '0');
    $('#railSeatClock').textContent = `${mm}:${ss}`;
  }

  function setRingPct(circleEl, valEl, pct) {
    const C = 2 * Math.PI * 18; // ~113.097
    const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
    circleEl.setAttribute('stroke-dasharray', C.toFixed(1));
    circleEl.setAttribute('stroke-dashoffset', (C * (1 - clamped / 100)).toFixed(1));
    valEl.textContent = `${Math.round(clamped)}%`;
  }

  function renderRailStats() {
    const caps = capturesInOrder();
    const allQs = Object.values(S.d.questions).filter(scored);
    const wrongQs = allQs.filter((q) => viewStatus(q) === 'wrong');
    const targetPages = state.overview?.tasks?.homework?.target_pages || 4;
    const donePages = caps.length;
    const hwPct = Math.min(100, Math.round((donePages / targetPages) * 100));

    $('#fnHwSub').textContent = `今日作业 ${donePages}/${targetPages}`;
    $('#fnHwPill').textContent = `${donePages}/${targetPages}`;
    $('#fnHwBar').style.width = `${hwPct}%`;
    $('#fnHwStat').textContent = `${donePages} / ${targetPages}`;

    const baseDue = 4;
    const dueTotal = baseDue + wrongQs.length;
    const freshPct = state.overview?.student?.mistake_freshness_pct ?? Math.max(58, 71 - wrongQs.length * 2);
    $('#fnMkSub').textContent = `${dueTotal} 题到期复习`;
    $('#fnMkPill').textContent = String(dueTotal);
    $('#fnMkBar').style.width = `${freshPct}%`;
    $('#fnMkStat').textContent = `保鲜 ${freshPct}%`;

    const noteScore = state.overview?.tasks?.notes?.last_score ?? 73;
    $('#fnNoteBar').style.width = `${noteScore}%`;
    $('#fnNoteStat').textContent = `上次 ${noteScore}`;
  }

  function renderOverview() {
    updateClockAndGreeting();
    renderRailStats();
    const ov = state.overview;
    if (!ov) return;

    $('#railStudentName').textContent = ov.student.name;
    $('#railStudentGrade').textContent = ov.student.grade;
    $('#railStreak').textContent = ov.student.streak_days;
    $('#kpiStreakVal').textContent = ov.student.streak_days;

    setRingPct($('#kpiLitCircle'), $('#kpiLitVal'), ov.student.knowledge_lit_pct);
    setRingPct($('#kpiFreshCircle'), $('#kpiFreshVal'), ov.student.mistake_freshness_pct);

    const m = ov.subjects.math;
    if (m) {
      $('#mathPublisher').textContent = m.publisher;
      $('#mathCurrentTitle').textContent = `${m.current.book_title} · ${m.current.chapter_no} ${m.current.chapter_title}`;
      $('#mathCurrentSub').textContent = `学到 ${m.current.section_code} ${m.current.section_title} · 本章掌握 ${m.current.chapter_mastery}%`;
      $('#mathProgBar').style.width = `${m.stats.progress_pct}%`;
      $('#mathProgPct').textContent = `${m.stats.progress_pct}%`;
      $('#mathLitCount').textContent = `${m.stats.lit_nodes} / ${m.stats.total_nodes} 节已点亮`;
    }

    const en = ov.subjects.en;
    if (en) {
      $('#enPublisher').textContent = en.publisher;
      $('#enCurrentTitle').textContent = `${en.current.book_title} · ${en.current.unit_code}`;
      $('#enCurrentSub').textContent = `${en.current.unit_title} · 被动语态`;
      $('#enProgBar').style.width = `${en.stats.progress_pct}%`;
      $('#enProgPct').textContent = `${en.stats.progress_pct}%`;
      $('#enLitCount').textContent = `${en.stats.lit_nodes} / ${en.stats.total_nodes} 单元 · 词汇 ${formatNum(en.stats.vocab_learned)} / ${formatNum(en.stats.vocab_total)}`;
    }

    const pill = $('#lampPill');
    pill.hidden = state.pillDismissed || state.route !== 'overview';
  }

  /* ---------- 知识图谱二级页面渲染 ---------- */
  function allNodesMap(g) {
    const map = new Map();
    if (!g) return map;
    if (g.subject === 'math') {
      for (const b of g.books) {
        for (const ch of b.chapters) {
          for (const sec of ch.sections) {
            map.set(sec.code, { ...sec, key: sec.code, book: b, chapter: ch });
          }
        }
      }
    } else {
      for (const b of g.books) {
        for (const u of b.units) {
          map.set(u.unit_id, { ...u, key: u.unit_id, book: b });
        }
      }
    }
    return map;
  }

  function matchesQuery(node, q, isMath) {
    if (!q) return true;
    const s = q.toLowerCase();
    if (isMath) {
      return (
        node.code.toLowerCase().includes(s) ||
        node.title.toLowerCase().includes(s) ||
        node.chapter.title.toLowerCase().includes(s) ||
        (node.key_points || []).some((k) => k.toLowerCase().includes(s))
      );
    }
    return (
      node.code.toLowerCase().includes(s) ||
      node.title.toLowerCase().includes(s) ||
      (node.cn_title || '').toLowerCase().includes(s) ||
      (node.grammar_focus || '').toLowerCase().includes(s) ||
      (node.key_vocab || []).some((v) => v.toLowerCase().includes(s))
    );
  }

  function renderGraphBanner(g) {
    const isMath = g.subject === 'math';
    const cur = g.current;
    const st = g.stats;
    const title = isMath
      ? `${cur.book_title} · ${cur.chapter_no}《${cur.chapter_title}》 → ${cur.section_code} ${cur.section_title}`
      : `${cur.book_title} · ${cur.unit_code} "${cur.unit_title}"`;
    const sub = cur.today_focus || '';
    const curKey = isMath ? cur.section_code : cur.unit_id;

    $('#kgBanner').innerHTML = `
      <div class="kgb-left">
        <div class="kgb-pulse">${isMath ? 'fx' : 'En'}</div>
        <div class="kgb-info">
          <div class="kgb-eyebrow">● 今日学习定位 · TODAY'S ANCHOR</div>
          <div class="kgb-title">${esc(title)}</div>
          <div class="kgb-sub">${esc(sub)}</div>
        </div>
      </div>
      <div class="kgb-right">
        <div class="kgb-stat">
          <strong class="num">${st.lit_nodes} / ${st.total_nodes}</strong>
          <span>${isMath ? '已点亮小节' : '已点亮单元'}（${st.progress_pct}%）</span>
        </div>
        ${!isMath ? `<div class="kgb-stat"><strong class="num">${formatNum(st.vocab_learned)} / ${formatNum(st.vocab_total)}</strong><span>课标核心词掌握</span></div>` : `<div class="kgb-stat"><strong class="num">${st.chapters_count} 章</strong><span>全套 ${st.books_count} 册人教体系</span></div>`}
        <button class="kgb-jump" type="button" data-jump-current="${esc(curKey)}">定位到今天所学</button>
      </div>
    `;
  }

  function renderBookTabs(g) {
    const isMath = g.subject === 'math';
    const allBtn = `<button type="button" role="tab" data-book-id="all" aria-selected="${state.kgBook === 'all'}">全部 ${g.books.length} 册 <small class="num">${g.stats.lit_nodes}/${g.stats.total_nodes}</small></button>`;
    const tabs = g.books.map((b) => {
      const lit = isMath ? b.lit_sections : b.lit_units;
      const tot = isMath ? b.total_sections : b.total_units;
      return `<button type="button" role="tab" data-book-id="${esc(b.book_id)}" aria-selected="${state.kgBook === b.book_id}">${esc(b.title)} <small class="num">${lit}/${tot}</small></button>`;
    }).join('');
    $('#kgBookTabs').innerHTML = allBtn + tabs;
  }

  function renderMapCanvas(g, nmap) {
    const isMath = g.subject === 'math';
    const books = state.kgBook === 'all' ? g.books : g.books.filter((b) => b.book_id === state.kgBook);
    const q = state.kgQuery.trim();
    const clusterSet = state.kgCluster
      ? new Set((g.grammar_clusters || []).find((c) => c.cluster_id === state.kgCluster)?.unit_ids || [])
      : null;

    let clustersHTML = '';
    if (!isMath && Array.isArray(g.grammar_clusters) && g.grammar_clusters.length) {
      clustersHTML = `<div class="kg-clusters-strip">
        <div class="kg-clusters-head">
          <h3>核心语法专题图谱（点击高亮跨册关联单元）</h3>
          <span>${state.kgCluster ? '<button class="mini" type="button" data-clear-cluster="1">清除专题筛选</button>' : '覆盖初中 10 大语法主干'}</span>
        </div>
        <div class="kg-clusters-list">
          ${g.grammar_clusters.map((gc) => `
            <button type="button" class="kg-cluster-chip${state.kgCluster === gc.cluster_id ? ' active' : ''}" data-cluster-id="${esc(gc.cluster_id)}">
              <strong>${esc(gc.title)}</strong>
              <small class="num">关联 ${gc.lit_count}/${gc.total_count} 单元 · 掌握 ${gc.mastery}%</small>
            </button>
          `).join('')}
        </div>
      </div>`;
    }

    const sectionsHTML = books.map((b) => {
      const lit = isMath ? b.lit_sections : b.lit_units;
      const tot = isMath ? b.total_sections : b.total_units;
      if (isMath) {
        const chRows = b.chapters.map((ch) => {
          const nodesHTML = ch.sections.map((sec) => {
            const full = nmap.get(sec.code);
            const hit = matchesQuery(full, q, true);
            const sel = state.selectedNodeKey === sec.code;
            const badge = sec.status === 'learning'
              ? '<span class="kgn-badge today">今天在学</span>'
              : sec.session_practice
                ? `<span class="kgn-badge today">刚练 ${sec.session_practice.right}/${sec.session_practice.total}</span>`
                : sec.status === 'mastered'
                  ? `<span class="kgn-badge pct num">${sec.mastery}%</span>`
                  : '';
            return `<button type="button" class="kg-node st-${esc(sec.status)}${sel ? ' selected' : ''}${!hit ? ' dimmed' : ''}" data-node-key="${esc(sec.code)}">
              <div class="kgn-top"><span class="kgn-code">${esc(sec.code)}</span>${badge}</div>
              <div class="kgn-title">${esc(sec.title)}</div>
              <div class="kgn-sub">${esc((sec.key_points && sec.key_points[0]) || sec.exam_weight || '')}</div>
            </button>`;
          }).join('');
          return `<div class="kg-ch-row ${esc(ch.status)}">
            <div class="kg-ch-meta">
              <div class="kg-ch-no"><span>${esc(ch.no)}</span><span class="kg-domain-tag">${esc(ch.domain || '')}</span></div>
              <div class="kg-ch-name">${esc(ch.title)}</div>
              <div class="kg-ch-stat num">点亮 ${ch.lit_sections}/${ch.total_sections} 节${ch.mastery ? ` · 掌握 ${ch.mastery}%` : ''}</div>
            </div>
            <div class="kg-nodes-flow">${nodesHTML}</div>
          </div>`;
        }).join('');

        return `<section class="kg-book-section" data-book-sec="${esc(b.book_id)}">
          <div class="kg-book-head">
            <div class="kg-book-title"><h3>${esc(b.title)}</h3><span>共 ${b.chapters.length} 章 · ${tot} 节</span></div>
            <div class="kg-book-prog"><span class="num">${lit}/${tot} 已点亮 (${b.progress_pct}%)</span><span class="mini-bar"><i style="width:${b.progress_pct}%"></i></span></div>
          </div>
          <div class="kg-chapters-grid">${chRows}</div>
        </section>`;
      }

      // 英语按册单元网格
      const uCards = b.units.map((u) => {
        const full = nmap.get(u.unit_id);
        const hit = matchesQuery(full, q, false) && (!clusterSet || clusterSet.has(u.unit_id));
        const sel = state.selectedNodeKey === u.unit_id;
        const badge = u.status === 'learning'
          ? '<span class="kgn-badge today">今天在学</span>'
          : u.session_practice
            ? `<span class="kgn-badge today">刚练 ${u.session_practice.right}/${u.session_practice.total}</span>`
            : u.status === 'mastered'
              ? `<span class="kgn-badge pct num">${u.mastery}%</span>`
              : '';
        return `<button type="button" class="kg-node st-${esc(u.status)}${sel ? ' selected' : ''}${!hit ? ' dimmed' : ''}" data-node-key="${esc(u.unit_id)}">
          <div class="kgn-top"><span class="kgn-code">${esc(u.code)}</span>${badge}</div>
          <div class="kgn-title">${esc(u.title)}</div>
          <div class="kgn-sub">${esc(u.cn_title || '')} · ${esc(u.grammar_focus || '')}</div>
        </button>`;
      }).join('');

      return `<section class="kg-book-section" data-book-sec="${esc(b.book_id)}">
        <div class="kg-book-head">
          <div class="kg-book-title"><h3>${esc(b.title)}</h3><span>共 ${tot} 单元</span></div>
          <div class="kg-book-prog"><span class="num">${lit}/${tot} 已点亮 (${b.progress_pct}%)</span><span class="mini-bar"><i style="width:${b.progress_pct}%"></i></span></div>
        </div>
        <div class="kg-units-grid">${uCards}</div>
      </section>`;
    }).join('');

    kgCanvas.innerHTML = `<div class="kg-map-stage" id="kgMapStage">
      ${clustersHTML}
      <svg class="kg-edges-svg" id="kgEdgesSvg" aria-hidden="true"></svg>
      ${sectionsHTML}
    </div>`;

    requestAnimationFrame(() => drawEdges(g));
  }

  function drawEdges(g) {
    const svg = $('#kgEdgesSvg');
    const stage = $('#kgMapStage');
    if (!svg || !stage || !g) return;
    const stageRect = stage.getBoundingClientRect();
    if (!stageRect.width || !stageRect.height) return;
    svg.setAttribute('viewBox', `0 0 ${stageRect.width} ${stageRect.height}`);

    const selKey = state.selectedNodeKey;
    const relevantEdges = (g.edges || []).filter((e) => e.from === selKey || e.to === selKey);

    let paths = '';
    for (const e of relevantEdges) {
      const fromEl = stage.querySelector(`[data-node-key="${CSS.escape(e.from)}"]`);
      const toEl = stage.querySelector(`[data-node-key="${CSS.escape(e.to)}"]`);
      if (!fromEl || !toEl) continue;
      const r1 = fromEl.getBoundingClientRect();
      const r2 = toEl.getBoundingClientRect();
      const x1 = r1.left + r1.width / 2 - stageRect.left;
      const y1 = r1.top + r1.height / 2 - stageRect.top;
      const x2 = r2.left + r2.width / 2 - stageRect.left;
      const y2 = r2.top + r2.height / 2 - stageRect.top;
      const dy = Math.max(28, Math.abs(y2 - y1) * 0.45);
      const cls = e.to === selKey ? 'active-prereq' : 'active-next';
      paths += `<path class="kg-edge ${cls}" d="M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${x1.toFixed(1)} ${(y1 + dy).toFixed(1)}, ${x2.toFixed(1)} ${(y2 - dy).toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}"/>`;
    }
    svg.innerHTML = paths;
  }

  function renderSyllabusCanvas(g, nmap) {
    const isMath = g.subject === 'math';
    const books = state.kgBook === 'all' ? g.books : g.books.filter((b) => b.book_id === state.kgBook);
    const q = state.kgQuery.trim();

    const html = books.map((b) => {
      if (isMath) {
        const rows = b.chapters.flatMap((ch) =>
          ch.sections.filter((sec) => matchesQuery(nmap.get(sec.code), q, true)).map((sec) => `
            <button type="button" class="kgs-row${state.selectedNodeKey === sec.code ? ' selected' : ''}" data-node-key="${esc(sec.code)}">
              <span class="num muted">${esc(sec.code)}</span>
              <strong>${esc(sec.title)}</strong>
              <span class="muted">${esc((sec.key_points || []).slice(0, 2).join(' · '))}</span>
              <span class="num">${sec.mastery ? sec.mastery + '%' : STATUS_LABEL[sec.status]}</span>
            </button>
          `)
        ).join('');
        return `<div class="kgs-card"><div class="kgs-head"><h3>${esc(b.title)}</h3><span class="muted num">已点亮 ${b.lit_sections}/${b.total_sections}</span></div><div class="kgs-rows">${rows || '<p class="muted">无匹配小节</p>'}</div></div>`;
      }
      const rows = b.units.filter((u) => matchesQuery(nmap.get(u.unit_id), q, false)).map((u) => `
        <button type="button" class="kgs-row${state.selectedNodeKey === u.unit_id ? ' selected' : ''}" data-node-key="${esc(u.unit_id)}">
          <span class="num muted">${esc(u.code)}</span>
          <strong>${esc(u.title)}</strong>
          <span class="muted">${esc(u.grammar_focus || '')}</span>
          <span class="num">${u.mastery ? u.mastery + '%' : STATUS_LABEL[u.status]}</span>
        </button>
      `).join('');
      return `<div class="kgs-card"><div class="kgs-head"><h3>${esc(b.title)}</h3><span class="muted num">已点亮 ${b.lit_units}/${b.total_units}</span></div><div class="kgs-rows">${rows || '<p class="muted">无匹配单元</p>'}</div></div>`;
    }).join('');

    kgCanvas.innerHTML = `<div class="kg-syllabus-table">${html}</div>`;
  }

  function renderInspector(g, nmap) {
    const isMath = g.subject === 'math';
    const node = nmap.get(state.selectedNodeKey) || [...nmap.values()][0];
    if (!node) {
      kgInspector.innerHTML = '<p class="muted">选择任意章节或单元查看核心考点与前置关系。</p>';
      return;
    }

    const prereqNodes = (node.prereqs || []).map((k) => nmap.get(k)).filter(Boolean);
    const nextNodes = (g.edges || [])
      .filter((e) => e.from === node.key)
      .map((e) => nmap.get(e.to))
      .filter(Boolean);

    if (isMath) {
      const formulasHTML = (node.formulas || []).length
        ? `<div class="kgi-sec"><h4>核心公式与定理</h4>${node.formulas.map((f) => `<div class="kgi-formula">${tex(f, false)}</div>`).join('')}</div>`
        : '';
      const practiceHTML = node.session_practice
        ? `<div class="kgi-sec"><h4>今晚台灯实拍作答</h4><p class="muted">本节关联 ${node.session_practice.total} 题，答对 ${node.session_practice.right} 题，错 ${node.session_practice.wrong} 题。</p></div>`
        : '';

      kgInspector.innerHTML = `
        <div class="kgi-eyebrow">
          <span>${esc(node.book.title)} · ${esc(node.chapter.no)}</span>
          <span class="kgi-status ${esc(node.status)}">${esc(STATUS_LABEL[node.status] || node.status)}</span>
        </div>
        <h3 class="kgi-title">${esc(node.code)} ${esc(node.title)}</h3>
        <p class="kgi-sub">${esc(node.chapter.title)} · ${esc(node.chapter.domain || '')} · ${esc(node.exam_weight || '核心考点')}</p>

        <div class="kgi-meter">
          <div class="kgi-meter-row"><span>当前掌握度</span><strong class="num">${node.mastery}%</strong></div>
          <div class="kgi-meter-bar"><i style="width:${node.mastery}%"></i></div>
        </div>

        <div class="kgi-sec">
          <h4>核心知识点清单</h4>
          <ul class="kgi-list">${(node.key_points || []).map((p) => `<li>${esc(p)}</li>`).join('')}</ul>
        </div>

        ${formulasHTML}
        ${practiceHTML}

        <div class="kgi-sec">
          <h4>前置基础节点（绿线溯源）</h4>
          <div class="kgi-rel-btns">
            ${prereqNodes.length ? prereqNodes.map((p) => `<button type="button" class="kgi-rel-btn" data-node-key="${esc(p.key)}">← ${esc(p.code)} ${esc(p.title)}</button>`).join('') : '<span class="muted" style="font-size:.78rem">起始基础节</span>'}
          </div>
        </div>

        <div class="kgi-sec">
          <h4>后续进阶节点（金线延展）</h4>
          <div class="kgi-rel-btns">
            ${nextNodes.length ? nextNodes.map((n) => `<button type="button" class="kgi-rel-btn" data-node-key="${esc(n.key)}">${esc(n.code)} ${esc(n.title)} →</button>`).join('') : '<span class="muted" style="font-size:.78rem">阶段综合收官节点</span>'}
          </div>
        </div>

        <div class="kgi-actions">
          <button type="button" class="kgi-cta" data-route-go="workbench" data-wb-tab="hw">去台灯批改本节配套作业 →</button>
        </div>
      `;
      return;
    }

    // 英语节点详情
    kgInspector.innerHTML = `
      <div class="kgi-eyebrow">
        <span>${esc(node.book.title)} · ${esc(node.code)}</span>
        <span class="kgi-status ${esc(node.status)}">${esc(STATUS_LABEL[node.status] || node.status)}</span>
      </div>
      <h3 class="kgi-title">${esc(node.title)}</h3>
      <p class="kgi-sub">${esc(node.cn_title || '')} · ${esc(node.topic || '')}</p>

      <div class="kgi-meter">
        <div class="kgi-meter-row"><span>单元综合掌握度（词汇 ${node.vocab_count || 30} 词）</span><strong class="num">${node.mastery}%</strong></div>
        <div class="kgi-meter-bar"><i style="width:${node.mastery}%"></i></div>
      </div>

      <div class="kgi-sec">
        <h4>核心语法 · ${esc(node.grammar_focus || '')}</h4>
        <ul class="kgi-list">${(node.grammar_points || []).map((p) => `<li>${esc(p)}</li>`).join('')}</ul>
      </div>

      <div class="kgi-sec">
        <h4>重点高频词汇</h4>
        <div class="kgi-vocab-chips">${(node.key_vocab || []).map((w) => `<span class="kgi-vocab">${esc(w)}</span>`).join('')}</div>
      </div>

      ${(node.key_patterns || []).length ? `<div class="kgi-sec"><h4>经典教材句型</h4><ul class="kgi-list">${node.key_patterns.map((s) => `<li>${esc(s)}</li>`).join('')}</ul></div>` : ''}

      <div class="kgi-sec">
        <h4>语法递进关联</h4>
        <div class="kgi-rel-btns">
          ${prereqNodes.map((p) => `<button type="button" class="kgi-rel-btn" data-node-key="${esc(p.key)}">← ${esc(p.book.title)} ${esc(p.code)}</button>`).join('')}
          ${nextNodes.map((n) => `<button type="button" class="kgi-rel-btn" data-node-key="${esc(n.key)}">${esc(n.book.title)} ${esc(n.code)} →</button>`).join('')}
        </div>
      </div>

      <div class="kgi-actions">
        <button type="button" class="kgi-cta" data-route-go="workbench" data-wb-tab="hw">去台灯批改本单元作业 →</button>
      </div>
    `;
  }

  function renderGraphView(scrollToSelected = false) {
    const g = state.graphs[state.kgSubj];
    $$('[data-kg-subj]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.kgSubj === state.kgSubj)));
    $$('[data-kg-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.kgMode === state.kgMode)));

    if (!g || !g.books?.length) {
      kgCanvas.innerHTML = '<p class="empty">正在加载知识图谱数据…</p>';
      return;
    }

    $('#kgPublisher').textContent = g.publisher;
    const nmap = allNodesMap(g);
    if (!state.selectedNodeKey || !nmap.has(state.selectedNodeKey)) {
      state.selectedNodeKey = g.subject === 'math' ? g.current.section_code : g.current.unit_id;
    }

    renderGraphBanner(g);
    renderBookTabs(g);
    if (state.kgMode === 'syllabus') renderSyllabusCanvas(g, nmap);
    else renderMapCanvas(g, nmap);
    renderInspector(g, nmap);

    if (scrollToSelected) {
      setTimeout(() => {
        const el = kgCanvas.querySelector(`[data-node-key="${CSS.escape(state.selectedNodeKey)}"]`);
        if (el) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }, 60);
    }
  }

  /* ---------- 单词背诵 / 笔记检查轻量二级页 ---------- */
  function renderFeatureView() {
    const body = $('#vfBody');
    if (state.route === 'vocab') {
      $('#vfTitle').textContent = '单词背诵 · 九年级全一册 Unit 5 智能推荐词单';
      const en = state.graphs.en;
      const u5 = en?.books?.flatMap((b) => b.units || []).find((u) => u.unit_id === 'en_9_u5');
      const words = u5?.key_vocab || [
        'glass n. 玻璃', 'cotton n. 棉；棉花', 'steel n. 钢；钢铁', 'silver n. 银；银器',
        'chopstick n. 筷子', 'coin n. 硬币', 'fork n. 餐叉', 'blouse n. 女式短上衣',
        'produce v. 生产；制造', 'widely adv. 广泛地', 'process v. 加工；处理', 'pack v. 包装；装箱',
      ];
      body.innerHTML = `
        <p class="muted">结合今日正在学的 <b>九年级全一册 · Unit 5 What are the shirts made of?</b> 与遗忘曲线排程，为你挑选以下 ${words.length} 个核心高频词。可将听写纸放到台灯下自动批改。</p>
        <div class="vf-grid">
          ${words.map((item, idx) => {
            const parts = item.split(/\s+/);
            const head = parts[0] || item;
            const rest = parts.slice(1).join(' ');
            return `<div class="vf-word-card"><small class="num muted">#${String(idx + 1).padStart(2, '0')} · 九全 Unit 5</small><strong>${esc(head)}</strong><span>${esc(rest)}</span></div>`;
          }).join('')}
        </div>
      `;
    } else {
      $('#vfTitle').textContent = '笔记检查 · 学习习惯诊断';
      body.innerHTML = `
        <p class="muted">上次检查得分 <b>73 分</b>（结构清晰度 82 · 核心公式提炼 76 · 错题红笔订正 64 · 卷面书写 72）。把课堂笔记本平铺在台灯下点击右上角进入实拍工作台即可抓拍诊断。</p>
        <div class="vf-grid">
          <div class="vf-word-card"><small class="muted">01 结构层次</small><strong>82 分</strong><span>章节标题与例题编号分明，建议补充思维导图小结</span></div>
          <div class="vf-word-card"><small class="muted">02 重点标注</small><strong>76 分</strong><span>已记录求根公式与判别式，需补充因式分解丢根警示</span></div>
          <div class="vf-word-card"><small class="muted">03 错题批注</small><strong>64 分</strong><span>部分错题仅抄写正答，建议写出“错因溯源”一步</span></div>
          <div class="vf-word-card"><small class="muted">04 书写规范</small><strong>72 分</strong><span>分式分数线与根号书写较工整，注意演草分区</span></div>
        </div>
      `;
    }
  }

  /* ---------- 事件绑定 ---------- */
  document.addEventListener('click', (e) => {
    const goBtn = e.target.closest('[data-route-go]');
    if (goBtn) {
      e.preventDefault();
      navigate(goBtn.dataset.routeGo, { wbTab: goBtn.dataset.wbTab });
      return;
    }
    const subjBtn = e.target.closest('[data-kg-subj]');
    if (subjBtn) {
      navigate(subjBtn.dataset.kgSubj);
      return;
    }
    const modeBtn = e.target.closest('[data-kg-mode]');
    if (modeBtn) {
      state.kgMode = modeBtn.dataset.kgMode;
      renderGraphView(false);
      return;
    }
    const bookBtn = e.target.closest('[data-book-id]');
    if (bookBtn) {
      state.kgBook = bookBtn.dataset.bookId;
      renderGraphView(false);
      return;
    }
    const clusterBtn = e.target.closest('[data-cluster-id]');
    if (clusterBtn) {
      const id = clusterBtn.dataset.clusterId;
      state.kgCluster = state.kgCluster === id ? null : id;
      renderGraphView(false);
      return;
    }
    if (e.target.closest('[data-clear-cluster]')) {
      state.kgCluster = null;
      renderGraphView(false);
      return;
    }
    const jumpBtn = e.target.closest('[data-jump-current]');
    if (jumpBtn) {
      state.kgBook = 'all';
      state.selectedNodeKey = jumpBtn.dataset.jumpCurrent;
      renderGraphView(true);
      return;
    }
    const nodeBtn = e.target.closest('[data-node-key]');
    if (nodeBtn && !viewGraph.hidden) {
      const key = nodeBtn.dataset.nodeKey;
      const g = state.graphs[state.kgSubj];
      const nmap = allNodesMap(g);
      const target = nmap.get(key);
      if (target && state.kgBook !== 'all' && target.book.book_id !== state.kgBook) {
        state.kgBook = 'all';
      }
      state.selectedNodeKey = key;
      renderGraphView(nodeBtn.classList.contains('kgi-rel-btn'));
    }
  });

  const openMath = () => navigate('math');
  const openEn = () => navigate('en');
  $('#cardMath').addEventListener('click', openMath);
  $('#cardMath').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openMath(); } });
  $('#cardEn').addEventListener('click', openEn);
  $('#cardEn').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openEn(); } });

  $('#kgBack').addEventListener('click', () => navigate('overview'));
  $('#lampPillClose').addEventListener('click', () => {
    state.pillDismissed = true;
    $('#lampPill').hidden = true;
  });

  $('#kgSearch').addEventListener('input', (e) => {
    state.kgQuery = e.target.value || '';
    renderGraphView(false);
  });

  kgCanvasWrap.addEventListener('scroll', () => {
    if (state.kgMode === 'map' && !viewGraph.hidden) drawEdges(state.graphs[state.kgSubj]);
  }, { passive: true });
  window.addEventListener('resize', () => {
    if (state.kgMode === 'map' && !viewGraph.hidden) drawEdges(state.graphs[state.kgSubj]);
  });

  // 支持 URL hash 直达或浏览器后退
  function applyInitialHash() {
    const h = (location.hash || '').replace(/^#/, '');
    if (h === 'math' || h === 'en' || h === 'vocab' || h === 'notes') {
      navigate(h, { fromPop: true });
    } else if (h.startsWith('workbench')) {
      const tab = h.split('/')[1] || 'hw';
      navigate('workbench', { wbTab: tab, fromPop: true });
    } else {
      navigate('overview', { fromPop: true });
    }
  }
  window.addEventListener('hashchange', applyInitialHash);

  function onSessionUpdated(evType) {
    renderRailStats();
    if (evType === 'page.grade_done' || evType === 'question.verdict' || evType === 'session.snapshot') {
      loadOverview();
      if (state.route === 'math' || state.route === 'en') {
        ensureGraph(state.kgSubj, true).then(() => renderGraphView(false));
      }
    }
  }

  return {
    state,
    navigate,
    init: applyInitialHash,
    tick: updateClockAndGreeting,
    onSessionUpdated,
  };
}
