// 左栏：按抓拍的试卷/页分组列题，状态点实时更新；错题本按知识点归类；报告只用真实会话数据。

import { S, viewStatus, captureStats, capturesInOrder, scored } from './store.js';
import { $, $$, esc, plain, subjectName } from './util.js';

const STAT = { none: '待识别', idle: '待批改', judging: '判定中', right: '对', wrong: '错', blank: '未作答', uncertain: '待确认', review: '作文点评', listen: '听力（不批改）', failed: '判题失败' };

export function createSidebar(app) {
  const body = $('#sideBody');

  function rowHTML(q) {
    const v = viewStatus(q);
    const fresh = S.ui.fresh.has(q.question_uid);
    const text = v === 'listen' ? '听力（不批改）' : plain(q.short_stem || q.stem || '');
    const sel = S.d.selected_question_uid === q.question_uid;
    return `<button class="row st-${v}${fresh ? ' fresh' : ''}" data-uid="${esc(q.question_uid)}" aria-pressed="${sel}" ` +
      `aria-label="第${esc(q.no)}题，${STAT[v] || ''}" title="${esc(text)}"><span class="n">${esc(q.no)}</span>` +
      `<span class="t">${esc(text)}</span><span class="sd"></span></button>`;
  }

  function scoreHTML(st) {
    if (!st.total && !st.review) return '';
    if (st.judging) return `判定中 ${st.done}/${st.total}`;
    if (st.done < st.total) {
      if (st.failed) return `<span class="r">${st.failed}题失败</span> · ${st.done}/${st.total}`;
      return st.done ? `${st.done}/${st.total}` : '待批改';
    }
    let s = `<b>对${st.right}</b> · <span class="${st.wrong ? 'r' : ''}">错${st.wrong}</span>`;
    if (st.uncertain) s += ` · 待确认${st.uncertain}`;
    if (st.blank) s += ` · 空${st.blank}`;
    if (st.review) s += ` · 评${st.review}`;
    return s;
  }

  function hwHTML() {
    const caps = capturesInOrder();
    if (!caps.length) return '<p class="empty">还没有作业。把作业纸平放到台灯下、保持静止，台灯会自动抓拍并逐题识别。</p>';
    return caps.map((c) => {
      const qs = (c.question_uids || []).map((u) => S.d.questions[u]).filter(Boolean);
      const title = c.paper_id ? `${subjectName(c.subject)} · 第${c.page_no}页` : (S.ui.unmatched.has(c.capture_id) ? '未匹配的页面' : '识别中…');
      const full = c.paper_title ? `${c.paper_title} · 第${c.page_no}页 · ${c.ts || ''}` : (c.ts || '');
      const st = captureStats(c);
      return `<div class="grp${S.ui.view === c.capture_id ? ' viewing' : ''}" data-cap="${esc(c.capture_id)}">` +
        `<button class="grp-h" data-cap="${esc(c.capture_id)}" title="${esc(full)}"><span class="gt">${esc(title)}${qs.length ? ' · ' + st.all + '题' : ''}</span><span class="sc">${scoreHTML(st)}</span></button>` +
        qs.map(rowHTML).join('') + '</div>';
    }).join('');
  }

  function mistakesHTML() {
    const wrong = Object.values(S.d.questions).filter((q) => viewStatus(q) === 'wrong');
    if (!wrong.length) return '<p class="empty">还没有错题。批改完成后，做错的题会按知识点收进这里。</p>';
    const groups = new Map();
    for (const q of wrong) {
      const k = q.tag || '其他';
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(q);
    }
    return [...groups].map(([tag, qs]) => `<div class="grp"><div class="grp-h"><span class="gt">${esc(tag)}</span><span class="sc">${qs.length}题</span></div>${qs.map(rowHTML).join('')}</div>`).join('');
  }

  function reportHTML() {
    const qs = Object.values(S.d.questions).filter(scored);
    const done = qs.filter((q) => ['right', 'wrong', 'blank'].includes(q.status));
    const right = done.filter((q) => q.status === 'right').length;
    const answered = done.filter((q) => q.status !== 'blank').length;
    const mins = Math.max(0, Math.floor((Date.now() / 1000 - (S.d.study_started_at || Date.now() / 1000)) / 60));
    const card = (v, u, l) => `<div class="card"><div class="v">${v}<small>${u}</small></div><div class="l">${l}</div></div>`;
    let h = '<div class="rep"><div class="rep-h"><span>今晚</span><span>实时统计</span></div><div class="cards">' +
      card(mins, '分钟', '学习时长') + card(capturesInOrder().length, '页', '抓拍页数') +
      card(done.length, '题', '已批改') + card(answered ? Math.round((right / answered) * 100) : '—', answered ? '%' : '', '正确率（不含未作答）') + '</div>';
    const tags = new Map();
    for (const q of done) {
      if (q.status === 'blank') continue;
      const k = q.tag || '其他';
      const t = tags.get(k) || { n: 0, ok: 0 };
      t.n++;
      if (q.status === 'right') t.ok++;
      tags.set(k, t);
    }
    if (tags.size) {
      h += '<div class="rep-h"><span>知识点正确率</span><span>按今晚作答</span></div><div class="bars">' +
        [...tags].sort((a, b) => a[1].ok / a[1].n - b[1].ok / b[1].n).map(([k, t]) => {
          const v = Math.round((t.ok / t.n) * 100);
          return `<div class="bar-r${v < 60 ? ' weak' : ''}"><span class="nm">${esc(k)}</span><span class="pc">${t.ok}/${t.n}</span><span class="track"><i style="width:${v}%"></i></span></div>`;
        }).join('') + '</div>';
    }
    return h + '</div>';
  }

  function render() {
    $$('.nav button').forEach((b) => b.setAttribute('aria-current', b.dataset.view === S.ui.nav ? 'page' : 'false'));
    const all = Object.values(S.d.questions).filter((q) => q.qtype !== 'listening');
    $('#cntHw').textContent = all.length || '';
    $('#cntMk').textContent = all.filter((q) => viewStatus(q) === 'wrong').length || '';
    const top = body.scrollTop;
    body.innerHTML = S.ui.nav === 'mistakes' ? mistakesHTML() : S.ui.nav === 'report' ? reportHTML() : hwHTML();
    body.scrollTop = top;
    S.ui.fresh.clear();
  }

  function scrollToCapture(cid) {
    const g = body.querySelector(`.grp[data-cap="${CSS.escape(cid)}"]`);
    if (g) body.scrollTo({ top: Math.max(0, g.offsetTop - 8), behavior: 'smooth' });
  }

  body.addEventListener('click', (e) => {
    const row = e.target.closest('.row');
    if (row) {
      const uid = row.dataset.uid;
      if (row.classList.contains('st-listen')) return;
      app.select(S.d.selected_question_uid === uid && !app.narrow() ? null : uid, { fromList: true });
      return;
    }
    const gh = e.target.closest('button.grp-h');
    if (gh) app.setView(gh.dataset.cap, { fromList: true });
  });
  $$('.nav button').forEach((b) => b.addEventListener('click', () => { S.ui.nav = b.dataset.view; render(); }));

  return { render, scrollToCapture };
}
