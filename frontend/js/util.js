// 通用工具与安全 Markdown/KaTeX 渲染，解析器全部本地加载。
import { createMarkdown } from './markdown.js';

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
export const esc = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const RM = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
export const NARROW = window.matchMedia('(max-width:900px)');

export const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* 隐私模式 */ } },
};

export const pad2 = (n) => String(n).padStart(2, '0');
export function hms(d = new Date(), sec = true) {
  return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + (sec ? ':' + pad2(d.getSeconds()) : '');
}

let seq = 0;
export const cid = (p = 'c') => p + '_' + Date.now().toString(36) + (++seq).toString(36);

/** 后端给的 /api/... 绝对路径转相对路径（前端挂在 / 时等价；挂在子路径时也能用）。 */
export const rel = (u) => (typeof u === 'string' && u.startsWith('/') && !u.startsWith('//') ? u.slice(1) : u || '');

export const SUBJECT = { math: '数学', en: '英语' };
export const subjectName = (s) => SUBJECT[s] || '试卷';

/* ---------- 数学文本 ---------- */

function texFallback(t) {
  return esc(
    t.replace(/\\text\{([^}]*)\}/g, '$1').replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, '$1/$2')
      .replace(/\\neq\s*/g, '≠').replace(/\\pm\s*/g, '±').replace(/\\cdot\s*/g, '·').replace(/\\times\s*/g, '×')
      .replace(/\\left|\\right/g, '').replace(/\^2/g, '²').replace(/\^3/g, '³').replace(/_1/g, '₁').replace(/_2/g, '₂')
      .replace(/\\;|\\,|\\quad|\\ /g, ' ').replace(/\\[a-zA-Z]+/g, '').replace(/[{}]/g, '')
  );
}

const texCache = new Map();
export function tex(src, display = false) {
  const key = (display ? 'D' : 'I') + src;
  const hit = texCache.get(key);
  if (hit) return hit;
  let html;
  try {
    if (window.katex) html = window.katex.renderToString(src, { throwOnError: false, displayMode: display, strict: 'ignore', output: 'html' });
  } catch { /* 走降级 */ }
  if (!html) html = '<span class="mfb">' + texFallback(src) + '</span>';
  if (texCache.size > 600) texCache.clear();
  texCache.set(key, html);
  return html;
}

const renderMarkdown = createMarkdown(tex);

/** 流式时重渲染累计正文；完整 Markdown 与公式各自按语法解析，代码内不解析公式。 */
export function rich(text) {
  return renderMarkdown(text);
}

/** 题干等转为一行纯文本（左栏列表、aria 用）。 */
export function plain(text) {
  return String(text ?? '')
    .replace(/\$\$?([^$]*)\$\$?/g, (_, t) => texFallbackPlain(t))
    .replace(/\\\(|\\\)/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}
function texFallbackPlain(t) {
  return t.replace(/\\text\{([^}]*)\}/g, '$1').replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, '$1/$2')
    .replace(/\\neq\s*/g, '≠').replace(/\\pm\s*/g, '±').replace(/\\cdot\s*/g, '·').replace(/\\left|\\right/g, '')
    .replace(/\^2/g, '²').replace(/\^3/g, '³').replace(/_1/g, '₁').replace(/_2/g, '₂')
    .replace(/\\;|\\,|\\quad|\\ /g, ' ').replace(/\\[a-zA-Z]+/g, '').replace(/[{}]/g, '').replace(/-/g, '−');
}

/** 合并同一帧内的多次重绘请求（不用 rAF：后台标签页里 rAF 会停）。 */
export function batcher(fn) {
  let queued = false;
  return () => {
    if (queued) return;
    queued = true;
    queueMicrotask(() => { queued = false; fn(); });
  };
}
