// gjs -m tools/check_markdown.mjs
// Parse/render tests, not a browser screenshot.
import GLib from 'gi://GLib';
globalThis.matchMedia = () => ({ matches: false });
// KaTeX's UMD build expects a script/global context, as in index.html.
const here = GLib.path_get_dirname(GLib.filename_from_uri(import.meta.url)[0]);
const [, katexSource] = GLib.file_get_contents(GLib.build_filenamev([here, '../frontend/vendor/katex/katex.min.js']));
(0, eval)(new TextDecoder().decode(katexSource));
const { rich } = await import('../frontend/js/util.js');
let count = 0;
function check(text, predicate, description) {
  const html = rich(text);
  if (!predicate(html)) throw new Error(description + ': ' + html);
  count++;
}
check('# 标题\n\n- 条目一\n- **条目二**', h => h.includes('<h1>') && h.includes('<ul>') && h.includes('<strong>'), 'heading/list/bold');
check('1. 第一项\n2. 第二项\n\n> 引用', h => h.includes('<ol>') && h.includes('<blockquote>'), 'ordered list/quote');
check('| 名称 | 数值 |\n|---|---|\n| x | 2 |', h => h.includes('<table>') && h.includes('<td>2</td>'), 'table');
check('结果为 $x^2=1$，因此 **有两个根**。', h => h.includes('katex') && h.includes('<strong>'), 'mixed math');
check('\\[\nx^2=1\n\\]', h => h.includes('math-block') && h.includes('katex-display'), 'bracket block math');
check('$$\nx_1=1\n$$', h => h.includes('katex-display'), 'dollar block math');
check('使用 \\(x=1\\) 验证。', h => h.includes('katex'), 'inline parentheses math');
check('`$x$` 和 $y$', h => h.includes('<code>$x$</code>') && h.includes('katex'), 'code is not math');
check('```python\nprint("$x$")\n```', h => h.includes('<pre><code') && !h.includes('katex'), 'fenced code is not math');
check('```html\n<script>alert(1)</script>', h => h.includes('&lt;script&gt;') && !h.includes('<script>'), 'unfinished safe code');
check('<img src=x onerror=alert(1)>', h => !h.includes('<img') && h.includes('&lt;img'), 'raw HTML disabled');
check('[点击](javascript:alert(1))', h => !h.includes('href='), 'javascript link blocked');
check('[点击](data:text/html,test)', h => !h.includes('href='), 'data link blocked');
check('![图片](https://example.org/track.png)', h => !h.includes('<img'), 'no remote image tracking');
check('[资料](https://example.org)', h => h.includes('href="https://example.org"') && h.includes('noopener'), 'safe link');
check('计算 $x^2', h => h.includes('$x^2') && !h.includes('katex'), 'unfinished formula');
check('计算 $x^2=1$ 完成', h => h.includes('katex'), 'completed streamed formula');
check('**还没写完', h => !h.includes('<script'), 'unfinished emphasis');
check('普通文字', h => h === '普通文字', 'inline consumers have no p wrapper');
print(`Markdown/math/security: ${count} checks passed.`);
