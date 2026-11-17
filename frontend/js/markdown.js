// Local markdown-it; raw HTML is disabled and remote images are not fetched.
import '../vendor/markdown-it/markdown-it.min.js';

export function createMarkdown(renderMath) {
  const md = globalThis.markdownit({ html: false, breaks: true, linkify: false, typographer: false });
  const safeLink = md.validateLink.bind(md);
  md.validateLink = (url) => safeLink(url) && !/^(?:data|javascript|vbscript|file):/i.test(url.trim());
  const escape = md.utils.escapeHtml;
  const linkOpen = md.renderer.rules.link_open || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    tokens[idx].attrSet('rel', 'noopener noreferrer');
    return linkOpen(tokens, idx, options, env, self);
  };
  // Model text must not trigger external tracking/image requests.
  md.renderer.rules.image = (tokens, idx) => escape(tokens[idx].content || '[图片]');

  function closing(src, delim, start) {
    let at = src.indexOf(delim, start);
    while (at >= 0) {
      let slashes = 0;
      for (let j = at - 1; j >= 0 && src[j] === '\\'; j--) slashes++;
      if (slashes % 2 === 0) return at;
      at = src.indexOf(delim, at + delim.length);
    }
    return -1;
  }

  md.inline.ruler.before('escape', 'math', (state, silent) => {
    const src = state.src, pos = state.pos;
    let open, close, display = false;
    if (src.startsWith('\\(', pos)) { open = '\\('; close = '\\)'; }
    else if (src.startsWith('\\[', pos)) { open = '\\['; close = '\\]'; display = true; }
    else if (src.startsWith('$$', pos)) { open = close = '$$'; display = true; }
    else if (src[pos] === '$') { open = close = '$'; }
    else return false;
    const end = closing(src, close, pos + open.length);
    if (end < 0) {
      // Incomplete streamed math remains literal, never fed to KaTeX or Markdown emphasis.
      if (!silent) state.push('text', '', 0).content = src.slice(pos);
      state.pos = src.length;
      return true;
    }
    if (!silent) {
      const token = state.push('math_inline', 'span', 0);
      token.content = src.slice(pos + open.length, end);
      token.meta = { display };
    }
    state.pos = end + close.length;
    return true;
  });
  md.renderer.rules.math_inline = (tokens, idx) => renderMath(tokens[idx].content, tokens[idx].meta.display);

  md.block.ruler.before('fence', 'math_block', (state, startLine, endLine, silent) => {
    const begin = state.bMarks[startLine] + state.tShift[startLine];
    if (state.sCount[startLine] - state.blkIndent >= 4) return false; // indented code
    const line = state.src.slice(begin, state.eMarks[startLine]);
    const open = line.startsWith('$$') ? '$$' : line.startsWith('\\[') ? '\\[' : null;
    if (!open) return false;
    const close = open === '$$' ? '$$' : '\\]';
    let body = line.slice(open.length), last = startLine;
    while (closing(body, close, 0) < 0 && last + 1 < endLine) {
      last++;
      body += '\n' + state.src.slice(state.bMarks[last], state.eMarks[last]);
    }
    const end = closing(body, close, 0);
    if (end < 0 || body.slice(end + close.length).trim()) return false;
    if (silent) return true;
    const token = state.push('math_block', 'div', 0);
    token.block = true;
    token.content = body.slice(0, end).trim();
    token.map = [startLine, last + 1];
    state.line = last + 1;
    return true;
  }, { alt: ['paragraph', 'reference', 'blockquote', 'list'] });
  md.renderer.rules.math_block = (tokens, idx) => '<div class="math-block">' + renderMath(tokens[idx].content, true) + '</div>\n';

  return (text) => {
    const tokens = md.parse(String(text ?? ''), {});
    // rich() is also used inside inline labels. Avoid invalid <p> in those simple cases.
    if (tokens.length === 3 && tokens[0].type === 'paragraph_open' && tokens[1].type === 'inline') {
      return md.renderer.renderInline(tokens[1].children, md.options, {});
    }
    return md.renderer.render(tokens, md.options, {});
  };
}
