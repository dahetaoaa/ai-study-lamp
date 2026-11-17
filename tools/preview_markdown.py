"""Screenshot the real Markdown/KaTeX renderer in an isolated headless Firefox profile."""
import functools
import http.server
from pathlib import Path
import subprocess
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
HTML = r'''<!doctype html><meta charset="utf-8">
<link rel="stylesheet" href="/css/styles.css"><link rel="stylesheet" href="/vendor/katex/katex.min.css">
<script src="/vendor/katex/katex.min.js"></script>
<style>body{display:block;padding:30px;overflow:auto}#result{width:820px;border:1px solid #ddd;padding:24px}</style>
<div id="result" class="explain"></div>
<script type="module">
import {rich} from '/js/util.js';
const text = '# Markdown 与公式验收\n\n- **关键结论**：有两个根\n- 方程 $x^2-1=0$\n\n1. 移项\n2. 开平方\n\n> 不遗漏负根。\n\n| 解 | 检验 |\n|---|---|\n| 1 | 正确 |\n| -1 | 正确 |\n\n$$x=\\pm 1$$\n\n```python\nprint("$x$ 保持代码原样")\n```\n\n[安全链接](https://example.org)\n\n<script>alert(1)<\/script>';
document.getElementById('result').innerHTML=rich(text);
</script>'''


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/__test.html':
            body = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='lamp-render-') as profile:
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=ROOT / 'frontend'))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        output = ROOT / 'data/diagnostics' / Path(profile).name
        output.mkdir(parents=True)
        image = output / 'markdown.png'
        try:
            subprocess.run(['firefox', '--headless', '--no-remote', '--profile', profile,
                            '--window-size', '1100,1200', '--screenshot', str(image),
                            f'http://127.0.0.1:{server.server_port}/__test.html'], check=True, timeout=40)
            print(image)
        finally:
            server.shutdown()
            server.server_close()
