// 单条多路复用 WebSocket：JSON 控制/流式事件 + 二进制 PCM 语音帧。断线指数退避重连，
// 重连后服务端首包就是 session.snapshot，由上层整体替换状态完成恢复。

export class WSTransport {
  constructor({ onJSON, onBinary, onState }) {
    this.onJSON = onJSON;
    this.onBinary = onBinary;
    this.onState = onState;
    this.ws = null;
    this.retry = 0;
    this.timer = 0;
    this.closedByUser = false;
  }

  get url() {
    const u = new URL('ws', location.href); // 相对当前页面目录，前端挂在 / 时即 /ws
    u.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    u.search = '';
    u.hash = '';
    return u.href;
  }

  get open() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  connect() {
    clearTimeout(this.timer);
    this.onState('connecting');
    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.schedule();
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      this.onState('open');
    };
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') {
        let data;
        try {
          data = JSON.parse(e.data);
        } catch {
          return;
        }
        if (data && typeof data === 'object') this.onJSON(data);
      } else if (e.data instanceof ArrayBuffer) {
        this.onBinary(e.data);
      }
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.onState('closed');
      if (!this.closedByUser) this.schedule();
    };
    ws.onerror = () => { /* onclose 会跟着触发 */ };
  }

  schedule() {
    const delay = Math.min(5000, 400 * 2 ** this.retry) + Math.random() * 250;
    this.retry = Math.min(this.retry + 1, 6);
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.connect(), delay);
  }

  send(obj) {
    if (!this.open) return false;
    this.ws.send(JSON.stringify(obj));
    return true;
  }

  sendBinary(buf) {
    if (!this.open) return false;
    // 背压保护：发送缓冲积压超过 ~1s 音频就丢帧，避免语音延迟越积越大。
    if (this.ws.bufferedAmount > 64 * 1024) return false;
    this.ws.send(buf);
    return true;
  }
}
