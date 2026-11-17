// 语音：下行 24kHz PCM16 无缝排队播放（只播 activeSeq 的帧）、ducking、打断清空；上行 AudioWorklet 16kHz 采集。
// 二进制帧：<BBH 头（frame_type, flags, turn_seq）+ PCM16LE。flags=1 且负载为空 = 句末终止帧。

export const FRAME_MIC = 0x01;
export const FRAME_TTS = 0x02;

export function micSupport() {
  if (!window.isSecureContext) return 'insecure';
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return 'unsupported';
  if (!window.AudioWorkletNode) return 'unsupported';
  return 'ok';
}

export class AudioEngine {
  constructor({ onPlayback, onMicFrame, onNeedGesture }) {
    this.onPlayback = onPlayback;
    this.onMicFrame = onMicFrame;
    this.onNeedGesture = onNeedGesture;
    this.ctx = null;
    this.out = null; // 播放增益（duck）
    this.sources = new Set();
    this.nextTime = 0;
    this.activeSeq = 0;
    this.playing = false;
    this.playingSeq = -1;
    this.ended = new Set(); // 已收到 tts.end 的轮次
    this.idleTimer = 0;
    this.gestureWarned = false;
    this.mic = null;
    this.workletReady = null;
  }

  ensureCtx() {
    if (this.ctx) return this.ctx;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    this.ctx = new AC({ latencyHint: 'interactive' });
    this.out = this.ctx.createGain();
    this.out.gain.value = 1;
    this.out.connect(this.ctx.destination);
    return this.ctx;
  }

  /** 需在用户手势里调用一次，解除浏览器自动播放限制。 */
  unlock() {
    const ctx = this.ensureCtx();
    if (ctx && ctx.state === 'suspended') ctx.resume().catch(() => {});
  }

  get running() {
    return !!this.ctx && this.ctx.state === 'running';
  }

  setActive(seq) {
    if (seq === this.activeSeq) return;
    // 新轮次开始：旧轮次残留音频不再播放。
    this.clearQueue();
    this.activeSeq = seq;
  }

  handleBinary(buf) {
    if (buf.byteLength < 4) return;
    const dv = new DataView(buf);
    const type = dv.getUint8(0);
    const seq = dv.getUint16(2, true);
    if (type !== FRAME_TTS || seq !== this.activeSeq) return; // 只播当前轮次
    const n = (buf.byteLength - 4) >> 1;
    if (n === 0) return; // 句末终止帧（flags=1，负载为空）
    const ctx = this.ensureCtx();
    if (!ctx) return;
    if (ctx.state !== 'running') {
      if (!this.gestureWarned) {
        this.gestureWarned = true;
        this.onNeedGesture?.();
      }
      return;
    }
    const pcm = new Int16Array(buf, 4, n);
    const audio = ctx.createBuffer(1, n, 24000);
    const ch = audio.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = audio;
    src.connect(this.out);
    const now = ctx.currentTime;
    if (this.nextTime < now + 0.02) this.nextTime = now + 0.04; // 首包/欠载时留一点抖动缓冲
    src.start(this.nextTime);
    this.nextTime += audio.duration;
    this.sources.add(src);
    src.onended = () => {
      this.sources.delete(src);
      if (!this.sources.size) this.maybeIdle();
    };
    clearTimeout(this.idleTimer);
    if (!this.playing) {
      this.playing = true;
      this.playingSeq = seq;
      this.onPlayback(true, seq);
    }
  }

  ttsEnd(seq) {
    this.ended.add(seq);
    if (this.ended.size > 32) this.ended.delete(this.ended.values().next().value);
    if (!this.sources.size) this.maybeIdle();
  }

  maybeIdle() {
    clearTimeout(this.idleTimer);
    if (!this.playing) return;
    const wait = this.ended.has(this.playingSeq) ? 0 : 2500; // 句间合成间隙不算停
    this.idleTimer = setTimeout(() => {
      if (!this.sources.size && this.playing) {
        this.playing = false;
        this.onPlayback(false, this.playingSeq);
      }
    }, wait);
  }

  /** voice.duck：约 80ms 平滑降到 0.3 / 恢复 1.0。 */
  duck(on) {
    if (!this.ctx || !this.out) return;
    const g = this.out.gain, t = this.ctx.currentTime;
    g.cancelScheduledValues(t);
    g.setValueAtTime(g.value, t);
    g.linearRampToValueAtTime(on ? 0.3 : 1, t + 0.08);
  }

  clearQueue() {
    for (const s of this.sources) {
      s.onended = null;
      try { s.stop(); } catch { /* 未开始的源 */ }
    }
    this.sources.clear();
    this.nextTime = 0;
    clearTimeout(this.idleTimer);
    const was = this.playing;
    this.playing = false;
    return was;
  }

  /** voice.interrupted：立即清空队列并切换轮次；播放状态由服务端自行置为 false。 */
  interrupt(newSeq) {
    this.clearQueue();
    this.activeSeq = newSeq;
    if (this.ctx && this.out) {
      const t = this.ctx.currentTime;
      this.out.gain.cancelScheduledValues(t);
      this.out.gain.setValueAtTime(1, t);
    }
  }

  /* ---------- 麦克风 ---------- */
  async startMic() {
    if (this.mic) return;
    const sup = micSupport();
    if (sup !== 'ok') throw Object.assign(new Error(sup), { code: sup });
    const ctx = this.ensureCtx();
    if (ctx.state === 'suspended') await ctx.resume().catch(() => {});
    if (!this.workletReady) this.workletReady = ctx.audioWorklet.addModule(new URL('./audio-worklet.js', import.meta.url));
    await this.workletReady;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'pcm-capture', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
    const sink = ctx.createGain();
    sink.gain.value = 0; // 保持节点被拉取，但不回放麦克风
    node.port.onmessage = (e) => this.onMicFrame(new Int16Array(e.data.pcm), e.data.level);
    src.connect(node);
    node.connect(sink);
    sink.connect(ctx.destination);
    this.mic = { stream, src, node, sink };
  }

  stopMic() {
    const m = this.mic;
    if (!m) return;
    this.mic = null;
    m.stream.getTracks().forEach((t) => t.stop());
    try { m.src.disconnect(); m.node.disconnect(); m.sink.disconnect(); } catch { /* 已断开 */ }
    m.node.port.onmessage = null;
  }

  get micOn() {
    return !!this.mic;
  }
}

/** 上行帧：4 字节头 + 320 个 Int16（共 644 字节）。 */
export function micFrame(int16, turnSeq) {
  const buf = new ArrayBuffer(4 + int16.byteLength);
  const dv = new DataView(buf);
  dv.setUint8(0, FRAME_MIC);
  dv.setUint8(1, 0);
  dv.setUint16(2, turnSeq & 0xffff, true);
  new Int16Array(buf, 4).set(int16);
  return buf;
}
