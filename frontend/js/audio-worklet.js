// AudioWorklet：把麦克风输入（通常 48kHz Float32）降采样到 16kHz，每 20ms（320 样本）打包成 Int16 发给主线程。
// 降采样用分数步长的盒式平均（先低通再抽取），对语音识别足够。

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.phase = 0;
    this.sum = 0;
    this.cnt = 0;
    this.frame = new Int16Array(320);
    this.n = 0;
    this.energy = 0;
  }

  push(v) {
    const s = v < -1 ? -1 : v > 1 ? 1 : v;
    this.frame[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    this.energy += s * s;
    if (this.n === 320) {
      const out = this.frame;
      this.port.postMessage({ pcm: out.buffer, level: Math.sqrt(this.energy / 320) }, [out.buffer]);
      this.frame = new Int16Array(320);
      this.n = 0;
      this.energy = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.sum += ch[i];
      this.cnt++;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        this.push(this.sum / this.cnt);
        this.sum = 0;
        this.cnt = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCapture);
