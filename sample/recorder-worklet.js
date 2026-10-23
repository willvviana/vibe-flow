// Runs on the audio thread. Copies the captured stereo signal, untouched, into 32-bit float blocks and reports peak/RMS levels.
class RecProc extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = false;
    this.buf = new Float32Array(2 * 4096);
    this.n = 0;
    this.lvl = { pl: 0, pr: 0, sl: 0, sr: 0, n: 0, clip: false };
    this.port.onmessage = (e) => {
      if (e.data.cmd === 'state') this.on = !!e.data.on;
      else if (e.data.cmd === 'flush') { this.flush(); this.port.postMessage({ flushed: true }); }
    };
  }
  flush() {
    if (!this.n) return;
    const out = this.buf.slice(0, this.n * 2);
    this.port.postMessage({ pcm: out.buffer, frames: this.n }, [out.buffer]);
    this.n = 0;
  }
  process(inputs) {
    const inp = inputs[0];
    if (!inp || !inp.length) return true;
    const L = inp[0], R = inp[1] || inp[0], len = L.length, v = this.lvl;
    for (let i = 0; i < len; i++) {
      const l = L[i], r = R[i], al = l < 0 ? -l : l, ar = r < 0 ? -r : r;
      if (al > v.pl) v.pl = al;
      if (ar > v.pr) v.pr = ar;
      v.sl += l * l; v.sr += r * r;
      if (al >= 0.999 || ar >= 0.999) v.clip = true;
      if (this.on) { this.buf[this.n * 2] = l; this.buf[this.n * 2 + 1] = r; this.n++; if (this.n >= 4096) this.flush(); }
    }
    v.n += len;
    if (v.n >= 2048) {
      this.port.postMessage({ lvl: { pl: v.pl, pr: v.pr, rl: Math.sqrt(v.sl / v.n), rr: Math.sqrt(v.sr / v.n), clip: v.clip } });
      this.lvl = { pl: 0, pr: 0, sl: 0, sr: 0, n: 0, clip: false };
    }
    return true;
  }
}
registerProcessor('rec-proc', RecProc);
