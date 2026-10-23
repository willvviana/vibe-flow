const test = require('node:test');
const assert = require('node:assert');
const D = require('../editor/dsp.js');

const SR = 22050;
const sine = (f, secs, amp = 0.5, sr = SR) => Float32Array.from({ length: Math.floor(secs * sr) }, (_, i) => amp * Math.sin(2 * Math.PI * f * i / sr));
function peakHz(x, sr = SR) {
  const n = 1 << Math.floor(Math.log2(x.length)), re = new Float64Array(n), im = new Float64Array(n), w = D.hann(n);
  for (let i = 0; i < n; i++) re[i] = x[i] * w[i];
  D.fft(re, im, false);
  let best = 1, bv = 0;
  for (let k = 1; k < n / 2; k++) { const m = Math.hypot(re[k], im[k]); if (m > bv) { bv = m; best = k; } }
  return best * sr / n;
}
const rms = (x) => D.rmsPeak(x).rms;
const db = (v) => 20 * Math.log10(v + 1e-12);
let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 * 2 - 1; };

test('fft/ifft round trip', () => {
  const n = 1024, re = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.3) + 0.2 * Math.cos(i * 1.7)), im = new Float64Array(n), orig = Float64Array.from(re);
  D.fft(re, im, false); D.fft(re, im, true);
  for (let i = 0; i < n; i++) assert.ok(Math.abs(re[i] - orig[i]) < 1e-9);
});

test('timeStretch changes duration but not pitch', () => {
  const x = sine(440, 3);
  for (const rate of [0.75, 1.25, 1.5]) {
    const y = D.timeStretch(x, rate);
    assert.ok(Math.abs(y.length / x.length - 1 / rate) < 0.01, `length for rate ${rate}: ${y.length / x.length}`);
    const f = peakHz(y.subarray(4000, y.length - 4000));
    assert.ok(Math.abs(f - 440) < 8, `pitch preserved at rate ${rate}: ${f}`);
    assert.ok(Math.abs(db(rms(y)) - db(rms(x))) < 1.5, 'level preserved');
  }
});

test('pitchShift moves pitch by the requested semitones and keeps the duration', () => {
  const x = sine(440, 3);
  for (const st of [-5, 3, 7]) {
    const y = D.pitchShift(x, st);
    assert.equal(y.length, x.length);
    const ratio = peakHz(y.subarray(4000, y.length - 4000)) / 440;
    assert.ok(Math.abs(ratio / Math.pow(2, st / 12) - 1) < 0.02, `${st} st -> ratio ${ratio}`);
  }
});

test('resample (speed with pitch) scales both', () => {
  const y = D.resample(sine(440, 2), 1.5);
  assert.ok(Math.abs(y.length - 2 * SR / 1.5) < 2);
  assert.ok(Math.abs(peakHz(y.subarray(2000, y.length - 2000)) - 660) < 8);
});

test('noiseReduce lowers the noise floor substantially and keeps the wanted tone', () => {
  const n = SR * 4, x = new Float32Array(n);
  const tone = sine(600, 4, 0.4);
  for (let i = 0; i < n; i++) {
    const burst = i > SR * 1.5 && i < SR * 3 ? tone[i] : 0;            // the tone plays only from 1.5 s to 3 s; the rest is the noise floor
    x[i] = burst + 0.03 * rnd();
  }
  const y = D.noiseReduce(x, SR, 0.8);
  const noiseBefore = rms(x.subarray(0, SR)), noiseAfter = rms(y.subarray(0, SR));
  assert.ok(db(noiseBefore) - db(noiseAfter) > 8, `noise reduced by ${db(noiseBefore) - db(noiseAfter)} dB`);
  const seg = (a) => a.subarray(Math.floor(SR * 1.8), Math.floor(SR * 2.7));
  assert.ok(Math.abs(db(rms(seg(y))) - db(rms(seg(x)))) < 2, 'tone level kept within 2 dB');
  assert.ok(Math.abs(peakHz(seg(y)) - 600) < 10);
  assert.equal(D.noiseReduce(x, SR, 0)[100], x[100], 'amount 0 is a pass-through');
});

test('silenceRegions finds the audible pieces with padding', () => {
  const x = new Float32Array(SR * 6);
  x.set(sine(300, 1, 0.5), SR * 1);              // 1..2 s
  x.set(sine(300, 1, 0.5), SR * 4);              // 4..5 s
  const r = D.silenceRegions(x, SR, -45, 0.3, 0.05);
  assert.equal(r.length, 2);
  assert.ok(Math.abs(r[0][0] - 0.95) < 0.02 && Math.abs(r[0][1] - 2.05) < 0.02, JSON.stringify(r));
  assert.ok(Math.abs(r[1][0] - 3.95) < 0.02 && Math.abs(r[1][1] - 5.05) < 0.02);
  const none = D.silenceRegions(new Float32Array(SR), SR);
  assert.equal(none.length, 0);
});

test('normalizeGain hits the RMS target without exceeding -1 dBFS peak', () => {
  const quiet = [sine(200, 1, 0.01)];
  const g = D.normalizeGain(quiet, -18);
  const y = quiet[0].map((v) => v * g);
  assert.ok(Math.abs(db(rms(y)) + 18) < 0.5 || D.rmsPeak(y).peak <= 0.8914);
  const peaky = [Float32Array.from({ length: 1000 }, (_, i) => (i === 500 ? 0.9 : 0.001))];
  const gp = D.normalizeGain(peaky, -18);
  assert.ok(0.9 * gp <= Math.pow(10, -1 / 20) + 1e-6, 'peak limited');
});

test('processClip applies window, reverse, speed and loops in the documented order', () => {
  const ramp = Float32Array.from({ length: SR * 4 }, (_, i) => i / (SR * 4));
  const out = D.processClip([ramp], SR, 1, 2, { reverse: true, speed: 1, loops: 2, keepPitch: true, pitch: 0, noiseReduce: 0, normalize: false });
  assert.equal(out[0].length, 2 * 2 * SR);                           // 2 s window x 2 loops
  assert.ok(out[0][0] > out[0][SR * 2 - 1], 'reversed: starts high and falls');
  assert.ok(Math.abs(out[0][0] - 0.75) < 0.01);                      // the window ended at 3 s = 0.75
  assert.equal(out[0][0], out[0][SR * 2]);                          // the loop repeats the processed window
  const sped = D.processClip([sine(440, 4)], SR, 0, 4, { speed: 2, keepPitch: true, loops: 1, reverse: false, pitch: 0, noiseReduce: 0, normalize: false });
  assert.ok(Math.abs(sped[0].length - 2 * SR) < 30);
  assert.ok(D.heavyKey('a', 0, 4, { speed: 2 }) !== D.heavyKey('a', 0, 4, { speed: 1 }));
  assert.equal(D.heavyKey('a', 0, 4, { speed: 1, gainDb: 5 }), D.heavyKey('a', 0, 4, { speed: 1, gainDb: -5 }), 'real-time params do not change the cache key');
});

test('encodeWav produces valid PCM and float files that round-trip', () => {
  const x = sine(440, 0.1);
  for (const bits of [16, 24, 32]) {
    const buf = D.encodeWav([x, x.map((v) => -v)], 44100, bits), v = new DataView(buf);
    assert.equal(String.fromCharCode(v.getUint8(0), v.getUint8(1), v.getUint8(2), v.getUint8(3)), 'RIFF');
    assert.equal(v.getUint16(22, true), 2); assert.equal(v.getUint32(24, true), 44100); assert.equal(v.getUint16(34, true), bits);
    assert.equal(buf.byteLength, 44 + x.length * 2 * bits / 8);
    const i = 100, off = 44 + i * 2 * bits / 8;
    const got = bits === 16 ? v.getInt16(off, true) / 32768 : bits === 32 ? v.getFloat32(off, true) : (((v.getUint8(off) | v.getUint8(off + 1) << 8 | v.getUint8(off + 2) << 16) << 8) >> 8) / 8388608;
    assert.ok(Math.abs(got - x[i]) < (bits === 16 ? 1e-4 : 1e-6), `${bits}-bit sample ${got} vs ${x[i]}`);
  }
});

test('peaks and activityDb', () => {
  const p = D.peaks(Float32Array.from([0.1, -0.5, 0.3, 0.9, -0.2, 0.2]), 3);
  assert.deepEqual(Array.from(p).map((v) => +v.toFixed(2)), [-0.5, 0.3, -0.2, 0.9]);
  const a = D.activityDb(sine(200, 1, 0.5), SR, 100);
  assert.equal(a.length, 100);
  assert.ok(Math.abs(a[50] - (-9.03)) < 0.3, 'a 0.5 amplitude sine is about -9 dBFS RMS: ' + a[50]);
});

test('look-ahead limiter: nothing exceeds the ceiling, quiet audio is untouched, and it recovers', () => {
  const SR2 = 48000, n = SR2 * 2;
  const loud = Float32Array.from({ length: n }, (_, i) => 1.6 * Math.sin(i / 15) * (i > SR2 / 2 && i < SR2 * 1.5 ? 1 : 0.2));
  const [l, r] = D.limit([loud.slice(), loud.map((v) => v * 0.7)], SR2, -1, 5, 80);
  const ceil = Math.pow(10, -1 / 20);
  assert.ok(D.rmsPeak(l).peak <= ceil + 1e-6 && D.rmsPeak(r).peak <= ceil + 1e-6, `peak ${D.rmsPeak(l).peak}`);
  const quiet = Float32Array.from({ length: n }, (_, i) => 0.3 * Math.sin(i / 20));
  const q = D.limit([quiet.slice()], SR2, -1, 5, 80)[0];
  assert.ok(quiet.every((v, i) => v === q[i]), 'a signal below the ceiling is returned unchanged');
  const burst = Float32Array.from({ length: n }, (_, i) => (i > SR2 && i < SR2 + 100 ? 1.5 : 0.2) * Math.sin(i / 20));
  const b = D.limit([burst.slice()], SR2, -1, 5, 80)[0];
  assert.ok(D.rmsPeak(b.subarray(0, SR2 / 2)).peak === D.rmsPeak(burst.subarray(0, SR2 / 2)).peak, 'audio far from the peak is untouched');
  assert.ok(D.rmsPeak(b.subarray(SR2 + 3000)).rms > D.rmsPeak(burst.subarray(SR2 + 3000)).rms * 0.9, 'gain recovers after the peak');
});
