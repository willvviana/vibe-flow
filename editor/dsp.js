/* Editor DSP: real, offline signal processing on Float32Array channels. Pure JS (no DOM), used from a Web Worker and from Node tests.
 *   timeStretch  phase vocoder (tempo change, pitch preserved)        pitchShift  stretch + resample (pitch change, duration preserved)
 *   resample     cubic interpolation (speed change WITH pitch)         noiseReduce spectral subtraction with a noise profile from the quietest frames
 *   silenceRegions, normalizeGain, reverse, peaks, activityDb (ducking trigger), makeImpulse (reverb IR), encodeWav, processClip (heavy chain)
 * Limits (be honest in the UI): the phase vocoder smears sharp transients for large factors (usable ~0.5x..2x); noise reduction can leave faint
 * "watery" artefacts at high amounts; pitch shifting is not formant-preserving.                                                           */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.EditorDSP = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const TWO_PI = Math.PI * 2;

  // ------------------------------------------------------------ FFT (radix-2, in place)
  const _tw = {};
  function twiddles(n) {
    if (_tw[n]) return _tw[n];
    const c = new Float64Array(n / 2), s = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { c[i] = Math.cos(TWO_PI * i / n); s[i] = -Math.sin(TWO_PI * i / n); }
    const rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
    return (_tw[n] = { c, s, rev });
  }
  function fft(re, im, inverse) {
    const n = re.length, { c, s, rev } = twiddles(n);
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const wr = c[k], wi = inverse ? -s[k] : s[k];
          const a = i + j, b = a + half;
          const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  const hann = (n) => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(TWO_PI * i / n); return w; };

  // ------------------------------------------------------------ basic helpers
  const reverse = (x) => { const y = new Float32Array(x.length); for (let i = 0; i < x.length; i++) y[i] = x[x.length - 1 - i]; return y; };
  function rmsPeak(x) { let s = 0, p = 0; for (let i = 0; i < x.length; i++) { const v = x[i]; s += v * v; if (Math.abs(v) > p) p = Math.abs(v); } return { rms: Math.sqrt(s / Math.max(1, x.length)), peak: p }; }
  /** Gain that brings the clip to a standard level: target RMS (default -18 dBFS), never pushing the peak above -1 dBFS. */
  function normalizeGain(chans, targetRmsDb) {
    let ss = 0, n = 0, pk = 0;
    for (const x of chans) { for (let i = 0; i < x.length; i++) { const v = x[i]; ss += v * v; if (Math.abs(v) > pk) pk = Math.abs(v); } n += x.length; }
    const rms = Math.sqrt(ss / Math.max(1, n));
    if (rms < 1e-6 || pk < 1e-6) return 1;
    return Math.min(Math.pow(10, (targetRmsDb == null ? -18 : targetRmsDb) / 20) / rms, Math.pow(10, -1 / 20) / pk);
  }
  /** Cubic (Catmull-Rom) resampling: output length = input / ratio; ratio > 1 plays faster and raises pitch. */
  function resample(x, ratio) {
    if (Math.abs(ratio - 1) < 1e-9) return Float32Array.from(x);
    const n = Math.max(1, Math.floor(x.length / ratio)), y = new Float32Array(n), L = x.length - 1;
    for (let i = 0; i < n; i++) {
      const p = i * ratio, i1 = Math.floor(p), f = p - i1;
      const a = x[Math.max(i1 - 1, 0)], b = x[Math.min(i1, L)], c = x[Math.min(i1 + 1, L)], d = x[Math.min(i1 + 2, L)];
      y[i] = b + 0.5 * f * (c - a + f * (2 * a - 5 * b + 4 * c - d + f * (3 * (b - c) + d - a)));
    }
    return y;
  }

  // ------------------------------------------------------------ phase vocoder
  /** rate > 1 = faster (shorter), rate < 1 = slower (longer); pitch preserved.
   *  Phase vocoder with identity phase locking (Laroche & Dolson): the phase of every spectral peak is advanced by its measured
   *  instantaneous frequency and the bins around a peak keep their analysis phase relationship to it, so partials stay coherent
   *  (a plain per-bin vocoder loses ~2 dB of level and sounds "phasey"). */
  function timeStretch(x, rate, nFft, hop) {
    if (Math.abs(rate - 1) < 1e-4) return Float32Array.from(x);
    nFft = nFft || 2048; hop = hop || 512;
    const win = hann(nFft), pad = nFft;
    const xp = new Float32Array(x.length + 2 * pad); xp.set(x, pad);
    const nOut = Math.floor(x.length / rate), nFrames = Math.ceil((nOut + 2 * pad) / hop) + 1;
    const out = new Float64Array(nOut + 4 * pad), norm = new Float64Array(out.length);
    const re1 = new Float64Array(nFft), im1 = new Float64Array(nFft), re2 = new Float64Array(nFft), im2 = new Float64Array(nFft);
    const bins = nFft / 2 + 1, omega = new Float64Array(bins);
    let phase = new Float64Array(bins), newPhase = new Float64Array(bins);
    const prevMag = new Float64Array(bins), mag = new Float64Array(bins), P1 = new Float64Array(bins), P2 = new Float64Array(bins);
    for (let k = 0; k < bins; k++) omega[k] = TWO_PI * hop * k / nFft;
    const frame = (re, im, pos) => { for (let i = 0; i < nFft; i++) { const v = xp[pos + i]; re[i] = (v === undefined ? 0 : v) * win[i]; im[i] = 0; } fft(re, im, false); };
    for (let f = 0; f < nFrames; f++) {
      const a = Math.min(Math.round(f * hop * rate), xp.length - nFft - hop - 1);
      if (a < 0) break;
      frame(re1, im1, a); frame(re2, im2, a + hop);
      for (let k = 0; k < bins; k++) { mag[k] = Math.hypot(re1[k], im1[k]); P1[k] = Math.atan2(im1[k], re1[k]); P2[k] = Math.atan2(im2[k], re2[k]); }
      const peaksIdx = [];
      for (let k = 2; k < bins - 2; k++) if (mag[k] > 1e-4 && mag[k] >= mag[k - 1] && mag[k] >= mag[k + 1] && mag[k] > mag[k - 2] && mag[k] > mag[k + 2]) peaksIdx.push(k);
      if (f === 0 || !peaksIdx.length) newPhase.set(P1);
      else {
        for (let i = 0; i < peaksIdx.length; i++) {
          const kp = peaksIdx[i];
          let np;
          if (prevMag[kp] < 1e-4) np = P1[kp];                  // a partial that was silent before: start from its measured phase
          else { let d = P2[kp] - P1[kp] - omega[kp]; d -= TWO_PI * Math.round(d / TWO_PI); np = phase[kp] + omega[kp] + d; }
          const lo = i === 0 ? 0 : ((peaksIdx[i - 1] + kp) >> 1) + 1, hi = i === peaksIdx.length - 1 ? bins - 1 : (kp + peaksIdx[i + 1]) >> 1;
          for (let k = lo; k <= hi; k++) newPhase[k] = np + (P1[k] - P1[kp]);
        }
      }
      for (let k = 0; k < bins; k++) {
        phase[k] = newPhase[k]; prevMag[k] = mag[k];
        re1[k] = mag[k] * Math.cos(phase[k]); im1[k] = mag[k] * Math.sin(phase[k]);
        if (k > 0 && k < nFft / 2) { re1[nFft - k] = re1[k]; im1[nFft - k] = -im1[k]; }
      }
      im1[0] = 0; im1[nFft / 2] = 0;
      fft(re1, im1, true);
      const o = f * hop;
      for (let i = 0; i < nFft && o + i < out.length; i++) { out[o + i] += re1[i] * win[i]; norm[o + i] += win[i] * win[i]; }
    }
    const y = new Float32Array(nOut);
    for (let i = 0; i < nOut; i++) { const j = i + pad; y[i] = norm[j] > 1e-6 ? out[j] / norm[j] : 0; }
    return y;
  }
  /** Shift pitch by `semitones` and keep the duration. */
  function pitchShift(x, semitones) {
    if (!semitones) return Float32Array.from(x);
    const f = Math.pow(2, semitones / 12);
    const y = resample(timeStretch(x, 1 / f), f);              // stretch to f x longer (same pitch), then play f x faster
    const out = new Float32Array(x.length); out.set(y.subarray(0, Math.min(y.length, x.length)));
    return out;
  }

  // ------------------------------------------------------------ noise reduction (spectral subtraction)
  function noiseReduce(x, sr, amount) {
    amount = Math.max(0, Math.min(1, amount));
    if (amount <= 0) return Float32Array.from(x);
    const nFft = 2048, hop = 512, win = hann(nFft), bins = nFft / 2 + 1;
    const nFrames = Math.max(1, Math.floor((x.length - nFft) / hop) + 1);
    const re = new Float64Array(nFft), im = new Float64Array(nFft);
    const mags = [], energies = [];
    for (let f = 0; f < nFrames; f++) {
      const o = f * hop; let e = 0;
      for (let i = 0; i < nFft; i++) { re[i] = (x[o + i] || 0) * win[i]; im[i] = 0; }
      fft(re, im, false);
      const m = new Float32Array(bins);
      for (let k = 0; k < bins; k++) { m[k] = Math.hypot(re[k], im[k]); e += m[k] * m[k]; }
      mags.push(m); energies.push(e);
    }
    // noise profile: mean magnitude of the quietest 15 % of frames (assumed to contain only the noise floor)
    const order = energies.map((e, i) => [e, i]).sort((a, b) => a[0] - b[0]);
    const take = Math.max(3, Math.floor(nFrames * 0.15));
    const noise = new Float64Array(bins);
    for (let t = 0; t < Math.min(take, order.length); t++) { const m = mags[order[t][1]]; for (let k = 0; k < bins; k++) noise[k] += m[k] / Math.min(take, order.length); }
    const over = 1 + 3 * amount, floor = Math.pow(10, -(8 + 22 * amount) / 20);
    const out = new Float64Array(x.length + nFft), norm = new Float64Array(out.length);
    let prevG = new Float64Array(bins).fill(1);
    for (let f = 0; f < nFrames; f++) {
      const o = f * hop;
      for (let i = 0; i < nFft; i++) { re[i] = (x[o + i] || 0) * win[i]; im[i] = 0; }
      fft(re, im, false);
      const m = mags[f], g = new Float64Array(bins);
      for (let k = 0; k < bins; k++) {
        const sub = Math.max(m[k] - over * noise[k], 0);
        g[k] = Math.max(m[k] > 1e-9 ? sub / m[k] : 1, floor);
      }
      for (let k = 0; k < bins; k++) {                         // smooth across frequency and time to avoid "musical noise"
        const sm = (g[Math.max(k - 1, 0)] + g[k] + g[Math.min(k + 1, bins - 1)]) / 3;
        g[k] = 0.6 * sm + 0.4 * prevG[k];
      }
      prevG = g;
      for (let k = 0; k < bins; k++) { re[k] *= g[k]; im[k] *= g[k]; if (k > 0 && k < nFft / 2) { re[nFft - k] = re[k]; im[nFft - k] = -im[k]; } }
      im[0] = 0; im[nFft / 2] = 0;
      fft(re, im, true);
      for (let i = 0; i < nFft; i++) { out[o + i] += re[i] * win[i]; norm[o + i] += win[i] * win[i]; }
    }
    const y = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = norm[i] > 1e-6 ? out[i] / norm[i] : x[i];
    return y;
  }

  // ------------------------------------------------------------ analysis
  /** Level (dBFS) of the signal per control step, e.g. 100 steps per second: trigger signal for ducking. */
  function activityDb(x, sr, rate) {
    const step = Math.max(1, Math.round(sr / rate)), n = Math.ceil(x.length / step), out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0; const a = i * step, b = Math.min(a + step, x.length);
      for (let j = a; j < b; j++) s += x[j] * x[j];
      out[i] = 10 * Math.log10(s / Math.max(1, b - a) + 1e-12);
    }
    return out;
  }
  /** Non-silent regions [[t0,t1],...] (seconds). Silence = below thresholdDb for at least minSilence seconds; regions are padded. */
  function silenceRegions(x, sr, thresholdDb, minSilence, pad) {
    thresholdDb = thresholdDb == null ? -45 : thresholdDb; minSilence = minSilence == null ? 0.3 : minSilence; pad = pad == null ? 0.05 : pad;
    const rate = 200, act = activityDb(x, sr, rate), minSteps = Math.round(minSilence * rate);
    const regions = []; let start = -1, quiet = 0;
    for (let i = 0; i < act.length; i++) {
      if (act[i] > thresholdDb) { if (start < 0) start = i; quiet = 0; }
      else if (start >= 0) { quiet++; if (quiet >= minSteps) { regions.push([start, i - quiet + 1]); start = -1; quiet = 0; } }
    }
    if (start >= 0) regions.push([start, act.length - quiet]);
    const dur = x.length / sr;
    const padded = regions.map(([a, b]) => [Math.max(0, a / rate - pad), Math.min(dur, b / rate + pad)]);
    const merged = [];
    for (const r of padded) { if (merged.length && r[0] <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], r[1]); else merged.push(r); }
    return merged;
  }
  /** Min/max per bucket for waveform drawing. Returns Float32Array [min0,max0,min1,max1,...]. */
  function peaks(x, bucket) {
    const n = Math.ceil(x.length / bucket), p = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      let mn = 1, mx = -1; const a = i * bucket, b = Math.min(a + bucket, x.length);
      for (let j = a; j < b; j++) { const v = x[j]; if (v < mn) mn = v; if (v > mx) mx = v; }
      p[2 * i] = mn > mx ? 0 : mn; p[2 * i + 1] = mn > mx ? 0 : mx;
    }
    return p;
  }
  /** Exponentially decaying stereo noise: a simple but real room impulse response for convolution reverb. */
  function makeImpulse(sr, seconds, decay) {
    const n = Math.max(1, Math.floor(sr * seconds)), L = new Float32Array(n), R = new Float32Array(n);
    let seed = 12345; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 * 2 - 1; };
    for (let i = 0; i < n; i++) { const e = Math.pow(1 - i / n, decay || 3); L[i] = rnd() * e; R[i] = rnd() * e; }
    return [L, R];
  }

  // ------------------------------------------------------------ look-ahead peak limiter (used for exports)
  /** Brick-wall look-ahead limiter: no output sample exceeds ceilingDb. Gain reduction starts `lookMs` before a peak and recovers over `releaseMs`.
   *  Works in place on the channels array (returns it). The WebAudio compressor used for live preview is not a true limiter; exports use this one. */
  function limit(chans, sr, ceilingDb, lookMs, releaseMs) {
    const ceil = Math.pow(10, (ceilingDb == null ? -1 : ceilingDb) / 20), L = Math.max(2, Math.round(sr * (lookMs || 5) / 1000)), n = chans[0].length;
    const g = new Float32Array(n);                                   // gain each sample needs on its own
    let any = false;
    for (let i = 0; i < n; i++) { let m = 0; for (const c of chans) { const a = Math.abs(c[i]); if (a > m) m = a; } if (m > ceil) { g[i] = ceil / m; any = true; } else g[i] = 1; }
    if (!any) return chans;
    // 1) sliding minimum over the NEXT L samples; 2) moving average of the last L of that minimum.
    // Every window of the average contains a minimum that is <= g[i], so the result is <= g[i]: the ceiling can never be exceeded.
    const slide = new Float32Array(n), dq = new Int32Array(n); let h = 0, t = 0;
    for (let j = 0; j < n; j++) { while (t > h && g[dq[t - 1]] >= g[j]) t--; dq[t++] = j; while (dq[h] <= j - L) h++; slide[j] = g[dq[h]]; }
    const mn = (i) => slide[Math.min(i + L - 1, n - 1)];
    const ring = new Float32Array(L).fill(1), rel = Math.exp(-1 / Math.max(1, sr * (releaseMs || 80) / 1000));
    let acc = L, prev = 1;
    for (let i = 0; i < n; i++) {
      const m = mn(i); acc += m - ring[i % L]; ring[i % L] = m;
      const ma = Math.min(1, acc / L);
      prev = ma < prev ? ma : rel * prev + (1 - rel) * ma;           // instant attack (already smoothed), slow release: never above the average
      for (const c of chans) c[i] *= prev;
    }
    return chans;
  }

  // ------------------------------------------------------------ WAV encoding
  /** channels: array of Float32Array; bitDepth 16 | 24 | 32 (32 = float). Returns an ArrayBuffer. */
  function encodeWav(channels, sr, bitDepth) {
    bitDepth = bitDepth || 16;
    const nCh = channels.length, n = channels[0].length, bps = bitDepth / 8, dataLen = n * nCh * bps;
    const buf = new ArrayBuffer(44 + dataLen), v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + dataLen, true); w(8, 'WAVE'); w(12, 'fmt '); v.setUint32(16, 16, true);
    v.setUint16(20, bitDepth === 32 ? 3 : 1, true); v.setUint16(22, nCh, true); v.setUint32(24, sr, true);
    v.setUint32(28, sr * nCh * bps, true); v.setUint16(32, nCh * bps, true); v.setUint16(34, bitDepth, true); w(36, 'data'); v.setUint32(40, dataLen, true);
    let o = 44;
    for (let i = 0; i < n; i++) for (let c = 0; c < nCh; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      if (bitDepth === 32) { v.setFloat32(o, channels[c][i], true); o += 4; }
      else if (bitDepth === 24) { const q = Math.round(s < 0 ? s * 0x800000 : s * 0x7fffff); v.setUint8(o, q & 255); v.setUint8(o + 1, (q >> 8) & 255); v.setUint8(o + 2, (q >> 16) & 255); o += 3; }
      else { v.setInt16(o, Math.round(s < 0 ? s * 0x8000 : s * 0x7fff), true); o += 2; }
    }
    return buf;
  }

  // ------------------------------------------------------------ heavy chain for one clip
  /** chans: source channels at sample rate sr. Applies the clip's window and the buffer-changing parameters, in a fixed order:
   *  window -> noise reduction -> reverse -> pitch -> speed -> normalize -> loops.  Returns new channels (same sr). */
  function processClip(chans, sr, srcStart, srcDur, p) {
    const a = Math.max(0, Math.floor(srcStart * sr)), b = Math.min(chans[0].length, Math.floor((srcStart + srcDur) * sr));
    let out = chans.map((x) => x.slice(a, Math.max(a + 1, b)));
    if (p.noiseReduce > 0) out = out.map((x) => noiseReduce(x, sr, p.noiseReduce));
    if (p.reverse) out = out.map(reverse);
    if (p.pitch) out = out.map((x) => pitchShift(x, p.pitch));
    const speed = p.speed || 1;
    if (Math.abs(speed - 1) > 1e-4) out = out.map((x) => (p.keepPitch === false ? resample(x, speed) : timeStretch(x, speed)));
    if (p.normalize) { const g = normalizeGain(out); out = out.map((x) => { const y = new Float32Array(x.length); for (let i = 0; i < x.length; i++) y[i] = x[i] * g; return y; }); }
    const loops = Math.max(1, p.loops | 0);
    if (loops > 1) out = out.map((x) => { const y = new Float32Array(x.length * loops); for (let l = 0; l < loops; l++) y.set(x, l * x.length); return y; });
    return out;
  }
  /** Cache key: only the parameters that change the processed buffer. */
  const heavyKey = (assetId, srcStart, srcDur, p) => [assetId, srcStart.toFixed(4), srcDur.toFixed(4), p.noiseReduce || 0, p.reverse ? 1 : 0, p.pitch || 0, p.speed || 1, p.keepPitch === false ? 0 : 1, p.normalize ? 1 : 0, Math.max(1, p.loops | 0)].join('|');

  return { fft, hann, reverse, rmsPeak, normalizeGain, limit, resample, timeStretch, pitchShift, noiseReduce, activityDb, silenceRegions, peaks, makeImpulse, encodeWav, processClip, heavyKey };
});
