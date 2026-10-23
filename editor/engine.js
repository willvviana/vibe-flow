/* Editor audio engine: ONE graph builder used for live playback AND offline rendering, so what you hear is what you export.
 *
 *   per clip   BufferSource(processed buffer) -> clipGain (volume + fades + crossfade) -> [voice enhance] -> EQ(5) -> compressor -> limiter
 *              -> filter -> pan -> (dry + optional reverb / delay) -> track bus
 *   per track  bus -> duck gain (automated from the trigger tracks) -> track volume -> track pan -> master
 *   master     -> safety limiter (fast compressor, optional) -> destination
 * Heavy, buffer-changing edits (speed, pitch, reverse, loops, noise reduction, normalise) are computed once in a worker and cached.
 * Honest limits: the "limiter" nodes are fast compressors, not look-ahead brick-wall limiters; mono sources are centred. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./core.js'), require('./dsp.js'));
  else root.EditorEngine = factory(root.EditorCore, root.EditorDSP);
})(typeof self !== 'undefined' ? self : this, function (C, D) {
  'use strict';
  const GLOBAL = typeof self !== 'undefined' ? self : globalThis;
  const db2lin = (d) => Math.pow(10, d / 20);
  /** The WebAudio DynamicsCompressorNode adds an automatic make-up gain: (1 / gain at 0 dBFS)^0.6 (Chromium's algorithm, which this ports).
   *  Returned so the engine can cancel it and make threshold/ceiling settings mean what the sliders say. */
  function dynMakeup(thDb, kneeDb, ratio) {
    const lin = (d) => Math.pow(10, d / 20), db = (x) => 20 * Math.log10(Math.max(x, 1e-12));
    const linTh = lin(thDb), kneeThDb = thDb + kneeDb, kneeTh = lin(kneeThDb), slope = 1 / ratio;
    const knee = (x, k) => (x < linTh ? x : linTh + (1 - Math.exp(-k * (x - linTh))) / k);
    const slopeAt = (x, k) => { if (x < linTh) return 1; const x2 = x * 1.001; return (db(knee(x2, k)) - db(knee(x, k))) / (db(x2) - db(x)); };
    let minK = 0.1, maxK = 10000, k = 5;
    for (let i = 0; i < 15; i++) { if (slopeAt(kneeTh, k) < slope) maxK = k; else minK = k; k = Math.sqrt(minK * maxK); }
    const yKneeDb = db(knee(kneeTh, k));
    const sat = (x) => (x < kneeTh ? knee(x, k) : lin(yKneeDb + slope * (db(x) - kneeThDb)));
    return Math.pow(1 / sat(1), 0.6);
  }
  const CTRL_RATE = 100;                                   // Hz: resolution of ducking curves

  // ------------------------------------------------------------------ gain envelope for one clip (fades + crossfades)
  /** Gain multiplier at time u (seconds since the clip start), for a clip of length dur. */
  function fadeGain(u, dur, fadeIn, fadeOut, xIn, xOut) {
    let g = 1;
    const fi = Math.max(fadeIn || 0, xIn || 0), fo = Math.max(fadeOut || 0, xOut || 0);
    if (fi > 0 && u < fi) { const r = Math.max(0, u / fi); g *= xIn >= fadeIn ? Math.sin(r * Math.PI / 2) : r; }        // equal power for crossfades, linear for user fades
    if (fo > 0 && u > dur - fo) { const r = Math.max(0, (dur - u) / fo); g *= xOut >= fadeOut ? Math.sin(r * Math.PI / 2) : r; }
    return g;
  }

  class Engine {
    constructor(opts) {
      opts = opts || {};
      this.sampleRate = opts.sampleRate || 48000;
      this.ctx = null; this.project = null; this.assets = {};      // assets: id -> {chans:[Float32Array], sampleRate, duration, name}
      this.processed = new Map();                                   // heavyKey -> {chans, sampleRate}
      this.pending = new Map();
      this.worker = null; this._wid = 0; this._wait = new Map();
      this.playing = false; this.pos = 0; this._origin = 0; this._from = 0; this.graph = null; this.onEnd = null;
      this.workerUrl = opts.workerUrl || 'editor/worker.js';
      this.buffers = new Map();                                     // per AudioContext: key -> AudioBuffer
      this.irCache = new Map();
    }
    // -------------------------------------------------------------- context & worker
    ensureCtx() {
      if (!this.ctx) this.ctx = new (GLOBAL.AudioContext || GLOBAL.webkitAudioContext)({ sampleRate: this.sampleRate, latencyHint: 'interactive' });
      return this.ctx;
    }
    _worker() {
      if (this.worker) return this.worker;
      this.worker = new Worker(this.workerUrl);
      this.worker.onmessage = (e) => { const w = this._wait.get(e.data.id); if (!w) return; this._wait.delete(e.data.id); e.data.ok ? w.resolve(e.data.chans) : w.reject(new Error(e.data.error)); };
      this.worker.onerror = (e) => { for (const [, w] of this._wait) w.reject(new Error('DSP worker failed: ' + (e.message || 'unknown'))); this._wait.clear(); this.worker = null; };
      return this.worker;
    }
    _process(asset, srcStart, srcDur, params) {
      const id = ++this._wid;
      return new Promise((resolve, reject) => {
        this._wait.set(id, { resolve, reject });
        // copy: the source channels must stay intact for the next job; the copy is transferred to the worker
        const chans = asset.chans.map((c) => c.slice());
        this._worker().postMessage({ id, chans, sr: asset.sampleRate, srcStart, srcDur, params: { speed: params.speed, keepPitch: params.keepPitch, pitch: params.pitch, reverse: params.reverse, loops: params.loops, normalize: params.normalize, noiseReduce: params.noiseReduce } }, chans.map((c) => c.buffer));
      });
    }
    /** Make sure every clip has its processed buffer. Reports progress (done, total). */
    async prepare(project, assets, onProgress) {
      this.project = project; this.assets = assets;
      const need = [];
      for (const c of project.clips) {
        const a = assets[c.assetId];
        if (!a) throw new Error(`Missing audio for "${c.name}". Re-import the file.`);
        const key = D.heavyKey(c.assetId, c.srcStart, c.srcDur, c.params);
        if (!this.processed.has(key)) need.push([key, c, a]);
      }
      const uniq = new Map(need.map((n) => [n[0], n]));
      let done = 0;
      for (const [key, c, a] of uniq.values()) {
        const trivial = !c.params.noiseReduce && !c.params.reverse && !c.params.pitch && Math.abs((c.params.speed || 1) - 1) < 1e-4 && !c.params.normalize && (c.params.loops | 0) <= 1;
        let chans;
        if (trivial) {                                              // no heavy edit: just the window, no worker round trip
          const a0 = Math.max(0, Math.floor(c.srcStart * a.sampleRate)), b0 = Math.min(a.chans[0].length, Math.floor((c.srcStart + c.srcDur) * a.sampleRate));
          chans = a.chans.map((x) => x.subarray(a0, Math.max(a0 + 1, b0)));
        } else {
          if (this.pending.has(key)) chans = await this.pending.get(key);
          else { const pr = this._process(a, c.srcStart, c.srcDur, c.params); this.pending.set(key, pr); try { chans = await pr; } finally { this.pending.delete(key); } }
        }
        this.processed.set(key, { chans, sampleRate: a.sampleRate });
        if (onProgress) onProgress(++done, uniq.size);
      }
    }
    _procFor(c) { return this.processed.get(D.heavyKey(c.assetId, c.srcStart, c.srcDur, c.params)); }
    _audioBuffer(ctx, key, proc) {
      let m = this.buffers.get(ctx); if (!m) { m = new Map(); this.buffers.set(ctx, m); }
      let b = m.get(key);
      if (!b) {
        b = ctx.createBuffer(proc.chans.length, proc.chans[0].length, proc.sampleRate);
        proc.chans.forEach((ch, i) => b.copyToChannel(ch, i));
        m.set(key, b);
      }
      return b;
    }
    _impulse(ctx, decay) {
      const k = ctx.sampleRate + ':' + decay.toFixed(2);
      if (!this.irCache.has(k)) {
        const [L, R] = D.makeImpulse(ctx.sampleRate, Math.max(0.2, decay), 3);
        const b = ctx.createBuffer(2, L.length, ctx.sampleRate); b.copyToChannel(L, 0); b.copyToChannel(R, 1);
        this.irCache.set(k, b);
      }
      return this.irCache.get(k);
    }

    // -------------------------------------------------------------- ducking
    /** Gain curve (Float32Array at CTRL_RATE) for a track, computed from the audio of its trigger tracks. */
    duckCurve(project, track, total) {
      const d = track.duck;
      if (!d || !d.on || !d.sources || !d.sources.length) return null;
      const n = Math.ceil(total * CTRL_RATE) + 2, act = new Float32Array(n).fill(-120);
      for (const c of project.clips) {
        if (!d.sources.includes(c.trackId)) continue;
        const src = project.tracks.find((t) => t.id === c.trackId);
        if (!src) continue;
        const proc = this._procFor(c);
        if (!proc) continue;
        const sr = proc.sampleRate, a0 = Math.floor(c.offset * sr), b0 = Math.min(proc.chans[0].length, Math.floor((c.offset + c.duration) * sr));
        const mixed = new Float32Array(Math.max(0, b0 - a0));
        for (const ch of proc.chans) for (let i = 0; i < mixed.length; i++) mixed[i] += ch[a0 + i] / proc.chans.length;
        const ad = D.activityDb(mixed, sr, CTRL_RATE);
        const gdb = c.params.gainDb + (src.volume || 0) + (c.params.mute ? -200 : 0);
        const i0 = Math.round(c.start * CTRL_RATE);
        for (let i = 0; i < ad.length && i0 + i < n; i++) act[i0 + i] = Math.max(act[i0 + i], ad[i] + gdb);
      }
      return C.duckGain(act, CTRL_RATE, { amountDb: d.amountDb, thresholdDb: d.thresholdDb, attack: d.attack, release: d.release, hold: d.hold });
    }

    // -------------------------------------------------------------- graph construction (live and offline)
    /** Build and schedule everything from timeline position `from`. `origin` = context time that corresponds to `from`.
     *  opts: {trackIds:Set|null (only these tracks, ignores mute/solo), ignoreMuteSolo, limiter, end:Number|null} Returns the graph object. */
    build(ctx, project, from, origin, opts) {
      opts = opts || {};
      const total = Math.max(C.projectDuration(project), from) + 1;
      const audible = opts.trackIds ? opts.trackIds : C.audibleTrackIds(project);
      const xf = C.crossfades(project);
      const master = ctx.createGain();
      let limiter = null, g_fix = null;
      if (opts.limiter !== false && project.settings.limiter !== false) {
        limiter = ctx.createDynamicsCompressor();
        limiter.threshold.value = -1; limiter.knee.value = 0; limiter.ratio.value = 20; limiter.attack.value = 0.002; limiter.release.value = 0.08;
        const fix = ctx.createGain(); fix.gain.value = 1 / dynMakeup(-1, 0, 20);
        master.connect(limiter); limiter.connect(fix); fix.connect(ctx.destination); g_fix = fix;
      } else master.connect(ctx.destination);
      const g = { ctx, master, limiter, fix: g_fix, tracks: new Map(), clips: new Map(), sources: [] };
      const soloMode = !opts.trackIds && project.tracks.some((t) => t.solo);
      for (const t of project.tracks) {
        const bus = ctx.createGain(), duck = ctx.createGain(), vol = ctx.createGain(), pan = ctx.createStereoPanner();
        vol.gain.value = db2lin(t.volume || 0); pan.pan.value = C.clamp(t.pan || 0, -1, 1);
        bus.connect(duck); duck.connect(vol); vol.connect(pan); pan.connect(master);
        g.tracks.set(t.id, { bus, duck, vol, pan, active: audible.has(t.id) });
        const curve = audible.has(t.id) ? this.duckCurve(project, t, total) : null;
        if (curve) {
          const i0 = Math.floor(from * CTRL_RATE), part = curve.subarray(i0);
          if (part.length > 1) {
            duck.gain.setValueAtTime(part[0], origin);
            duck.gain.setValueCurveAtTime(part, origin, (part.length - 1) / CTRL_RATE);
          }
        }
      }
      for (const c of project.clips) {
        const tr = g.tracks.get(c.trackId);
        if (!tr || !tr.active) continue;
        const end = C.clipEnd(c);
        if (end <= from + 1e-6) continue;
        if (opts.end != null && c.start >= opts.end) continue;
        if (opts.clipIds && !opts.clipIds.has(c.id)) continue;
        const proc = this._procFor(c);
        if (!proc) continue;
        const key = D.heavyKey(c.assetId, c.srcStart, c.srcDur, c.params);
        const buf = this._audioBuffer(ctx, key, proc);
        const startT = Math.max(c.start, from), bufOffset = c.offset + (startT - c.start), dur = end - startT;
        const when = origin + (startT - from);
        const node = this._clipChain(ctx, c, tr.bus, g, xf[c.id] || {}, startT, from, origin);
        const src = ctx.createBufferSource(); src.buffer = buf;
        src.connect(node.input);
        src.start(when, Math.min(bufOffset, Math.max(0, buf.duration - 1e-4)), Math.max(0, Math.min(dur, buf.duration - bufOffset)));
        g.sources.push(src);
        g.clips.set(c.id, node);
      }
      return g;
    }
    _clipChain(ctx, c, bus, g, x, startT, from, origin) {
      const p = c.params, dur = c.duration;
      const input = ctx.createGain();
      // gain + fades + crossfade, sampled as a value curve over the audible part of the clip
      const gain = ctx.createGain(), vol = ctx.createGain(), base = p.mute ? 0 : db2lin(p.gainDb || 0);
      vol.gain.value = base;                                       // static volume: updated live from the slider
      const n = Math.max(2, Math.ceil((C.clipEnd(c) - startT) * CTRL_RATE) + 1), curve = new Float32Array(n);
      for (let i = 0; i < n; i++) curve[i] = fadeGain(Math.min(startT - c.start + i / CTRL_RATE, dur), dur, p.fadeIn, p.fadeOut, x.fadeIn || 0, x.fadeOut || 0);
      const t0 = origin + (startT - from);
      gain.gain.setValueAtTime(curve[0], t0);
      gain.gain.setValueCurveAtTime(curve, t0, (n - 1) / CTRL_RATE);
      input.connect(gain); gain.connect(vol);
      let last = vol;
      const chain = (node) => { last.connect(node); last = node; return node; };
      const nodes = { input, gain, vol, base, fadeIn: p.fadeIn, fadeOut: p.fadeOut, enhanced: !!p.enhanceVoice, xf: JSON.stringify(x) };
      // voice enhance: gentle high-pass, low-mid cut, presence lift, light compression
      if (p.enhanceVoice) {
        for (let i = 0; i < 2; i++) { const hp = chain(ctx.createBiquadFilter()); hp.type = 'highpass'; hp.frequency.value = 85; hp.Q.value = 0.707; }
        const mud = chain(ctx.createBiquadFilter()); mud.type = 'peaking'; mud.frequency.value = 220; mud.gain.value = -2; mud.Q.value = 1;
        const pres = chain(ctx.createBiquadFilter()); pres.type = 'peaking'; pres.frequency.value = 3500; pres.gain.value = 3; pres.Q.value = 0.9;
        const cmp = chain(ctx.createDynamicsCompressor()); cmp.threshold.value = -26; cmp.ratio.value = 2.5; cmp.knee.value = 10; cmp.attack.value = 0.01; cmp.release.value = 0.2;
        const cmpFix = chain(ctx.createGain()); cmpFix.gain.value = 1 / dynMakeup(-26, 10, 2.5);
      }
      nodes.eq = C.EQ_BANDS.map((b, i) => { const f = chain(ctx.createBiquadFilter()); f.type = b.type; f.frequency.value = b.f; f.Q.value = 1; f.gain.value = p.eq.on ? p.eq.gains[i] : 0; return f; });
      const comp = chain(ctx.createDynamicsCompressor()); nodes.comp = comp;
      const makeup = chain(ctx.createGain()); nodes.makeup = makeup;
      const lim = chain(ctx.createDynamicsCompressor()); nodes.lim = lim;
      const limFix = chain(ctx.createGain()); nodes.limFix = limFix;
      const flt = chain(ctx.createBiquadFilter()); nodes.flt = flt;
      const pan = chain(ctx.createStereoPanner()); nodes.pan = pan;
      // routing: dry always; reverb and delay only when enabled (toggling them rebuilds the graph)
      const mix = ctx.createGain(); mix.connect(bus);
      const dry = ctx.createGain(); last.connect(dry); dry.connect(mix); nodes.dry = dry;
      if (p.reverb.on) {
        const conv = ctx.createConvolver(); conv.buffer = this._impulse(ctx, p.reverb.decay);
        const wet = ctx.createGain(); last.connect(conv); conv.connect(wet); wet.connect(mix); nodes.revWet = wet; nodes.conv = conv;
      }
      if (p.delay.on) {
        const dl = ctx.createDelay(2); const fb = ctx.createGain(); const wet = ctx.createGain();
        last.connect(dl); dl.connect(fb); fb.connect(dl); dl.connect(wet); wet.connect(mix);
        nodes.delay = dl; nodes.delayFb = fb; nodes.delayWet = wet;
      }
      this._applyRealtime(nodes, p);
      return nodes;
    }
    /** Push the real-time parameters into existing nodes (also used for live updates while playing). */
    _applyRealtime(nodes, p) {
      const set = (param, v) => { param.value = v; };
      nodes.eq.forEach((f, i) => set(f.gain, p.eq.on ? p.eq.gains[i] : 0));
      const cp = p.comp;
      if (cp.on) { set(nodes.comp.threshold, cp.threshold); set(nodes.comp.ratio, cp.ratio); set(nodes.comp.attack, cp.attack); set(nodes.comp.release, cp.release); set(nodes.comp.knee, cp.knee); set(nodes.makeup.gain, db2lin(cp.makeup) / dynMakeup(cp.threshold, cp.knee, cp.ratio)); }
      else { set(nodes.comp.threshold, 0); set(nodes.comp.ratio, 1); set(nodes.comp.knee, 0); set(nodes.makeup.gain, 1); }
      if (p.limiter.on) { set(nodes.lim.threshold, p.limiter.ceiling); set(nodes.lim.knee, 0); set(nodes.lim.ratio, 20); set(nodes.lim.attack, 0.001); set(nodes.lim.release, 0.05); set(nodes.limFix.gain, 1 / dynMakeup(p.limiter.ceiling, 0, 20)); }
      else { set(nodes.lim.threshold, 0); set(nodes.lim.ratio, 1); set(nodes.lim.knee, 0); set(nodes.limFix.gain, 1); }
      const f = p.filter;
      if (f.type === 'none') { nodes.flt.type = 'allpass'; nodes.flt.frequency.value = 1000; }
      else { nodes.flt.type = f.type; set(nodes.flt.frequency, f.freq); set(nodes.flt.Q, f.q); }
      set(nodes.pan.pan, C.clamp(p.pan || 0, -1, 1));
      if (nodes.revWet) { set(nodes.revWet.gain, p.reverb.mix); set(nodes.dry.gain, 1 - 0.5 * p.reverb.mix); } else set(nodes.dry.gain, 1);
      if (nodes.delay) { set(nodes.delay.delayTime, C.clamp(p.delay.time, 0.01, 1.9)); set(nodes.delayFb.gain, C.clamp(p.delay.feedback, 0, 0.9)); set(nodes.delayWet.gain, p.delay.mix); }
    }
    /** Live update of one clip's real-time params. Returns false if a rebuild is needed (reverb/delay toggled, ...). */
    updateClip(clip) {
      if (!this.graph) return true;
      const n = this.graph.clips.get(clip.id);
      if (!n) return true;
      const p = clip.params;
      if (!!p.reverb.on !== !!n.revWet || !!p.delay.on !== !!n.delay || !!p.enhanceVoice !== !!n.enhanced) return false;
      if (p.fadeIn !== n.fadeIn || p.fadeOut !== n.fadeOut) return false;          // fades are scheduled curves: rebuild
      this._applyRealtime(n, p);
      n.vol.gain.value = p.mute ? 0 : db2lin(p.gainDb || 0);
      return true;
    }
    updateTrack(track) {
      if (!this.graph) return;
      const t = this.graph.tracks.get(track.id); if (!t) return;
      t.vol.gain.value = db2lin(track.volume || 0); t.pan.pan.value = C.clamp(track.pan || 0, -1, 1);
    }

    // -------------------------------------------------------------- transport
    async play(from) {
      const ctx = this.ensureCtx();
      if (ctx.state === 'suspended') await ctx.resume();
      this.stopGraph();
      this._from = from; this.pos = from;
      this._origin = ctx.currentTime + 0.06;                  // small lead so every source can be scheduled before it starts
      this.graph = this.build(ctx, this.project, from, this._origin, {});
      this.analyser = ctx.createAnalyser(); this.analyser.fftSize = 1024;       // level meter (a parallel tap on the master bus)
      this.graph.master.connect(this.analyser);
      this.playing = true;
      const end = C.projectDuration(this.project);
      this._endTimer = setTimeout(() => { if (this.playing && this.position() >= end - 0.02) { this.pause(); if (this.onEnd) this.onEnd(); } }, Math.max(0, (end - from) * 1000) + 120);
    }
    stopGraph() {
      clearTimeout(this._endTimer);
      if (!this.graph) return;
      for (const s of this.graph.sources) { try { s.stop(); } catch (e) { /* already stopped */ } try { s.disconnect(); } catch (e) { /* */ } }
      try { this.graph.master.disconnect(); } catch (e) { /* */ }
      if (this.graph.limiter) try { this.graph.limiter.disconnect(); } catch (e) { /* */ }
      if (this.graph.fix) try { this.graph.fix.disconnect(); } catch (e) { /* */ }
      this.graph = null;
    }
    pause() { if (this.playing) { this.pos = this.position(); } this.stopGraph(); this.playing = false; }
    position() {
      if (!this.playing) return this.pos;
      return this._from + Math.max(0, this.ctx.currentTime - this._origin);
    }
    async seek(t) { const was = this.playing; this.pos = Math.max(0, t); if (was) await this.play(this.pos); }
    async rebuild() { if (this.playing) await this.play(this.position()); }

    // -------------------------------------------------------------- offline render (export)
    /** Render [from, to] of the timeline at `sampleRate`. opts: {trackIds:Set|null, limiter, tail, onProgress}. Returns {chans:[L,R], sampleRate, stats}. */
    async render(from, to, sampleRate, opts) {
      opts = opts || {};
      const project = this.project;
      const tail = opts.tail == null ? 2.5 : opts.tail;
      const len = Math.max(1, Math.ceil((to - from + tail) * sampleRate));
      const ctx = new (GLOBAL.OfflineAudioContext || GLOBAL.webkitOfflineAudioContext)(2, len, sampleRate);
      this.build(ctx, project, from, 0, { trackIds: opts.trackIds || null, clipIds: opts.clipIds || null, limiter: false, end: to });
      if (opts.onProgress) {
        const steps = Math.min(40, Math.max(1, Math.floor(len / sampleRate / 1.5)));
        for (let i = 1; i < steps; i++) { const t = (len / sampleRate) * i / steps; ctx.suspend(t).then(() => { opts.onProgress(i / steps); ctx.resume(); }); }
      }
      const out = await ctx.startRendering();
      const keep = Math.max(1, Math.ceil((to - from + tail) * sampleRate));
      const chans = [out.getChannelData(0).slice(0, keep), out.getChannelData(1).slice(0, keep)];
      // exports use a true look-ahead brick-wall limiter (the live preview uses a compressor-based approximation of it)
      if (opts.limiter !== false && project.settings.limiter !== false) D.limit(chans, sampleRate, -1, 5, 80);
      return { chans, sampleRate, stats: this.stats(chans) };
    }
    stats(chans) {
      let peak = 0, clipped = 0, ss = 0, n = 0;
      for (const ch of chans) for (let i = 0; i < ch.length; i++) { const v = Math.abs(ch[i]); if (v > peak) peak = v; if (v >= 0.999) clipped++; ss += ch[i] * ch[i]; n++; }
      return { peakDb: 20 * Math.log10(peak + 1e-9), rmsDb: 10 * Math.log10(ss / Math.max(1, n) + 1e-12), clipped, seconds: chans[0].length };
    }
  }

  return { Engine, fadeGain, CTRL_RATE };
});
