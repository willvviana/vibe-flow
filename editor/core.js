/* Editor core: project model, non-destructive clip operations, undo/redo, snapping, shortcuts.
 * Pure data + functions (no DOM, no Web Audio) so it runs in the browser and in Node tests.
 *
 * Time model: all times are seconds (doubles) on the PROJECT timeline.  A clip is a non-destructive reference to an asset:
 *   assetId           which imported audio
 *   srcStart, srcDur  the window of the ORIGINAL asset that is processed (seconds in the asset)
 *   params            heavy (speed/pitch/reverse/loops/noise reduction/normalize) + realtime (gain, fades, EQ, ...) settings
 *   start             where the clip begins on the timeline
 *   offset, duration  the part of the PROCESSED buffer that is audible (processed-buffer seconds == timeline seconds)
 * Trimming/splitting only changes offset/duration/start; the processed buffer is never cut. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.EditorCore = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const MIN_CLIP = 0.01;
  let _n = 0;
  const uid = (p) => `${p}${Date.now().toString(36)}${(++_n).toString(36)}${Math.random().toString(36).slice(2, 5)}`;
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  // ---------------------------------------------------------------- defaults
  function defaultParams() {
    return {
      gainDb: 0, fadeIn: 0, fadeOut: 0, mute: false, pan: 0,
      speed: 1, keepPitch: true, pitch: 0, reverse: false, loops: 1, normalize: false, noiseReduce: 0, enhanceVoice: false,
      eq: { on: false, gains: [0, 0, 0, 0, 0] },
      comp: { on: false, threshold: -24, ratio: 4, attack: 0.01, release: 0.25, knee: 6, makeup: 0 },
      limiter: { on: false, ceiling: -1 },
      reverb: { on: false, mix: 0.25, decay: 1.8 },
      delay: { on: false, time: 0.3, feedback: 0.35, mix: 0.3 },
      filter: { type: 'none', freq: 1000, q: 0.7 },
    };
  }
  const EQ_BANDS = [{ f: 80, type: 'lowshelf' }, { f: 250, type: 'peaking' }, { f: 1000, type: 'peaking' }, { f: 4000, type: 'peaking' }, { f: 10000, type: 'highshelf' }];

  function newTrack(name, role) {
    return { id: uid('t'), name: name || 'Track', volume: 0, pan: 0, mute: false, solo: false, lock: false, role: role || 'music', duck: null };
  }
  function newProject(name) {
    const p = { version: 1, id: uid('p'), name: name || 'Untitled project', created: Date.now(), modified: Date.now(),
      tracks: [], clips: [], assets: {}, settings: { snap: true, grid: 0, crossfade: true, loop: false, pxPerSec: 80, limiter: true } };
    p.tracks.push(newTrack('Track 1'));
    return p;
  }

  // ---------------------------------------------------------------- queries
  const clipEnd = (c) => c.start + c.duration;
  const byId = (arr, id) => arr.find((x) => x.id === id);
  const trackOf = (p, clip) => byId(p.tracks, clip.trackId);
  function projectDuration(p) { return p.clips.reduce((m, c) => Math.max(m, clipEnd(c)), 0); }
  function clipsOnTrack(p, trackId) { return p.clips.filter((c) => c.trackId === trackId).sort((a, b) => a.start - b.start); }
  /** Which tracks are heard: solo wins over mute; a muted track is silent unless it is soloed... (solo overrides mute, like most DAWs). */
  function audibleTrackIds(p) {
    const solo = p.tracks.filter((t) => t.solo);
    if (solo.length) return new Set(solo.map((t) => t.id));
    return new Set(p.tracks.filter((t) => !t.mute).map((t) => t.id));
  }
  /** Automatic equal-power crossfade over the overlap of two clips on the SAME track (later clip fades in, earlier fades out). */
  function crossfades(p) {
    const out = {};
    if (!p.settings.crossfade) return out;
    for (const t of p.tracks) {
      const cs = clipsOnTrack(p, t.id);
      for (let i = 0; i < cs.length; i++) {
        for (let j = i + 1; j < cs.length; j++) {
          const a = cs[i], b = cs[j];
          if (b.start >= clipEnd(a)) continue;
          if (b.start < a.start) continue;
          const L = Math.min(clipEnd(a) - b.start, a.duration, b.duration);
          if (L <= 0.002 || clipEnd(b) < clipEnd(a)) continue;           // fully contained clips are layered, not crossfaded
          out[a.id] = Object.assign(out[a.id] || {}, { fadeOut: L });
          out[b.id] = Object.assign(out[b.id] || {}, { fadeIn: L });
        }
      }
    }
    return out;
  }

  // ---------------------------------------------------------------- history (snapshot based)
  class History {
    constructor(project) { this.p = project; this.stack = [this._snap()]; this.i = 0; }
    _snap() { return JSON.stringify({ tracks: this.p.tracks, clips: this.p.clips, settings: this.p.settings }); }
    _restore(s) { const o = JSON.parse(s); this.p.tracks = o.tracks; this.p.clips = o.clips; this.p.settings = o.settings; }
    /** Call AFTER a completed user action (not on every mouse move). No-op when nothing changed. */
    commit() {
      const s = this._snap();
      if (s === this.stack[this.i]) return false;
      this.stack = this.stack.slice(0, this.i + 1); this.stack.push(s); this.i++;
      if (this.stack.length > 200) { this.stack.shift(); this.i--; }
      this.p.modified = Date.now();
      return true;
    }
    canUndo() { return this.i > 0; }
    canRedo() { return this.i < this.stack.length - 1; }
    undo() { if (!this.canUndo()) return false; this.i--; this._restore(this.stack[this.i]); this.p.modified = Date.now(); return true; }
    redo() { if (!this.canRedo()) return false; this.i++; this._restore(this.stack[this.i]); this.p.modified = Date.now(); return true; }
    reset() { this.stack = [this._snap()]; this.i = 0; }
  }

  // ---------------------------------------------------------------- track operations
  function addTrack(p, name, role) {
    const t = newTrack(name || `Track ${p.tracks.length + 1}`, role);
    p.tracks.push(t);
    return t;
  }
  function removeTrack(p, id) {
    const t = byId(p.tracks, id);
    if (!t || t.lock) return false;
    if (p.tracks.length <= 1) return false;
    p.tracks = p.tracks.filter((x) => x.id !== id);
    p.clips = p.clips.filter((c) => c.trackId !== id);
    for (const o of p.tracks) if (o.duck) o.duck.sources = o.duck.sources.filter((s) => s !== id);
    return true;
  }

  // ---------------------------------------------------------------- clip operations
  function makeClip(p, asset, trackId, start) {
    const dur = asset.duration;
    return { id: uid('c'), trackId, assetId: asset.id, name: asset.name, start: Math.max(0, start), offset: 0, duration: dur, srcStart: 0, srcDur: dur, params: defaultParams() };
  }
  function addClip(p, asset, trackId, start) {
    const t = byId(p.tracks, trackId);
    if (!t || t.lock) return null;
    const c = makeClip(p, asset, trackId, start);
    p.clips.push(c);
    return c;
  }
  /** Length of the processed buffer for a clip, i.e. how much timeline the clip could fill at most. */
  function processedLength(c) { return (c.srcDur / (c.params.speed || 1)) * Math.max(1, c.params.loops | 0); }

  function splitClip(p, id, t) {
    const c = byId(p.clips, id);
    if (!c) return null;
    const tr = trackOf(p, c);
    if (!tr || tr.lock) return null;
    if (t <= c.start + MIN_CLIP || t >= clipEnd(c) - MIN_CLIP) return null;
    const left = c, d = t - c.start;
    const right = clone(c);
    right.id = uid('c');
    right.start = t; right.offset = c.offset + d; right.duration = c.duration - d;
    left.duration = d;
    // fades belong to the outer edges only
    right.params.fadeIn = 0; left.params.fadeOut = 0;
    right.params.fadeOut = c.params.fadeOut;
    p.clips.push(right);
    return [left, right];
  }
  function trimStart(p, id, newStart) {
    const c = byId(p.clips, id); if (!c || trackOf(p, c).lock) return false;
    const end = clipEnd(c);
    newStart = clamp(newStart, Math.max(0, c.start - c.offset), end - MIN_CLIP);        // cannot reveal audio before the processed buffer starts
    const d = newStart - c.start;
    c.start = newStart; c.offset += d; c.duration -= d;
    return true;
  }
  function trimEnd(p, id, newEnd) {
    const c = byId(p.clips, id); if (!c || trackOf(p, c).lock) return false;
    const maxDur = processedLength(c) - c.offset;
    newEnd = clamp(newEnd, c.start + MIN_CLIP, c.start + maxDur);
    c.duration = newEnd - c.start;
    return true;
  }
  function moveClips(p, ids, dt, dTrack) {
    const sel = ids.map((i) => byId(p.clips, i)).filter(Boolean);
    if (!sel.length) return false;
    for (const c of sel) if (trackOf(p, c).lock) return false;
    const minStart = Math.min(...sel.map((c) => c.start));
    dt = Math.max(dt, -minStart);
    const idx = (c) => p.tracks.findIndex((t) => t.id === c.trackId);
    let dT = dTrack | 0;
    for (const c of sel) {                         // clamp the vertical move so every clip stays on an existing, unlocked track
      const j = idx(c) + dT;
      if (j < 0) dT = -idx(c);
      if (j >= p.tracks.length) dT = p.tracks.length - 1 - idx(c);
    }
    for (const c of sel) if (p.tracks[idx(c) + dT].lock) dT = 0;
    for (const c of sel) { c.start += dt; c.trackId = p.tracks[idx(c) + dT].id; }
    return dt !== 0 || dT !== 0;
  }
  function deleteClips(p, ids) {
    const s = new Set(ids);
    const before = p.clips.length;
    p.clips = p.clips.filter((c) => !(s.has(c.id) && !trackOf(p, c).lock));
    return before !== p.clips.length;
  }
  function duplicateClips(p, ids) {
    const sel = ids.map((i) => byId(p.clips, i)).filter(Boolean);
    const out = [];
    for (const c of sel) {
      const n = clone(c); n.id = uid('c'); n.start = clipEnd(c);
      if (trackOf(p, c).lock) continue;
      p.clips.push(n); out.push(n);
    }
    return out;
  }
  /** Clipboard holds clip definitions relative to the earliest selected clip. */
  function copyClips(p, ids) {
    const sel = ids.map((i) => byId(p.clips, i)).filter(Boolean);
    if (!sel.length) return null;
    const t0 = Math.min(...sel.map((c) => c.start));
    const i0 = Math.min(...sel.map((c) => p.tracks.findIndex((t) => t.id === c.trackId)));
    return sel.map((c) => ({ clip: clone(c), dt: c.start - t0, dTrack: p.tracks.findIndex((t) => t.id === c.trackId) - i0 }));
  }
  function pasteClips(p, clipboard, time, trackId) {
    if (!clipboard || !clipboard.length) return [];
    const base = Math.max(0, p.tracks.findIndex((t) => t.id === trackId));
    const out = [];
    for (const it of clipboard) {
      const tr = p.tracks[clamp(base + it.dTrack, 0, p.tracks.length - 1)];
      if (tr.lock) continue;
      const n = clone(it.clip); n.id = uid('c'); n.trackId = tr.id; n.start = Math.max(0, time + it.dt);
      p.clips.push(n); out.push(n);
    }
    return out;
  }
  /** Parameters that change the processed buffer; when they change, offset/duration are rescaled so the audible part stays the same. */
  function setHeavyParam(c, key, value) {
    const oldLen = processedLength(c);
    c.params[key] = value;
    const newLen = processedLength(c);
    if (oldLen > 0 && newLen > 0 && Math.abs(oldLen - newLen) > 1e-9) {
      const f = newLen / oldLen;
      c.offset *= f; c.duration *= f;
      c.duration = Math.min(c.duration, newLen - c.offset);
    }
  }
  function setParam(c, path, value) {
    const keys = path.split('.');
    let o = c.params;
    for (let i = 0; i < keys.length - 1; i++) o = o[keys[i]];
    o[keys[keys.length - 1]] = value;
  }
  /** Replace a clip by the non-silent pieces found by silence detection. regions = [[t0,t1],...] in processed-buffer seconds. */
  function keepRegions(p, id, regions) {
    const c = byId(p.clips, id);
    if (!c || trackOf(p, c).lock || !regions.length) return [];
    const out = [];
    for (const [a, b] of regions) {
      const s = Math.max(a, c.offset), e = Math.min(b, c.offset + c.duration);
      if (e - s < MIN_CLIP) continue;
      const n = clone(c); n.id = uid('c'); n.start = c.start + (s - c.offset); n.offset = s; n.duration = e - s;
      n.params.fadeIn = 0; n.params.fadeOut = 0;
      out.push(n);
    }
    if (!out.length) return [];
    p.clips = p.clips.filter((x) => x.id !== id).concat(out);
    return out;
  }

  // ---------------------------------------------------------------- snapping
  function snapTime(p, t, o) {
    o = o || {};
    if (o.enabled === false) return { t, snapped: false };
    const tolSec = (o.tolPx || 8) / (o.pxPerSec || p.settings.pxPerSec || 80);
    const cand = [0];
    if (o.playhead != null) cand.push(o.playhead);
    const ex = o.exclude || new Set();
    for (const c of p.clips) if (!ex.has(c.id)) { cand.push(c.start, clipEnd(c)); }
    let best = null;
    for (const x of cand) if (Math.abs(x - t) <= tolSec && (best == null || Math.abs(x - t) < Math.abs(best - t))) best = x;
    if (best != null) return { t: best, snapped: true };
    const g = o.grid != null ? o.grid : p.settings.grid;
    if (g > 0) { const q = Math.round(t / g) * g; if (Math.abs(q - t) <= tolSec) return { t: q, snapped: true }; }
    return { t, snapped: false };
  }

  // ---------------------------------------------------------------- ducking envelope (pure: works on activity arrays)
  /** activity: Float32Array of the trigger level in dBFS per control step (rate Hz). Returns gain (linear) per step.
   *  Gain drops by amountDb while the level is above thresholdDb; attack/release smooth the transition. */
  function duckGain(activityDb, rate, opts) {
    const amount = Math.pow(10, (opts.amountDb || -12) / 20);
    const att = Math.exp(-1 / Math.max(1, (opts.attack || 0.05) * rate)), rel = Math.exp(-1 / Math.max(1, (opts.release || 0.4) * rate));
    const hold = Math.round((opts.hold == null ? 0.15 : opts.hold) * rate);
    const out = new Float32Array(activityDb.length);
    let g = 1, h = 0;
    for (let i = 0; i < activityDb.length; i++) {
      if (activityDb[i] > (opts.thresholdDb == null ? -40 : opts.thresholdDb)) h = hold; else if (h > 0) h--;
      const target = h > 0 || activityDb[i] > (opts.thresholdDb == null ? -40 : opts.thresholdDb) ? amount : 1;
      const k = target < g ? att : rel;
      g = k * g + (1 - k) * target;
      out[i] = g;
    }
    return out;
  }

  // ---------------------------------------------------------------- shortcuts
  const DEFAULT_SHORTCUTS = {
    playPause: 'Space', undo: 'Ctrl+Z', redo: 'Ctrl+Shift+Z', copy: 'Ctrl+C', paste: 'Ctrl+V', cut: 'Ctrl+X', selectAll: 'Ctrl+A',
    delete: 'Delete', duplicate: 'Ctrl+D', save: 'Ctrl+S', export: 'Ctrl+E', split: 'S', muteTrack: 'M', home: 'Home', end: 'End',
    left: 'ArrowLeft', right: 'ArrowRight',
  };
  function comboOf(e) {
    let k = e.key;
    if (k === ' ') k = 'Space';
    if (k.length === 1) k = k.toUpperCase();
    const parts = [];
    if (e.ctrlKey || e.metaKey) parts.push('Ctrl');
    if (e.altKey) parts.push('Alt');
    if (e.shiftKey) parts.push('Shift');
    if (['Control', 'Shift', 'Alt', 'Meta'].includes(k)) return null;
    parts.push(k);
    return parts.join('+');
  }
  function actionFor(e, map) {
    const combo = comboOf(e);
    if (!combo) return null;
    for (const [action, c] of Object.entries(map)) if (c.toLowerCase() === combo.toLowerCase()) return action;
    return null;
  }
  const isTextTarget = (el) => !!el && (/^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) && !(el.type === 'range' || el.type === 'checkbox' || el.type === 'button') || el.isContentEditable === true);

  function fmtTime(s, withMs) {
    s = Math.max(0, s);
    const m = Math.floor(s / 60), r = s - m * 60;
    return `${m}:${r.toFixed(withMs === false ? 0 : 1).padStart(withMs === false ? 2 : 4, '0')}`;
  }

  return { MIN_CLIP, uid, clone, clamp, defaultParams, EQ_BANDS, newTrack, newProject, clipEnd, byId, trackOf, projectDuration, clipsOnTrack, audibleTrackIds,
    crossfades, History, addTrack, removeTrack, makeClip, addClip, processedLength, splitClip, trimStart, trimEnd, moveClips, deleteClips, duplicateClips,
    copyClips, pasteClips, setHeavyParam, setParam, keepRegions, snapTime, duckGain, DEFAULT_SHORTCUTS, comboOf, actionFor, isTextTarget, fmtTime };
});
