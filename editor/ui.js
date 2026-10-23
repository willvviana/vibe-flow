/* Editor application core: state, timeline rendering, pointer interactions, transport, history, persistence glue.
 * Panels (library/properties/toolbar) are in ui-panels.js, dialogs/export/projects/settings in ui-export.js. */
(function () {
  'use strict';
  const C = window.EditorCore, D = window.EditorDSP, S = window.EditorStorage;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const LANE_H = 72;

  const App = window.EditorApp = {
    C, D, S, $, $$, esc, LANE_H,
    project: null, hist: null, engine: null, assets: {}, pyr: new Map(),
    sel: new Set(), selTrack: null, clipboard: null, region: null,
    shortcuts: Object.assign({}, C.DEFAULT_SHORTCUTS), dirty: false, busy: 0, drag: null,
    left: 'audio', leftSub: 'songs', props: 'basic',
  };

  // ------------------------------------------------------------------ small helpers
  App.toast = (msg, err) => {
    const t = $('#toast'); t.textContent = msg; t.className = 'ed-toast' + (err ? ' err' : ''); t.hidden = false;
    clearTimeout(App._tt); App._tt = setTimeout(() => { t.hidden = true; }, err ? 6000 : 2600);
  };
  App.pps = () => App.project.settings.pxPerSec;
  App.px = (t) => t * App.pps();
  App.timeAt = (x) => Math.max(0, x / App.pps());
  App.clips = () => [...App.sel].map((id) => C.byId(App.project.clips, id)).filter(Boolean);
  App.setDirty = (v) => { App.dirty = v; $('#save-state').textContent = v ? 'Unsaved changes' : (App._savedAt ? 'Saved' : ''); };

  /** Commit one user action to the undo history, persist a recovery copy shortly after, and refresh. */
  App.commit = (opts) => {
    if (App.hist.commit()) { App.setDirty(true); App.scheduleAutosave(); }
    if (!opts || !opts.quiet) App.renderAll();
    App.updateToolbar();
  };
  App.scheduleAutosave = () => {
    clearTimeout(App._as);
    App._as = setTimeout(async () => { try { await S.saveRecovery(App.serialize()); } catch (e) { /* storage may be unavailable */ } }, 1200);
  };
  App.serialize = () => {
    const p = C.clone({ ...App.project, assets: {} });
    const used = new Set(App.project.clips.map((c) => c.assetId));
    for (const id of used) if (App.project.assets[id]) p.assets[id] = App.project.assets[id];
    p.modified = Date.now();
    return p;
  };

  // ------------------------------------------------------------------ assets
  /** Decode bytes into an asset (Float32 channels at the engine rate) and register it in the project. */
  App.addAssetFromBytes = async (id, name, kind, bytes, source, persist) => {
    const ctx = App.engine.ensureCtx();
    let buf;
    try { buf = await ctx.decodeAudioData(bytes.slice(0)); }
    catch (e) { throw new Error(`Could not decode "${name}". Supported: WAV, MP3, FLAC, OGG/Opus, M4A (browser dependent).`); }
    const chans = []; for (let i = 0; i < Math.min(2, buf.numberOfChannels); i++) chans.push(buf.getChannelData(i).slice());
    const meta = { id, name, kind, duration: buf.duration, sampleRate: buf.sampleRate, channels: chans.length, created: Date.now(), source: source || null };
    App.assets[id] = { chans, sampleRate: buf.sampleRate, duration: buf.duration, name };
    App.project.assets[id] = { id, name, kind, duration: buf.duration, source: source || null };
    if (persist !== false) { try { await S.putAsset(meta, bytes); } catch (e) { App.toast('Could not store this audio for later (storage full or disabled); it works for this session only.', true); } }
    return meta;
  };
  App.ensureAsset = async (id) => {
    if (App.assets[id]) return App.assets[id];
    const bytes = await S.getAssetBytes(id);
    if (!bytes) throw new Error('The audio for this clip is no longer stored. Re-import it.');
    const m = App.project.assets[id] || {};
    await App.addAssetFromBytes(id, m.name || id, m.kind || 'local', bytes, m.source, false);
    return App.assets[id];
  };

  // ------------------------------------------------------------------ processing (worker) + waveform peaks
  /** Make processed buffers for every clip, then refresh waveforms; rebuild the live graph if playing. */
  App.process = async () => {
    App.busy++; $('#save-state').textContent = 'Processing…';
    try {
      await App.engine.prepare(App.project, App.assets);
      App.pyr.clear();
      App.drawAllWaveforms();
      if (App.engine.playing) await App.engine.rebuild();
    } catch (e) { App.toast(e.message, true); }
    finally { App.busy--; App.setDirty(App.dirty); }
  };
  /** Min/max pyramid of a clip's processed audio (mono mix) for fast drawing at any zoom. */
  function pyramid(clip) {
    const key = D.heavyKey(clip.assetId, clip.srcStart, clip.srcDur, clip.params);
    if (App.pyr.has(key)) return App.pyr.get(key);
    const proc = App.engine.processed.get(key);
    if (!proc) return null;
    const n = proc.chans[0].length, mono = new Float32Array(n);
    for (const ch of proc.chans) for (let i = 0; i < n; i++) mono[i] += ch[i] / proc.chans.length;
    const levels = []; let bucket = 128, p = D.peaks(mono, bucket);
    levels.push({ bucket, p });
    while (p.length / 2 > 64) {                                   // halve the resolution each level
      const q = new Float32Array(Math.ceil(p.length / 4) * 2);
      for (let i = 0; i < q.length / 2; i++) {
        let mn = 1, mx = -1;
        for (let j = 0; j < 2; j++) { const k = (2 * i + j) * 2; if (k < p.length) { if (p[k] < mn) mn = p[k]; if (p[k + 1] > mx) mx = p[k + 1]; } }
        q[2 * i] = mn > mx ? 0 : mn; q[2 * i + 1] = mn > mx ? 0 : mx;
      }
      bucket *= 2; p = q; levels.push({ bucket, p });
    }
    const out = { levels, sr: proc.sampleRate };
    App.pyr.set(key, out);
    return out;
  }
  function drawWave(canvas, clip, widthPx) {
    const pyr = pyramid(clip);
    const w = Math.max(1, Math.round(widthPx)), h = canvas.clientHeight || 50, dpr = window.devicePixelRatio || 1;
    canvas.width = Math.min(w, 8192) * dpr; canvas.height = h * dpr;
    const g = canvas.getContext('2d'); g.scale(canvas.width / w < dpr ? canvas.width / w : dpr, dpr); g.clearRect(0, 0, w, h);
    if (!pyr) return;
    const sr = pyr.sr, spp = (clip.duration * sr) / w;                 // samples per pixel
    let lv = pyr.levels[0];
    for (const l of pyr.levels) if (l.bucket <= spp) lv = l;
    const x0 = clip.offset * sr;
    g.fillStyle = 'rgba(255,255,255,.85)';
    const mid = h / 2;
    const cw = Math.min(w, 8192), scale = w / cw;
    for (let x = 0; x < cw; x++) {
      const a = Math.floor((x0 + x * scale * spp) / lv.bucket), b = Math.max(a + 1, Math.floor((x0 + (x + 1) * scale * spp) / lv.bucket));
      let mn = 1, mx = -1;
      for (let k = a; k < b && 2 * k + 1 < lv.p.length; k++) { if (lv.p[2 * k] < mn) mn = lv.p[2 * k]; if (lv.p[2 * k + 1] > mx) mx = lv.p[2 * k + 1]; }
      if (mn > mx) continue;
      const y1 = mid - mx * mid * 0.95, y2 = mid - mn * mid * 0.95;
      g.fillRect(x * (w / cw), y1, Math.max(1, w / cw), Math.max(1, y2 - y1));
    }
  }
  App.drawAllWaveforms = () => {
    for (const el of $$('.ed-clip')) { const c = C.byId(App.project.clips, el.dataset.id); const cv = $('canvas', el); if (c && cv) drawWave(cv, c, el.clientWidth); }
  };

  // ------------------------------------------------------------------ rendering: tracks, clips, ruler, playhead
  App.renderAll = () => {
    App.renderTracks(); App.renderRuler(); App.updatePlayhead(); App.renderRegion(); App.renderProps(); App.updateTime();
    $('#proj-name').value = App.project.name;
    const duck = $('#duck-out'); if (duck) duck.textContent = '';
  };
  App.renderTracks = () => {
    const p = App.project, heads = $('#head-list'), lanes = $('#lanes');
    heads.innerHTML = ''; lanes.innerHTML = '';
    const total = Math.max(60, C.projectDuration(p) + 20);
    $('#canvas').style.width = App.px(total) + 'px';
    p.tracks.forEach((t) => {
      const h = document.createElement('div');
      h.className = 'ed-head' + (App.selTrack === t.id ? ' sel' : ''); h.dataset.id = t.id;
      h.innerHTML = `<div class="nm" title="Double-click to rename">${esc(t.name)}</div>
        <div class="ed-hb"><button data-a="mute" aria-pressed="${t.mute}" title="Mute (M)">M</button><button data-a="solo" aria-pressed="${t.solo}" title="Solo">S</button>
        <button data-a="lock" aria-pressed="${t.lock}" title="Lock track">L</button><button data-a="del" title="Delete track" aria-label="Delete track">×</button></div>
        <input type="range" min="-40" max="12" step="0.5" value="${t.volume}" title="Track volume (${t.volume} dB)" aria-label="Track volume">`;
      heads.appendChild(h);
      const lane = document.createElement('div');
      lane.className = 'ed-lane' + (App.selTrack === t.id ? ' sel' : '') + (t.lock ? ' locked' : ''); lane.dataset.id = t.id;
      lanes.appendChild(lane);
      for (const c of C.clipsOnTrack(p, t.id)) lane.appendChild(App.clipEl(c, t));
    });
    App.syncHeadScroll();
  };
  App.clipEl = (c, t) => {
    const el = document.createElement('div');
    const xf = C.crossfades(App.project)[c.id] || {};
    el.className = 'ed-clip' + (t.role === 'voice' ? ' voice' : '') + (App.sel.has(c.id) ? ' sel' : '') + (c.params.mute ? ' muted' : '');
    el.dataset.id = c.id; el.style.left = App.px(c.start) + 'px'; el.style.width = Math.max(4, App.px(c.duration)) + 'px';
    const fi = Math.max(c.params.fadeIn || 0, xf.fadeIn || 0), fo = Math.max(c.params.fadeOut || 0, xf.fadeOut || 0);
    el.innerHTML = `<div class="ct">${esc(c.name)}</div><canvas></canvas>${fi ? `<div class="fi" style="width:${App.px(fi)}px"></div>` : ''}${fo ? `<div class="fo" style="right:0;width:${App.px(fo)}px;left:auto"></div>` : ''}<div class="h l" data-h="l"></div><div class="h r" data-h="r"></div>`;
    el.title = `${c.name}\n${C.fmtTime(c.start)} – ${C.fmtTime(C.clipEnd(c))}`;
    requestAnimationFrame(() => { const cv = $('canvas', el); if (cv) drawWave(cv, c, el.clientWidth); });
    return el;
  };
  App.syncHeadScroll = () => { const sc = $('#scroll'); $('#head-list').style.transform = `translateY(${-sc.scrollTop}px)`; };
  App.renderRuler = () => {
    const cv = $('#ruler-c'), w = Math.max($('#scroll').clientWidth, parseFloat($('#canvas').style.width) || 0), dpr = window.devicePixelRatio || 1;
    cv.style.width = w + 'px'; cv.width = w * dpr; cv.height = 28 * dpr;
    const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, w, 28);
    const pps = App.pps();
    const steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    const major = steps.find((s) => s * pps >= 90) || 600, minor = major / 5;
    g.fillStyle = '#a3a3bd'; g.font = '10px Inter, sans-serif'; g.textBaseline = 'top';
    for (let t = 0; t * pps < w; t += minor) {
      const x = Math.round(t * pps) + 0.5, isMajor = Math.abs(t / major - Math.round(t / major)) < 1e-6;
      g.strokeStyle = isMajor ? '#6b6b88' : '#34344a';
      g.beginPath(); g.moveTo(x, isMajor ? 12 : 20); g.lineTo(x, 28); g.stroke();
      if (isMajor) g.fillText(major < 1 ? C.fmtTime(t) : `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, '0')}`, x + 3, 2);
    }
  };
  App.updatePlayhead = () => {
    const x = App.px(App.engine.position());
    $('#ph').style.left = x + 'px'; $('#ph-handle').style.left = x + 'px';
  };
  App.updateTime = () => { $('#time-cur').textContent = C.fmtTime(App.engine.position()); $('#time-tot').textContent = C.fmtTime(C.projectDuration(App.project)); };
  App.renderRegion = () => {
    const r = $('#region');
    if (!App.region) { r.hidden = true; return; }
    r.hidden = false; r.style.left = App.px(App.region[0]) + 'px'; r.style.width = App.px(App.region[1] - App.region[0]) + 'px';
  };

  // ------------------------------------------------------------------ transport
  App.play = async () => {
    if (App.engine.playing) { App.engine.pause(); App.updatePlay(); return; }
    if (!App.project.clips.length) { App.toast('Add audio to the timeline first.'); return; }
    try {
      await App.engine.prepare(App.project, App.assets);
      let from = App.engine.pos, end = C.projectDuration(App.project);
      if (from >= end - 0.02) from = 0;
      if (App.region && App.project.settings.loop) from = App.region[0];
      await App.engine.play(from);
    } catch (e) { App.toast(e.message, true); return; }
    App.updatePlay(); App.tick();
  };
  App.updatePlay = () => { const b = $('#btn-play'); b.textContent = App.engine.playing ? '❚❚' : '▶'; b.setAttribute('aria-label', App.engine.playing ? 'Pause' : 'Play'); };
  App.tick = () => {
    cancelAnimationFrame(App._raf);
    const loop = () => {
      App.updatePlayhead(); App.updateTime(); App.drawMeter();
      if (App.engine.playing) {
        const sc = $('#scroll'), x = App.px(App.engine.position());
        if (x > sc.scrollLeft + sc.clientWidth - 40 || x < sc.scrollLeft) sc.scrollLeft = Math.max(0, x - 80);     // keep the needle in view while playing
        const pos = App.engine.position();
        if (App.project.settings.loop) {
          const [a, b] = App.region || [0, C.projectDuration(App.project)];
          if (pos >= b - 0.01) { App.engine.play(a); }
        }
        App._raf = requestAnimationFrame(loop);
      } else App.drawMeter();
    };
    App._raf = requestAnimationFrame(loop);
  };
  App.drawMeter = () => {
    const cv = $('#meter'), g = cv.getContext('2d'); g.clearRect(0, 0, cv.width, cv.height);
    const an = App.engine.analyser; if (!an || !App.engine.playing) return;
    const buf = new Float32Array(an.fftSize); an.getFloatTimeDomainData(buf);
    let pk = 0; for (const v of buf) pk = Math.max(pk, Math.abs(v));
    const db = 20 * Math.log10(pk + 1e-6), w = Math.max(0, Math.min(1, (db + 60) / 60)) * cv.width;
    g.fillStyle = db > -1 ? '#f87171' : db > -9 ? '#fcd34d' : '#34d399'; g.fillRect(0, 0, w, cv.height);
  };
  App.seek = async (t, fromUser) => {
    t = Math.max(0, t);
    await App.engine.seek(t);
    App.updatePlayhead(); App.updateTime();
  };
  App.snap = (t, exclude) => C.snapTime(App.project, t, { enabled: App.project.settings.snap, pxPerSec: App.pps(), playhead: App.engine.position(), exclude, grid: App.project.settings.grid });

  // ------------------------------------------------------------------ clip operations (all go through history)
  const unlockedSelection = () => App.clips().filter((c) => !C.trackOf(App.project, c).lock);
  App.addClipFromAsset = (assetId, trackId, start) => {
    const asset = App.project.assets[assetId];
    const track = C.byId(App.project.tracks, trackId) || App.project.tracks[0];
    if (track.lock) { App.toast('That track is locked.'); return null; }
    const clip = C.addClip(App.project, { id: assetId, name: asset.name, duration: asset.duration }, track.id, start);
    App.sel = new Set([clip.id]); App.selTrack = track.id;
    return clip;
  };
  App.act = {
    split() {
      const t = App.engine.position(); let n = 0;
      const targets = App.sel.size ? App.clips() : App.project.clips.filter((c) => t > c.start && t < C.clipEnd(c));
      const right = [];
      for (const c of targets) { const r = C.splitClip(App.project, c.id, t); if (r) { n++; right.push(r[1].id); } }
      if (n) App.sel = new Set(right);                     // keep only the right-hand pieces selected, so each half can be dragged on its own
      if (!n) { App.toast('Place the playhead inside a clip to split it.'); return; }
      App.commit();
    },
    delete() { const ids = [...App.sel]; if (C.deleteClips(App.project, ids)) { App.sel.clear(); App.commit(); } else if (ids.length) App.toast('Those clips are on a locked track.'); },
    duplicate() { const d = C.duplicateClips(App.project, [...App.sel]); if (d.length) { App.sel = new Set(d.map((c) => c.id)); App.commit(); App.process(); } },
    copy() { App.clipboard = C.copyClips(App.project, [...App.sel]); if (App.clipboard) App.toast(`Copied ${App.clipboard.length} clip(s)`); },
    cut() { App.act.copy(); if (App.clipboard) App.act.delete(); },
    paste() {
      if (!App.clipboard) { App.toast('Nothing to paste.'); return; }
      const tid = App.selTrack || App.project.tracks[0].id;
      const out = C.pasteClips(App.project, App.clipboard, App.engine.position(), tid);
      if (out.length) { App.sel = new Set(out.map((c) => c.id)); App.commit(); App.process(); }
    },
    selectAll() { App.sel = new Set(App.project.clips.map((c) => c.id)); App.renderAll(); },
    trimStartToPlayhead() { let n = 0; for (const c of App.clips()) if (C.trimStart(App.project, c.id, App.engine.position())) n++; if (n) App.commit(); },
    trimEndToPlayhead() { let n = 0; for (const c of App.clips()) if (C.trimEnd(App.project, c.id, App.engine.position())) n++; if (n) App.commit(); },
    undo() { if (App.hist.undo()) { App.sel = new Set([...App.sel].filter((id) => C.byId(App.project.clips, id))); App.setDirty(true); App.scheduleAutosave(); App.renderAll(); App.process(); App.updateToolbar(); } },
    redo() { if (App.hist.redo()) { App.sel = new Set([...App.sel].filter((id) => C.byId(App.project.clips, id))); App.setDirty(true); App.scheduleAutosave(); App.renderAll(); App.process(); App.updateToolbar(); } },
    muteTrack() { const t = C.byId(App.project.tracks, App.selTrack || (App.clips()[0] && App.clips()[0].trackId)); if (t) { t.mute = !t.mute; App.commit(); if (App.engine.playing) App.engine.rebuild(); } },
    addTrack(name, role) { const t = C.addTrack(App.project, name, role); App.selTrack = t.id; App.commit(); return t; },
    toggleSnap() { App.project.settings.snap = !App.project.settings.snap; App.commit({ quiet: true }); App.updateToolbar(); },
    toggleLoop() { App.project.settings.loop = !App.project.settings.loop; App.commit({ quiet: true }); App.updateToolbar(); },
    silence() {
      const clips = App.clips(); let n = 0;
      for (const c of clips) {
        const proc = App.engine.processed.get(D.heavyKey(c.assetId, c.srcStart, c.srcDur, c.params));
        if (!proc) continue;
        const mono = new Float32Array(proc.chans[0].length); for (const ch of proc.chans) for (let i = 0; i < mono.length; i++) mono[i] += ch[i] / proc.chans.length;
        const regions = D.silenceRegions(mono, proc.sampleRate, -45, 0.3, 0.05);
        if (!regions.length) { App.toast('This clip is silent: nothing to keep.', true); continue; }
        const out = C.keepRegions(App.project, c.id, regions);
        if (out.length) { n += out.length; App.sel.delete(c.id); out.forEach((o) => App.sel.add(o.id)); }
      }
      if (n) { App.commit(); App.toast(`Silence removed: ${n} piece(s) kept`); } else App.toast('No silence longer than 0.3 s found.');
    },
  };

  // ------------------------------------------------------------------ pointer interaction on the timeline
  function laneAt(y) {
    const lanes = $$('#lanes .ed-lane');
    for (const l of lanes) { const r = l.getBoundingClientRect(); if (y >= r.top && y < r.bottom) return l.dataset.id; }
    return null;
  }
  function canvasX(e) { const r = $('#canvas').getBoundingClientRect(); return e.clientX - r.left; }
  let rafPending = false;
  const redrawSoon = () => { if (!rafPending) { rafPending = true; requestAnimationFrame(() => { rafPending = false; App.renderTracks(); App.renderRegion(); App.updatePlayhead(); }); } };

  App.bindTimeline = () => {
    const lanes = $('#lanes'), ruler = $('#ruler'), scroll = $('#scroll');
    scroll.addEventListener('scroll', () => App.syncHeadScroll());
    // ---- clips: select, move (also between tracks), trim
    lanes.addEventListener('pointerdown', (e) => {
      const clipEl = e.target.closest('.ed-clip');
      if (!clipEl) {                                         // empty lane area: select the track and place the playhead
        const tid = e.target.closest('.ed-lane') && e.target.closest('.ed-lane').dataset.id;
        if (tid) { App.selTrack = tid; if (!e.shiftKey) App.sel.clear(); App.seek(App.snap(App.timeAt(canvasX(e))).t); App.renderAll(); }
        return;
      }
      const c = C.byId(App.project.clips, clipEl.dataset.id);
      if (!c) return;
      let collapse = null;
      if (e.shiftKey) { App.sel.has(c.id) ? App.sel.delete(c.id) : App.sel.add(c.id); }
      else if (!App.sel.has(c.id)) App.sel = new Set([c.id]);
      else if (App.sel.size > 1) collapse = c.id;           // a plain click inside a multi-selection narrows it to this clip (a drag still moves them all)
      App.selTrack = c.trackId;
      const mode = e.target.dataset.h === 'l' ? 'trimL' : e.target.dataset.h === 'r' ? 'trimR' : 'move';
      const ids = mode === 'move' ? [...App.sel] : [c.id];
      App.drag = { mode, id: c.id, ids, x0: e.clientX, y0: e.clientY, orig: ids.map((id) => C.clone(C.byId(App.project.clips, id))), moved: false, collapse };
      try { clipEl.setPointerCapture(e.pointerId); } catch (err) { /* synthetic or already released pointer */ }
      App.renderAll();
      e.preventDefault();
    });
    window.addEventListener('pointermove', (e) => {
      const d = App.drag; if (!d) return;
      const dx = (e.clientX - d.x0) / App.pps();
      if (!d.moved && Math.abs(e.clientX - d.x0) + Math.abs(e.clientY - d.y0) < 3) return;
      d.moved = true;
      const p = App.project;
      // restore originals, then apply the new delta (so the result never accumulates drift)
      for (const o of d.orig) Object.assign(C.byId(p.clips, o.id), { start: o.start, offset: o.offset, duration: o.duration, trackId: o.trackId });
      let snapped = null;
      if (d.mode === 'move') {
        const first = d.orig[0], ex = new Set(d.ids);
        let dt = dx;
        const a = App.snap(first.start + dx, ex), b = App.snap(first.start + first.duration + dx, ex);
        if (a.snapped) { dt = a.t - first.start; snapped = a.t; } else if (b.snapped) { dt = b.t - first.start - first.duration; snapped = b.t; }
        const laneId = laneAt(e.clientY);
        const fromIdx = p.tracks.findIndex((t) => t.id === first.trackId), toIdx = laneId ? p.tracks.findIndex((t) => t.id === laneId) : fromIdx;
        C.moveClips(p, d.ids, dt, toIdx - fromIdx);
      } else {
        const o = d.orig[0];
        if (d.mode === 'trimL') { const s = App.snap(o.start + dx, new Set([o.id])); C.trimStart(p, o.id, s.t); snapped = s.snapped ? s.t : null; }
        else { const s = App.snap(o.start + o.duration + dx, new Set([o.id])); C.trimEnd(p, o.id, s.t); snapped = s.snapped ? s.t : null; }
      }
      const sl = $('#snapline'); if (snapped != null) { sl.hidden = false; sl.style.left = App.px(snapped) + 'px'; } else sl.hidden = true;
      redrawSoon();
    });
    window.addEventListener('pointerup', async () => {
      const d = App.drag; if (!d) return;
      App.drag = null; $('#snapline').hidden = true;
      if (d.moved) { App.commit(); if (App.engine.playing) await App.engine.rebuild(); }
      else { if (d.collapse) App.sel = new Set([d.collapse]); App.renderAll(); }
    });
    lanes.addEventListener('dblclick', (e) => {                // double-click a clip: select it and open the Basic tab
      const el = e.target.closest('.ed-clip'); if (el) { App.setProps('basic'); }
    });
    // ---- track headers
    $('#head-list').addEventListener('click', (e) => {
      const h = e.target.closest('.ed-head'); if (!h) return;
      const t = C.byId(App.project.tracks, h.dataset.id), a = e.target.dataset.a;
      App.selTrack = t.id;
      if (a === 'mute') t.mute = !t.mute; else if (a === 'solo') t.solo = !t.solo; else if (a === 'lock') t.lock = !t.lock;
      else if (a === 'del') {
        if (t.lock) { App.toast('Unlock the track before deleting it.'); return; }
        if (App.project.tracks.length <= 1) { App.toast('A project needs at least one track.'); return; }
        const n = App.project.clips.filter((c) => c.trackId === t.id).length;
        if (n && !confirm(`Delete "${t.name}" and its ${n} clip(s)?`)) return;
        C.removeTrack(App.project, t.id); App.sel = new Set([...App.sel].filter((id) => C.byId(App.project.clips, id)));
      } else { App.renderAll(); return; }
      App.commit(); if (App.engine.playing) App.engine.rebuild();
    });
    $('#head-list').addEventListener('input', (e) => {
      if (e.target.type !== 'range') return;
      const t = C.byId(App.project.tracks, e.target.closest('.ed-head').dataset.id); t.volume = +e.target.value; e.target.title = `Track volume (${t.volume} dB)`;
      App.engine.updateTrack(t);
    });
    $('#head-list').addEventListener('change', (e) => { if (e.target.type === 'range') { App.commit({ quiet: true }); } });
    $('#head-list').addEventListener('dblclick', (e) => {
      const nm = e.target.closest('.nm'); if (!nm) return;
      const t = C.byId(App.project.tracks, nm.closest('.ed-head').dataset.id);
      nm.contentEditable = 'true'; nm.focus(); document.getSelection().selectAllChildren(nm);
      const done = () => { nm.contentEditable = 'false'; const v = nm.textContent.trim().slice(0, 40) || t.name; if (v !== t.name) { t.name = v; App.commit(); } else nm.textContent = t.name; };
      nm.onblur = done; nm.onkeydown = (ev) => { if (ev.key === 'Enter') { ev.preventDefault(); nm.blur(); } ev.stopPropagation(); };
    });
    $('#btn-addtrack').addEventListener('click', () => App.act.addTrack());
    // ---- ruler: click/drag = playhead (snaps to clip edges); Shift+drag = region
    let rdrag = null;
    ruler.addEventListener('pointerdown', (e) => {
      const t0 = App.timeAt(canvasX(e));
      if (e.shiftKey) { rdrag = { mode: 'region', a: t0 }; App.region = [t0, t0]; }
      else { rdrag = { mode: 'ph' }; App.seek(App.snap(t0).t); }
      try { ruler.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ } e.preventDefault();
    });
    ruler.addEventListener('pointermove', (e) => {
      if (!rdrag) return;
      const t = App.timeAt(canvasX(e));
      if (rdrag.mode === 'region') { const s = App.snap(t).t; App.region = [Math.min(rdrag.a, s), Math.max(rdrag.a, s)]; App.renderRegion(); }
      else App.seek(App.snap(t).t);
    });
    const end = () => { if (rdrag && rdrag.mode === 'region' && App.region && App.region[1] - App.region[0] < 0.02) { App.region = null; App.renderRegion(); } rdrag = null; };
    ruler.addEventListener('pointerup', end); ruler.addEventListener('pointercancel', end);
    // ---- zoom with Ctrl + wheel (around the pointer)
    scroll.addEventListener('wheel', (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const r = scroll.getBoundingClientRect(), t = App.timeAt(e.clientX - r.left + scroll.scrollLeft);
      App.setZoom(App.pps() * (e.deltaY < 0 ? 1.2 : 1 / 1.2));
      scroll.scrollLeft = App.px(t) - (e.clientX - r.left);
    }, { passive: false });
    // ---- drag & drop from the library
    lanes.addEventListener('dragover', (e) => { if (!e.dataTransfer.types.includes('application/x-vf-asset')) return; e.preventDefault(); const id = laneAt(e.clientY); $$('.ed-lane').forEach((l) => l.classList.toggle('drop', l.dataset.id === id)); });
    lanes.addEventListener('dragleave', () => $$('.ed-lane').forEach((l) => l.classList.remove('drop')));
    lanes.addEventListener('drop', async (e) => {
      const raw = e.dataTransfer.getData('application/x-vf-asset'); if (!raw) return;
      e.preventDefault(); $$('.ed-lane').forEach((l) => l.classList.remove('drop'));
      const tid = laneAt(e.clientY) || App.project.tracks[0].id, t = App.snap(App.timeAt(canvasX(e))).t;
      await App.insertLibraryItem(JSON.parse(raw), tid, t);
    });
  };
  App.setZoom = (pps) => { App.project.settings.pxPerSec = C.clamp(pps, 4, 1200); $('#zoom') && ($('#zoom').value = Math.log(App.pps())); App.renderTracks(); App.renderRuler(); App.updatePlayhead(); App.renderRegion(); };

  // ------------------------------------------------------------------ keyboard
  App.bindKeys = () => {
    document.addEventListener('keydown', (e) => {
      if (App.capturing) return;
      if (!$('#modal').hidden) { if (e.key === 'Escape') App.closeDialog(); return; }
      if (C.isTextTarget(e.target) || (e.target.isContentEditable)) return;
      const a = C.actionFor(e, App.shortcuts);
      if (!a) return;
      const map = {
        playPause: () => App.play(), undo: App.act.undo, redo: App.act.redo, copy: App.act.copy, paste: App.act.paste, cut: App.act.cut, selectAll: App.act.selectAll,
        delete: App.act.delete, duplicate: App.act.duplicate, save: () => App.saveProject(), export: () => App.openExport(), split: App.act.split, muteTrack: App.act.muteTrack,
        home: () => App.seek(0), end: () => App.seek(C.projectDuration(App.project)),
        left: () => App.seek(App.engine.position() - (e.shiftKey ? 1 : 0.1)), right: () => App.seek(App.engine.position() + (e.shiftKey ? 1 : 0.1)),
      };
      if (!map[a]) return;
      e.preventDefault(); map[a]();
    });
  };

  // ------------------------------------------------------------------ project lifecycle
  App.newProject = async (name) => {
    if (App.engine.playing) App.engine.pause();
    App.project = C.newProject(name); App.hist = new C.History(App.project); App.assets = {}; App.sel.clear(); App.selTrack = App.project.tracks[0].id; App.region = null;
    App.engine.project = App.project; App.engine.assets = App.assets; App.engine.pos = 0;
    App.setDirty(false); App._savedAt = null;
    App.renderAll(); App.updateToolbar(); App.updatePlay();
  };
  App.loadProjectData = async (data) => {
    if (App.engine.playing) App.engine.pause();
    const p = C.clone(data); p.settings = Object.assign({ snap: true, grid: 0, crossfade: true, loop: false, pxPerSec: 80, limiter: true }, p.settings || {});
    App.project = p; App.hist = new C.History(p); App.assets = {}; App.sel.clear(); App.region = null; App.selTrack = p.tracks[0] && p.tracks[0].id;
    App.engine.project = p; App.engine.assets = App.assets; App.engine.pos = 0;
    const missing = [];
    for (const id of new Set(p.clips.map((c) => c.assetId))) { try { await App.ensureAsset(id); } catch (e) { missing.push((p.assets[id] || {}).name || id); } }
    if (missing.length) {
      p.clips = p.clips.filter((c) => !missing.includes((p.assets[c.assetId] || {}).name || c.assetId));
      App.hist.reset(); App.toast(`Some audio could not be restored and was removed: ${missing.join(', ')}`, true);
    }
    App.renderAll(); App.updateToolbar(); App.updatePlay(); App.setDirty(false);
    await App.process();
  };
  App.saveProject = async () => {
    try { await S.saveProject(App.serialize()); await S.clearRecovery(); App._savedAt = Date.now(); App.setDirty(false); App.toast('Project saved'); }
    catch (e) { App.toast('Could not save the project: ' + e.message, true); }
  };

  App.init = async () => {
    try { const sc = JSON.parse(localStorage.getItem('vf-editor-shortcuts') || 'null'); if (sc) Object.assign(App.shortcuts, sc); } catch (e) { /* defaults */ }
    App.engine = new window.EditorEngine.Engine({ sampleRate: 48000 });
    await App.newProject('Untitled project');
    App.bindTimeline(); App.bindKeys(); App.bindChrome();
    window.addEventListener('beforeunload', (e) => { if (App.dirty) { e.preventDefault(); e.returnValue = ''; } });
    window.addEventListener('resize', () => { App.renderRuler(); App.drawAllWaveforms(); });
    App.renderLeft(); App.renderToolbar(); App.renderProps();
    // offer to restore an interrupted session
    try {
      const rec = await S.loadRecovery();
      if (rec && rec.project && rec.project.clips && rec.project.clips.length) App.offerRecovery(rec);
    } catch (e) { /* storage unavailable */ }
    // a project opened from elsewhere (e.g. Create)
    const m = /[?&]open=([^&]+)/.exec(location.search);
    if (m) { try { const pr = await S.loadProject(decodeURIComponent(m[1])); if (pr) await App.loadProjectData(pr); } catch (e) { /* ignore */ } }
  };
})();
