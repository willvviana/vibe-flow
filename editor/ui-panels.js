/* Editor panels: toolbar, library (left), properties (right), microphone recording. Extends window.EditorApp (see ui.js). */
(function () {
  'use strict';
  const App = window.EditorApp, { C, D, S, $, $$, esc } = App;
  const HEAVY = new Set(['speed', 'keepPitch', 'pitch', 'reverse', 'loops', 'normalize', 'noiseReduce']);
  const fmtDur = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;

  // ================================================================== toolbar
  const TOOLS = [
    { id: 'copy', t: '⧉', title: 'Copy (Ctrl+C)', run: () => App.act.copy(), need: 'sel' },
    { id: 'cut', t: '✂', title: 'Cut (Ctrl+X)', run: () => App.act.cut(), need: 'sel' },
    { id: 'paste', t: '⎘', title: 'Paste at the playhead (Ctrl+V)', run: () => App.act.paste(), need: 'clip' },
    { id: 'undo', t: '↶', title: 'Undo (Ctrl+Z)', run: () => App.act.undo(), need: 'undo' },
    { id: 'redo', t: '↷', title: 'Redo (Ctrl+Shift+Z)', run: () => App.act.redo(), need: 'redo' },
    '|',
    { id: 'split', t: '⟊ Split', title: 'Split at the playhead (S)', run: () => App.act.split() },
    { id: 'trimL', t: '⇥', title: 'Trim the start of the selected clip to the playhead', run: () => App.act.trimStartToPlayhead(), need: 'sel' },
    { id: 'trimR', t: '⇤', title: 'Trim the end of the selected clip to the playhead', run: () => App.act.trimEndToPlayhead(), need: 'sel' },
    { id: 'del', t: '🗑', title: 'Delete (Delete)', run: () => App.act.delete(), need: 'sel' },
    { id: 'dup', t: '⧉+', title: 'Duplicate (Ctrl+D)', run: () => App.act.duplicate(), need: 'sel' },
    { id: 'all', t: '☰', title: 'Select all (Ctrl+A)', run: () => App.act.selectAll() },
    { id: 'sil', t: '🔇 Silence', title: 'Remove silence from the selected clips', run: () => App.act.silence(), need: 'sel' },
    '|',
    { id: 'snap', t: '🧲', title: 'Snap to clip edges and the playhead', run: () => App.act.toggleSnap(), pressed: () => App.project.settings.snap },
    { id: 'loop', t: '⟲', title: 'Loop playback (region if set, else the whole project)', run: () => App.act.toggleLoop(), pressed: () => App.project.settings.loop },
    { id: 'rec', t: '🎙', title: 'Record from the microphone', run: () => App.toggleRecord(), pressed: () => !!App.recording },
    { id: 'clearreg', t: '▭ Region', title: 'Clear the export region (set one with Shift+drag on the ruler)', run: () => { App.region = null; App.renderRegion(); }, need: 'region' },
  ];
  App.renderToolbar = () => {
    const tb = $('#toolbar'); tb.innerHTML = '';
    for (const t of TOOLS) {
      if (t === '|') { const s = document.createElement('span'); s.className = 'ed-sep'; tb.appendChild(s); continue; }
      const b = document.createElement('button'); b.className = 'ed-tool'; b.id = 'tool-' + t.id; b.textContent = t.t; b.title = t.title; b.setAttribute('aria-label', t.title);
      b.onclick = t.run; tb.appendChild(b);
    }
    const sep = document.createElement('span'); sep.className = 'ed-sep'; tb.appendChild(sep);
    const zo = document.createElement('button'); zo.className = 'ed-tool'; zo.textContent = '−'; zo.title = 'Zoom out (Ctrl+wheel)'; zo.setAttribute('aria-label', 'Zoom out'); zo.onclick = () => App.setZoom(App.pps() / 1.4);
    const z = document.createElement('input'); z.type = 'range'; z.id = 'zoom'; z.className = 'ed-zoom'; z.min = Math.log(4); z.max = Math.log(1200); z.step = 0.01; z.value = Math.log(App.pps()); z.title = 'Zoom'; z.setAttribute('aria-label', 'Timeline zoom');
    z.oninput = () => App.setZoom(Math.exp(+z.value));
    const zi = document.createElement('button'); zi.className = 'ed-tool'; zi.textContent = '+'; zi.title = 'Zoom in (Ctrl+wheel)'; zi.setAttribute('aria-label', 'Zoom in'); zi.onclick = () => App.setZoom(App.pps() * 1.4);
    const fit = document.createElement('button'); fit.className = 'ed-tool'; fit.textContent = 'Fit'; fit.title = 'Fit the whole project in view';
    fit.onclick = () => { const d = Math.max(C.projectDuration(App.project), 5); App.setZoom(($('#scroll').clientWidth - 40) / d); $('#scroll').scrollLeft = 0; };
    tb.append(zo, z, zi, fit);
    mountCorner();
    App.updateToolbar();
  };
  App.updateToolbar = () => {
    if (!App.hist) return;
    const has = { sel: App.sel.size > 0, clip: !!App.clipboard, undo: App.hist.canUndo(), redo: App.hist.canRedo(), region: !!App.region };
    for (const t of TOOLS) {
      if (t === '|') continue;
      const b = $('#tool-' + t.id); if (!b) continue;
      if (t.need) b.disabled = !has[t.need];
      if (t.pressed) b.setAttribute('aria-pressed', !!t.pressed());
    }
    const u = $('#under'); if (u) { u.value = App.underVoice(); $('#under-out').textContent = u.value + '%'; }
    $('#btn-play').setAttribute('aria-label', App.engine.playing ? 'Pause' : 'Play');
  };

  // ================================================================== "under voice" ducking quick control
  App.underVoice = () => {
    const t = App.project.tracks.find((x) => x.role !== 'voice' && x.duck && x.duck.on && x.duck.auto);
    return t ? Math.round(-t.duck.amountDb / 24 * 100) : 0;
  };
  App.setUnderVoice = (pct) => {
    const voices = App.project.tracks.filter((t) => t.role === 'voice').map((t) => t.id);
    for (const t of App.project.tracks) {
      if (t.role === 'voice') continue;
      if (pct > 0 && voices.length) t.duck = Object.assign(t.duck && !t.duck.auto ? t.duck : { thresholdDb: -40, attack: 0.05, release: 0.4, hold: 0.15 }, { on: true, auto: true, sources: voices, amountDb: -pct / 100 * 24 });
      else if (t.duck && t.duck.auto) t.duck = null;
    }
    if (pct > 0 && !voices.length) App.toast('Mark a track as "Voice" (Track tab) so the music can duck under it.', true);
  };
  function mountCorner() {
    const tb = $('#toolbar'); if (!tb || $('#under')) return;
    const l = document.createElement('label'); l.className = 'ed-muted ed-under'; l.title = 'Lowers the music tracks while a Voice track is speaking (automatic ducking)';
    l.innerHTML = `Under voice <input type="range" id="under" min="0" max="100" step="5" value="0" aria-label="Under voice ducking amount"><output id="under-out">0%</output>`;
    tb.appendChild(l);
    const u = $('#under');
    u.oninput = () => { $('#under-out').textContent = u.value + '%'; };
    u.onchange = async () => { App.setUnderVoice(+u.value); App.commit({ quiet: true }); if (App.engine.playing) await App.engine.rebuild(); };
  }

  // ================================================================== left panel
  App.renderLeft = () => {    $$('#left-tabs button').forEach((b) => b.setAttribute('aria-selected', b.dataset.t === App.left));
    const body = $('#left-body');
    const fn = { import: leftImport, audio: leftAudio, voice: leftVoice, effects: leftEffects, filters: leftFilters }[App.left];
    body.innerHTML = ''; fn(body);
  };
  $('#left-tabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { App.left = b.dataset.t; App.renderLeft(); } });

  function dropZone(el) {
    el.innerHTML = `<div class="ed-drop" id="dz">Drop audio files here<br><span class="ed-muted">WAV, MP3, FLAC, OGG, M4A</span><br><br><label class="ed-btn" style="display:inline-block">Choose files<input type="file" id="file-in" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.opus,.webm" multiple hidden></label></div>`;
    const dz = $('#dz', el), fi = $('#file-in', el);
    fi.onchange = () => { App.importFiles([...fi.files]); fi.value = ''; };
    dz.ondragover = (e) => { e.preventDefault(); dz.classList.add('over'); };
    dz.ondragleave = () => dz.classList.remove('over');
    dz.ondrop = (e) => { e.preventDefault(); dz.classList.remove('over'); App.importFiles([...e.dataTransfer.files]); };
  }
  App.importFiles = async (files) => {
    files = files.filter((f) => f.type.startsWith('audio/') || /\.(wav|mp3|flac|ogg|opus|m4a|webm)$/i.test(f.name));
    if (!files.length) { App.toast('No audio files found.', true); return; }
    let i = 0;
    for (const f of files) {
      try {
        const bytes = await f.arrayBuffer(), id = C.uid('a');
        await App.addAssetFromBytes(id, f.name, 'local', bytes, { type: 'local' });
        // the first file lands on the selected track, every further file on its own new track at the same time: instant layering
        let tid = App.selTrack || App.project.tracks[0].id;
        if (i > 0 || App.project.clips.some((c) => c.trackId === tid && C.clipEnd(c) > App.engine.position() && c.start <= App.engine.position())) tid = App.act.addTrack(`Track ${App.project.tracks.length + 1}`).id;
        App.addClipFromAsset(id, tid, App.engine.position());
        App.commit(); i++;
      } catch (e) { App.toast(e.message, true); }
    }
    await App.process();
    if (App.left === 'import' || (App.left === 'audio' && ['imports'].includes(App.leftSub))) App.renderLeft();
  };

  async function leftImport(body) {
    const top = document.createElement('div'); body.appendChild(top); dropZone(top);
    const list = document.createElement('div'); body.appendChild(list);
    const items = (await S.listAssets(['local'])).map((a) => ({ src: 'asset', id: a.id, name: a.name, dur: a.duration, meta: 'Imported' }));
    renderItems(list, items, 'No imported files yet.');
  }
  const SUBS = [['songs', 'Your Songs'], ['samples', 'Samples'], ['recordings', 'Recordings'], ['voices', 'Voices'], ['imports', 'Imports']];
  async function leftAudio(body) {
    body.innerHTML = `<div class="ed-sub" role="group" aria-label="Library source">${SUBS.map(([k, l]) => `<button class="ed-pill" data-s="${k}" aria-pressed="${App.leftSub === k}">${l}</button>`).join('')}</div>
      <input class="ed-search" id="lib-q" placeholder="Search…" aria-label="Search the library"><div id="lib-list"><div class="ed-hint">Loading…</div></div>`;
    $$('.ed-pill', body).forEach((b) => b.onclick = () => { App.leftSub = b.dataset.s; App.renderLeft(); });
    const list = $('#lib-list', body);
    let items = [], empty = 'Nothing here yet.';
    try {
      if (App.leftSub === 'songs') {
        const r = await fetch('/api/library'); const tracks = r.ok ? await r.json() : [];
        items = tracks.map((t) => ({ src: 'song', id: t.id, name: t.title, dur: t.audio && t.audio.duration, meta: (t.created || '').slice(0, 10) + (t.kind ? ' · ' + t.kind : '') }));
        empty = 'No songs yet. Generate music on the Create page, or save an export to Your Songs.';
      } else if (App.leftSub === 'samples') {
        const r = await fetch('/api/samples'); const ss = r.ok ? await r.json() : [];
        items = ss.map((s) => ({ src: 'sample', id: s.id, name: s.name, dur: s.duration, meta: `${s.kind} · ${(s.created || '').slice(0, 10)}` }));
        empty = 'No samples yet. Export → "Send to Create as Sample" saves one here.';
      } else {
        const kinds = { recordings: ['recording', 'voice'], voices: ['voice'], imports: ['local'] }[App.leftSub];
        items = (await S.listAssets(kinds)).map((a) => ({ src: 'asset', id: a.id, name: a.name, dur: a.duration, meta: a.kind + ' · ' + new Date(a.created).toLocaleDateString() }));
        empty = App.leftSub === 'voices' ? 'No voices yet. Record with the microphone (Voice tab) and tick "Save as voice".' : 'Nothing here yet.';
      }
    } catch (e) { list.innerHTML = `<div class="ed-hint ed-err">${esc(e.message)}</div>`; return; }
    const draw = (q) => renderItems(list, items.filter((i) => !q || i.name.toLowerCase().includes(q.toLowerCase())), empty);
    $('#lib-q', body).oninput = (e) => draw(e.target.value); draw('');
  }
  function renderItems(list, items, empty) {
    list.innerHTML = '';
    if (!items.length) { list.innerHTML = `<div class="ed-hint">${esc(empty)}</div>`; return; }
    for (const it of items) {
      const row = document.createElement('div'); row.className = 'ed-item'; row.draggable = true;
      row.innerHTML = `<button class="ed-pv" title="Preview" aria-label="Preview ${esc(it.name)}">▶</button><div class="nm"><b title="${esc(it.name)}">${esc(it.name)}</b><span>${it.dur ? fmtDur(it.dur) + ' · ' : ''}${esc(it.meta || '')}</span></div><button class="ed-add" title="Add to the timeline at the playhead" aria-label="Add ${esc(it.name)} to the timeline">+</button>`;
      row.ondragstart = (e) => { e.dataTransfer.setData('application/x-vf-asset', JSON.stringify(it)); e.dataTransfer.effectAllowed = 'copy'; };
      $('.ed-add', row).onclick = () => App.insertLibraryItem(it, App.selTrack || App.project.tracks[0].id, App.engine.position());
      $('.ed-pv', row).onclick = (e) => App.previewItem(it, e.currentTarget);
      list.appendChild(row);
    }
  }
  let pv = null;
  App.previewItem = async (it, btn) => {
    if (pv) { pv.audio.pause(); pv.btn.textContent = '▶'; const same = pv.btn === btn; pv = null; if (same) return; }
    let url;
    if (it.src === 'song') url = `/api/library/${it.id}/audio?format=wav`;
    else if (it.src === 'sample') url = `/api/samples/${it.id}/audio`;
    else { const bytes = await S.getAssetBytes(it.id); if (!bytes) { App.toast('Audio not found.', true); return; } url = URL.createObjectURL(new Blob([bytes])); }
    const audio = new Audio(url); pv = { audio, btn }; btn.textContent = '❚❚';
    audio.onended = () => { btn.textContent = '▶'; pv = null; };
    audio.play().catch((e) => { App.toast('Preview failed: ' + e.message, true); btn.textContent = '▶'; pv = null; });
  };
  /** Resolve a library item to an asset (download/decode once) and place a clip. */
  App.insertLibraryItem = async (it, trackId, time) => {
    try {
      let id;
      if (it.src === 'asset') { id = it.id; if (!App.project.assets[id]) { const m = (await S.listAssets()).find((a) => a.id === id); App.project.assets[id] = { id, name: m.name, duration: m.duration, kind: m.kind, source: m.source }; } await App.ensureAsset(id); }
      else {
        id = `${it.src}:${it.id}`;
        if (!App.assets[id]) {
          const url = it.src === 'song' ? `/api/library/${it.id}/audio?format=wav` : `/api/samples/${it.id}/audio`;
          const r = await fetch(url); if (!r.ok) throw new Error('Could not load that audio from the library.');
          await App.addAssetFromBytes(id, it.name, it.src, await r.arrayBuffer(), { type: it.src, ref: it.id });
        }
      }
      App.addClipFromAsset(id, trackId, time); App.commit(); await App.process();
    } catch (e) { App.toast(e.message, true); }
  };

  // ------------------------------------------------------------------ microphone recording
  App.toggleRecord = async () => {
    if (App.recording) { App.recording.stop(); return; }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === 'undefined') { App.toast('Recording is not supported by this browser.', true); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
    catch (e) { App.toast('Microphone unavailable: ' + (e.name === 'NotAllowedError' ? 'permission was denied.' : e.message), true); return; }
    const rec = new MediaRecorder(stream), chunks = [], t0 = Date.now(), at = App.engine.position();
    const asVoice = !$('#rec-voice') || $('#rec-voice').checked;
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop()); clearInterval(timer); App.recording = null; App.updateToolbar(); if ($('#rec-state')) $('#rec-state').textContent = 'Processing…';
      try {
        const bytes = await new Blob(chunks, { type: rec.mimeType }).arrayBuffer(), id = C.uid('a'), name = `Recording ${new Date().toLocaleTimeString()}`;
        await App.addAssetFromBytes(id, name, asVoice ? 'voice' : 'recording', bytes, { type: 'mic' });
        let tr = App.project.tracks.find((t) => t.id === App.selTrack && t.role === 'voice') || App.project.tracks.find((t) => t.role === 'voice');
        if (!tr) tr = App.act.addTrack('Voice', 'voice');
        App.addClipFromAsset(id, tr.id, at); App.commit(); await App.process();
        App.toast('Recording added to the timeline'); if ($('#rec-state')) $('#rec-state').textContent = '';
      } catch (e) { App.toast(e.message, true); }
    };
    const timer = setInterval(() => { if ($('#rec-state')) $('#rec-state').textContent = '● ' + fmtDur((Date.now() - t0) / 1000); }, 250);
    App.recording = { stop: () => rec.stop() }; rec.start(); App.updateToolbar();
    if ($('#rec-btn')) $('#rec-btn').textContent = 'Stop recording';
  };
  function leftVoice(body) {
    body.innerHTML = `<p class="ed-muted">Record your voice straight onto a Voice track at the playhead. Then use <b>Enhance voice</b> and <b>Reduce noise</b> (Basic tab) and the <b>Under voice</b> slider to lower the music while you speak.</p>
      <div class="ed-actions"><button class="ed-btn" id="rec-btn">${App.recording ? 'Stop recording' : '🎙 Start recording'}</button><span id="rec-state" class="ed-muted"></span></div>
      <label class="ed-chk"><input type="checkbox" id="rec-voice" checked> Save as voice <small>(appears under Audio → Voices)</small></label>
      <h4 style="margin:14px 0 6px">Quick voice fixes for the selected clip</h4>
      <div class="ed-chips"><button data-q="clean">Clean voice</button><button data-q="podcast">Podcast tone</button><button data-q="quiet">Reduce noise only</button></div>`;
    $('#rec-btn', body).onclick = () => App.toggleRecord();
    $$('[data-q]', body).forEach((b) => b.onclick = () => applyPreset({ clean: { enhanceVoice: true, noiseReduce: 0.6 }, podcast: { enhanceVoice: true, normalize: true, 'comp.on': true, 'comp.threshold': -22, 'comp.ratio': 3, 'eq.on': true, 'eq.gains': [-2, 0, 1, 2, 1] }, quiet: { noiseReduce: 0.7 } }[b.dataset.q]));
  }

  // ------------------------------------------------------------------ presets (Effects / Filters tabs)
  async function applyPreset(patch) {
    const clips = App.clips().filter((c) => !C.trackOf(App.project, c).lock);
    if (!clips.length) { App.toast('Select a clip first.', true); return; }
    let heavy = false;
    for (const c of clips) for (const [k, v] of Object.entries(patch)) {
      if (HEAVY.has(k)) { C.setHeavyParam(c, k, v); heavy = true; } else C.setParam(c, k, Array.isArray(v) ? v.slice() : v);
    }
    App.commit(); if (heavy) await App.process(); else if (App.engine.playing) await App.engine.rebuild();
  }
  function chips(body, intro, list) {
    body.innerHTML = `<p class="ed-muted">${intro}</p><div class="ed-chips">${list.map(([n]) => `<button>${n}</button>`).join('')}</div>`;
    $$('button', body).forEach((b, i) => b.onclick = () => applyPreset(list[i][1]));
  }
  function leftEffects(body) {
    chips(body, 'Click a preset to apply it to the selected clips. Fine-tune in the properties panel.', [
      ['Small room', { 'reverb.on': true, 'reverb.mix': 0.2, 'reverb.decay': 0.8 }], ['Hall', { 'reverb.on': true, 'reverb.mix': 0.35, 'reverb.decay': 2.6 }],
      ['Echo', { 'delay.on': true, 'delay.time': 0.35, 'delay.feedback': 0.4, 'delay.mix': 0.35 }], ['Slapback', { 'delay.on': true, 'delay.time': 0.12, 'delay.feedback': 0.15, 'delay.mix': 0.3 }],
      ['Compress', { 'comp.on': true, 'comp.threshold': -24, 'comp.ratio': 4, 'comp.makeup': 4 }], ['Limit peaks', { 'limiter.on': true, 'limiter.ceiling': -1 }],
      ['Warm', { 'eq.on': true, 'eq.gains': [3, 2, 0, -1, -2] }], ['Bright', { 'eq.on': true, 'eq.gains': [-2, -1, 0, 3, 5] }], ['Bass boost', { 'eq.on': true, 'eq.gains': [7, 3, 0, 0, 0] }],
      ['Reset effects', { 'reverb.on': false, 'delay.on': false, 'comp.on': false, 'limiter.on': false, 'eq.on': false, 'eq.gains': [0, 0, 0, 0, 0] }]]);
  }
  function leftFilters(body) {
    chips(body, 'Filters shape the frequency range of the selected clips.', [
      ['Muffled (low-pass)', { 'filter.type': 'lowpass', 'filter.freq': 1200, 'filter.q': 0.7 }], ['Thin (high-pass)', { 'filter.type': 'highpass', 'filter.freq': 400, 'filter.q': 0.7 }],
      ['Telephone', { 'filter.type': 'bandpass', 'filter.freq': 1800, 'filter.q': 1.2 }], ['Radio', { 'filter.type': 'bandpass', 'filter.freq': 2500, 'filter.q': 0.6 }],
      ['Cut hum', { 'filter.type': 'notch', 'filter.freq': 60, 'filter.q': 8 }], ['No filter', { 'filter.type': 'none' }]]);
  }

  // ================================================================== properties panel
  App.setProps = (t) => { App.props = t; App.renderProps(); };
  $('#props-tabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) App.setProps(b.dataset.t); });
  const row = (label, path, min, max, step, unit, v, extra) => `<div class="ed-row"><label for="p-${path}">${label}</label><input id="p-${path}" type="range" data-p="${path}" min="${min}" max="${max}" step="${step}" value="${v}" ${extra || ''}><output>${fmtVal(v, unit)}</output></div>`;
  const chk = (label, path, v, hint) => `<label class="ed-chk"><input type="checkbox" data-p="${path}" ${v ? 'checked' : ''}> ${label} ${hint ? `<small>${hint}</small>` : ''}</label>`;
  const fmtVal = (v, unit) => (unit === 's' ? (+v).toFixed(2) + ' s' : unit === 'x' ? (+v).toFixed(2) + '×' : unit === 'st' ? (+v > 0 ? '+' : '') + v + ' st' : unit === 'hz' ? (+v >= 1000 ? (v / 1000).toFixed(1) + ' kHz' : Math.round(v) + ' Hz') : unit === '' ? String(v) : (+v).toFixed(1) + ' ' + unit);
  const get = (o, path) => path.split('.').reduce((a, k) => a[k], o);

  App.renderProps = () => {
    $$('#props-tabs button').forEach((b) => b.setAttribute('aria-selected', b.dataset.t === App.props));
    const body = $('#props-body'), clips = App.clips(), c = clips[0];
    if (App.props === 'track') { body.innerHTML = trackHtml(); return bindTrack(body); }
    if (!c) { body.innerHTML = '<div class="ed-hint">Select a clip on the timeline to edit it.<br><span class="ed-muted">Shift+click selects several clips; changes apply to all of them.</span></div>'; return; }
    const p = c.params, many = clips.length > 1;
    const head = `<div class="ed-muted" style="margin-bottom:6px">${many ? `${clips.length} clips selected` : `${esc(c.name)} · ${C.fmtTime(c.start)} – ${C.fmtTime(C.clipEnd(c))}`}</div>`;
    let h = head;
    if (App.props === 'basic') {
      h += row('Volume', 'gainDb', -40, 12, 0.5, 'dB', p.gainDb) + row('Fade in', 'fadeIn', 0, 10, 0.05, 's', p.fadeIn) + row('Fade out', 'fadeOut', 0, 10, 0.05, 's', p.fadeOut) + row('Pan', 'pan', -1, 1, 0.05, '', p.pan)
        + chk('Normalize loudness', 'normalize', p.normalize, 'brings the clip to a standard level')
        + chk('Enhance voice', 'enhanceVoice', p.enhanceVoice, 'clearer, present speech')
        + chk('Reduce noise', 'noiseRed', p.noiseReduce > 0, 'filters hiss and hum') + row('Noise amount', 'noiseReduce', 0, 1, 0.05, '', p.noiseReduce)
        + chk('Reverse', 'reverse', p.reverse) + chk('Mute', 'mute', p.mute) + row('Loop', 'loops', 1, 16, 1, '', p.loops, 'title="Repeat the clip this many times"')
        + `<div class="ed-actions"><button class="ed-btn" data-do="split">Split at playhead</button><button class="ed-btn" data-do="silence">Remove silence</button><button class="ed-btn" data-do="dup">Duplicate</button><button class="ed-btn" data-do="del">Delete</button></div>`;
    } else if (App.props === 'voice') {
      h += row('Pitch', 'pitch', -12, 12, 1, 'st', p.pitch) + `<p class="ed-muted">Changes the pitch without changing the speed (phase vocoder; not formant-preserving, so large shifts sound less natural).</p>
        <div class="ed-chips"><button data-set="pitch:-4">Deeper</button><button data-set="pitch:4">Higher</button><button data-set="pitch:8">Chipmunk</button><button data-set="pitch:-8">Giant</button><button data-set="pitch:0">Reset</button></div>`;
    } else if (App.props === 'speed') {
      h += row('Speed', 'speed', 0.5, 2, 0.05, 'x', p.speed) + chk('Keep pitch', 'keepPitch', p.keepPitch !== false, 'off = tape-style speed and pitch change together')
        + `<div class="ed-chips"><button data-set="speed:0.75">0.75×</button><button data-set="speed:1">1×</button><button data-set="speed:1.25">1.25×</button><button data-set="speed:1.5">1.5×</button><button data-set="speed:2">2×</button></div>`;
    } else if (App.props === 'eq') {
      const f = ['80 Hz', '250 Hz', '1 kHz', '4 kHz', '10 kHz'];
      h += chk('Equalizer', 'eq.on', p.eq.on) + f.map((n, i) => row(n, `eq.gains.${i}`, -12, 12, 0.5, 'dB', p.eq.gains[i])).join('')
        + chk('Compressor', 'comp.on', p.comp.on) + row('Threshold', 'comp.threshold', -60, 0, 1, 'dB', p.comp.threshold) + row('Ratio', 'comp.ratio', 1, 20, 0.5, ':1', p.comp.ratio) + row('Attack', 'comp.attack', 0, 0.5, 0.005, 's', p.comp.attack)
        + row('Release', 'comp.release', 0.01, 1, 0.01, 's', p.comp.release) + row('Make-up', 'comp.makeup', 0, 24, 0.5, 'dB', p.comp.makeup)
        + chk('Limiter', 'limiter.on', p.limiter.on, 'fast peak control, not look-ahead') + row('Ceiling', 'limiter.ceiling', -20, 0, 0.5, 'dB', p.limiter.ceiling);
    } else if (App.props === 'space') {
      h += chk('Reverb', 'reverb.on', p.reverb.on) + row('Mix', 'reverb.mix', 0, 1, 0.01, '', p.reverb.mix) + row('Decay', 'reverb.decay', 0.2, 6, 0.1, 's', p.reverb.decay)
        + chk('Delay / echo', 'delay.on', p.delay.on) + row('Time', 'delay.time', 0.02, 1.5, 0.01, 's', p.delay.time) + row('Feedback', 'delay.feedback', 0, 0.9, 0.01, '', p.delay.feedback) + row('Mix', 'delay.mix', 0, 1, 0.01, '', p.delay.mix);
    } else if (App.props === 'filter') {
      h += `<div class="ed-row"><label for="p-ft">Type</label><select class="ed-sel" id="p-ft" data-p="filter.type">${['none', 'lowpass', 'highpass', 'bandpass', 'notch'].map((t) => `<option ${p.filter.type === t ? 'selected' : ''}>${t}</option>`).join('')}</select><span></span></div>`
        + row('Frequency', 'filter.freq', 20, 20000, 1, 'hz', p.filter.freq) + row('Resonance (Q)', 'filter.q', 0.1, 18, 0.1, '', p.filter.q);
    }
    body.innerHTML = h;
    bindProps(body);
  };
  let rebuildT = null;
  const rebuildSoon = () => { clearTimeout(rebuildT); rebuildT = setTimeout(() => { if (App.engine.playing) App.engine.rebuild(); }, 150); };
  function bindProps(body) {
    $$('[data-p]', body).forEach((el) => {
      const path = el.dataset.p;
      const read = () => (el.type === 'checkbox' ? el.checked : el.type === 'range' ? +el.value : el.value);
      el.addEventListener('input', () => {
        if (el.type === 'checkbox' || el.tagName === 'SELECT') return;
        const out = el.parentElement.querySelector('output'); if (out) out.textContent = fmtVal(+el.value, outUnit(path));
        if (HEAVY.has(path)) return;                                  // heavy edits are applied when the slider is released
        for (const c of App.clips()) { C.setParam(c, path, +el.value); if (!App.engine.updateClip(c)) rebuildSoon(); }
      });
      el.addEventListener('change', async () => {
        const v = read();
        if (path === 'noiseRed') { for (const c of App.clips()) C.setHeavyParam(c, 'noiseReduce', v ? (c.params.noiseReduce || 0.6) || 0.6 : 0); App.commit(); await App.process(); return; }
        if (path === 'normalize' || path === 'reverse' || path === 'noiseReduce' || path === 'speed' || path === 'pitch' || path === 'loops' || path === 'keepPitch') {
          for (const c of App.clips()) C.setHeavyParam(c, path, v);
          App.commit(); await App.process(); return;
        }
        for (const c of App.clips()) { C.setParam(c, path, v); }
        const needRebuild = App.clips().some((c) => !App.engine.updateClip(c));
        App.commit({ quiet: el.type !== 'checkbox' && el.tagName !== 'SELECT' });
        if (needRebuild) rebuildSoon();
        if (el.type === 'checkbox' || el.tagName === 'SELECT') App.renderProps();
      });
    });
    $$('[data-set]', body).forEach((b) => b.onclick = async () => { const [k, v] = b.dataset.set.split(':'); for (const c of App.clips()) C.setHeavyParam(c, k, +v); App.commit(); await App.process(); });
    $$('[data-do]', body).forEach((b) => b.onclick = () => ({ split: App.act.split, silence: App.act.silence, dup: App.act.duplicate, del: App.act.delete })[b.dataset.do]());
  }
  function outUnit(path) {
    if (/gainDb|eq\.gains|threshold|makeup|ceiling/.test(path)) return 'dB';
    if (/fadeIn|fadeOut|attack|release|decay|delay\.time/.test(path)) return 's';
    if (path === 'speed') return 'x'; if (path === 'pitch') return 'st'; if (/freq/.test(path)) return 'hz'; if (/ratio/.test(path)) return ':1';
    return '';
  }

  // ------------------------------------------------------------------ track tab
  function trackHtml() {
    const t = C.byId(App.project.tracks, App.selTrack) || App.project.tracks[0];
    if (!t) return '<div class="ed-hint">No track.</div>';
    const others = App.project.tracks.filter((x) => x.id !== t.id), d = t.duck || { on: false, sources: [], amountDb: -12, thresholdDb: -40, attack: 0.05, release: 0.4 };
    const r = (label, key, min, max, step, unit, v) => `<div class="ed-row"><label>${label}</label><input type="range" data-t="${key}" min="${min}" max="${max}" step="${step}" value="${v}"><output>${fmtVal(v, unit)}</output></div>`;
    return `<div class="ed-field"><label>Name</label><input type="text" id="t-name" value="${esc(t.name)}" maxlength="40" style="background:var(--bg);border:1px solid var(--line);border-radius:8px;color:var(--text);padding:6px 10px"></div>
      <div class="ed-field"><label>Type</label><select class="ed-sel" id="t-role"><option value="music" ${t.role === 'music' ? 'selected' : ''}>Music / instrument</option><option value="voice" ${t.role === 'voice' ? 'selected' : ''}>Voice</option></select></div>
      ${r('Volume', 'volume', -40, 12, 0.5, 'dB', t.volume)}${r('Pan', 'pan', -1, 1, 0.05, '', t.pan)}
      <h4 style="margin:12px 0 4px">Ducking (lower this track while others play)</h4>
      <label class="ed-chk"><input type="checkbox" id="d-on" ${d.on ? 'checked' : ''}> Duck this track</label>
      <div class="ed-muted">Trigger tracks:</div>${others.length ? others.map((o) => `<label class="ed-chk"><input type="checkbox" data-src="${o.id}" ${d.sources.includes(o.id) ? 'checked' : ''}> ${esc(o.name)}</label>`).join('') : '<div class="ed-muted">Add another track to use as the trigger.</div>'}
      <div class="ed-row"><label>Reduction</label><input type="range" id="d-amt" min="-36" max="-1" step="1" value="${d.amountDb}"><output>${d.amountDb} dB</output></div>
      <div class="ed-row"><label>Threshold</label><input type="range" id="d-thr" min="-70" max="-10" step="1" value="${d.thresholdDb ?? -40}"><output>${d.thresholdDb ?? -40} dB</output></div>
      <div class="ed-row"><label>Attack</label><input type="range" id="d-att" min="0.005" max="0.5" step="0.005" value="${d.attack}"><output>${(+d.attack).toFixed(2)} s</output></div>
      <div class="ed-row"><label>Release</label><input type="range" id="d-rel" min="0.05" max="2" step="0.05" value="${d.release}"><output>${(+d.release).toFixed(2)} s</output></div>`;
  }
  function bindTrack(body) {
    const t = C.byId(App.project.tracks, App.selTrack) || App.project.tracks[0]; if (!t) return;
    const ensureDuck = () => { if (!t.duck) t.duck = { on: false, sources: [], amountDb: -12, thresholdDb: -40, attack: 0.05, release: 0.4, hold: 0.15 }; delete t.duck.auto; return t.duck; };
    const after = async () => { App.commit({ quiet: true }); if (App.engine.playing) await App.engine.rebuild(); };
    $('#t-name', body).onchange = (e) => { t.name = e.target.value.trim().slice(0, 40) || t.name; App.commit(); };
    $('#t-role', body).onchange = (e) => { t.role = e.target.value; App.commit(); };
    $$('[data-t]', body).forEach((el) => { el.oninput = () => { t[el.dataset.t] = +el.value; el.parentElement.querySelector('output').textContent = fmtVal(+el.value, el.dataset.t === 'volume' ? 'dB' : ''); App.engine.updateTrack(t); }; el.onchange = () => { $('.ed-head[data-id="' + t.id + '"] input') && ($('.ed-head[data-id="' + t.id + '"] input').value = t.volume); App.commit({ quiet: true }); }; });
    $('#d-on', body).onchange = (e) => { ensureDuck().on = e.target.checked; after(); };
    $$('[data-src]', body).forEach((el) => el.onchange = () => { const d = ensureDuck(); d.sources = el.checked ? [...new Set([...d.sources, el.dataset.src])] : d.sources.filter((s) => s !== el.dataset.src); after(); });
    for (const [id, key, unit] of [['d-amt', 'amountDb', 'dB'], ['d-thr', 'thresholdDb', 'dB'], ['d-att', 'attack', 's'], ['d-rel', 'release', 's']]) {
      const el = $('#' + id, body);
      el.oninput = () => { ensureDuck()[key] = +el.value; el.parentElement.querySelector('output').textContent = fmtVal(+el.value, unit); };
      el.onchange = after;
    }
  }
})();
