/* Sample page: capture (getDisplayMedia / microphone -> AudioWorklet -> float32 PCM streamed to the local server), live transcript,
   processing progress, and the persistent recordings library (players, waveforms, tags, transcripts, hand-off to Create). */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const mmss = (s) => { s = Math.max(0, s || 0); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), sec = Math.floor(s % 60); return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + String(sec).padStart(2, '0'); };
  const dateFmt = (iso) => new Date(iso).toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  const langName = (c) => { try { return new Intl.DisplayNames(['en'], { type: 'language' }).of(c) || c; } catch { return c; } };

  async function api(url, opt = {}) {
    let r;
    try { r = await fetch(url, opt); } catch { throw new Error('The local engine is offline. Start it with: npm start'); }
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || 'Something went wrong.');
    return d;
  }
  const jsonOpt = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  let toastT;
  function toast(msg, err) { const t = $('#toast'); t.textContent = msg; t.className = 'toast' + (err ? ' err' : ''); t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => { t.hidden = true; }, err ? 7000 : 3500); }
  function alertBox(msg, kind) { const a = $('#cap-alert'); if (!msg) { a.hidden = true; return; } a.textContent = msg; a.className = 'alert' + (kind === 'info' ? ' info' : ''); a.hidden = false; }

  // ============================================================ transcript helpers
  const segsToText = (segs) => (segs || []).map((s) => `[${mmss(s.start)}] ${s.text}`).join('\n');
  function textToSegs(text, old) {
    const lines = String(text).split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const re = /^\[(?:(\d+):)?(\d+):(\d\d)\]\s*(.*)$/;
    const parsed = lines.map((l, i) => { const m = re.exec(l); return m ? { start: (+m[1] || 0) * 3600 + +m[2] * 60 + +m[3], text: m[4] } : { start: null, text: l }; });
    if (old && old.length === parsed.length) return parsed.map((p, i) => ({ start: old[i].start, end: old[i].end, text: p.text }));
    let last = 0;
    return parsed.map((p, i) => { const start = p.start ?? last; last = start; return { start, end: start, text: p.text }; })
      .map((s, i, a) => ({ ...s, end: a[i + 1] ? a[i + 1].start : s.start + 2 }));
  }
  const srtTime = (s) => { const ms = Math.round(s * 1000), h = Math.floor(ms / 3600000), m = Math.floor(ms % 3600000 / 60000), sec = Math.floor(ms % 60000 / 1000); return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')},${String(ms % 1000).padStart(3, '0')}`; };
  const segsToSrt = (segs) => segs.map((s, i) => `${i + 1}\n${srtTime(s.start)} --> ${srtTime(Math.max(s.end, s.start + 0.5))}\n${s.text}\n`).join('\n');
  function download(name, text, type = 'text/plain') { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: type + ';charset=utf-8' })); a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500); }
  async function copyText(text) { try { await navigator.clipboard.writeText(text); toast('Copied.'); } catch { toast('Copy failed. Select the text and press Ctrl+C.', true); } }
  const safeName = (s) => String(s || 'transcript').replace(/[^\w.\- ()]/g, '_').slice(0, 60);

  // ============================================================ capture
  const R = { sys: false, ws: 0, st: 'idle', ctx: null, stream: null, node: null, id: null, sr: 48000, frames: 0, pend: [], pendFrames: 0, queue: [], sending: false, uploadErr: null,
    wave: [], waveAcc: { mn: 0, mx: 0, n: 0 }, lvl: { pl: 0, pr: 0, hold: 0, clipUntil: 0 }, since: 0, segs: [], liveTimer: null, startedAt: 0, kind: 'display' };
  let current = null;                                    // { id, segs, edited } the recording shown in the transcript box
  const ind = $('#rec-ind');
  const setState = (st, text) => {
    R.st = st;
    ind.className = 'rec-ind' + (st === 'rec' ? ' on' : st === 'paused' ? ' paused' : '');
    $('#rec-state').textContent = text;
    document.title = (st === 'rec' ? '● REC – ' : st === 'paused' ? '❚❚ Paused – ' : '') + 'Sample – Vibe Flow';
    const live = st === 'rec' || st === 'paused';
    $('#btn-start').disabled = st !== 'idle';
    $('#btn-pause').disabled = st !== 'rec'; $('#btn-pause').hidden = st === 'paused';
    $('#btn-resume').disabled = st !== 'paused'; $('#btn-resume').hidden = st !== 'paused';
    $('#btn-stop').disabled = !live;
    $('#src').disabled = st !== 'idle';
  };

  let SYS = { available: false };
  function srcNote() {
    const k = $('#src').value, p = navigator.platform || '', win = /Win/i.test(p), mac = /Mac/i.test(p);
    $('#src-note').textContent = k === 'system'
      ? (SYS.available ? `Records everything this computer plays through its default output${SYS.device ? ' (' + SYS.device + ')' : ''}: music, browser, apps and system sounds, with no screen or tab share. Your microphone is not recorded. Capture runs only between Start and Stop.` : (SYS.reason || 'System audio capture is not available on this computer.'))
      : k === 'mic'
      ? 'Records from your microphone (the browser asks for permission). Echo cancellation, noise suppression and auto-gain are switched off so the signal stays untouched.'
      : win ? 'Windows: in the picker choose “Entire screen” and tick “Share system audio” to capture everything you hear (music, browser, apps), or pick one tab and tick “Share tab audio”. Chrome or Edge required.'
        : mac ? 'macOS: browsers can only share the audio of a single tab or Chrome window, not the whole system. To capture everything, install a virtual loopback device (e.g. BlackHole), set it as the output, and choose it under Microphone.'
          : 'Linux/other: Chrome shares tab audio; whole-system audio needs a loopback/monitor source selected as the Microphone device (PulseAudio “Monitor of …”).';
  }
  function capability() {
    const md = navigator.mediaDevices;
    if (!window.isSecureContext) return 'This page must be opened from http://localhost (or HTTPS) to use audio capture.';
    if (!md || !md.getUserMedia) return 'This browser does not support audio capture.';
    if (!window.AudioWorkletNode) return 'This browser does not support AudioWorklet, which is needed for lossless capture. Use a current Chrome, Edge or Firefox.';
    return null;
  }

  async function getStream(kind) {
    const raw = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 2, sampleRate: 48000 };
    if (kind === 'mic') return navigator.mediaDevices.getUserMedia({ audio: raw });
    if (!navigator.mediaDevices.getDisplayMedia) throw Object.assign(new Error('x'), { name: 'NO_DISPLAY' });
    try {
      return await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 2, max: 5 } }, audio: { ...raw, suppressLocalAudioPlayback: false }, systemAudio: 'include', selfBrowserSurface: 'include' });
    } catch (e) { if (e.name === 'TypeError') return navigator.mediaDevices.getDisplayMedia({ video: true, audio: true }); throw e; }
  }
  function explain(e) {
    switch (e && e.name) {
      case 'NotAllowedError': return 'Permission was denied or the share dialog was closed. Press Start again, choose a screen or tab, and tick “Share audio”.';
      case 'NotFoundError': return 'No audio input was found.';
      case 'NotReadableError': return 'The audio device is busy or blocked by another application.';
      case 'AbortError': return 'Capture was cancelled.';
      case 'NO_DISPLAY': return 'This browser cannot share screen or tab audio. Use Chrome or Edge, or switch the source to Microphone.';
      case 'NO_AUDIO': return 'No audio was shared. In the picker, choose “Entire screen” (Windows) or a tab, and tick “Share system audio” / “Share tab audio” before pressing Share.';
      default: return (e && e.message) || 'Could not start recording.';
    }
  }

  async function start(kind, injected) {
    if (R.st !== 'idle') return;
    alertBox('');
    if (kind === 'system') return startSystem();
    const cap = capability(); if (cap) { alertBox(cap); return; }
    setState('starting', 'Waiting for permission…');
    let stream = null, ctx = null;
    try {
      stream = injected || await getStream(kind);
      const at = stream.getAudioTracks()[0];
      if (!at) { stream.getTracks().forEach((t) => t.stop()); throw Object.assign(new Error('x'), { name: 'NO_AUDIO' }); }
      ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
      await ctx.audioWorklet.addModule('sample/recorder-worklet.js');
      const rec = await api(`/api/recordings?sr=${ctx.sampleRate}&ch=2&source=${encodeURIComponent(kind)}`, { method: 'POST' });
      Object.assign(R, { ctx, stream, id: rec.id, sr: ctx.sampleRate, frames: 0, pend: [], pendFrames: 0, queue: [], sending: false, uploadErr: null, wave: [], waveAcc: { mn: 0, mx: 0, n: 0 }, since: 0, segs: [], kind });
      const node = new AudioWorkletNode(ctx, 'rec-proc', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 2, channelCountMode: 'explicit', outputChannelCount: [2] });
      const mute = ctx.createGain(); mute.gain.value = 0;
      ctx.createMediaStreamSource(new MediaStream([at])).connect(node); node.connect(mute).connect(ctx.destination);
      node.port.onmessage = onNodeMessage; R.node = node;
      stream.getVideoTracks().forEach((t) => { try { t.applyConstraints({ frameRate: 1 }); } catch {} });
      at.addEventListener('ended', () => { if (R.st === 'rec' || R.st === 'paused') { toast('Sharing was stopped, so the recording was ended.'); stop(); } });
      await ctx.resume();
      node.port.postMessage({ cmd: 'state', on: true });
      current = { id: rec.id, segs: [], edited: false }; showTranscript(''); $('#tr-lang').hidden = true;
      $('#proc').hidden = true;
      setState('rec', 'Recording'); R.startedAt = performance.now();
      R.liveTimer = setInterval(pollLive, 1500); requestAnimationFrame(frame);
    } catch (e) {
      if (stream && !injected) stream.getTracks().forEach((t) => t.stop());
      if (ctx) ctx.close().catch(() => {});
      R.ctx = R.stream = R.node = null; R.id = null;
      setState('idle', 'Not recording'); alertBox(explain(e));
    }
  }

  // ---- system-output capture runs on the server (WASAPI loopback); the page only controls it and shows what it reports
  async function startSystem() {
    setState('starting', 'Starting system audio capture…');
    try {
      const rec = await api('/api/recordings?source=system', { method: 'POST' });
      Object.assign(R, { ctx: null, stream: null, node: null, id: rec.id, sr: rec.sample_rate, frames: 0, wave: [], since: 0, ws: 0, segs: [], kind: 'system', sys: true });
      current = { id: rec.id, segs: [], edited: false }; showTranscript(''); $('#tr-lang').hidden = true; $('#proc').hidden = true;
      setState('rec', 'Recording system audio'); R.startedAt = performance.now();
      R.liveTimer = setInterval(pollLive, 150); requestAnimationFrame(frame);
    } catch (e) { R.id = null; R.sys = false; setState('idle', 'Not recording'); alertBox(e.message); }
  }
  async function stopSystem() {
    const id = R.id; setState('stopping', 'Finishing…'); clearInterval(R.liveTimer);
    try {
      $('#tr-state').textContent = 'Finishing the transcript…';
      const meta = await api(`/api/recordings/${id}/stop`, jsonOpt('POST', {}));
      R.id = null; R.sys = false; setState('idle', 'Not recording');
      toast('Recording saved. Processing…'); watch(meta.id);
    } catch (e) { R.id = null; R.sys = false; setState('idle', 'Not recording'); alertBox(e.message); loadLibrary(true); }
  }

  function onNodeMessage(ev) {
    const d = ev.data;
    if (d.lvl) {
      const l = R.lvl; l.pl = Math.max(d.lvl.pl, l.pl * 0.9); l.pr = Math.max(d.lvl.pr, l.pr * 0.9);
      if (d.lvl.clip) l.clipUntil = performance.now() + 3000;
      l.last = d.lvl; return;
    }
    if (!d.pcm) return;
    const a = new Float32Array(d.pcm);
    R.frames += d.frames;
    let mn = 0, mx = 0; for (let i = 0; i < a.length; i += 2) { const v = (a[i] + a[i + 1]) * 0.5; if (v < mn) mn = v; if (v > mx) mx = v; }
    R.wave.push([mn, mx]); if (R.wave.length > 720) R.wave.shift();
    R.pend.push(a); R.pendFrames += d.frames;
    if (R.pendFrames >= R.sr) flushPending();
  }
  function flushPending() {
    if (!R.pendFrames) return;
    const out = new Float32Array(R.pendFrames * 2); let o = 0;
    for (const a of R.pend) { out.set(a, o); o += a.length; }
    R.pend = []; R.pendFrames = 0;
    R.queue.push(out.buffer); pump();
  }
  async function pump() {
    if (R.sending) return; R.sending = true;
    try {
      while (R.queue.length) {
        const body = R.queue[0];
        try {
          const r = await fetch(`/api/recordings/${R.id}/pcm`, { method: 'POST', body });
          if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.error || 'upload failed'); }
          R.queue.shift(); R.uploadErr = null;
        } catch (e) { R.uploadErr = e.message; await sleep(2000); if (R.st === 'idle' || R.st === 'done') break; }
      }
    } finally { R.sending = false; }
  }
  const pause = async () => {
    if (R.st !== 'rec') return;
    if (R.sys) { try { await api(`/api/recordings/${R.id}/pause`, { method: 'POST' }); setState('paused', 'Paused (nothing is being recorded)'); } catch (e) { toast(e.message, true); } return; }
    R.node.port.postMessage({ cmd: 'state', on: false }); R.node.port.postMessage({ cmd: 'flush' }); setState('paused', 'Paused (sharing is still active)'); };
  const resume = async () => {
    if (R.st !== 'paused') return;
    if (R.sys) { try { await api(`/api/recordings/${R.id}/resume`, { method: 'POST' }); setState('rec', 'Recording system audio'); } catch (e) { toast(e.message, true); } return; }
    R.node.port.postMessage({ cmd: 'state', on: true }); setState('rec', 'Recording'); };

  async function stop() {
    if (R.st !== 'rec' && R.st !== 'paused') return;
    if (R.sys) return stopSystem();
    const id = R.id; setState('stopping', 'Finishing…');
    clearInterval(R.liveTimer);
    try {
      R.node.port.postMessage({ cmd: 'state', on: false });
      await new Promise((res) => { const prev = R.node.port.onmessage; const t = setTimeout(res, 1500); R.node.port.onmessage = (e) => { prev(e); if (e.data.flushed) { clearTimeout(t); res(); } }; R.node.port.postMessage({ cmd: 'flush' }); });
      flushPending();
      for (let i = 0; i < 300 && (R.queue.length || R.sending); i++) await sleep(200);      // let the last audio reach the server
      if (R.queue.length) throw new Error('Some audio could not be sent to the local server. Check that it is still running.');
      R.stream.getTracks().forEach((t) => t.stop()); R.node.disconnect(); await R.ctx.close().catch(() => {});
      R.ctx = R.node = R.stream = null;
      $('#tr-state').textContent = 'Finishing the transcript…';
      const meta = await api(`/api/recordings/${id}/stop`, jsonOpt('POST', {}));
      R.id = null; setState('idle', 'Not recording');
      toast('Recording saved. Processing…');
      watch(meta.id);
    } catch (e) {
      if (R.stream) R.stream.getTracks().forEach((t) => t.stop());
      if (R.ctx) R.ctx.close().catch(() => {});
      R.ctx = R.node = R.stream = null; R.id = null; setState('idle', 'Not recording');
      alertBox(e.message); loadLibrary(true);
    }
  }

  // ---- live waveform + meters
  const wave = $('#live-wave');
  function drawWave(live) {
    const dpr = window.devicePixelRatio || 1, w = wave.clientWidth, h = wave.clientHeight;
    if (wave.width !== Math.round(w * dpr)) { wave.width = Math.round(w * dpr); wave.height = Math.round(h * dpr); }
    const c = wave.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
    c.fillStyle = 'rgba(255,255,255,.07)'; c.fillRect(0, h / 2, w, 1);
    if (!live || !R.wave.length) return;
    const n = Math.floor(w / 3), data = R.wave.slice(-n), g = c.createLinearGradient(0, 0, w, 0); g.addColorStop(0, '#8b5cf6'); g.addColorStop(1, '#22d3ee'); c.fillStyle = g;
    data.forEach(([mn, mx], i) => { const x = w - (data.length - i) * 3, top = h / 2 - Math.min(1, mx * 1.3) * h / 2, bot = h / 2 - Math.max(-1, mn * 1.3) * h / 2; c.fillRect(x, top, 2, Math.max(1, bot - top)); });
  }
  const dbPct = (v) => Math.max(0, Math.min(100, (20 * Math.log10(Math.max(v, 1e-5)) + 60) / 60 * 100));
  function frame() {
    const live = R.st === 'rec' || R.st === 'paused';
    const l = R.lvl;
    if (!live) { l.pl *= 0.8; l.pr *= 0.8; }
    $('#m-l').style.width = (100 - dbPct(l.pl)) + '%'; $('#m-r').style.width = (100 - dbPct(l.pr)) + '%';
    l.hold = Math.max(dbPct(Math.max(l.pl, l.pr)), l.hold - 0.6);
    $('#m-lp').style.left = Math.max(0, dbPct(l.pl) - 1) + '%'; $('#m-rp').style.left = Math.max(0, dbPct(l.pr) - 1) + '%';
    const pk = Math.max(l.pl, l.pr); $('#m-db').textContent = pk > 1e-5 ? (20 * Math.log10(pk)).toFixed(1) + ' dB' : '−∞ dB';
    $('#clip').hidden = performance.now() > l.clipUntil;
    $('#rec-time').textContent = mmss(R.frames / R.sr);
    if (R.uploadErr && live) $('#rec-state').textContent = 'Recording — cannot reach the local server, retrying…';
    drawWave(live);
    if (live || l.pl > 0.001 || l.pr > 0.001) requestAnimationFrame(frame); else { $('#m-l').style.width = $('#m-r').style.width = '100%'; $('#m-db').textContent = '−∞ dB'; drawWave(false); }
  }

  // ---- live transcript
  const box = $('#tr-box');
  function showTranscript(text) { box.value = text; box.scrollTop = box.scrollHeight; }
  async function pollLive() {
    if (!R.id) return;
    try {
      const d = await api(`/api/recordings/${R.id}/live?since=${R.since}&ws=${R.ws}`);
      if (!d.active) return;
      if (d.sys) {
        const y = d.sys; R.lvl.pl = y.pl; R.lvl.pr = y.pr; if (y.clip) R.lvl.clipUntil = performance.now() + 500; R.ws = y.seq;
        for (const w of y.wave) R.wave.push(w); if (R.wave.length > 720) R.wave.splice(0, R.wave.length - 720);
        R.frames = d.seconds * R.sr; $('#rec-time').textContent = mmss(d.seconds);
        if (y.ended && (R.st === 'rec' || R.st === 'paused')) { toast('System audio capture ended unexpectedly. Stopping.', true); stop(); return; }
      }
      if (d.segments.length) { R.segs.push(...d.segments); R.since = d.total; current.segs = R.segs; const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 30; if (box.readOnly) { showTranscript(segsToText(R.segs)); if (!stick) {} } }
      if (d.language) { const c = $('#tr-lang'); c.hidden = false; c.textContent = `${langName(d.language)}${d.language_probability ? ' · ' + Math.round(d.language_probability * 100) + '%' : ''}`; }
      $('#tr-state').textContent = d.error ? 'A segment could not be transcribed' : d.state === 'listening' ? `Transcribed up to ${mmss(d.transcribed_to)} of ${mmss(d.seconds)}` : d.state[0].toUpperCase() + d.state.slice(1) + '…';
    } catch { /* the server may be busy; try again on the next tick */ }
  }

  // ---- processing progress (after Stop)
  const STAGES = { queued: 'Waiting to start…', separate: 'Separating the instrumental and the voice', tag: 'Analysing genre, instruments, mood, tempo and key', transcribe: 'Transcribing the isolated voice', done: 'Finishing' };
  const watchers = new Set();
  async function watch(id) {
    if (watchers.has(id)) return; watchers.add(id);
    const p = $('#proc'); p.hidden = false;
    try {
      for (;;) {
        const m = await api(`/api/recordings/${id}`).catch(() => null);
        if (!m) { await sleep(2000); continue; }
        if (m.status === 'processing' || m.status === 'finalizing') {
          $('#proc-title').textContent = STAGES[m.stage] || 'Processing';
          $('#proc-pct').textContent = (m.pct || 0) + '%' + (m.eta_s ? ` · about ${m.eta_s > 90 ? Math.round(m.eta_s / 60) + ' min' : Math.round(m.eta_s) + ' s'} left` : '');
          $('#proc-bar').style.width = (m.pct || 0) + '%'; p.querySelector('[role=progressbar]').setAttribute('aria-valuenow', m.pct || 0);
          $('#proc-note').textContent = 'You can keep using the page, start another recording, or leave: processing continues on this computer and the result is saved.';
          if (current && current.id === id) { const segs = (m.transcript && m.transcript.segments) || []; if (segs.length && box.readOnly && R.st !== 'rec') showTranscript(segsToText(segs)); }
        } else {
          if (m.status === 'ready') {
            p.hidden = true; toast('Ready: full, instrumental and voice versions saved.');
            if (current && current.id === id && m.transcript) { current.segs = m.transcript.segments; showTranscript(segsToText(m.transcript.segments)); if (m.transcript.language) { const c = $('#tr-lang'); c.hidden = false; c.textContent = `${langName(m.transcript.language)}`; } $('#tr-state').textContent = m.transcript.segments.length ? `Final transcript (${m.transcript.source})` : 'No speech or singing was detected'; }
          } else { $('#proc-title').textContent = 'Processing failed'; $('#proc-pct').textContent = ''; $('#proc-note').textContent = m.error || 'Unknown error. The original recording is safe; use Retry in the library.'; }
          await loadLibrary(true); break;
        }
        if (Date.now() % 3 === 0) loadLibrary(false);
        await sleep(1500);
      }
    } finally { watchers.delete(id); }
  }

  // ---- transcript box actions
  const trEdit = $('#tr-edit');
  trEdit.addEventListener('click', async () => {
    if (R.st === 'rec' || R.st === 'paused') { toast('Editing is available when the recording is stopped.'); return; }
    if (box.readOnly) { box.readOnly = false; trEdit.textContent = 'Save changes'; box.focus(); return; }
    box.readOnly = true; trEdit.textContent = 'Edit';
    if (!current) return;
    const segs = textToSegs(box.value, current.segs); current.segs = segs;
    if (current.id) { try { await api(`/api/recordings/${current.id}`, jsonOpt('PATCH', { transcript: { segments: segs } })); toast('Transcript saved.'); loadLibrary(true); } catch (e) { toast(e.message, true); } }
  });
  $('#tr-copy').addEventListener('click', () => copyText(box.value));
  $('#tr-txt').addEventListener('click', () => box.value ? download(safeName('transcript') + '.txt', box.value) : toast('Nothing to download yet.', true));
  $('#tr-srt').addEventListener('click', () => { const s = textToSegs(box.value, current && current.segs); s.length ? download('transcript.srt', segsToSrt(s)) : toast('Nothing to download yet.', true); });

  // ============================================================ library
  const VERS = [['full', 'Full recording', 'original, unmodified'], ['instrumental', 'Instrumental', 'vocals removed'], ['voice', 'Voice only', 'speech and singing']];
  const cards = new Map();                                // id -> { el, sig, audios }
  let playing = null;                                     // { id, ver, audio }
  const sigOf = (m) => JSON.stringify([m.status, m.stage, m.name, m.duration, m.tags, m.transcript && m.transcript.edited, m.transcript && m.transcript.segments.length, m.error]);

  async function loadLibrary(force) {
    let list; try { list = await api('/api/recordings'); } catch (e) { $('#lib').innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
    $('#lib-count').textContent = list.length ? `${list.length} saved` : '';
    const lib = $('#lib');
    if (!list.length) { cards.forEach((c) => disposeCard(c)); cards.clear(); lib.innerHTML = '<div class="empty">No recordings yet. Press Start recording above, play something, and press Stop.</div>'; return; }
    $$('.empty, :scope > p', lib).forEach((n) => n.remove());
    const ids = new Set(list.map((m) => m.id));
    for (const [id, c] of cards) if (!ids.has(id)) { disposeCard(c); c.el.remove(); cards.delete(id); }
    let prev = null;
    for (const m of list) {
      const sig = sigOf(m); let c = cards.get(m.id);
      const focused = c && c.el.contains(document.activeElement) && /INPUT|TEXTAREA/.test(document.activeElement.tagName);
      if (!c || (c.sig !== sig && !focused && !(playing && playing.id === m.id && m.status === 'ready' && c.m.status === 'ready' && !force))) {
        const el = buildCard(m);
        if (c) { disposeCard(c); c.el.replaceWith(el.el); } else if (prev) prev.after(el.el); else lib.prepend(el.el);
        c = el; cards.set(m.id, c);
      }
      c.m = m; prev = c.el;
      if ((m.status === 'processing' || m.status === 'finalizing') && !watchers.has(m.id)) watch(m.id);
    }
  }
  function disposeCard(c) { for (const a of Object.values(c.audios || {})) { a.pause(); a.removeAttribute('src'); a.load(); } if (playing && c.m && playing.id === c.m.id) playing = null; c.ro && c.ro.disconnect(); }

  const patch = async (m, body) => { const r = await api(`/api/recordings/${m.id}`, jsonOpt('PATCH', body)); Object.assign(m, r); return r; };

  function buildCard(m) {
    const el = document.createElement('article'); el.className = 'rc'; el.dataset.id = m.id;
    const ready = m.status === 'ready', hasFull = !!(m.files && m.files.includes('full'));
    const stcls = m.status === 'ready' ? 'ready' : m.status === 'failed' ? 'failed' : 'processing';
    const vocalsLine = m.vocals ? (m.vocals.present ? `Vocals detected${m.vocals.language ? ' · ' + esc(langName(m.vocals.language)) : ''}` : 'No vocals detected') : '';
    const stems = m.stems ? `Separation: Demucs ${esc(m.stems.model)} on ${esc(m.stems.device)} in ${esc(m.stems.seconds)} s. Instrumental + voice add back to the original within ${esc(m.stems.reconstruction_residual_db)} dB (lower is closer). Some vocal remnants can remain in the instrumental and drum bleed in the voice.` : '';
    el.innerHTML = `
      <div class="rc-head">
        <div>
          <input class="rc-name" value="${esc(m.name)}" maxlength="80" aria-label="Recording name">
          <div class="rc-sub">${esc(dateFmt(m.created))} · ${m.duration ? mmss(m.duration) : '—'} · ${m.source === 'mic' ? 'Microphone' : m.source === 'display' ? 'Computer audio' : esc(m.source || '')} · <span class="st ${stcls}">${ready ? 'Ready' : m.status === 'failed' ? 'Failed' : 'Processing ' + (m.pct || 0) + '%'}</span>${m.recovered ? ' · recovered after an interruption' : ''}</div>
        </div>
        <div class="rc-actions">
          ${ready || m.status === 'failed' ? '<button class="btn btn-ghost sm" data-a="reprocess" title="Run separation, analysis and transcription again">Reprocess</button>' : ''}
          <button class="btn btn-ghost sm" data-a="delete">Delete</button>
        </div>
      </div>
      ${m.status === 'failed' ? `<div class="alert">${esc(m.error || 'Processing failed.')} The original recording is safe.</div>` : ''}
      <div class="spcmp" ${ready ? '' : 'hidden'}>Compare, same position:
        <div class="seg" role="group" aria-label="Switch version while playing">${VERS.map(([v, l]) => `<button data-sw="${v}" aria-pressed="false">${l}</button>`).join('')}</div>
      </div>
      <div class="vers">${VERS.map(([v, l, sub]) => {
        const avail = v === 'full' ? hasFull : ready;
        return `<div class="ver" data-v="${v}">
          <button class="pl" data-a="play" aria-label="Play ${l}" ${avail ? '' : 'disabled'}>▶</button>
          <div class="vn">${l}<small>${sub}</small></div>
          <canvas data-wave aria-label="${l} waveform. Click to seek."></canvas>
          <div class="vt" data-t>0:00 / ${m.duration ? mmss(m.duration) : '—'}</div>
          <div class="va">
            <a class="btn btn-ghost" data-a="dl" ${avail ? `href="/api/recordings/${m.id}/audio/${v}?download=1" download` : 'aria-disabled="true" style="opacity:.4;pointer-events:none"'}>Download</a>
            <button class="btn btn-primary" data-a="use" ${avail ? '' : 'disabled'}>Use as sample</button>
          </div></div>`;
      }).join('')}</div>
      ${stems ? `<p class="stems-note">${stems}</p>` : ''}
      ${m.tags ? tagsHtml(m) : (ready ? '' : '<p class="note">Genre, instruments, mood, tempo and key appear when processing finishes.</p>')}
      ${vocalsLine ? `<div class="row"><span class="chip">${vocalsLine}</span></div>` : ''}
      <details class="tx"><summary>Transcription${m.transcript && m.transcript.language ? ' · ' + esc(langName(m.transcript.language)) : ''}${m.transcript && m.transcript.edited ? ' (edited)' : ''}</summary>
        <textarea class="tr-box" rows="7" readonly aria-label="Saved transcription">${esc(segsToText(m.transcript && m.transcript.segments))}</textarea>
        <div class="row"><button class="btn btn-ghost sm" data-a="tedit">Edit</button><button class="btn btn-ghost sm" data-a="tcopy">Copy</button><button class="btn btn-ghost sm" data-a="ttxt">.txt</button><button class="btn btn-ghost sm" data-a="tsrt">.srt</button>
          <span class="muted">${m.transcript && m.transcript.source ? 'Source: ' + esc(m.transcript.source) : ''}${m.transcript && !m.transcript.segments.length ? 'No speech or singing was detected.' : ''}</span></div>
      </details>`;
    const c = { el, m, sig: sigOf(m), audios: {}, peaks: {} };
    wireCard(c);
    return c;
  }

  function tagsHtml(m) {
    const t = m.tags, cat = (title, key) => `<div class="tg"><h4>${title}</h4><div class="tl" data-cat="${key}">${(t[key] || []).map((x, i) => `<span class="tag">${esc(x.label)}${typeof x.score === 'number' && !x.edited ? ` <em>${Math.round(x.score * 100)}%</em>` : ''}<button data-rm="${key}:${i}" aria-label="Remove ${esc(x.label)}">×</button></span>`).join('')}<input class="inp tag-add" data-add="${key}" placeholder="+ add" aria-label="Add to ${title}" maxlength="40"></div></div>`;
    return `<div class="sptags">${cat('Genres', 'genres')}${cat('Subgenres', 'subgenres')}${cat('Instruments', 'instruments')}${cat('Mood', 'moods')}
      <div class="tg"><h4>Tempo &amp; key</h4><div class="kv"><label>BPM<input class="inp" data-kv="bpm" type="number" min="30" max="300" step="0.1" value="${t.bpm ? esc(t.bpm.value) : ''}"></label><label>Key<input class="inp" data-kv="key" value="${t.key ? esc(t.key.value) : ''}" maxlength="12"></label></div>
      <p class="stems-note">${t.bpm && t.bpm.confidence != null && !t.bpm.edited ? `Tempo confidence ${Math.round(t.bpm.confidence * 100)}%` : ''}${t.key && t.key.confidence != null && !t.key.edited ? ` · key confidence ${Math.round(t.key.confidence * 100)}%` : ''}</p></div></div>
      <p class="stems-note">Tags come from a local zero-shot audio model (CLAP). Percentages are relative rankings among the candidates, not calibrated probabilities. Edit freely: your tags are what Create uses.</p>`;
  }

  function wireCard(c) {
    const { el, m } = c;
    const nameIn = $('.rc-name', el);
    const rename = async () => { const v = nameIn.value.trim(); if (!v || v === m.name) { nameIn.value = m.name; return; } try { await patch(m, { name: v }); c.sig = sigOf(m); toast('Renamed.'); } catch (e) { toast(e.message, true); nameIn.value = m.name; } };
    nameIn.addEventListener('blur', rename); nameIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') nameIn.blur(); if (e.key === 'Escape') { nameIn.value = m.name; nameIn.blur(); } });

    // players + waveforms
    for (const row of $$('.ver', el)) {
      const v = row.dataset.v, cv = $('canvas', row), btn = $('[data-a=play]', row), tt = $('[data-t]', row);
      const a = new Audio(); a.preload = 'none'; a.dataset.v = v; c.audios[v] = a;
      const avail = v === 'full' ? (m.files || []).includes('full') : m.status === 'ready';
      if (avail) a.src = `/api/recordings/${m.id}/audio/${v}`;
      const draw = () => drawPeaks(cv, c.peaks[v], a.duration && isFinite(a.duration) ? a.currentTime / a.duration : (m.duration ? a.currentTime / m.duration : 0), v);
      c.draws = c.draws || {}; c.draws[v] = draw;
      const dur = () => (isFinite(a.duration) && a.duration > 0 ? a.duration : m.duration || 0);
      a.addEventListener('timeupdate', () => { tt.textContent = `${mmss(a.currentTime)} / ${mmss(dur())}`; });
      a.addEventListener('play', () => { row.classList.add('playing'); btn.textContent = '❚❚'; btn.setAttribute('aria-label', 'Pause ' + v); markSwitch(c); tick(c, v); });
      a.addEventListener('pause', () => { row.classList.remove('playing'); btn.textContent = '▶'; btn.setAttribute('aria-label', 'Play ' + v); draw(); markSwitch(c); });
      a.addEventListener('ended', () => { a.currentTime = 0; draw(); });
      a.addEventListener('error', () => { if (a.src && avail) toast('This version could not be played.', true); });
      btn.addEventListener('click', () => togglePlay(c, v));
      const seek = (e) => { const r = cv.getBoundingClientRect(), f = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); a.currentTime = f * dur(); draw(); tt.textContent = `${mmss(a.currentTime)} / ${mmss(dur())}`; };
      cv.addEventListener('pointerdown', (e) => { if (avail) seek(e); });
      if (avail) api(`/api/recordings/${m.id}/peaks/${v}`).then((p) => { c.peaks[v] = p; draw(); }).catch(() => { c.peaks[v] = null; draw(); });
      draw();
    }
    c.ro = new ResizeObserver(() => { for (const f of Object.values(c.draws || {})) f(); }); c.ro.observe(el);

    el.addEventListener('click', async (e) => {
      const b = e.target.closest('[data-a],[data-sw],[data-rm]'); if (!b) return;
      try {
        if (b.dataset.sw) { switchVersion(c, b.dataset.sw); return; }
        if (b.dataset.rm) { const [k, i] = b.dataset.rm.split(':'); const tags = { [k]: m.tags[k].filter((_, j) => j !== +i) }; await patch(m, { tags }); rerender(c); return; }
        const a = b.dataset.a;
        if (a === 'use') { location.href = `create.html?rec=${m.id}&v=${b.closest('.ver').dataset.v}`; }
        else if (a === 'delete') { if (!confirm(`Delete “${m.name}” and all three audio versions? This cannot be undone.`)) return; await api(`/api/recordings/${m.id}`, { method: 'DELETE' }); if (current && current.id === m.id) { current = null; showTranscript(''); } toast('Deleted.'); loadLibrary(true); }
        else if (a === 'reprocess') { await api(`/api/recordings/${m.id}/reprocess`, { method: 'POST' }); loadLibrary(true); }
        else if (a === 'tedit') {
          const ta = $('.tr-box', el);
          if (ta.readOnly) { ta.readOnly = false; b.textContent = 'Save changes'; ta.focus(); }
          else { ta.readOnly = true; b.textContent = 'Edit'; await patch(m, { transcript: { segments: textToSegs(ta.value, m.transcript && m.transcript.segments) } }); c.sig = sigOf(m); toast('Transcript saved.'); }
        }
        else if (a === 'tcopy') copyText($('.tr-box', el).value);
        else if (a === 'ttxt') download(safeName(m.name) + '.txt', $('.tr-box', el).value);
        else if (a === 'tsrt') download(safeName(m.name) + '.srt', segsToSrt(textToSegs($('.tr-box', el).value, m.transcript && m.transcript.segments)));
      } catch (err) { toast(err.message, true); }
    });
    el.addEventListener('keydown', async (e) => {
      const inp = e.target.closest('[data-add]'); if (!inp || e.key !== 'Enter') return;
      const v = inp.value.trim(); if (!v) return; const k = inp.dataset.add;
      try { await patch(m, { tags: { [k]: [...(m.tags[k] || []), { label: v, score: null, edited: true }] } }); rerender(c); } catch (err) { toast(err.message, true); }
    });
    el.addEventListener('change', async (e) => {
      const kv = e.target.closest('[data-kv]'); if (!kv) return;
      try {
        const val = kv.dataset.kv === 'bpm' ? +kv.value : kv.value.trim();
        if (kv.dataset.kv === 'bpm' && !(val >= 30 && val <= 300)) { toast('BPM must be between 30 and 300.', true); kv.value = m.tags.bpm ? m.tags.bpm.value : ''; return; }
        await patch(m, { tags: { [kv.dataset.kv]: { value: val, confidence: null } } }); c.sig = sigOf(m); toast('Saved.');
      } catch (err) { toast(err.message, true); }
    });
  }
  function rerender(c) {                                   // rebuild the tag area only, keep players untouched
    const wrap = document.createElement('div'); wrap.innerHTML = tagsHtml(c.m);
    const old = $('.sptags', c.el); if (!old) return;
    const oldNote = old.nextElementSibling; if (oldNote && oldNote.classList.contains('stems-note')) oldNote.remove();
    old.replaceWith(...wrap.children); c.sig = sigOf(c.m);
  }

  function drawPeaks(cv, pk, frac, ver) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight; if (!w) return;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
    c.fillStyle = 'rgba(255,255,255,.08)'; c.fillRect(0, h / 2, w, 1);
    if (!pk) return;
    const col = ver === 'instrumental' ? ['#22d3ee', '#0e7490'] : ver === 'voice' ? ['#f472b6', '#9d174d'] : ['#8b5cf6', '#5b21b6'];
    const N = pk.bins, px = Math.floor(w / 2);
    const amp = Math.max(0.05, ...pk.max, ...pk.min.map(Math.abs));
    for (let x = 0; x < px; x++) {
      const a = Math.floor(x / px * N), b = Math.max(a + 1, Math.floor((x + 1) / px * N));
      let mn = 0, mx = 0; for (let i = a; i < b && i < N; i++) { if (pk.min[i] < mn) mn = pk.min[i]; if (pk.max[i] > mx) mx = pk.max[i]; }
      c.fillStyle = x / px <= frac ? col[0] : col[1];
      const top = h / 2 - mx / amp * (h / 2 - 2), bot = h / 2 - mn / amp * (h / 2 - 2);
      c.fillRect(x * 2, top, 1.4, Math.max(1, bot - top));
    }
    c.fillStyle = '#fff'; c.fillRect(Math.round(frac * w), 0, 1.5, h);
  }
  function tick(c, v) { const a = c.audios[v]; if (a.paused) return; c.draws[v](); requestAnimationFrame(() => tick(c, v)); }
  function togglePlay(c, v) {
    const a = c.audios[v];
    if (!a.paused) { a.pause(); return; }
    stopOthers(c.m.id, v); playing = { id: c.m.id, ver: v, audio: a };
    a.play().catch((e) => toast('Playback failed: ' + e.message, true));
  }
  function stopOthers(id, ver) { for (const [cid, c] of cards) for (const [v, a] of Object.entries(c.audios)) if (!(cid === id && v === ver) && !a.paused) a.pause(); }
  function switchVersion(c, v) {                           // A/B: continue at the same position in another version
    const cur = Object.entries(c.audios).find(([, a]) => !a.paused) || Object.entries(c.audios).find(([k, a]) => a.currentTime > 0 && k !== v);
    const t = cur ? cur[1].currentTime : 0, was = cur && !cur[1].paused;
    for (const a of Object.values(c.audios)) a.pause();
    const a = c.audios[v]; if (!a.src) return;
    const go = () => { a.currentTime = t; if (was) { stopOthers(c.m.id, v); playing = { id: c.m.id, ver: v, audio: a }; a.play().catch(() => {}); } else c.draws[v](); };
    a.readyState >= 1 ? go() : (a.addEventListener('loadedmetadata', go, { once: true }), a.load());
  }
  function markSwitch(c) { const on = Object.entries(c.audios).find(([, a]) => !a.paused); $$('[data-sw]', c.el).forEach((b) => b.setAttribute('aria-pressed', on && on[0] === b.dataset.sw ? 'true' : 'false')); }

  // ============================================================ init
  $('#src').addEventListener('change', srcNote); srcNote();
  $('#btn-start').addEventListener('click', () => start($('#src').value));
  $('#btn-pause').addEventListener('click', pause); $('#btn-resume').addEventListener('click', resume); $('#btn-stop').addEventListener('click', stop);
  window.addEventListener('beforeunload', (e) => { if (R.st === 'rec' || R.st === 'paused') { e.preventDefault(); e.returnValue = ''; } });
  api('/api/recordings/capabilities').then((c) => {
    SYS = c.system || { available: false };
    const opt = $('#src option[value=system]');
    if (!SYS.available) { opt.disabled = true; opt.textContent += ' (unavailable)'; $('#src').value = 'display'; } else $('#src').value = 'system';
    srcNote();
    if (!SYS.available) { const m = capability(); if (m) alertBox(m); }
  }).catch(() => { $('#src option[value=system]').disabled = true; $('#src').value = 'display'; srcNote(); });
  api('/api/recordings/models').then((m) => {
    const missing = Object.entries(m).filter(([, ok]) => !ok).map(([k]) => ({ stems: 'stem separation (Demucs)', transcription: 'transcription (Whisper)', tagging: 'audio tagging (CLAP)' }[k]));
    if (missing.length) alertBox(`Missing local models: ${missing.join(', ')}. Install them with: npm run install-sampling`, 'info');
  }).catch((e) => alertBox(e.message));
  setState('idle', 'Not recording'); drawWave(false); loadLibrary(true);
  setInterval(() => { if ([...cards.values()].some((c) => c.m && (c.m.status === 'processing' || c.m.status === 'finalizing')) && !watchers.size) loadLibrary(false); }, 4000);
  window.addEventListener('resize', () => drawWave(R.st === 'rec'));
  window.SamplePage = { startWithStream: (stream, label) => start(label || 'display', stream), state: () => R.st, stop, pause, resume, loadLibrary };
})();
