// Sample page backend: live capture upload, live transcription, post-recording processing (stems, tags, transcript) and the library.
// Everything is a local file under userdata/recordings/<id>/ ; the heavy lifting is done by the Python engine (engine/sampling.py).
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');

module.exports = function createRecordings({ ROOT, PY, engine, readBody, readJson, send, serveFile, scrub, uploads, UPLOADS }) {
  const DIR = process.env.VIBEFLOW_RECORDINGS || path.join(ROOT, 'userdata', 'recordings');
  fs.mkdirSync(DIR, { recursive: true });
  const FILES = { full: 'full.wav', instrumental: 'instrumental.wav', voice: 'vocals.wav', lead: 'lead.wav', backing: 'backing.wav' };
  const CHUNK_MAX = 64 * 1024 * 1024;
  const MAX_PCM_BYTES = 6 * 3600 * 48000 * 2 * 4;               // refuse to grow one recording beyond 6 h of 48 kHz stereo float
  const env = { ...process.env, PYTHONIOENCODING: 'utf-8', TOKENIZERS_PARALLELISM: 'false' };

  const dirOf = id => path.join(DIR, id);
  const metaPath = id => path.join(dirOf(id), 'meta.json');
  const readMeta = id => { try { return JSON.parse(fs.readFileSync(metaPath(id), 'utf8')); } catch { return null; } };
  const writeMeta = m => { const t = metaPath(m.id) + '.tmp'; fs.writeFileSync(t, JSON.stringify(m)); fs.renameSync(t, metaPath(m.id)); };
  const live = new Map();                                       // id -> { committed, segments[], lang, busy, bytes, ... } (only while recording)

  // ---------------------------------------------------------------- live transcription worker (warm, exits when idle)
  let lw = null, nextReq = 1;
  function liveWorker() {
    if (lw && !lw.dead) return lw;
    const proc = spawn(PY, ['-m', 'engine.cli', 'sample-serve'], { cwd: ROOT, env });
    const w = lw = { proc, pending: new Map(), dead: false };
    let buf = '', err = '';
    proc.stdout.on('data', d => {
      buf += d; let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line.startsWith('{')) continue;
        let e; try { e = JSON.parse(line); } catch { continue; }
        if (e.event === 'idle_exit') w.dead = true;
        else if ((e.event === 'result' || e.event === 'error') && w.pending.has(e.id)) { const p = w.pending.get(e.id); w.pending.delete(e.id); e.event === 'result' ? p.resolve(e) : p.reject(new Error(scrub(e.message))); }
      }
    });
    proc.stderr.on('data', d => { err = (err + d).slice(-1500); });
    proc.on('error', () => {});
    proc.on('close', () => { w.dead = true; if (lw === w) lw = null; for (const p of w.pending.values()) p.reject(new Error('The transcription worker stopped.')); w.pending.clear(); });
    return w;
  }
  function liveAsk(req) {
    if (!fs.existsSync(PY)) return Promise.reject(new Error('Python environment missing. Run: npm run setup'));
    return new Promise((resolve, reject) => {
      const w = liveWorker(), id = nextReq++;
      w.pending.set(id, { resolve, reject });
      w.proc.stdin.write(JSON.stringify({ id, cmd: 'live', ...req }) + '\n');
    });
  }

  // Transcribe what has been recorded since the last committed point. Window <= 28 s. Segments that end right at the live edge may be cut
  // mid-word, so they are held back until more audio arrives (committed only when they end >= 1 s before the window end, or at the end).
  async function pump(id, final = false) {
    const L = live.get(id); if (!L || L.busy) return;
    L.busy = true;
    try {
      for (;;) {
        const total = L.bytes / (4 * L.ch * L.sr);
        const a = L.committed;
        if (!final && total - a < 6) break;
        if (final && total - a < 0.4) break;
        const b = Math.min(total, a + 28);
        L.state = L.loaded ? 'transcribing' : 'loading speech model';
        let r;
        try { r = await liveAsk({ pcm: path.join(dirOf(id), 'live.pcm'), sr: L.sr, ch: L.ch, from: a, to: b, lang: L.lang || null }); }
        catch (e) { L.error = e.message; L.committed = b; continue; }      // skip a failing window rather than stalling the whole recording
        L.loaded = true; L.error = null;
        const atEdge = b >= total - 0.01, keepAll = final && atEdge;
        let end = a;
        for (const s of r.segments) if (keepAll || s.end <= b - 1.0 || !atEdge) { L.segments.push(s); end = Math.max(end, s.end); }
        if (r.language && r.language_probability >= 0.85 && r.segments.length) L.lang = r.language;
        if (r.language && !L.language) L.language = r.language;
        L.language_probability = r.language_probability;
        L.committed = keepAll ? total : (end > a ? end : (b - a >= 20 ? b - 1 : a));
        if (L.committed <= a) break;
      }
    } finally { L.busy = false; L.state = 'listening'; }
  }

  // ---------------------------------------------------------------- system-output capture (WASAPI loopback, no screen share)
  // A "warm" capture process has the device open but records NOTHING until it is told to start, so Start begins within milliseconds.
  let warm = null;
  function spawnWarm() {
    const proc = spawn(PY, ['-m', 'engine.cli', 'sample-loopback'], { cwd: ROOT, env });
    const w = { proc, info: null, dead: false, err: '', buf: '', onEvent: null };
    w.ready = new Promise((resolve, reject) => {
      const timer = setTimeout(() => { reject(new Error('System audio capture did not become ready.')); proc.kill(); }, 20000);
      proc.stdout.on('data', d => {
        w.buf += d; let i;
        while ((i = w.buf.indexOf('\n')) >= 0) {
          const line = w.buf.slice(0, i).trim(); w.buf = w.buf.slice(i + 1);
          if (!line.startsWith('{')) continue;
          let e; try { e = JSON.parse(line); } catch { continue; }
          if (e.event === 'ready') { clearTimeout(timer); w.info = e; resolve(w); }
          else if (w.onEvent) w.onEvent(e);
          if (e.event === 'idle_exit') w.dead = true;
        }
      });
      proc.stderr.on('data', d => { w.err = (w.err + d).slice(-1500); });
      proc.on('error', e => { clearTimeout(timer); w.dead = true; reject(e); });
      proc.on('close', code => { clearTimeout(timer); w.dead = true; w.closed = true; reject(new Error(scrub(w.err) || 'System audio capture is not available.')); if (w.onClose) w.onClose(code); });
    });
    w.ready.catch(() => {});
    return w;
  }
  function ensureWarm() { if (!warm || warm.dead) warm = spawnWarm(); return warm; }
  let capCache = null;
  async function capabilities() {
    if (capCache && capCache.v.system.available && Date.now() - capCache.t < 20000 && warm && !warm.dead) return capCache.v;
    let v;
    if (process.platform !== 'win32') { try { v = { system: (await engine(['sample-loopback-info'])) }; delete v.system.event; } catch (e) { v = { system: { available: false, reason: scrub(e.message) } }; } }
    else {
      try { const w = ensureWarm(); await w.ready; v = { system: { available: true, device: w.info.device, sample_rate: w.info.sr, channels: 2 } }; }
      catch (e) { warm = null; try { v = { system: (await engine(['sample-loopback-info'])) }; delete v.system.event; } catch (e2) { v = { system: { available: false, reason: scrub(e2.message) } }; } }
    }
    capCache = { t: Date.now(), v }; return v;
  }
  async function startLoopback(id, pcmPath) {
    const w = ensureWarm(); await w.ready;                           // instant when the page already prepared it
    warm = null;                                                     // this process now belongs to the recording
    return new Promise((resolve, reject) => {
      const proc = w.proc;
      const sys = { proc, sr: w.info.sr, ch: 2, level: { pl: 0, pr: 0, clip: false, clipUntil: 0 }, wave: [], seq: 0, frames: 0, paused: false, done: null, device: w.info.device };
      let started = false;
      const timer = setTimeout(() => { if (!started) { proc.kill(); reject(new Error('System audio capture did not start.')); } }, 8000);
      w.onClose = () => { clearTimeout(timer); sys.done = 0; if (!started) reject(new Error(scrub(w.err) || 'System audio capture could not start.')); };
      w.onEvent = e => {
        if (e.event === 'started') { started = true; clearTimeout(timer); resolve(sys); }
        else if (e.event === 'level') {
          sys.frames = e.frames; sys.paused = e.paused;
          sys.level.pl = Math.max(e.pl, sys.level.pl * 0.9); sys.level.pr = Math.max(e.pr, sys.level.pr * 0.9);
          if (e.clip) sys.level.clipUntil = Date.now() + 3000;
          for (const x of e.wave) { sys.wave.push(x); sys.seq++; }
          if (sys.wave.length > 900) sys.wave.splice(0, sys.wave.length - 900);
          const L = live.get(id); if (L) { L.bytes = e.frames * 8; pump(id).catch(() => {}); }
        } else if (e.event === 'result') { sys.frames = e.frames; const L = live.get(id); if (L) L.bytes = e.frames * 8; }
      };
      proc.stdin.write('start ' + pcmPath + '\n');
    });
  }
  const stopLoopback = sys => new Promise(res => {
    if (!sys || sys.done !== null) return res();
    sys.proc.once('close', () => res());
    try { sys.proc.stdin.write('stop\n'); } catch { sys.proc.kill(); }
    setTimeout(() => { if (sys.done === null) sys.proc.kill(); }, 8000);
  });
  process.on('exit', () => { for (const L of live.values()) if (L.sys && L.sys.done === null) L.sys.proc.kill(); if (warm) warm.proc.kill(); });

  // ---------------------------------------------------------------- processing queue (one heavy job at a time)
  const queue = []; let running = null;
  function enqueue(id) { if (!queue.includes(id) && running !== id) queue.push(id); const m = readMeta(id); if (m) { m.status = 'processing'; m.stage = 'queued'; m.pct = 0; delete m.error; writeMeta(m); } next(); }
  function next() {
    if (running || !queue.length) return;
    const id = queue.shift(); running = id;
    const m = readMeta(id); if (!m) { running = null; return next(); }
    const dir = dirOf(id);
    let timer, proc, buf = '', result = null, error = null, stderr = '';
    const set = (o) => { const cur = readMeta(id); if (cur) writeMeta(Object.assign(cur, o)); };
    try { proc = spawn(PY, ['-m', 'engine.cli', 'sample-process', path.join(dir, FILES.full), dir], { cwd: ROOT, env }); }
    catch (e) { set({ status: 'failed', error: e.message }); running = null; return next(); }
    timer = setTimeout(() => { proc.killedByUs = true; proc.kill(); }, 4 * 3600 * 1000);
    let last = 0;
    proc.stdout.on('data', d => {
      buf += d; let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line.startsWith('{')) continue;
        let e; try { e = JSON.parse(line); } catch { continue; }
        if (e.event === 'progress') {
          const f = e.fraction || 0;
          const pct = e.stage === 'separate' ? f * 75 : e.stage === 'tag' ? 80 : e.stage === 'transcribe' ? 80 + 20 * f : 100;
          if (Date.now() - last > 700 || f >= 1) { last = Date.now(); set({ stage: e.stage, pct: Math.round(pct), eta_s: e.eta_s }); }
        } else if (e.event === 'result') result = e; else if (e.event === 'error') error = e;
      }
    });
    proc.stderr.on('data', d => { stderr = (stderr + d).slice(-2000); });
    proc.on('error', e => { error = { message: e.message }; });
    proc.on('close', code => {
      clearTimeout(timer); running = null;
      if (result) {
        try {
          const r = JSON.parse(fs.readFileSync(path.join(dir, 'result.json'), 'utf8'));
          const cur = readMeta(id) || m;
          const mk = (arr = []) => arr.map(t => ({ label: t.label, score: t.score, edited: false }));
          const t = r.tags || {};
          Object.assign(cur, {
            status: 'ready', stage: 'done', pct: 100, error: undefined, eta_s: undefined, stems: r.stems, vocals: r.vocals, vocal_layers: r.vocal_layers || undefined, processing_seconds: r.processing_seconds,
            tags: { genres: mk(t.genres), subgenres: mk(t.subgenres), instruments: mk(t.instruments), moods: mk(t.moods), bpm: t.bpm, key: t.key, model: t.model },
          });
          if (r.final_transcript && r.final_transcript.segments.length) {         // the vocal stem gives a cleaner pass than the live window decoding
            cur.transcript = { language: r.final_transcript.language, language_probability: r.final_transcript.language_probability, segments: r.final_transcript.segments, source: 'vocal stem (full pass)', edited: false };
          }
          writeMeta(cur);
        } catch (e) { set({ status: 'failed', error: 'Processing finished but its result could not be read.' }); }
      } else {
        const msg = proc.killedByUs ? 'Processing took too long and was stopped.' : scrub(error ? error.message : 'The engine stopped unexpectedly. See the server console.');
        if (!error) console.error('[sample-process]', code, scrub(stderr));
        set({ status: 'failed', error: msg });
      }
      next();
    });
  }

  // ---------------------------------------------------------------- finalize a capture (PCM -> WAV) and queue processing
  async function finalize(id, name) {
    const m = readMeta(id); if (!m) throw Object.assign(new Error('Recording not found.'), { code: 404 });
    const L = live.get(id);
    if (L && L.sys) { await stopLoopback(L.sys); if (process.platform === 'win32') ensureWarm(); }
    if (L) { await pump(id, true).catch(() => {}); }
    const pcm = path.join(dirOf(id), 'live.pcm');
    const bytes = fs.existsSync(pcm) ? fs.statSync(pcm).size : 0;
    if (bytes < 4 * m.channels * m.sample_rate) {                 // under one second: nothing useful to keep
      live.delete(id); fs.rmSync(dirOf(id), { recursive: true, force: true });
      throw Object.assign(new Error('The recording was shorter than one second and was discarded.'), { code: 400 });
    }
    m.status = 'finalizing'; writeMeta(m);
    const r = await engine(['sample-pcm2wav', pcm, String(m.sample_rate), String(m.channels), path.join(dirOf(id), FILES.full)]);
    fs.unlinkSync(pcm);
    Object.assign(m, { duration: r.duration, files: Object.keys(FILES) });
    if (name) m.name = String(name).slice(0, 80);
    if (L) m.transcript = { language: L.language || null, language_probability: L.language_probability || 0, segments: L.segments, source: 'live', edited: false };
    live.delete(id);
    writeMeta(m);
    enqueue(id);
    return readMeta(id);
  }

  // recover captures that were interrupted by a crash or a closed tab: keep what was recorded
  for (const id of fs.readdirSync(DIR)) {
    if (!/^[a-f0-9]{16}$/.test(id)) continue;
    const m = readMeta(id); if (!m) continue;
    if (m.status === 'recording' || m.status === 'finalizing') { m.recovered = true; writeMeta(m); finalize(id).catch(() => {}); }
    else if (m.status === 'processing') enqueue(id);
  }

  const pubMeta = m => m;

  // a tab that was closed (or a crashed browser) stops sending audio: keep what exists and process it
  setInterval(() => { for (const [id, L] of live) if (Date.now() - L.last > 90000) finalize(id).catch(() => { live.delete(id); }); }, 15000).unref();

  // ---------------------------------------------------------------- HTTP
  return async function handle(req, res, url) {
    const p = url.pathname, mth = req.method;
    let g;
    if (!p.startsWith('/api/recordings')) return false;
    if (mth === 'GET' && p === '/api/recordings/models') { send(res, 200, (await engine(['sample-status'])).models); return true; }
    if (mth === 'GET' && p === '/api/recordings/capabilities') { send(res, 200, await capabilities()); return true; }
    if (mth === 'GET' && p === '/api/recordings') {
      const out = [];
      for (const id of fs.readdirSync(DIR)) { if (!/^[a-f0-9]{16}$/.test(id)) continue; const m = readMeta(id); if (m && m.status !== 'recording') out.push(m); }
      send(res, 200, out.sort((a, b) => (b.created || '').localeCompare(a.created || ''))); return true;
    }
    if (mth === 'POST' && p === '/api/recordings' && url.searchParams.get('source') === 'system') {
      if (live.size) { send(res, 409, { error: 'A recording is already in progress.' }); return true; }
      const cap = (await capabilities()).system;
      if (!cap.available) { send(res, 400, { error: cap.reason || 'System audio capture is not available.' }); return true; }
      const id = crypto.randomBytes(8).toString('hex');
      fs.mkdirSync(dirOf(id), { recursive: true });
      const pcm = path.join(dirOf(id), 'live.pcm'); fs.writeFileSync(pcm, '');
      let sys;
      try { sys = await startLoopback(id, pcm); } catch (e) { fs.rmSync(dirOf(id), { recursive: true, force: true }); send(res, 500, { error: scrub(e.message) }); return true; }
      const name = 'Recording ' + new Date().toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
      const m = { id, name, created: new Date().toISOString(), status: 'recording', sample_rate: sys.sr, channels: 2, source: 'system', device: sys.device };
      writeMeta(m);
      live.set(id, { sr: sys.sr, ch: 2, bytes: 0, committed: 0, segments: [], lang: null, language: null, busy: false, state: 'listening', loaded: false, error: null, last: Date.now(), sys });
      send(res, 200, m); return true;
    }
    if (mth === 'POST' && p === '/api/recordings') {
      const sr = Math.round(+url.searchParams.get('sr')), ch = Math.round(+url.searchParams.get('ch'));
      if (!(sr >= 8000 && sr <= 192000) || !(ch === 1 || ch === 2)) { send(res, 400, { error: 'Bad capture format.' }); return true; }
      if ([...live.values()].length >= 1) { send(res, 409, { error: 'A recording is already in progress.' }); return true; }
      const id = crypto.randomBytes(8).toString('hex');
      fs.mkdirSync(dirOf(id), { recursive: true });
      fs.writeFileSync(path.join(dirOf(id), 'live.pcm'), '');
      const name = 'Recording ' + new Date().toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
      const m = { id, name, created: new Date().toISOString(), status: 'recording', sample_rate: sr, channels: ch, source: String(url.searchParams.get('source') || 'unknown').slice(0, 40) };
      writeMeta(m);
      live.set(id, { sr, ch, bytes: 0, committed: 0, segments: [], lang: null, language: null, busy: false, state: 'listening', loaded: false, error: null, last: Date.now() });
      send(res, 200, m); return true;
    }
    if ((g = p.match(/^\/api\/recordings\/([a-f0-9]{16})(?:\/(pcm|live|stop|discard|pause|resume|audio|peaks|use|reprocess)(?:\/(full|instrumental|voice|lead|backing))?)?$/))) {
      const id = g[1], act = g[2], ver = g[3], m = readMeta(id);
      if (!m) { send(res, 404, { error: 'Recording not found.' }); return true; }
      if (act === 'pcm' && mth === 'POST') {
        const L = live.get(id); if (!L) { send(res, 409, { error: 'This recording is not active.' }); return true; }
        const body = await readBody(req, CHUNK_MAX);
        if (body.length % (4 * L.ch)) { send(res, 400, { error: 'Chunk is not whole frames.' }); return true; }
        if (L.bytes + body.length > MAX_PCM_BYTES) { send(res, 413, { error: 'Recording limit reached (6 hours).' }); return true; }
        fs.appendFileSync(path.join(dirOf(id), 'live.pcm'), body);
        L.bytes += body.length; L.last = Date.now();
        pump(id).catch(() => {});
        send(res, 200, { seconds: L.bytes / (4 * L.ch * L.sr) }); return true;
      }
      if (act === 'live' && mth === 'GET') {
        const L = live.get(id);
        if (!L) { send(res, 200, { active: false, status: m.status, segments: (m.transcript && m.transcript.segments) || [], total: ((m.transcript && m.transcript.segments) || []).length }); return true; }
        const since = Math.max(0, +url.searchParams.get('since') || 0);
        L.last = Date.now();
        let sysOut;
        if (L.sys) { const ws = Math.max(0, +url.searchParams.get('ws') || 0), n = Math.min(L.sys.wave.length, L.sys.seq - ws); sysOut = { pl: L.sys.level.pl, pr: L.sys.level.pr, clip: Date.now() < L.sys.level.clipUntil, paused: L.sys.paused, ended: L.sys.done !== null, seq: L.sys.seq, wave: n > 0 ? L.sys.wave.slice(-n) : [] }; }
        send(res, 200, { sys: sysOut, active: true, state: L.state, error: L.error, language: L.language, language_probability: L.language_probability, seconds: L.bytes / (4 * L.ch * L.sr), transcribed_to: L.committed, total: L.segments.length, segments: L.segments.slice(since) }); return true;
      }
      if ((act === 'pause' || act === 'resume') && mth === 'POST') {
        const L = live.get(id); if (!L || !L.sys || L.sys.done !== null) { send(res, 409, { error: 'This recording is not active.' }); return true; }
        L.sys.proc.stdin.write(act + '\n'); L.sys.paused = act === 'pause'; send(res, 200, { paused: L.sys.paused }); return true;
      }
      if (act === 'stop' && mth === 'POST') {
        let b = {}; try { b = await readJson(req); } catch {}
        try { send(res, 200, await finalize(id, b.name)); } catch (e) { send(res, typeof e.code === 'number' ? e.code : 500, { error: scrub(e.message) }); }
        return true;
      }
      if (act === 'discard' && mth === 'POST') { const L0 = live.get(id); if (L0 && L0.sys) { await stopLoopback(L0.sys); if (process.platform === 'win32') ensureWarm(); } live.delete(id); fs.rmSync(dirOf(id), { recursive: true, force: true }); send(res, 200, { ok: true }); return true; }
      if (act === 'audio' && mth === 'GET') {
        const f = FILES[ver || 'full']; const file = path.join(dirOf(id), f || '');
        if (!f || !fs.existsSync(file)) { send(res, 404, { error: 'That version is not available (yet).' }); return true; }
        if (url.searchParams.get('download')) res.setHeader('Content-Disposition', `attachment; filename="${(m.name || 'recording').replace(/[^\w.\- ()]/g, '_')} - ${ver || 'full'}.wav"`);
        serveFile(req, res, file); return true;
      }
      if (act === 'peaks' && mth === 'GET') {
        const f = path.join(dirOf(id), `peaks.${ver || 'full'}.json`);
        if (!ver || !FILES[ver] || !fs.existsSync(f)) { send(res, 404, { error: 'No waveform yet.' }); return true; }
        serveFile(req, res, f); return true;
      }
      if (act === 'use' && mth === 'POST') {
        const v = url.searchParams.get('version') || 'full', f = FILES[v], src = f && path.join(dirOf(id), f);
        if (!src || !fs.existsSync(src)) { send(res, 404, { error: 'That version is not available (yet).' }); return true; }
        const uid = crypto.randomBytes(8).toString('hex'), dest = path.join(UPLOADS, uid + '.wav');
        try { await engine(['sample-reference', src, dest]); } catch (e) { send(res, 400, { error: e.message }); return true; }
        const label = { full: 'Full recording', instrumental: 'Instrumental', voice: 'Voice only', lead: 'Lead vocal', backing: 'Backing vocals' }[v];
        const name = `${m.name} - ${label}.wav`;
        uploads.set(uid, { path: dest, name });
        let info = {}; try { info = (await engine(['probe', dest])).info; } catch {}
        send(res, 200, { id: uid, name, version: v, recording: id, ...info, analysis: { tags: m.tags || null, vocals: m.vocals || null, vocal_layers: m.vocal_layers ? { stereo: !!m.vocal_layers.stereo, backing_share: m.vocal_layers.backing_share } : null } }); return true;
      }
      if (act === 'reprocess' && mth === 'POST') {
        if (!fs.existsSync(path.join(dirOf(id), FILES.full))) { send(res, 400, { error: 'The original recording file is missing.' }); return true; }
        enqueue(id); send(res, 200, readMeta(id)); return true;
      }
      if (!act && mth === 'GET') { send(res, 200, pubMeta(m)); return true; }
      if (!act && mth === 'PATCH') {
        const b = await readJson(req);
        if (typeof b.name === 'string' && b.name.trim()) m.name = b.name.trim().slice(0, 80);
        if (b.tags && typeof b.tags === 'object') {
          m.tags = m.tags || {};
          for (const k of ['genres', 'subgenres', 'instruments', 'moods']) if (Array.isArray(b.tags[k])) m.tags[k] = b.tags[k].slice(0, 20).map(t => ({ label: String(t.label || '').slice(0, 40), score: typeof t.score === 'number' ? t.score : null, edited: !!t.edited })).filter(t => t.label);
          for (const k of ['bpm', 'key']) if (b.tags[k] && typeof b.tags[k] === 'object') m.tags[k] = { value: b.tags[k].value, confidence: b.tags[k].confidence ?? null, edited: true };
        }
        if (b.transcript && Array.isArray(b.transcript.segments)) {
          m.transcript = { ...(m.transcript || {}), segments: b.transcript.segments.slice(0, 20000).map(s => ({ start: +s.start || 0, end: +s.end || 0, text: String(s.text || '').slice(0, 2000) })), edited: true };
        }
        writeMeta(m); send(res, 200, m); return true;
      }
      if (!act && mth === 'DELETE') {
        const qi = queue.indexOf(id); if (qi >= 0) queue.splice(qi, 1);
        if (running === id) { send(res, 409, { error: 'This recording is being processed. Try again when it finishes.' }); return true; }
        live.delete(id); fs.rmSync(dirOf(id), { recursive: true, force: true }); send(res, 200, { ok: true }); return true;
      }
    }
    send(res, 404, { error: 'Not found' }); return true;
  };
};
