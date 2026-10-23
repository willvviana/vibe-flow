// Vibe Flow server: static site + JSON API in front of the local Python music engine (./engine).
// Everything runs on this machine; no cloud inference. Bound to localhost only.
const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');

const ROOT = __dirname;
const PORT = process.env.PORT || 3000;
const PY = process.env.PYTHON || path.join(ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const UPLOADS = path.join(ROOT, 'uploads');
const JOBS_DIR = path.join(ROOT, '.jobs');
fs.mkdirSync(UPLOADS, { recursive: true });
fs.mkdirSync(JOBS_DIR, { recursive: true });

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.txt': 'text/plain', '.xml': 'application/xml', '.webmanifest': 'application/manifest+json',
  '.wav': 'audio/wav', '.json': 'application/json', '.flac': 'audio/flac', '.mp3': 'audio/mpeg'
};
const PUBLIC = new Set(['sample.html', 'sample.css', 'sample/app.js', 'sample/recorder-worklet.js', 'editor.html', 'editor.css', 'editor/core.js', 'editor/dsp.js', 'editor/engine.js', 'editor/worker.js', 'editor/storage.js', 'editor/ui.js', 'editor/ui-panels.js', 'editor/ui-export.js', 'editor/selftest.js', 'vendor/lame.min.js', 'index.html', 'about.html', 'create.html', 'styles.css', 'script.js', 'studio.js', 'favicon.svg', 'og-image.png',
  'apple-touch-icon.png', 'robots.txt', 'sitemap.xml', 'site.webmanifest']);

// Never leak filesystem layout to the browser.
const scrub = m => String(m).split(ROOT).join('<app>').split(require('os').homedir()).join('~').replace(/[A-Za-z]:[\\/][^\s'"]+/g, '<path>');
const MAX_JOB_MS = 30 * 60 * 1000;          // hard stop for any engine job
const UPLOAD_TTL_MS = 24 * 3600 * 1000;
const EDITOR_MAX = 400 * 1024 * 1024;            // rendered mixes can be long and 24/32-bit
const SAMPLES = process.env.VIBEFLOW_SAMPLES || path.join(ROOT, 'userdata', 'samples');
fs.mkdirSync(SAMPLES, { recursive: true });

const send = (res, code, obj) => {
  res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(obj));
};

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = []; let n = 0;
    req.on('data', c => { n += c.length; if (n > limit) { reject(Object.assign(new Error('too large'), { code: 413 })); req.destroy(); } else chunks.push(c); });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}
const readJson = async req => { try { return JSON.parse((await readBody(req, 100_000)).toString() || '{}'); } catch { throw Object.assign(new Error('Invalid request.'), { code: 400 }); } };

// ---- run an engine command; resolves with its final "result" event -----------------
function engine(args, { onProgress, onSpawn } = {}) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(PY)) return reject(new Error('Python environment missing. Run: npm run setup'));
    const p = spawn(PY, ['-m', 'engine.cli', ...args], { cwd: ROOT, env: { ...process.env, PYTHONIOENCODING: 'utf-8', TOKENIZERS_PARALLELISM: 'false' } });
    onSpawn && onSpawn(p);
    let buf = '', result = null, error = null, stderr = '';
    p.stdout.on('data', d => {
      buf += d;
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line.startsWith('{')) continue;
        try {
          const e = JSON.parse(line);
          if (e.event === 'progress') onProgress && onProgress(e);
          else if (e.event === 'result') result = e;
          else if (e.event === 'error') error = e;
        } catch {}
      }
    });
    p.stderr.on('data', d => { stderr = (stderr + d).slice(-2000); });
    p.on('error', reject);
    p.on('close', code => {
      if (p.killedByUs) return reject(Object.assign(new Error('Cancelled.'), { cancelled: true }));
      if (result) return resolve(result);
      if (!error) console.error('[engine]', args[0], 'exit', code, scrub(stderr));
      reject(Object.assign(new Error(scrub(error ? error.message : 'The engine stopped unexpectedly. See the server console for details.')), { code: error && error.kind === 'FileNotFoundError' ? 404 : undefined }));
    });
  });
}

// ---- warm generation worker ----------------------------------------------------------
// Generation runs in one long-lived Python process so the model stays loaded between jobs (saves 6-20 s per job).
// The worker exits by itself after VIBEFLOW_IDLE_S idle seconds, releasing ~4 GB of RAM. Cancelling kills it.
let worker = null;   // { proc, current: {resolve, reject, onProgress, started}, dead }
function getWorker() {
  if (worker && !worker.dead) return worker;
  const proc = spawn(PY, ['-m', 'engine.cli', 'serve'], { cwd: ROOT, env: { ...process.env, PYTHONIOENCODING: 'utf-8', TOKENIZERS_PARALLELISM: 'false' } });
  const w = worker = { proc, current: null, dead: false };
  let buf = '', stderr = '';
  proc.stdout.on('data', d => {
    buf += d; let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
      if (!line.startsWith('{')) continue;
      let e; try { e = JSON.parse(line); } catch { continue; }
      const cur = w.current;
      if (e.event === 'idle_exit') w.dead = true;
      else if (!cur) continue;
      else if (e.event === 'progress') { cur.started = true; cur.onProgress && cur.onProgress(e); }
      else if (e.event === 'result') { w.current = null; cur.resolve(e); }
      else if (e.event === 'error') { w.current = null; cur.reject(Object.assign(new Error(scrub(e.message)), { code: e.kind === 'FileNotFoundError' ? 404 : undefined })); }
    }
  });
  proc.stderr.on('data', d => { stderr = (stderr + d).slice(-2000); });
  proc.on('error', () => {});
  proc.on('close', code => {
    w.dead = true; if (worker === w) worker = null;
    const cur = w.current; w.current = null;
    if (!cur) return;
    if (proc.killedByUs) return cur.reject(Object.assign(new Error('Cancelled.'), { cancelled: true }));
    if (!cur.started && !cur.retried) { cur.retried = true; return workerRun(cur).then(cur.resolve, cur.reject); }   // worker idled out just as we sent the job
    console.error('[worker] exit', code, scrub(stderr));
    cur.reject(new Error('The engine stopped unexpectedly. See the server console for details.'));
  });
  return w;
}
function workerRun(cur) {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(PY)) return reject(new Error('Python environment missing. Run: npm run setup'));
    Object.assign(cur, { resolve, reject });
    const w = getWorker();
    w.current = cur;
    cur.onSpawn && cur.onSpawn(w.proc);
    w.proc.stdin.write(JSON.stringify({ cmd: 'generate', job: cur.jobFile }) + '\n');
  });
}

// ---- jobs (one heavy job at a time: models are large) -----------------------------
const jobs = new Map();
let active = null;
function startJob(kind, args, extra = {}) {
  if (active && jobs.get(active).status === 'running') throw Object.assign(new Error('Another job is already running. Wait for it or cancel it.'), { code: 409 });
  const id = crypto.randomBytes(6).toString('hex');
  const job = { id, kind, status: 'running', stage: 'queued', pct: 0, message: 'Starting…', started: Date.now(), ...extra };
  jobs.set(id, job); active = id;
  for (const [k, v] of jobs) if (jobs.size > 50 && v.status !== 'running') jobs.delete(k);   // bound memory
  (kind === 'generate' ? workerRun({ jobFile: args[1], started: false,
    onSpawn: p => { job.proc = p; clearTimeout(job.timer); job.timer = setTimeout(() => { p.killedByUs = true; job.timedOut = true; p.kill(); }, MAX_JOB_MS); },
    onProgress: e => Object.assign(job, { stage: e.stage, pct: Math.round(e.pct ?? job.pct), message: e.message || job.message }) })
  : engine(args, {
    onSpawn: p => { job.proc = p; job.timer = setTimeout(() => { p.killedByUs = true; job.timedOut = true; p.kill(); }, MAX_JOB_MS); },
    onProgress: e => Object.assign(job, { stage: e.stage, pct: Math.round(e.pct ?? job.pct), message: e.message || job.message })
  })).then(r => Object.assign(job, { status: 'complete', pct: 100, stage: 'done', message: 'Done.', track: r.track }))
    .catch(e => Object.assign(job, { status: e.cancelled && !job.timedOut ? 'cancelled' : 'failed', error: job.timedOut ? 'The job took too long and was stopped.' : e.message, message: job.timedOut ? 'Timed out.' : e.message }))
    .finally(() => {
      clearTimeout(job.timer); delete job.proc;
      if (extra.tmp) fs.unlink(extra.tmp, () => {});
      if (kind === 'generate') engine(['cleanup', '30']).catch(() => {});   // drop partial files left by a killed/failed job
    });
  return job;
}
const pub = j => { const { proc, tmp, timer, timedOut, ...rest } = j; return rest; };

// ---- uploads ---------------------------------------------------------------------
const uploads = new Map();   // id -> { path, info }
async function upload(req, res) {
  const name = decodeURIComponent(req.headers['x-filename'] || 'audio').replace(/[^\w.\- ]/g, '_').slice(-80);
  const ext = path.extname(name).toLowerCase();
  if (!['.wav', '.flac', '.mp3', '.ogg'].includes(ext)) return send(res, 400, { error: 'Use a WAV, FLAC, MP3 or OGG file.' });
  if (+req.headers['content-length'] > 26 * 1024 * 1024) return send(res, 413, { error: 'File is larger than 25 MB.' });
  let body;
  try { body = await readBody(req, 26 * 1024 * 1024); } catch { return send(res, 413, { error: 'File is larger than 25 MB.' }); }
  const id = crypto.randomBytes(8).toString('hex');
  const file = path.join(UPLOADS, id + ext);
  fs.writeFileSync(file, body);
  try {
    const r = await engine(['probe', file]);
    uploads.set(id, { path: file, name });
    send(res, 200, { id, name, ...r.info });
  } catch (e) {
    fs.unlink(file, () => {});
    send(res, 400, { error: e.message });
  }
}

// ---- audio streaming with Range support -------------------------------------------
function serveFile(req, res, file) {
  const st = fs.statSync(file);
  const type = TYPES[path.extname(file)] || 'application/octet-stream';
  const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '');
  if (m) {
    const start = m[1] ? +m[1] : 0, end = m[2] ? Math.min(+m[2], st.size - 1) : st.size - 1;
    if (start > end || start >= st.size) { res.writeHead(416); return res.end(); }
    res.writeHead(206, { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${st.size}`, 'Content-Length': end - start + 1 });
    return fs.createReadStream(file, { start, end }).pipe(res);
  }
  res.writeHead(200, { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Length': st.size });
  fs.createReadStream(file).pipe(res);
}

function sweepUploads() {                       // uploads are temporary inputs, not library content
  for (const f of fs.readdirSync(UPLOADS)) {
    const p = path.join(UPLOADS, f);
    try { if (Date.now() - fs.statSync(p).mtimeMs > UPLOAD_TTL_MS) fs.unlinkSync(p); } catch {}
  }
}
sweepUploads(); setInterval(sweepUploads, 3600 * 1000).unref();
for (const f of fs.readdirSync(JOBS_DIR)) fs.unlink(path.join(JOBS_DIR, f), () => {});   // stale job files from a crash
engine(['cleanup', '60']).catch(() => {});

let statusCache = null;
const recordingsApi = require('./recordings')({ ROOT, PY, engine, readBody, readJson, send, serveFile, scrub, uploads, UPLOADS });
async function api(req, res, url) {
  const p = url.pathname, m = req.method;
  let g;
  if (p.startsWith('/api/recordings')) {
    if (await recordingsApi(req, res, url)) return;
  }

  if (m === 'GET' && p === '/api/status') {
    if (!statusCache || Date.now() - statusCache.t > 15000) statusCache = { t: Date.now(), v: await engine(['status']) };
    return send(res, 200, statusCache.v);
  }
  if (m === 'POST' && p === '/api/models/install') {
    const b = await readJson(req);
    if (!/^[\w-]{1,40}$/.test(b.model || '')) return send(res, 400, { error: 'Unknown model.' });
    statusCache = null;
    return send(res, 200, pub(startJob('install', ['install', b.model])));
  }
  if (m === 'POST' && (g = p.match(/^\/api\/models\/(repair|verify)$/))) {
    const b = await readJson(req);
    if (!/^[\w-]{1,40}$/.test(b.model || '')) return send(res, 400, { error: 'Unknown model.' });
    statusCache = null;
    return send(res, 200, pub(startJob(g[1], [g[1], b.model])));
  }
  if (m === 'POST' && p === '/api/upload') return upload(req, res);

  if (m === 'POST' && p === '/api/generate') {
    const b = await readJson(req);
    const settings = { ...(b.settings || {}) };
    let refName;
    if (settings.reference) {
      const u = uploads.get(String(settings.reference));
      if (!u) return send(res, 400, { error: 'Reference upload expired. Upload the file again.' });
      settings.reference = u.path; refName = u.name;
    }
    const tmp = path.join(JOBS_DIR, crypto.randomBytes(6).toString('hex') + '.json');
    fs.writeFileSync(tmp, JSON.stringify({ settings, model: b.model || undefined, reference_name: refName }));
    statusCache = null;
    try { return send(res, 200, pub(startJob('generate', ['generate', tmp], { tmp }))); }
    catch (e) { fs.unlink(tmp, () => {}); throw e; }
  }
  if ((g = p.match(/^\/api\/jobs\/(\w+)(\/cancel)?$/))) {
    const job = jobs.get(g[1]);
    if (!job) return send(res, 404, { error: 'Job not found.' });
    if (g[2] && m === 'POST') {
      if (job.proc) { job.proc.killedByUs = true; job.proc.kill(); }
      return send(res, 200, pub(job));
    }
    if (m === 'GET') return send(res, 200, pub(job));
  }

  if (m === 'GET' && p === '/api/library') return send(res, 200, (await engine(['library'])).tracks);
  if ((g = p.match(/^\/api\/library\/([a-z0-9-]{6,40})(?:\/(audio))?$/))) {
    const id = g[1];
    if (g[2] && m === 'GET') {
      const part = url.searchParams.get('part');       // adapt tracks also keep the original sample window and the added layers (WAV only)
      if (part) {
        if (!['source', 'layers'].includes(part)) return send(res, 400, { error: 'Bad part.' });
        const pf = path.join(process.env.VIBEFLOW_LIBRARY || path.join(ROOT, 'library'), `${id}.${part}.wav`);
        if (!fs.existsSync(pf)) return send(res, 404, { error: 'Not found.' });
        if (url.searchParams.get('download')) res.setHeader('Content-Disposition', `attachment; filename="${id}.${part}.wav"`);
        return serveFile(req, res, pf);
      }
      const fmt = url.searchParams.get('format') || 'wav';
      if (!['wav', 'flac', 'mp3'].includes(fmt)) return send(res, 400, { error: 'Bad format.' });
      const norm = url.searchParams.get('normalize') === '1';
      let r;
      try { r = await engine(['export', id, fmt, ...(norm ? ['--normalize'] : [])]); }
      catch (e) { return send(res, e.code === 404 ? 404 : 500, { error: e.code === 404 ? 'Track not found.' : e.message }); }
      const file = path.join(process.env.VIBEFLOW_LIBRARY || path.join(ROOT, 'library'), r.file);
      if (url.searchParams.get('download')) res.setHeader('Content-Disposition', `attachment; filename="${r.file}"`);
      return serveFile(req, res, file);
    }
    if (m === 'PATCH') { const b = await readJson(req); return send(res, 200, (await engine(['rename', id, String(b.title || '')])).track); }
    if (m === 'DELETE') { await engine(['delete', id]); return send(res, 200, { ok: true }); }
  }
  // ---- editor: encode / save to library / samples (all local files; WAV in, small JSON or encoded audio out) ----
  if (m === 'POST' && p === '/api/editor/encode') {
    const fmt = url.searchParams.get('format');
    if (!['flac', 'wav'].includes(fmt)) return send(res, 400, { error: 'Unsupported format.' });
    const body = await readBody(req, EDITOR_MAX);
    const inp = path.join(JOBS_DIR, crypto.randomBytes(6).toString('hex') + '.wav'), outp = inp.replace(/\.wav$/, '.' + fmt);
    fs.writeFileSync(inp, body);
    try {
      await engine(['convert', inp, fmt, outp, url.searchParams.get('bits') === '24' ? '24' : '16']);
      const data = fs.readFileSync(outp);
      res.writeHead(200, { 'Content-Type': TYPES['.' + fmt], 'Content-Length': data.length, 'Cache-Control': 'no-store' });
      return res.end(data);
    } catch (e) { return send(res, 400, { error: e.message }); }
    finally { fs.unlink(inp, () => {}); fs.unlink(outp, () => {}); }
  }
  if (m === 'POST' && p === '/api/editor/save') {
    const body = await readBody(req, EDITOR_MAX);
    const inp = path.join(JOBS_DIR, crypto.randomBytes(6).toString('hex') + '.wav');
    fs.writeFileSync(inp, body);
    try { return send(res, 200, (await engine(['import-track', inp, String(url.searchParams.get('title') || 'Edited mix').slice(0, 80)])).track); }
    catch (e) { return send(res, 400, { error: e.message }); }
    finally { fs.unlink(inp, () => {}); }
  }
  if (m === 'GET' && p === '/api/samples') {
    const out = [];
    for (const f of fs.readdirSync(SAMPLES)) if (f.endsWith('.json')) { try { out.push(JSON.parse(fs.readFileSync(path.join(SAMPLES, f), 'utf8'))); } catch {} }
    return send(res, 200, out.sort((a, b) => (b.created || '').localeCompare(a.created || '')));
  }
  if (m === 'POST' && p === '/api/samples') {
    const kind = ['full', 'vocals', 'instrumental'].includes(url.searchParams.get('kind')) ? url.searchParams.get('kind') : 'full';
    const name = String(url.searchParams.get('name') || 'Sample').replace(/[^\w.\- ()]/g, '_').slice(0, 80);
    const body = await readBody(req, EDITOR_MAX);
    const id = crypto.randomBytes(8).toString('hex'), file = path.join(SAMPLES, id + '.wav');
    fs.writeFileSync(file, body);
    try {
      const info = (await engine(['probe', file])).info;
      const meta = { id, name, kind, created: new Date().toISOString(), ...info };
      fs.writeFileSync(path.join(SAMPLES, id + '.json'), JSON.stringify(meta));
      return send(res, 200, meta);
    } catch (e) { fs.unlink(file, () => {}); return send(res, 400, { error: e.message }); }
  }
  if ((g = p.match(/^\/api\/samples\/([a-f0-9]{16})(?:\/(audio|use))?$/))) {
    const id = g[1], wavf = path.join(SAMPLES, id + '.wav'), jsonf = path.join(SAMPLES, id + '.json');
    if (!fs.existsSync(wavf)) return send(res, 404, { error: 'Sample not found.' });
    if (g[2] === 'audio' && m === 'GET') return serveFile(req, res, wavf);
    if (g[2] === 'use' && m === 'POST') {                       // make a stored sample available to Create without another upload
      const meta = JSON.parse(fs.readFileSync(jsonf, 'utf8')), uid = crypto.randomBytes(8).toString('hex'), dest = path.join(UPLOADS, uid + '.wav');
      fs.copyFileSync(wavf, dest);
      uploads.set(uid, { path: dest, name: meta.name + '.wav' });
      return send(res, 200, { id: uid, name: meta.name + '.wav', duration: meta.duration, sample_rate: meta.sample_rate, channels: meta.channels, format: meta.format, size_mb: meta.size_mb });
    }
    if (!g[2] && m === 'DELETE') { fs.unlinkSync(wavf); fs.unlink(jsonf, () => {}); return send(res, 200, { ok: true }); }
  }
  send(res, 404, { error: 'Not found' });
}

http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (url.pathname.startsWith('/api/')) {
      // The server only listens on localhost, but any web page the user visits could still try to call it
      // (CSRF) or rebind its own hostname to 127.0.0.1. Require a local Host and a same-origin Origin.
      const host = (req.headers.host || '').replace(/:\d+$/, '');
      if (!['localhost', '127.0.0.1', '[::1]'].includes(host)) return send(res, 403, { error: 'Forbidden host.' });
      const origin = req.headers.origin;
      if (origin && new URL(origin).host !== req.headers.host) return send(res, 403, { error: 'Cross-origin request blocked.' });
      return await api(req, res, url);
    }
    const name = url.pathname === '/' ? 'index.html' : ({ '/about': 'about.html', '/create': 'create.html', '/editor': 'editor.html', '/sample': 'sample.html' })[url.pathname] || url.pathname.slice(1);
    if (!PUBLIC.has(name)) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(name)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });   // always revalidate: scripts and pages change between versions
    fs.createReadStream(path.join(ROOT, name)).pipe(res);
  } catch (e) {
    if (!e.code || typeof e.code !== 'number') console.error(e);
    if (!res.headersSent) send(res, typeof e.code === 'number' ? e.code : 500, { error: scrub(e.message || 'Server error') });
  }
}).listen(PORT, '127.0.0.1', () => console.log(`Vibe Flow on http://localhost:${PORT}`));

// stop any running engine child when the server exits
process.on('exit', () => { for (const j of jobs.values()) j.proc && j.proc.kill(); if (worker) worker.proc.kill(); });
process.on('SIGINT', () => process.exit(0));
