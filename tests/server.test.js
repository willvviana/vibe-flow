// UI-to-backend integration: real HTTP server + real Python engine + real model.
const test = require('node:test');
const assert = require('node:assert');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const PORT = 3999, BASE = `http://127.0.0.1:${PORT}`;
const lib = fs.mkdtempSync(path.join(os.tmpdir(), 'vf-lib-'));
let srv;

const j = async (url, opt) => { const r = await fetch(BASE + url, opt); return { r, d: await r.json().catch(() => ({})) }; };
const post = (url, body) => j(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
async function wait(id, ms = 300000) {
  const t = Date.now();
  while (Date.now() - t < ms) { const { d } = await j('/api/jobs/' + id); if (d.status !== 'running') return d; await new Promise(r => setTimeout(r, 1000)); }
  throw new Error('timeout');
}
function wav(seconds, sr = 32000) {            // minimal 16-bit mono PCM WAV with a 440 Hz tone
  const n = seconds * sr, b = Buffer.alloc(44 + n * 2);
  b.write('RIFF', 0); b.writeUInt32LE(36 + n * 2, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(9000 * Math.sin(2 * Math.PI * 440 * i / sr)), 44 + i * 2);
  return b;
}

test.before(async () => {
  srv = spawn(process.execPath, ['server.js'], { cwd: path.join(__dirname, '..'), env: { ...process.env, PORT, VIBEFLOW_LIBRARY: lib, VIBEFLOW_IDLE_S: '8', VIBEFLOW_SAMPLES: path.join(lib, 'samples'), VIBEFLOW_LISTENING: path.join(lib, 'listening'), VIBEFLOW_RECORDINGS: path.join(lib, 'rec') } });
  for (let i = 0; i < 50; i++) { try { await fetch(BASE + '/'); return; } catch { await new Promise(r => setTimeout(r, 200)); } }
  throw new Error('server did not start');
});
test.after(() => srv && srv.kill());

test('static files served, private files are not', async () => {
  assert.equal((await fetch(BASE + '/')).status, 200);
  for (const p of ['/.env', '/server.js', '/engine/cli.py', '/models/x']) assert.equal((await fetch(BASE + p)).status, 404);
});

test('status reports hardware and models', async () => {
  const { d } = await j('/api/status');
  assert.ok(d.hardware.device && d.models.length >= 2 && d.unsupported.length);
});

test('validation errors reach the UI', async () => {
  const { d } = await post('/api/generate', { settings: { prompt: 'x', lyrics: 'la la' } });
  assert.equal(d.status, 'running');
  const done = await wait(d.id);
  assert.equal(done.status, 'failed');
  assert.match(done.error, /not available|not supported/i);
});

test('bad uploads are rejected', async () => {
  const bad = await j('/api/upload', { method: 'POST', headers: { 'x-filename': 'a.wav' }, body: Buffer.from('nope') });
  assert.equal(bad.r.status, 400);
  const ext = await j('/api/upload', { method: 'POST', headers: { 'x-filename': 'a.exe' }, body: Buffer.from('x') });
  assert.equal(ext.r.status, 400);
});

test('generate -> progress -> play -> export -> rename -> delete (real model)', { timeout: 600000 }, async () => {
  const { d: job } = await post('/api/generate', { settings: { prompt: 'soft piano', genres: ['ambient'], duration: 3, seed: 5, format: 'flac' } });
  assert.equal(job.status, 'running');
  const { r: busy } = await post('/api/generate', { settings: { prompt: 'second' } });
  assert.equal(busy.status, 409);                         // one heavy job at a time
  const done = await wait(job.id);
  assert.equal(done.status, 'complete', done.error);
  const t = done.track;
  const audio = await fetch(`${BASE}/api/library/${t.id}/audio?format=wav`, { headers: { Range: 'bytes=0-99' } });
  assert.equal(audio.status, 206);
  const flac = await fetch(`${BASE}/api/library/${t.id}/audio?format=flac`);
  assert.equal((await flac.arrayBuffer()).byteLength > 1000, true);
  assert.equal((await j('/api/library')).d.length, 1);
  const rn = await j(`/api/library/${t.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: 'Renamed' }) });
  assert.equal(rn.d.title, 'Renamed');
  assert.equal((await j(`/api/library/${t.id}`, { method: 'DELETE' })).d.ok, true);
  assert.equal((await j('/api/library')).d.length, 0);
});

test('reference upload + continue workflow + cancellation (real model)', { timeout: 600000 }, async () => {
  const up = await j('/api/upload', { method: 'POST', headers: { 'x-filename': 'tone.wav' }, body: wav(3) });
  assert.equal(up.r.status, 200, JSON.stringify(up.d));
  const { d: job } = await post('/api/generate', { settings: { prompt: 'continue', duration: 3, mode: 'continue', reference: up.d.id } });
  const done = await wait(job.id);
  assert.equal(done.status, 'complete', done.error);
  assert.equal(done.track.reference.workflow, 'continue');

  const { d: j2 } = await post('/api/generate', { settings: { prompt: 'cancel me', duration: 20 } });
  await new Promise(r => setTimeout(r, 4000));
  await post(`/api/jobs/${j2.id}/cancel`, {});
  assert.equal((await wait(j2.id, 30000)).status, 'cancelled');
  const { d: j3 } = await post('/api/generate', { settings: { prompt: 'after cancel', duration: 2 } });   // engine recovers
  assert.equal((await wait(j3.id)).status, 'complete');
});

test('security: foreign origin, rebinding host, traversal and path leaks are rejected', async () => {
  const body = JSON.stringify({ settings: { prompt: 'x' } });
  const foreign = await fetch(BASE + '/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://evil.example' }, body });
  assert.equal(foreign.status, 403);
  const del = await fetch(BASE + '/api/library/20260101-000000-abcdef', { method: 'DELETE', headers: { Origin: 'https://evil.example' } });
  assert.equal(del.status, 403);
  // Host header of a rebinding attack (raw http because fetch forbids overriding Host)
  const http = require('http');
  const code = await new Promise(res => http.get({ host: '127.0.0.1', port: PORT, path: '/api/status', headers: { Host: 'evil.example' } }, r => { r.resume(); res(r.statusCode); }));
  assert.equal(code, 403);
  for (const p of ['/api/library/..%2F..%2Fserver.js/audio', '/api/library/../x', '/api/library/NOPE!/audio', '/api/jobs/../../x'])
    assert.ok([400, 404].includes((await fetch(BASE + p)).status), p);
  assert.equal((await fetch(BASE + '/api/library/20260101-000000-abcdef/audio?format=exe')).status, 400);
  const miss = await j('/api/library/20260101-000000-abcdef/audio?format=wav');
  assert.ok(!/[A-Za-z]:[\/]|vibe-flow/i.test(JSON.stringify(miss.d)), 'error leaks a path: ' + JSON.stringify(miss.d));
  // oversized upload is refused up front
  const big = await j('/api/upload', { method: 'POST', headers: { 'x-filename': 'big.wav' }, body: Buffer.alloc(27 * 1024 * 1024) });
  assert.equal(big.r.status, 413);
  // hostile filename cannot escape the uploads directory
  const evil = await j('/api/upload', { method: 'POST', headers: { 'x-filename': encodeURIComponent('..\..\evil.wav') }, body: wav(2) });
  assert.equal(evil.r.status, 200);
  assert.ok(!/[\/]/.test(evil.d.name), evil.d.name);
});

test('library survives a server restart', { timeout: 300000 }, async () => {
  const { d: job } = await post('/api/generate', { settings: { prompt: 'persist me', duration: 2, seed: 2 } });
  const done = await wait(job.id);
  assert.equal(done.status, 'complete', done.error);
  srv.kill();
  await new Promise(r => setTimeout(r, 500));
  srv = spawn(process.execPath, ['server.js'], { cwd: path.join(__dirname, '..'), env: { ...process.env, PORT, VIBEFLOW_LIBRARY: lib, VIBEFLOW_IDLE_S: '8', VIBEFLOW_SAMPLES: path.join(lib, 'samples'), VIBEFLOW_LISTENING: path.join(lib, 'listening'), VIBEFLOW_RECORDINGS: path.join(lib, 'rec') } });
  for (let i = 0; i < 50; i++) { try { await fetch(BASE + '/'); break; } catch { await new Promise(r => setTimeout(r, 200)); } }
  const list = (await j('/api/library')).d;
  assert.ok(list.some(t => t.id === done.track.id && t.seed === 2));
  const a = await fetch(`${BASE}/api/library/${done.track.id}/audio?format=wav`);
  assert.equal(a.status, 200);
  assert.ok(fs.readdirSync(lib).every(f => !f.includes('.part')), 'no partial files left');
});


test('warm worker reuses the loaded model, and recovers after it idles out', { timeout: 600000 }, async () => {
  const gen = async seed => { const { d } = await post('/api/generate', { settings: { prompt: 'warm test', duration: 2, seed } }); const r = await wait(d.id); assert.equal(r.status, 'complete', r.error); return r.track.timing.load_s; };
  await gen(21);                                     // may be cold or warm depending on earlier tests
  const warm = await gen(22);                        // immediately after: model must still be in memory
  assert.ok(warm < 1, 'second job should not reload the model, load_s=' + warm);
  await new Promise(r => setTimeout(r, 11000));      // worker exits after 8 s idle and frees its RAM
  const cold = await gen(23);                        // server must transparently start a new worker
  assert.ok(cold > 2, 'after idle exit the model must be reloaded, load_s=' + cold);
});

test('adapt a sample over HTTP: 100% keeps the sample, parts are downloadable, vocals are refused (real model)', { timeout: 600000 }, async () => {
  const sample = fs.readFileSync(path.join(__dirname, '..', 'samples', 'musicgen_xgenre', 'g12.wav'));
  const up = await j('/api/upload', { method: 'POST', headers: { 'x-filename': 'g12.wav' }, body: sample });
  assert.equal(up.r.status, 200, JSON.stringify(up.d));
  const bad = await post('/api/generate', { settings: { prompt: 'x', mode: 'adapt', reference: up.d.id, adherence: 50, vocals: true } });
  const badDone = await wait(bad.d.id);
  assert.equal(badDone.status, 'failed');
  assert.match(badDone.error, /Music with Vocals is not available/);
  const { d: job } = await post('/api/generate', { settings: { mode: 'adapt', reference: up.d.id, adherence: 100, instruments: ['piano'], prompt: '', seed: 3, layer_level_db: -6 } });
  const done = await wait(job.id);
  assert.equal(done.status, 'complete', done.error);
  const t = done.track;
  assert.equal(t.kind, 'adapt'); assert.equal(t.vocals, false);
  assert.ok(t.plan.retained_bars.every(x => x === 1), 'every bar of the sample is retained at 100%');
  assert.ok(t.evaluation.source_integrity_max_residual < 1e-5);
  for (const part of ['source', 'layers']) {
    const r = await fetch(`${BASE}/api/library/${t.id}/audio?part=${part}`);
    assert.equal(r.status, 200); assert.ok((await r.arrayBuffer()).byteLength > 1000);
  }
  assert.equal((await fetch(`${BASE}/api/library/${t.id}/audio?part=../x`)).status, 400);
  assert.equal((await j(`/api/library/${t.id}`, { method: 'DELETE' })).d.ok, true);
  assert.ok(fs.readdirSync(lib).every(f => !f.startsWith(t.id)), 'delete removes the mix, source and layers files');
});

test('editor endpoints: FLAC encode, save to Your Songs, samples store and hand-off to Create', { timeout: 120000 }, async () => {
  const body = wav(2);
  const flac = await fetch(BASE + '/api/editor/encode?format=flac', { method: 'POST', body });
  assert.equal(flac.status, 200);
  const fb = Buffer.from(await flac.arrayBuffer());
  assert.equal(fb.slice(0, 4).toString(), 'fLaC'); assert.ok(fb.length > 100 && fb.length < body.length);
  assert.equal((await fetch(BASE + '/api/editor/encode?format=mp3', { method: 'POST', body })).status, 400);
  const garbage = await j('/api/editor/encode?format=flac', { method: 'POST', body: Buffer.from('not audio') });
  assert.equal(garbage.r.status, 400); assert.ok(!/[A-Za-z]:[\/]/.test(JSON.stringify(garbage.d)));

  const saved = await j('/api/editor/save?title=' + encodeURIComponent('Edited mix'), { method: 'POST', body });
  assert.equal(saved.r.status, 200, JSON.stringify(saved.d));
  assert.equal(saved.d.kind, 'editor'); assert.equal(saved.d.title, 'Edited mix'); assert.ok(saved.d.audio.duration > 1.9);
  assert.ok((await j('/api/library')).d.some((t) => t.id === saved.d.id));
  const back = await fetch(`${BASE}/api/library/${saved.d.id}/audio?format=wav`); assert.equal(back.status, 200);
  assert.equal(Buffer.compare(Buffer.from(await back.arrayBuffer()), body), 0, 'the saved WAV is the rendered WAV, byte for byte');
  await j(`/api/library/${saved.d.id}`, { method: 'DELETE' });

  const s = await j('/api/samples?name=' + encodeURIComponent('My edit') + '&kind=instrumental', { method: 'POST', body });
  assert.equal(s.r.status, 200, JSON.stringify(s.d)); assert.equal(s.d.kind, 'instrumental'); assert.ok(s.d.duration > 1.9);
  assert.ok((await j('/api/samples')).d.some((x) => x.id === s.d.id));
  const audio = await fetch(`${BASE}/api/samples/${s.d.id}/audio`); assert.equal(audio.status, 200);
  const use = await j(`/api/samples/${s.d.id}/use`, { method: 'POST' });
  assert.equal(use.r.status, 200); assert.ok(use.d.id && use.d.duration > 1.9);
  // the registered upload id is accepted by a generation request as a valid reference (it validates, it does not need to run)
  const job = await post('/api/generate', { settings: { mode: 'adapt', reference: use.d.id, adherence: 50, instruments: ['piano'], prompt: 'x', vocals: true } });
  const jd = await wait(job.d.id); assert.match(jd.error, /Music with Vocals is not available/);   // reached validation => the reference id resolved
  assert.equal((await j(`/api/samples/${s.d.id}`, { method: 'DELETE' })).d.ok, true);
  assert.equal((await fetch(`${BASE}/api/samples/${s.d.id}/audio`)).status, 404);
  for (const bad of ['/api/samples/..%2F..%2Fserver.js/audio', '/api/samples/ZZZZ/audio', '/api/samples/0123456789abcdef0/audio']) assert.ok([400, 404].includes((await fetch(BASE + bad)).status), bad);
  const evil = await fetch(BASE + '/api/samples?name=x', { method: 'POST', headers: { Origin: 'https://evil.example' }, body });
  assert.equal(evil.status, 403);
});

test('listening pack over HTTP: validation, real run at two levels, blind files, hidden key, ratings (real model)', { timeout: 900000 }, async () => {
  assert.deepEqual((await j('/api/listening')).d, []);
  assert.equal((await post('/api/listening/run', { sources: [] })).r.status, 400);
  assert.equal((await post('/api/listening/run', { sources: [{ type: 'recording', id: '0123456789abcdef', version: 'full' }] })).r.status, 400);
  assert.equal((await post('/api/listening/run', { sources: [{ type: 'upload', id: 'nope' }], levels: ['extreme'] })).r.status, 400);
  const up = await j('/api/upload', { method: 'POST', headers: { 'x-filename': 'tone.wav' }, body: wav(8) });
  const srcs = [{ type: 'upload', id: up.d.id, kind: 'instrumental' }];
  assert.equal((await post('/api/listening/run', { sources: srcs })).r.status, 400, 'no instrument/genre/mood: rejected before any generation');
  const { d: job } = await post('/api/listening/run', { name: 'T', sources: srcs, levels: ['low', 'strict'], seeds: [1], seconds: 8, instruments: ['piano'] });
  assert.equal(job.kind, 'listening');
  const done = await wait(job.id, 800000);
  assert.equal(done.status, 'complete', done.error);
  const list = (await j('/api/listening')).d;
  assert.equal(list.length, 1); assert.equal(list[0].id, done.run);
  const rep = (await j('/api/listening/' + done.run)).d;
  assert.deepEqual(Object.keys(rep.levels), ['low', 'strict']);
  const src = rep.sources[0], checks = Object.fromEntries(src.checks.map(c => [c.id, c.pass]));
  assert.equal(checks['mechanism strict'], true); assert.equal(checks['mechanism low'], true); assert.equal(checks.integrity, true);
  assert.ok(src.summary.strict.mix.adherence_index[0] > src.summary.low.mix.adherence_index[0]);
  assert.equal(rep.blind.length, 2);
  const f = await fetch(`${BASE}/api/listening/${done.run}/file?path=blind/${rep.blind[0].code}.wav`);
  assert.equal(f.status, 200); assert.equal(f.headers.get('content-type'), 'audio/wav');
  for (const bad of ['key.json', '../report.json', 'lib/../key.json']) assert.equal((await fetch(`${BASE}/api/listening/${done.run}/file?path=${encodeURIComponent(bad)}`)).status, 400);
  assert.ok(!JSON.stringify(rep).includes('"level":"low","seed":1,"track"'), 'the blind key is not in the report');
  const ratings = Object.fromEntries(rep.blind.map((b, i) => [b.code, { closeness: i + 1, quality: 3 }]));
  const sc = await post(`/api/listening/${done.run}/ratings`, { ratings });
  assert.equal(sc.r.status, 200); assert.equal(Object.values(sc.d.per_level).reduce((a, v) => a + v.n, 0), 2);
  assert.equal((await j(`/api/listening/${done.run}`, { method: 'DELETE' })).r.status, 200);
  assert.deepEqual((await j('/api/listening')).d, []);
});
