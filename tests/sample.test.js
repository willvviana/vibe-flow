// Sample page backend: real HTTP server + real Python engine + real Demucs / Whisper / CLAP models (CPU, a few minutes).
// Input is a 30 s sung clip generated earlier by ACE-Step (samples/ace/s2...), streamed to the server as raw float32 PCM like the browser does.
const test = require('node:test');
const assert = require('node:assert');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const PORT = 3998, BASE = `http://127.0.0.1:${PORT}`;
const CLIP = path.join(__dirname, '..', 'samples', 'ace', 's2_sft_codes_30s_32steps.wav');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'vf-rec-'));
let srv;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const j = async (url, opt) => { const r = await fetch(BASE + url, opt); return { r, d: await r.json().catch(() => ({})) }; };

function readWav(file) {                       // tiny WAV reader: PCM16 / PCM24 / float32 -> { sr, ch, data: Float32Array interleaved }
  const b = fs.readFileSync(file); let p = 12, fmt, data;
  while (p < b.length - 8) {
    const id = b.toString('ascii', p, p + 4), len = b.readUInt32LE(p + 4);
    if (id === 'fmt ') fmt = { tag: b.readUInt16LE(p + 8), ch: b.readUInt16LE(p + 10), sr: b.readUInt32LE(p + 12), bits: b.readUInt16LE(p + 22) };
    if (id === 'data') { data = b.subarray(p + 8, p + 8 + len); break; }
    p += 8 + len + (len & 1);
  }
  const bytes = fmt.bits / 8, n = Math.floor(data.length / bytes), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * bytes;
    out[i] = fmt.bits === 16 ? data.readInt16LE(o) / 32768 : fmt.bits === 24 ? data.readIntLE(o, 3) / 8388608 : fmt.tag === 3 ? data.readFloatLE(o) : data.readInt32LE(o) / 2147483648;
  }
  return { sr: fmt.sr, ch: fmt.ch, data: out };
}

test.before(async () => {
  srv = spawn(process.execPath, ['server.js'], { cwd: path.join(__dirname, '..'), env: { ...process.env, PORT, VIBEFLOW_RECORDINGS: path.join(tmp, 'rec'), VIBEFLOW_SAMPLES: path.join(tmp, 'samples'), VIBEFLOW_LIBRARY: path.join(tmp, 'lib') } });
  for (let i = 0; i < 50; i++) { try { await fetch(BASE + '/'); return; } catch { await sleep(200); } }
  throw new Error('server did not start');
});
test.after(() => { srv && srv.kill(); });

test('record -> live transcript -> stems, tags, transcript -> library -> use as sample', { skip: !fs.existsSync(CLIP) && 'the sung test clip is not present', timeout: 900000 }, async () => {
  assert.deepEqual((await j('/api/recordings/models')).d, { stems: true, transcription: true, tagging: true });
  const wav = readWav(CLIP), seconds = wav.data.length / wav.ch / wav.sr;
  const start = (await j(`/api/recordings?sr=${wav.sr}&ch=${wav.ch}&source=test`, { method: 'POST' })).d;
  assert.match(start.id, /^[a-f0-9]{16}$/);
  assert.equal((await j(`/api/recordings?sr=${wav.sr}&ch=${wav.ch}`, { method: 'POST' })).r.status, 409, 'only one capture at a time');
  assert.equal((await j(`/api/recordings/${start.id}/pcm`, { method: 'POST', body: Buffer.alloc(5) })).r.status, 400, 'partial frames are rejected');

  // stream the audio in 2 s chunks at ~4x real time, as the browser would while capturing
  const frames = wav.sr * 2 * wav.ch;
  for (let o = 0; o < wav.data.length; o += frames) {
    const part = wav.data.subarray(o, o + frames);
    const r = await fetch(`${BASE}/api/recordings/${start.id}/pcm`, { method: 'POST', body: Buffer.from(part.buffer, part.byteOffset, part.byteLength) });
    assert.equal(r.status, 200);
    await sleep(500);
  }
  const lv = (await j(`/api/recordings/${start.id}/live`)).d;
  assert.ok(lv.active && Math.abs(lv.seconds - seconds) < 0.01, `server holds all audio: ${lv.seconds}`);
  let text = '';
  for (let i = 0; i < 60; i++) {                                         // the live transcript catches up while recording / shortly after
    const d = (await j(`/api/recordings/${start.id}/live`)).d; text = d.segments.map(s => s.text).join(' ');
    if (d.transcribed_to > seconds - 8 && text) break; await sleep(2000);
  }
  assert.match(text.toLowerCase(), /street|moon|walking|hold me/, 'live transcript has the sung words: ' + text);

  const stop = await j(`/api/recordings/${start.id}/stop`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: 'Test take' }) });
  assert.equal(stop.r.status, 200); assert.equal(stop.d.name, 'Test take');
  assert.ok(Math.abs(stop.d.duration - seconds) < 0.01);
  let m;
  for (let i = 0; i < 400; i++) { m = (await j(`/api/recordings/${start.id}`)).d; if (m.status === 'ready' || m.status === 'failed') break; await sleep(2000); }
  assert.equal(m.status, 'ready', m.error);

  // three versions exist and the original is bit-exact (float32 WAV of exactly what was captured)
  const sizes = {};
  for (const v of ['full', 'instrumental', 'voice']) { const r = await fetch(`${BASE}/api/recordings/${start.id}/audio/${v}`); assert.equal(r.status, 200); sizes[v] = (await r.arrayBuffer()).byteLength; }
  assert.ok(sizes.instrumental > 100000 && sizes.voice > 100000);
  const full = readWav(path.join(tmp, 'rec', start.id, 'full.wav'));
  assert.equal(full.data.length, wav.data.length);
  let maxDiff = 0; for (let i = 0; i < full.data.length; i += 97) maxDiff = Math.max(maxDiff, Math.abs(full.data[i] - wav.data[i]));
  assert.ok(maxDiff < 1e-6, 'full recording is untouched, max diff ' + maxDiff);
  const vox = readWav(path.join(tmp, 'rec', start.id, 'vocals.wav')), ins = readWav(path.join(tmp, 'rec', start.id, 'instrumental.wav'));
  const rms = a => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
  assert.ok(Math.abs(vox.data.length - ins.data.length) < 4 && vox.sr === 44100);
  assert.ok(m.stems.reconstruction_residual_db < -25, 'stems add back up to the original: ' + m.stems.reconstruction_residual_db + ' dB');
  assert.ok(rms(vox.data) > 0.01 && rms(ins.data) > 0.005);

  assert.equal(m.vocals.present, true);
  assert.equal(m.vocals.language, 'en');
  assert.match(m.transcript.segments.map(s => s.text).join(' ').toLowerCase(), /street lights/);
  assert.ok(m.tags.genres.length && m.tags.instruments.length && m.tags.moods.length && m.tags.bpm.value > 40 && m.tags.key.value);

  // library, rename, edit tags, edit transcript
  assert.ok((await j('/api/recordings')).d.some(x => x.id === start.id));
  const up = (await j(`/api/recordings/${start.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: 'Renamed', tags: { genres: [{ label: 'folk pop', score: null, edited: true }] }, transcript: { segments: [{ start: 0, end: 1, text: 'edited' }] } }) })).d;
  assert.equal(up.name, 'Renamed'); assert.equal(up.tags.genres[0].label, 'folk pop'); assert.equal(up.transcript.edited, true);

  // "Use as sample": each version becomes a Create-compatible reference
  for (const v of ['full', 'instrumental', 'voice']) {
    const u = await j(`/api/recordings/${start.id}/use?version=${v}`, { method: 'POST' });
    assert.equal(u.r.status, 200); assert.match(u.d.id, /^[a-f0-9]{16}$/); assert.equal(u.d.version, v); assert.ok(u.d.duration > 25 && u.d.analysis.tags.moods.length);
  }
  assert.equal((await j(`/api/recordings/${start.id}/use?version=nope`, { method: 'POST' })).r.status, 404);
  assert.equal((await j(`/api/recordings/${start.id}`, { method: 'DELETE' })).r.status, 200);
  assert.equal((await j(`/api/recordings/${start.id}`)).r.status, 404);
});

test('a recording shorter than one second is discarded with a clear message', async () => {
  const s = (await j('/api/recordings?sr=48000&ch=2', { method: 'POST' })).d;
  await fetch(`${BASE}/api/recordings/${s.id}/pcm`, { method: 'POST', body: Buffer.alloc(4 * 2 * 1000) });
  const r = await j(`/api/recordings/${s.id}/stop`, { method: 'POST' });
  assert.equal(r.r.status, 400); assert.match(r.d.error, /shorter than one second/);
});

test('bad ids and capture formats are rejected', async () => {
  assert.equal((await j('/api/recordings?sr=5&ch=2', { method: 'POST' })).r.status, 400);
  assert.equal((await j('/api/recordings/../../etc', {})).r.status, 404);
  assert.equal((await j('/api/recordings/0123456789abcdef/audio/full')).r.status, 404);
});

// System-output capture (Windows WASAPI loopback). Plays a short sound through the default speakers, so it is opt-in: VIBEFLOW_TEST_LOOPBACK=1
test('system audio capture records what the computer plays', { skip: process.env.VIBEFLOW_TEST_LOOPBACK !== '1' && 'set VIBEFLOW_TEST_LOOPBACK=1 (plays audio through the speakers)', timeout: 120000 }, async () => {
  const cap = (await j('/api/recordings/capabilities')).d;
  assert.equal(cap.system.available, true, cap.system.reason);
  const s = (await j('/api/recordings?source=system', { method: 'POST' })).d;
  assert.equal(s.source, 'system'); assert.equal(s.channels, 2);
  const player = spawn('powershell', ['-NoProfile', '-Command', `$p=New-Object System.Media.SoundPlayer('${CLIP}'); $p.PlaySync()`], { stdio: 'ignore' });
  let lv; let peak = 0;
  for (let i = 0; i < 25; i++) { await sleep(200); lv = (await j(`/api/recordings/${s.id}/live`)).d; peak = Math.max(peak, lv.sys.pl, lv.sys.pr); }
  assert.ok(peak > 0.05, 'meters saw the playback: ' + peak);
  assert.equal((await j(`/api/recordings/${s.id}/pause`, { method: 'POST' })).d.paused, true);
  const mid = (await j(`/api/recordings/${s.id}/live`)).d.seconds; await sleep(1500);
  assert.ok(Math.abs((await j(`/api/recordings/${s.id}/live`)).d.seconds - mid) < 0.3, 'nothing is recorded while paused');
  await j(`/api/recordings/${s.id}/resume`, { method: 'POST' });
  await sleep(500);
  const stop = await j(`/api/recordings/${s.id}/stop`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: 'loopback' }) });
  player.kill();
  assert.equal(stop.r.status, 200);
  const w = readWav(path.join(tmp, 'rec', s.id, 'full.wav'));
  const rms = Math.sqrt(w.data.reduce((a, v) => a + v * v, 0) / w.data.length);
  assert.ok(w.ch === 2 && w.sr >= 44100 && rms > 0.005, `recorded the playback: rms ${rms}`);
  await j(`/api/recordings/${s.id}`, { method: 'DELETE' }).catch(() => {});
});
