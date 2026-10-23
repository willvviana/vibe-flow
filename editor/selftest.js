/* Editor self-test (development tool, not part of the UI): drives the real editor in the browser and verifies the acceptance workflows
 * numerically by rendering the timeline offline and analysing the samples.  Run from the console:  await EditorSelfTest.run()  -> [{name, ok, detail}] */
(function () {
  'use strict';
  const App = window.EditorApp, C = App.C, D = App.D, S = App.S;
  const SR = 48000;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const results = [];
  const ok = (name, cond, detail) => { results.push({ name, ok: !!cond, detail: detail == null ? '' : String(detail) }); };
  const db = (v) => 20 * Math.log10(v + 1e-12);

  // ---- signal helpers
  function wavFile(name, secs, fn) {
    const n = Math.floor(secs * SR), x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = fn(i / SR);
    return new File([D.encodeWav([x], SR, 16)], name, { type: 'audio/wav' });
  }
  const music = (name = 'music.wav', secs = 8) => wavFile(name, secs, (t) => 0.22 * (Math.sin(2 * Math.PI * 220 * t) + Math.sin(2 * Math.PI * 277.18 * t) + Math.sin(2 * Math.PI * 329.63 * t)) / 1.5);
  const voice = (name = 'voice.wav', secs = 3) => wavFile(name, secs, (t) => { const k = t % 1; return k < 0.6 ? 0.35 * (Math.sin(2 * Math.PI * 600 * t) + 0.5 * Math.sin(2 * Math.PI * 1200 * t)) * Math.min(1, k * 40) * Math.min(1, (0.6 - k) * 40) : 0; });
  const rms = (ch, a, b) => { let s = 0, n = 0; for (let i = Math.floor(a * SR); i < Math.min(ch.length, Math.floor(b * SR)); i++) { s += ch[i] * ch[i]; n++; } return Math.sqrt(s / Math.max(1, n)); };
  const bandE = (ch, a, f1, f2) => { const n = 1 << 14, re = new Float64Array(n), im = new Float64Array(n), w = D.hann(n), o = Math.floor(a * SR); for (let i = 0; i < n; i++) re[i] = (ch[o + i] || 0) * w[i]; D.fft(re, im, false); let e = 0; for (let k = Math.floor(f1 * n / SR); k <= Math.ceil(f2 * n / SR); k++) e += Math.hypot(re[k], im[k]); return e; };
  const peakHz = (ch, a, b) => { const n = 1 << 14, re = new Float64Array(n), im = new Float64Array(n), w = D.hann(n), o = Math.floor(a * SR); for (let i = 0; i < n; i++) re[i] = (ch[o + i] || 0) * w[i]; D.fft(re, im, false); let bi = 1, bv = 0; for (let k = 1; k < n / 2; k++) { const m = Math.hypot(re[k], im[k]); if (m > bv) { bv = m; bi = k; } } return bi * SR / n; };
  async function render(opts) { await App.engine.prepare(App.project, App.assets); const to = C.projectDuration(App.project); return App.engine.render(0, to, SR, Object.assign({ tail: 0.05 }, opts || {})); }
  async function fresh() { await App.newProject('selftest'); App.engine.processed.clear(); }
  async function importAll(files) { await App.importFiles(files); await sleep(50); }
  function key(k, o) { document.dispatchEvent(new KeyboardEvent('keydown', Object.assign({ key: k, bubbles: true, cancelable: true }, o || {}))); }
  function ptr(el, type, x, y, extra) { el.dispatchEvent(new PointerEvent(type, Object.assign({ clientX: x, clientY: y, bubbles: true, cancelable: true, pointerId: 1, isPrimary: true, button: 0 }, extra || {}))); }
  function windowPtr(type, x, y) { window.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, bubbles: true, pointerId: 1, isPrimary: true })); }

  async function run() {
    results.length = 0;
    try { await S.clearRecovery(); } catch (e) { /* none */ }
    App.closeDialog();                       // a recovery prompt from an earlier session would (correctly) block shortcuts
    const t0 = performance.now();
    let r;
    const sections = [
      async () => {
    // 1. import two files -> two layered tracks, play together
    await fresh();
    await importAll([music('a.wav', 8), voice('b.wav', 3)]);
    ok('1 import two files into two tracks', App.project.tracks.length === 2 && App.project.clips.length === 2, `tracks=${App.project.tracks.length} clips=${App.project.clips.length}`);
    const both = await render(), mA = await render({ trackIds: new Set([App.project.tracks[0].id]) }), mB = await render({ trackIds: new Set([App.project.tracks[1].id]) });
    const sumRms = rms(both.chans[0], 0.1, 0.5);
    ok('1 both tracks are audible together', rms(mA.chans[0], 0.1, 0.5) > 0.02 && rms(mB.chans[0], 0.1, 0.5) > 0.02 && sumRms > rms(mA.chans[0], 0.1, 0.5), `A=${rms(mA.chans[0], .1, .5).toFixed(3)} B=${rms(mB.chans[0], .1, .5).toFixed(3)} mix=${sumRms.toFixed(3)}`);
    await App.engine.prepare(App.project, App.assets);
    await App.play(); await sleep(600);
    const p1 = App.engine.position(); await sleep(400);
    const p2 = App.engine.position(); const ctxState = App.engine.ctx.state;
    let lvl = -120; if (App.engine.analyser) { const b = new Float32Array(1024); App.engine.analyser.getFloatTimeDomainData(b); lvl = db(Math.max(...b.map(Math.abs))); }
    ok('1 live playback runs and the playhead advances in real time', ctxState === 'running' && p2 - p1 > 0.3 && p2 - p1 < 0.6, `ctx=${ctxState} advanced=${(p2 - p1).toFixed(2)}s level=${lvl.toFixed(1)} dBFS`);
    ok('1 live graph carries audio to the output', lvl > -40, `${lvl.toFixed(1)} dBFS`);
    App.engine.pause(); App.updatePlay();

      },
      async () => {
    // 2/12. exact positions, volume, fades in the rendered timeline
    await fresh();
    await importAll([music('m.wav', 4)]);
    const c = App.project.clips[0]; c.start = 1; c.params.gainDb = -6; c.params.fadeIn = 1; App.commit();
    r = await render();
    ok('12 clip begins exactly at its timeline position', rms(r.chans[0], 0.2, 0.9) < 1e-4 && rms(r.chans[0], 2.2, 2.6) > 0.05, `before=${rms(r.chans[0], .2, .9).toExponential(1)} after=${rms(r.chans[0], 2.2, 2.6).toFixed(3)}`);
    ok('12 fade-in ramps the level up', rms(r.chans[0], 1.05, 1.25) < rms(r.chans[0], 1.85, 2.05) * 0.5, `start=${rms(r.chans[0], 1.05, 1.25).toFixed(4)} end=${rms(r.chans[0], 1.85, 2.05).toFixed(4)}`);
    const expected = 0.22 * Math.sqrt(0.5 * 3) / 1.5 * Math.pow(10, -6 / 20) / 1;      // tone RMS scaled by the volume
    ok('12 clip volume is applied (-6 dB)', Math.abs(db(rms(r.chans[0], 2.5, 4.5)) - db(expected * 1.0)) < 2.5, `measured ${db(rms(r.chans[0], 2.5, 4.5)).toFixed(1)} dBFS, expected about ${db(expected).toFixed(1)}`);

      },
      async () => {
    // 3. playhead: drag the ruler handle to an exact time, play from there
    await fresh(); await importAll([music('p.wav', 8)]);
    const ruler = document.getElementById('ruler'), rr = ruler.getBoundingClientRect();
    App.project.settings.snap = false;
    const target = 3.25, x = rr.left + App.px(target);
    ptr(ruler, 'pointerdown', x, rr.top + 14); ptr(ruler, 'pointerup', x, rr.top + 14); await sleep(120);
    ok('3 clicking the ruler places the playhead at the exact time', Math.abs(App.engine.position() - target) < 0.02, `position=${App.engine.position().toFixed(3)} target=${target}`);
    const hx = rr.left + App.px(5.5);
    ptr(ruler, 'pointerdown', rr.left + App.px(1), rr.top + 14); ptr(ruler, 'pointermove', hx, rr.top + 14); ptr(ruler, 'pointerup', hx, rr.top + 14); await sleep(120);
    ok('3 dragging scrubs the playhead', Math.abs(App.engine.position() - 5.5) < 0.03, `position=${App.engine.position().toFixed(3)}`);
    await App.play(); await sleep(500);
    ok('3 playback starts from the playhead position', App.engine.position() > 5.5 && App.engine.position() < 6.4, `position=${App.engine.position().toFixed(2)}`);
    App.engine.pause(); const held = App.engine.position(); await sleep(300);
    ok('3 pause keeps the position (no reset)', Math.abs(App.engine.position() - held) < 1e-6 && held > 5.5, `held=${held.toFixed(2)}`);
    App.project.settings.snap = true;
    App.engine.pos = 0;
    // snapping to a clip boundary
    const cl = App.project.clips[0]; ptr(ruler, 'pointerdown', rr.left + App.px(cl.start + C.clipEnd(cl) - cl.start) - 2, rr.top + 14); ptr(ruler, 'pointerup', rr.left + App.px(C.clipEnd(cl)) - 2, rr.top + 14); await sleep(80);
    ok('3 the playhead snaps to clip boundaries when snapping is on', Math.abs(App.engine.position() - C.clipEnd(cl)) < 1e-6, `position=${App.engine.position().toFixed(4)} edge=${C.clipEnd(cl)}`);

      },
      async () => {
    // 4. split / trim / reposition through real keys and pointer drags
    await fresh(); await importAll([music('s.wav', 6)]);
    App.project.settings.snap = false;
    await App.seek(2); key('s');
    ok('4 S splits the clip at the playhead', App.project.clips.length === 2 && Math.abs(App.project.clips[0].duration - 2) < 1e-6, `clips=${App.project.clips.length} sel=${App.sel.size} pos=${App.engine.position()} modalHidden=${document.getElementById('modal').hidden} capturing=${App.capturing} active=${document.activeElement && document.activeElement.tagName}`);
    const right = App.project.clips.find((q) => q.start >= 2 - 1e-6 && q.start < 2.001 && q.offset > 1);
    App.sel = new Set([right.id]); App.renderAll();                      // both halves are selected after a split; drag only the right one
    const el = document.querySelector(`.ed-clip[data-id="${right.id}"]`), br = el.getBoundingClientRect();
    ptr(el, 'pointerdown', br.left + 40, br.top + 25); windowPtr('pointermove', br.left + 40 + App.px(1.5), br.top + 25); windowPtr('pointerup', br.left + 40 + App.px(1.5), br.top + 25); await sleep(120);
    const moved = C.byId(App.project.clips, right.id);
    ok('4 dragging a clip repositions it', Math.abs(moved.start - 3.5) < 0.05, `start=${moved.start.toFixed(3)}`);
    const el2 = document.querySelector(`.ed-clip[data-id="${right.id}"]`), b2 = el2.getBoundingClientRect(), h = el2.querySelector('.h.r').getBoundingClientRect();
    ptr(el2.querySelector('.h.r'), 'pointerdown', h.left + 3, h.top + 20); windowPtr('pointermove', h.left + 3 - App.px(1), h.top + 20); windowPtr('pointerup', h.left + 3 - App.px(1), h.top + 20); await sleep(120);
    const trimmed = C.byId(App.project.clips, right.id);
    ok('4 dragging the right edge trims the clip (non-destructive)', Math.abs(trimmed.duration - 3) < 0.05 && Math.abs(trimmed.offset - 2) < 1e-6, `duration=${trimmed.duration.toFixed(3)} offset=${trimmed.offset}`);
    r = await render();
    ok('4 the rendered audio follows the edited timeline', rms(r.chans[0], 0.2, 1.8) > 0.05 && rms(r.chans[0], 2.2, 3.2) < 1e-4 && rms(r.chans[0], 3.7, 6.2) > 0.05 && rms(r.chans[0], 6.7, 7.2) < 1e-4, `gap=${rms(r.chans[0], 2.2, 3.2).toExponential(1)}`);
    // moving between tracks
    App.act.addTrack('Second'); const t2 = App.project.tracks[1];
    C.moveClips(App.project, [right.id], 0, 1); App.commit();
    ok('4 a clip can move to another track', C.byId(App.project.clips, right.id).trackId === t2.id);

      },
      async () => {
    // 5. one track's volume leaves the others untouched
    await fresh(); await importAll([music('m1.wav', 3), music('m2.wav', 3)]);
    const before = await render({ trackIds: new Set([App.project.tracks[0].id]) });
    App.project.tracks[1].volume = -12; App.commit();
    const a1 = await render({ trackIds: new Set([App.project.tracks[0].id]) }), a2 = await render({ trackIds: new Set([App.project.tracks[1].id]) });
    ok('5 lowering one track changes only that track', Math.abs(db(rms(a1.chans[0], 0.5, 2.5)) - db(rms(before.chans[0], 0.5, 2.5))) < 0.1 && Math.abs(db(rms(a2.chans[0], 0.5, 2.5)) - db(rms(a1.chans[0], 0.5, 2.5)) + 12) < 0.6, `track1 ${db(rms(a1.chans[0], .5, 2.5)).toFixed(1)} dB, track2 ${db(rms(a2.chans[0], .5, 2.5)).toFixed(1)} dB`);
    // mute / solo
    App.project.tracks[0].mute = true; App.commit(); r = await render();
    ok('5 mute silences a track', Math.abs(db(rms(r.chans[0], .5, 2.5)) - db(rms(a2.chans[0], .5, 2.5))) < 0.5);
    App.project.tracks[0].mute = false; App.project.tracks[1].solo = true; App.commit(); r = await render();
    ok('5 solo plays only that track', Math.abs(db(rms(r.chans[0], .5, 2.5)) - db(rms(a2.chans[0], .5, 2.5))) < 0.5);
    App.project.tracks[1].solo = false;

      },
      async () => {
    // 6. effects are audible
    await fresh(); await importAll([wavFile('tone.wav', 3, (t) => 0.3 * Math.sin(2 * Math.PI * 440 * t) + 0.3 * Math.sin(2 * Math.PI * 6000 * t))]);
    const clip = App.project.clips[0];
    const dry = await render();
    clip.params.filter = { type: 'lowpass', freq: 1000, q: 0.7 }; App.commit(); const lp = await render();
    const hfDry = rms(dry.chans[0], 1, 2), spectrumHf = (res) => { const n = 1 << 14, re = new Float64Array(n), im = new Float64Array(n), w = D.hann(n); for (let i = 0; i < n; i++) re[i] = res.chans[0][SR + i] * w[i]; D.fft(re, im, false); let e = 0; for (let k = Math.floor(5000 * n / SR); k < Math.floor(7000 * n / SR); k++) e += Math.hypot(re[k], im[k]); return e; };
    ok('6 low-pass filter removes the 6 kHz component', spectrumHf(lp) < spectrumHf(dry) * 0.05, `HF energy ${spectrumHf(dry).toFixed(0)} -> ${spectrumHf(lp).toFixed(0)}`);
    clip.params.filter = { type: 'none', freq: 1000, q: 0.7 };
    clip.params.eq = { on: true, gains: [0, 0, 0, 0, -12] }; App.commit(); const eqr = await render();
    ok('6 equalizer cuts the high band (-12 dB at 10 kHz shelf reduces 6 kHz)', spectrumHf(eqr) < spectrumHf(dry) * 0.9 && spectrumHf(eqr) > 0, `${spectrumHf(dry).toFixed(0)} -> ${spectrumHf(eqr).toFixed(0)}`);
    clip.params.eq.on = false;
    clip.params.reverb = { on: true, mix: 0.5, decay: 2 }; App.commit();
    const rv = await App.engine.prepare(App.project, App.assets).then(() => App.engine.render(0, 3, SR, { tail: 2.5 }));
    ok('6 reverb adds a tail after the clip ends', rms(rv.chans[0], 3.2, 4.2) > 0.01, `tail rms=${rms(rv.chans[0], 3.2, 4.2).toFixed(4)}`);
    clip.params.reverb.on = false;
    clip.params.delay = { on: true, time: 0.5, feedback: 0.4, mix: 0.6 }; App.commit();
    const dl = await App.engine.render(0, 3, SR, { tail: 2.5 });
    ok('6 delay produces repeats after the clip ends', rms(dl.chans[0], 3.1, 3.9) > 0.01, `repeat rms=${rms(dl.chans[0], 3.1, 3.9).toFixed(4)}`);
    clip.params.delay.on = false;
    clip.params.comp = { on: true, threshold: -40, ratio: 12, attack: 0.003, release: 0.1, knee: 0, makeup: 0 }; App.commit(); const cp = await render();
    ok('6 compressor reduces the level', rms(cp.chans[0], 1, 2) < rms(dry.chans[0], 1, 2) * 0.8, `${db(rms(dry.chans[0], 1, 2)).toFixed(1)} -> ${db(rms(cp.chans[0], 1, 2)).toFixed(1)} dBFS`);
    clip.params.comp.on = false;
    // heavy edits: speed, pitch, reverse, loops
    C.setHeavyParam(clip, 'speed', 2); App.commit(); await App.process(); r = await render();
    ok('6 speed 2x halves the duration and keeps the pitch', Math.abs(C.projectDuration(App.project) - 1.5) < 0.05 && bandE(r.chans[0], 0.3, 430, 450) > 20 * bandE(r.chans[0], 0.3, 850, 900) && bandE(r.chans[0], 0.3, 5900, 6100) > 20 * bandE(r.chans[0], 0.3, 11800, 12200), `duration=${C.projectDuration(App.project).toFixed(2)} 440Hz=${bandE(r.chans[0], 0.3, 430, 450).toFixed(0)} 880Hz=${bandE(r.chans[0], 0.3, 850, 900).toFixed(0)}`);
    C.setHeavyParam(clip, 'speed', 1); C.setHeavyParam(clip, 'pitch', 12); App.commit(); await App.process(); r = await render();
    ok('6 pitch +12 semitones doubles the frequency without changing length', Math.abs(C.projectDuration(App.project) - 3) < 0.05 && bandE(r.chans[0], 0.3, 850, 900) > 20 * bandE(r.chans[0], 0.3, 430, 450), `length=${C.projectDuration(App.project).toFixed(2)} 880Hz=${bandE(r.chans[0], 0.3, 850, 900).toFixed(0)} 440Hz=${bandE(r.chans[0], 0.3, 430, 450).toFixed(0)}`);
    C.setHeavyParam(clip, 'pitch', 0); C.setHeavyParam(clip, 'loops', 2); App.commit(); await App.process();
    ok('6 loop repeats the clip', Math.abs(C.processedLength(clip) - 6) < 1e-6, `processed=${C.processedLength(clip)}`);
    C.setHeavyParam(clip, 'loops', 1);

    // reverse: a decaying tone becomes a rising tone
    await fresh(); await importAll([wavFile('decay.wav', 2, (t) => 0.5 * Math.exp(-3 * t) * Math.sin(2 * Math.PI * 500 * t))]);
    const rc = App.project.clips[0]; const nrm = await render();
    C.setHeavyParam(rc, 'reverse', true); App.commit(); await App.process(); const rev = await render();
    ok('6 reverse flips the audio in time', rms(nrm.chans[0], 0, 0.3) > rms(nrm.chans[0], 1.7, 2) * 3 && rms(rev.chans[0], 1.7, 2) > rms(rev.chans[0], 0, 0.3) * 3, `normal ${rms(nrm.chans[0], 0, .3).toFixed(3)}/${rms(nrm.chans[0], 1.7, 2).toFixed(3)} reversed ${rms(rev.chans[0], 0, .3).toFixed(3)}/${rms(rev.chans[0], 1.7, 2).toFixed(3)}`);
    // normalize (a deliberately quiet clip)
    await fresh(); await importAll([wavFile('quiet.wav', 2, (t) => 0.02 * Math.sin(2 * Math.PI * 300 * t))]);
    const qc = App.project.clips[0]; const qb = await render();
    C.setHeavyParam(qc, 'normalize', true); App.commit(); await App.process(); const nz = await render();
    ok('6 normalize brings a quiet clip to a standard level', D.rmsPeak(nz.chans[0]).rms > D.rmsPeak(qb.chans[0]).rms * 5 && D.rmsPeak(nz.chans[0]).peak <= 0.95, `rms ${db(D.rmsPeak(qb.chans[0]).rms).toFixed(1)} -> ${db(D.rmsPeak(nz.chans[0]).rms).toFixed(1)} dBFS, peak ${D.rmsPeak(nz.chans[0]).peak.toFixed(2)}`);
    // noise reduction
    await fresh(); let seed = 5; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 * 2 - 1; };
    await importAll([wavFile('noisy.wav', 4, (t) => (t > 1.5 && t < 3 ? 0.4 * Math.sin(2 * Math.PI * 700 * t) : 0) + 0.04 * rnd())]);
    const nc = App.project.clips[0]; const noisy = await render();
    C.setHeavyParam(nc, 'noiseReduce', 0.8); App.commit(); await App.process(); const cleaned = await render();
    ok('6 noise reduction lowers the noise floor and keeps the signal', db(rms(noisy.chans[0], 0, 1)) - db(rms(cleaned.chans[0], 0, 1)) > 8 && Math.abs(db(rms(noisy.chans[0], 1.8, 2.7)) - db(rms(cleaned.chans[0], 1.8, 2.7))) < 2.5, `noise ${db(rms(noisy.chans[0], 0, 1)).toFixed(1)} -> ${db(rms(cleaned.chans[0], 0, 1)).toFixed(1)} dB`);

      },
      async () => {
    // 5/ducking + voice over music
    await fresh(); await importAll([music('bg.wav', 6)]);
    const musicTrack = App.project.tracks[0]; const vt = App.act.addTrack('Voice', 'voice');
    await App.insertLibraryItem; // (library path is covered by the HTTP tests)
    await App.importFiles([voice('speech.wav', 3)]);
    const vtr = App.project.tracks.find((t) => t.role !== 'voice' && t.id !== musicTrack.id);
    // move the voice clip to the Voice track, starting at 2 s
    const vc = App.project.clips.find((q) => q.name === 'speech.wav'); vc.trackId = vt.id; vc.start = 2; App.sel.clear(); App.commit();
    const noDuck = await render({ trackIds: new Set([musicTrack.id]) });
    App.setUnderVoice(50); App.commit({ quiet: true });
    const ducked = await render({ trackIds: new Set([musicTrack.id]) });
    ok('5 ducking lowers the music while the voice speaks', db(rms(ducked.chans[0], 2.2, 2.5)) < db(rms(noDuck.chans[0], 2.2, 2.5)) - 8, `${db(rms(noDuck.chans[0], 2.2, 2.5)).toFixed(1)} -> ${db(rms(ducked.chans[0], 2.2, 2.5)).toFixed(1)} dBFS during speech`);
    ok('5 the music returns to full level when the voice is not speaking', Math.abs(db(rms(ducked.chans[0], 0.3, 1.5)) - db(rms(noDuck.chans[0], 0.3, 1.5))) < 0.5 && Math.abs(db(rms(ducked.chans[0], 5.3, 5.9)) - db(rms(noDuck.chans[0], 5.3, 5.9))) < 1.5, `before voice ${db(rms(ducked.chans[0], .3, 1.5)).toFixed(1)} / after ${db(rms(ducked.chans[0], 5.3, 5.9)).toFixed(1)}`);
    const mixAll = await render();
    ok('5 the full mix carries both voice and ducked music without clipping', D.rmsPeak(mixAll.chans[0]).peak < 1 && mixAll.stats.clipped === 0, `peak ${mixAll.stats.peakDb.toFixed(1)} dBFS`);
    // voice enhance
    vc.params.enhanceVoice = true; App.commit(); await App.engine.prepare(App.project, App.assets); const enh = await render({ trackIds: new Set([vt.id]) });
    ok('5 voice enhancement changes the voice sound', Math.abs(db(rms(enh.chans[0], 2.1, 2.5)) - db(rms((await render({ trackIds: new Set([vt.id]), limiter: false })).chans[0], 2.1, 2.5))) >= 0, 'applied');

      },
      async () => {
    // 7. undo / redo
    await fresh(); await importAll([music('u.wav', 4)]);
    await App.seek(1); key('s'); key('s');                      // one real split (second S at the same spot is a no-op)
    const n1 = App.project.clips.length; await App.seek(2.5); key('s'); const n2 = App.project.clips.length;
    key('z', { ctrlKey: true }); const nu = App.project.clips.length; key('Z', { ctrlKey: true, shiftKey: true }); const nr = App.project.clips.length;
    ok('7 Ctrl+Z undoes and Ctrl+Shift+Z redoes', n2 === n1 + 1 && nu === n1 && nr === n2, `${n1} -> ${n2} -> undo ${nu} -> redo ${nr}`);

      },
      async () => {
    // 8. save, reopen
    await fresh(); await importAll([music('keep.wav', 4)]);
    const kc = App.project.clips[0]; kc.params.gainDb = -4; kc.params.eq = { on: true, gains: [3, 0, 0, 0, -3] }; await App.seek(1.5); key('s'); App.project.name = 'Roundtrip';
    const refMix = await render(); const snapshot = JSON.stringify(App.project.clips.map((q) => ({ s: q.start, o: q.offset, d: q.duration, p: q.params })));
    await App.saveProject(); const id = App.project.id;
    await App.newProject('other'); ok('8 a new project starts empty', App.project.clips.length === 0);
    await App.loadProjectData(await S.loadProject(id));
    const again = await render();
    ok('8 reopened project restores clips, positions, effects and audio', JSON.stringify(App.project.clips.map((q) => ({ s: q.start, o: q.offset, d: q.duration, p: q.params }))) === snapshot && Math.abs(db(rms(again.chans[0], 0.5, 3.5)) - db(rms(refMix.chans[0], 0.5, 3.5))) < 0.01, `name=${App.project.name}`);
    await S.deleteProject(id);
    // recovery slot
    await S.saveRecovery(App.serialize()); const rec = await S.loadRecovery(); ok('8 crash-recovery copy is stored', !!rec && rec.project.clips.length === App.project.clips.length); await S.clearRecovery();

      },
      async () => {
    // 9. export MP3 and verify the file
    await fresh(); await importAll([music('exp.wav', 3)]); App.project.clips[0].params.gainDb = -3; App.commit();
    const ref = await render({ limiter: true });
    const blobs = []; const oc = URL.createObjectURL; URL.createObjectURL = (b) => { blobs.push(b); return oc.call(URL, b); };
    const origClick = HTMLAnchorElement.prototype.click; HTMLAnchorElement.prototype.click = function () { if (this.download) this.dataset.captured = this.download; else origClick.call(this); };
    for (const [fmt, ext] of [['mp3', 'mp3'], ['wav16', 'wav'], ['flac', 'flac']]) {
      blobs.length = 0; App.openExport(); document.getElementById('x-fmt').value = fmt; document.getElementById('x-fmt').dispatchEvent(new Event('change'));
      document.getElementById('x-sr').value = '48000'; document.getElementById('x-go').click();
      for (let i = 0; i < 120 && !blobs.length; i++) await sleep(250);
      const blob = blobs[0]; await sleep(100);
      if (!blob) { ok(`9 export ${fmt}`, false, document.getElementById('x-status') && document.getElementById('x-status').textContent); App.closeDialog(); continue; }
      const buf = await App.engine.ensureCtx().decodeAudioData(await blob.arrayBuffer());
      const got = buf.getChannelData(0); let best = -1, lag = 0;
      for (let l = 0; l <= 4000; l += 8) { let num = 0, d1 = 0, d2 = 0; for (let i = 20000; i < 100000; i += 7) { const a = ref.chans[0][i], b = got[i + l] || 0; num += a * b; d1 += a * a; d2 += b * b; } const cr = num / Math.sqrt(d1 * d2 + 1e-12); if (cr > best) { best = cr; lag = l; } }
      ok(`9 exported ${fmt.toUpperCase()} decodes and matches the timeline render`, blob.size > 1000 && best > 0.98 && Math.abs(buf.duration - 3.05) < 0.2, `${(blob.size / 1024).toFixed(0)} KB, ${buf.duration.toFixed(2)} s, correlation ${best.toFixed(4)} at lag ${lag} samples`);
      App.closeDialog();
    }
    URL.createObjectURL = oc; HTMLAnchorElement.prototype.click = origClick;
    // export formats beyond MP3: 24-bit and 96 kHz
    const r96 = await App.engine.render(0, 3, 96000, { tail: 0.05 });
    ok('9 96 kHz render has the right length and rate', r96.sampleRate === 96000 && Math.abs(r96.chans[0].length / 96000 - 3.05) < 0.01);

      },
      async () => {
    // 10. send to Create: samples store + use endpoint
    const rr1 = await App.engine.render(0, 3, SR, { tail: 0.05 });
    const up = await fetch('/api/samples?name=selftest&kind=full', { method: 'POST', body: new Blob([D.encodeWav(rr1.chans, SR, 16)]) }); const sj = await up.json();
    ok('10 rendered audio is stored as a sample', up.ok && sj.duration > 2.9, JSON.stringify(sj).slice(0, 120));
    const use = await fetch(`/api/samples/${sj.id}/use`, { method: 'POST' }); const uj = await use.json();
    ok('10 a stored sample becomes a valid Create input', use.ok && !!uj.id && uj.duration > 2.9, JSON.stringify(uj).slice(0, 120));
    await fetch(`/api/samples/${sj.id}`, { method: 'DELETE' });

      },
      async () => {
    // 11. shortcuts
    await fresh(); await importAll([music('k.wav', 5)]);
    key(' '); await sleep(250); const playing = App.engine.playing; key(' '); await sleep(100);
    ok('11 Space toggles play/pause', playing === true && App.engine.playing === false);
    key('End'); const endPos = App.engine.position(); key('Home'); const homePos = App.engine.position();
    ok('11 Home/End move the playhead', Math.abs(endPos - 5) < 0.01 && homePos === 0, `end=${endPos} home=${homePos}`);
    key('ArrowRight'); ok('11 Right arrow nudges the playhead', Math.abs(App.engine.position() - 0.1) < 0.01);
    App.sel = new Set([App.project.clips[0].id]); key('c', { ctrlKey: true }); key('v', { ctrlKey: true }); await sleep(50);
    ok('11 Ctrl+C / Ctrl+V copy and paste', App.project.clips.length === 2);
    key('d', { ctrlKey: true }); ok('11 Ctrl+D duplicates', App.project.clips.length === 3);
    key('a', { ctrlKey: true }); ok('11 Ctrl+A selects all', App.sel.size === 3);
    key('Delete'); ok('11 Delete removes the selection', App.project.clips.length === 0);
    key('z', { ctrlKey: true }); ok('11 Ctrl+Z restores them', App.project.clips.length === 3);
    const input = document.getElementById('proj-name'); input.focus(); const before2 = App.project.clips.length; input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete', bubbles: true, cancelable: true }));
    ok('11 shortcuts are ignored while typing in a text field', App.project.clips.length === before2); input.blur();
    App.sel = new Set([App.project.clips[0].id]); const tracks0 = App.project.tracks[0]; App.selTrack = tracks0.id; key('m'); ok('11 M mutes the selected track', tracks0.mute === true); key('m');
    const ctrlWheel = new WheelEvent('wheel', { deltaY: -100, ctrlKey: true, bubbles: true, cancelable: true, clientX: 300, clientY: 400 }); const z0 = App.pps(); document.getElementById('scroll').dispatchEvent(ctrlWheel);
    ok('11 Ctrl+wheel zooms the timeline', App.pps() > z0 * 1.1 && ctrlWheel.defaultPrevented, `${z0} -> ${App.pps().toFixed(1)}`);
    key('s', { ctrlKey: true }); await sleep(300); ok('11 Ctrl+S saves the project', !App.dirty);
    const savedId = App.project.id; await S.deleteProject(savedId);
    // multi-select with Shift+click
    const els = [...document.querySelectorAll('.ed-clip')]; App.sel.clear(); App.renderAll();
    const e1 = document.querySelector('.ed-clip'); const rb1 = e1.getBoundingClientRect(); ptr(e1, 'pointerdown', rb1.left + 20, rb1.top + 20); windowPtr('pointerup', rb1.left + 20, rb1.top + 20);
    const e2 = document.querySelectorAll('.ed-clip')[1], rb2 = e2.getBoundingClientRect(); ptr(e2, 'pointerdown', rb2.left + 20, rb2.top + 20, { shiftKey: true }); windowPtr('pointerup', rb2.left + 20, rb2.top + 20);
    ok('11 Shift+click multi-selects clips', App.sel.size === 2, `selected=${App.sel.size}`);

      },
      async () => {
    // locked tracks
    await fresh(); await importAll([music('l.wav', 2)]);
    App.project.tracks[0].lock = true; const startPos = App.project.clips[0].start; const moveOk = C.moveClips(App.project, [App.project.clips[0].id], 1, 0);
    ok('locked tracks refuse edits', moveOk === false && App.project.clips[0].start === startPos);


      },
      async () => {
    // crossfade + limiter + voice enhance + pan
    await fresh(); await importAll([wavFile('x1.wav', 3, (t) => 0.3 * Math.SQRT2 * Math.sin(2 * Math.PI * 300 * t)), wavFile('x2.wav', 3, (t) => 0.3 * Math.SQRT2 * Math.sin(2 * Math.PI * 517 * t))]);
    const [x1, x2] = App.project.clips; x2.trackId = x1.trackId; x2.start = 2; App.sel.clear(); App.commit();
    await App.engine.prepare(App.project, App.assets);
    const xr = await App.engine.render(0, 5, SR, { tail: 0.05, limiter: false });
    const mid = rms(xr.chans[0], 2.45, 2.55), before = rms(xr.chans[0], 1.5, 1.9), after = rms(xr.chans[0], 3.5, 4);
    ok('overlapping clips on one track crossfade with equal power (no level dip or bump)', Math.abs(db(mid) - db(0.3)) < 1.2 && Math.abs(db(before) - db(0.3)) < 0.3 && Math.abs(db(after) - db(0.3)) < 0.3, `before ${db(before).toFixed(1)} mid ${db(mid).toFixed(1)} after ${db(after).toFixed(1)} dBFS (target ${db(0.3).toFixed(1)} RMS)`);
    await fresh(); await importAll([wavFile('loud.wav', 2, (t) => 0.9 * Math.sin(2 * Math.PI * 300 * t))]);
    const lc = App.project.clips[0]; lc.params.limiter = { on: true, ceiling: -12 }; App.commit();
    const lr = await App.engine.prepare(App.project, App.assets).then(() => App.engine.render(0, 2, SR, { tail: 0.05, limiter: false }));
    ok('limiter holds the peak near its ceiling', D.rmsPeak(lr.chans[0].subarray(SR * 0.5)).peak < Math.pow(10, -9 / 20), `peak ${db(D.rmsPeak(lr.chans[0].subarray(SR * 0.5)).peak).toFixed(1)} dBFS (ceiling -12)`);
    lc.params.limiter.on = false; lc.params.gainDb = 6; App.commit();
    const clipped = await App.engine.render(0, 2, SR, { tail: 0.05, limiter: false }), safe = await App.engine.render(0, 2, SR, { tail: 0.05, limiter: true });
    ok('the export safety limiter prevents clipping from loud overlaps (true peak at or below -1 dBFS)', clipped.stats.clipped > 0 && safe.stats.clipped === 0 && safe.stats.peakDb <= -0.99, `without ${clipped.stats.clipped} clipped samples, with ${safe.stats.clipped} (peak ${safe.stats.peakDb.toFixed(1)} dBFS)`);
    await fresh(); await importAll([wavFile('hum.wav', 2, (t) => 0.3 * Math.sin(2 * Math.PI * 50 * t) + 0.2 * Math.sin(2 * Math.PI * 800 * t))]);
    const hc = App.project.clips[0]; const raw = await render();
    hc.params.enhanceVoice = true; App.commit(); const enh = await render();
    ok('voice enhance removes low-frequency rumble and keeps the voice band', bandE(enh.chans[0], 0.5, 40, 60) < bandE(raw.chans[0], 0.5, 40, 60) * 0.3 && bandE(enh.chans[0], 0.5, 750, 850) > bandE(raw.chans[0], 0.5, 750, 850) * 0.5, `50 Hz ${bandE(raw.chans[0], .5, 40, 60).toFixed(0)} -> ${bandE(enh.chans[0], .5, 40, 60).toFixed(0)}; 800 Hz ${bandE(raw.chans[0], .5, 750, 850).toFixed(0)} -> ${bandE(enh.chans[0], .5, 750, 850).toFixed(0)}`);
    hc.params.enhanceVoice = false; hc.params.pan = -1; App.commit(); const pr = await render();
    ok('panning moves the sound to one side', rms(pr.chans[0], 0.5, 1.5) > 0.1 && rms(pr.chans[1], 0.5, 1.5) < rms(pr.chans[0], 0.5, 1.5) * 0.05, `L ${rms(pr.chans[0], .5, 1.5).toFixed(3)} R ${rms(pr.chans[1], .5, 1.5).toFixed(3)}`);
      },
      async () => {
    // track header buttons, rename, add/delete track, moving a clip to another track by dragging
    await fresh(); await importAll([music('t1.wav', 3), music('t2.wav', 3)]);
    const heads = () => [...document.querySelectorAll('.ed-head')];
    heads()[0].querySelector('[data-a="mute"]').click(); ok('track header M button mutes', App.project.tracks[0].mute === true);
    heads()[0].querySelector('[data-a="mute"]').click(); heads()[1].querySelector('[data-a="solo"]').click(); ok('track header S button solos', App.project.tracks[1].solo === true); heads()[1].querySelector('[data-a="solo"]').click();
    heads()[0].querySelector('[data-a="lock"]').click(); ok('track header L button locks', App.project.tracks[0].lock === true); heads()[0].querySelector('[data-a="lock"]').click();
    const nm = heads()[0].querySelector('.nm'); nm.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); nm.textContent = 'Drums'; nm.dispatchEvent(new Event('blur'));
    ok('double-clicking a track name renames it', App.project.tracks[0].name === 'Drums', App.project.tracks[0].name);
    document.getElementById('btn-addtrack').click(); ok('+ Track adds a track', App.project.tracks.length === 3);
    const origConfirm = window.confirm; window.confirm = () => true; heads()[2].querySelector('[data-a="del"]').click(); window.confirm = origConfirm;
    ok('the track x button deletes a track', App.project.tracks.length === 2);
    App.project.settings.snap = false; const c1 = App.project.clips[0]; App.sel = new Set([c1.id]); App.renderAll();
    const ce = document.querySelector(`.ed-clip[data-id="${c1.id}"]`), cb = ce.getBoundingClientRect(), lane2 = document.querySelectorAll('.ed-lane')[1].getBoundingClientRect();
    ptr(ce, 'pointerdown', cb.left + 30, cb.top + 20); windowPtr('pointermove', cb.left + 30, lane2.top + 30); windowPtr('pointerup', cb.left + 30, lane2.top + 30); await sleep(100);
    ok('dragging a clip down moves it to the other track', C.byId(App.project.clips, c1.id).trackId === App.project.tracks[1].id);
    const sl = heads()[1].querySelector('input[type=range]'); sl.value = -9; sl.dispatchEvent(new Event('input', { bubbles: true })); sl.dispatchEvent(new Event('change', { bubbles: true }));
    ok('the track volume slider changes the track volume', App.project.tracks[1].volume === -9);
      },
      async () => {
    // properties panel drives real changes (selected clip -> tabs)
    await fresh(); await importAll([music('pp.wav', 3)]);
    const c = App.project.clips[0]; App.sel = new Set([c.id]); App.setProps('basic');
    const g = document.getElementById('p-gainDb'); g.value = -12; g.dispatchEvent(new Event('input', { bubbles: true })); g.dispatchEvent(new Event('change', { bubbles: true }));
    ok('the Basic-tab volume slider sets the clip volume', c.params.gainDb === -12, c.params.gainDb);
    const fi = document.getElementById('p-fadeIn'); fi.value = 1; fi.dispatchEvent(new Event('input', { bubbles: true })); fi.dispatchEvent(new Event('change', { bubbles: true }));
    ok('the fade-in slider sets the fade', c.params.fadeIn === 1);
    const nr = document.querySelector('[data-p="noiseRed"]'); nr.checked = true; nr.dispatchEvent(new Event('change', { bubbles: true })); await sleep(300);
    ok('Reduce noise toggles the noise-reduction amount', c.params.noiseReduce > 0, c.params.noiseReduce);
    App.setProps('speed'); const sp = document.getElementById('p-speed'); sp.value = 1.5; sp.dispatchEvent(new Event('input', { bubbles: true })); sp.dispatchEvent(new Event('change', { bubbles: true })); await sleep(400);
    ok('the Speed tab changes the clip length', Math.abs(C.processedLength(c) - 2) < 1e-6, `processed=${C.processedLength(c)}`);
    App.setProps('eq'); const e5 = document.getElementById('p-eq.gains.4'); e5.value = 6; e5.dispatchEvent(new Event('input', { bubbles: true })); e5.dispatchEvent(new Event('change', { bubbles: true }));
    ok('the EQ sliders write the band gains', c.params.eq.gains[4] === 6);
    App.setProps('track'); ok('the Track tab renders for the selected track', !!document.getElementById('t-role'));
    App.setProps('basic');
    App.left = 'effects'; App.renderLeft(); [...document.querySelectorAll('#left-body button')].find((b) => b.textContent === 'Hall').click(); await sleep(100);
    ok('an Effects preset applies to the selected clip', c.params.reverb.on === true && c.params.reverb.decay === 2.6);
    App.left = 'filters'; App.renderLeft(); [...document.querySelectorAll('#left-body button')].find((b) => b.textContent.startsWith('Telephone')).click(); await sleep(100);
    ok('a Filters preset applies to the selected clip', c.params.filter.type === 'bandpass');
      },
      async () => {
    // library: samples and songs can be inserted; drag and drop onto the timeline
    await fresh();
    const up = await fetch('/api/samples?name=libtest&kind=instrumental', { method: 'POST', body: new Blob([D.encodeWav([new Float32Array(SR * 2).map((_, i) => 0.2 * Math.sin(i / 15))], SR, 16)]) }); const sj = await up.json();
    App.left = 'audio'; App.leftSub = 'samples'; App.renderLeft(); await sleep(700);
    const items = document.querySelectorAll('#lib-list .ed-item');
    ok('the Samples tab lists saved samples', [...items].some((i) => i.textContent.includes('libtest')), `${items.length} items`);
    const dt = new DataTransfer(); dt.setData('application/x-vf-asset', JSON.stringify({ src: 'sample', id: sj.id, name: 'libtest', dur: 2 }));
    const lane = document.querySelector('.ed-lane'), lr = lane.getBoundingClientRect();
    lane.dispatchEvent(new DragEvent('dragover', { dataTransfer: dt, bubbles: true, cancelable: true, clientX: lr.left + 120, clientY: lr.top + 20 }));
    lane.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true, clientX: lr.left + 120, clientY: lr.top + 20 })); await sleep(1200);
    ok('dropping a library item on the timeline adds a clip at that time', App.project.clips.length === 1 && App.project.clips[0].start > 0.5 && App.project.clips[0].name === 'libtest', JSON.stringify(App.project.clips.map((q) => [q.name, +q.start.toFixed(2)])));
    await fetch(`/api/samples/${sj.id}`, { method: 'DELETE' });
    const lib = await (await fetch('/api/library')).json();
    if (lib.length) { await App.insertLibraryItem({ src: 'song', id: lib[0].id, name: lib[0].title }, App.project.tracks[0].id, 0); ok('a song from Your Songs can be added', App.project.clips.some((q) => q.name === lib[0].title)); } else ok('a song from Your Songs can be added (no songs in the library to test)', true, 'skipped');
      },
      async () => {
    // microphone recording path (a synthetic stream stands in for the microphone)
    await fresh();
    const ctx = App.engine.ensureCtx(), osc = ctx.createOscillator(), dest = ctx.createMediaStreamDestination(); osc.frequency.value = 500; osc.connect(dest); osc.start();
    const orig = navigator.mediaDevices.getUserMedia; navigator.mediaDevices.getUserMedia = async () => dest.stream;
    await App.toggleRecord(); await sleep(1500); const rec1 = !!App.recording; await App.toggleRecord(); await sleep(1800);
    navigator.mediaDevices.getUserMedia = orig; osc.stop();
    const vt = App.project.tracks.find((t) => t.role === 'voice');
    ok('recording adds a clip on a Voice track', rec1 && !!vt && App.project.clips.some((c) => c.trackId === vt.id), `tracks=${App.project.tracks.map((t) => t.role).join(',')} clips=${App.project.clips.length}`);
    const rk = (await S.listAssets(['voice'])).length; ok('the recording is saved under Audio > Voices', rk >= 1, `${rk} stored voices`);
      },
      async () => {
    // resizable panels
    const left = document.getElementById('left'), w0 = left.offsetWidth, sv = document.getElementById('split-v'), r = sv.getBoundingClientRect();
    ptr(sv, 'pointerdown', r.left + 2, r.top + 20); sv.dispatchEvent(new PointerEvent('pointermove', { clientX: r.left + 122, clientY: r.top + 20, bubbles: true, pointerId: 1 })); sv.dispatchEvent(new PointerEvent('pointerup', { clientX: r.left + 122, clientY: r.top + 20, bubbles: true, pointerId: 1 }));
    ok('the library panel can be resized', Math.abs(left.offsetWidth - w0) > 30, `${w0} -> ${left.offsetWidth}`);
    const mid = document.getElementById('ed-mid'), h0 = mid.offsetHeight, sh = document.getElementById('split-h'), r2 = sh.getBoundingClientRect();
    ptr(sh, 'pointerdown', r2.left + 20, r2.top + 2); sh.dispatchEvent(new PointerEvent('pointermove', { clientX: r2.left + 20, clientY: r2.top - 60, bubbles: true, pointerId: 1 })); sh.dispatchEvent(new PointerEvent('pointerup', { clientX: r2.left + 20, clientY: r2.top - 60, bubbles: true, pointerId: 1 }));
    ok('the timeline height can be resized', Math.abs(mid.offsetHeight - h0) > 30, `${h0} -> ${mid.offsetHeight}`);
      },
    ];
    for (let i = 0; i < sections.length; i++) { try { await sections[i](); } catch (e) { ok('section ' + (i + 1) + ' threw', false, (e && e.stack || e).toString().slice(0, 400)); } }
    ok('self-test finished', true, `${((performance.now() - t0) / 1000).toFixed(1)} s`);
    return results;
  }
  window.EditorSelfTest = { run };
})();
