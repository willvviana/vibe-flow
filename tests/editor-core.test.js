const test = require('node:test');
const assert = require('node:assert');
const C = require('../editor/core.js');

const asset = (id = 'a1', d = 10) => ({ id, name: id + '.wav', duration: d });
function proj() {
  const p = C.newProject('t');
  C.addTrack(p, 'Voice', 'voice');
  return { p, h: new C.History(p) };
}

test('add clip, split at the playhead keeps total audible length and source continuity', () => {
  const { p } = proj();
  const c = C.addClip(p, asset(), p.tracks[0].id, 2);
  const [l, r] = C.splitClip(p, c.id, 5);
  assert.equal(l.start, 2); assert.equal(l.duration, 3);
  assert.equal(r.start, 5); assert.equal(r.offset, 3); assert.equal(r.duration, 7);
  assert.equal(l.duration + r.duration, 10);
  assert.equal(C.splitClip(p, l.id, 2.001), null, 'cannot split at the very edge');
  assert.equal(C.splitClip(p, l.id, 99), null);
  assert.equal(p.clips.length, 2);
});

test('trim start/end are non-destructive and limited to the processed audio', () => {
  const { p } = proj();
  const c = C.addClip(p, asset(), p.tracks[0].id, 4);
  assert.ok(C.trimStart(p, c.id, 6));
  assert.equal(c.start, 6); assert.equal(c.offset, 2); assert.equal(c.duration, 8);
  assert.ok(C.trimEnd(p, c.id, 9));
  assert.equal(c.duration, 3);
  C.trimEnd(p, c.id, 999);                                   // cannot extend past the processed buffer (10 s - 2 s offset)
  assert.equal(c.duration, 8);
  C.trimStart(p, c.id, 0);                                   // cannot reveal audio before the buffer start: start - offset = 4
  assert.equal(c.start, 4); assert.equal(c.offset, 0);
});

test('move clips horizontally and between tracks; locked tracks and bounds are respected', () => {
  const { p } = proj();
  const [t0, t1] = p.tracks;
  const a = C.addClip(p, asset(), t0.id, 1), b = C.addClip(p, asset('a2'), t0.id, 12);
  C.moveClips(p, [a.id, b.id], -5, 0);
  assert.equal(a.start, 0); assert.equal(b.start, 11);          // the group stops at 0 together (keeps relative spacing)
  C.moveClips(p, [a.id], 0, 1);
  assert.equal(a.trackId, t1.id);
  C.moveClips(p, [a.id], 0, 5);                                 // clamped to the last track
  assert.equal(a.trackId, t1.id);
  t1.lock = true;
  assert.equal(C.moveClips(p, [a.id], 3, 0), false);
  assert.equal(C.deleteClips(p, [a.id]), false, 'locked tracks cannot be edited');
  assert.equal(C.splitClip(p, a.id, 3), null);
});

test('duplicate, copy/paste and delete', () => {
  const { p } = proj();
  const [t0, t1] = p.tracks;
  const a = C.addClip(p, asset(), t0.id, 0);
  const b = C.addClip(p, asset('a2', 4), t1.id, 2);
  const [d] = C.duplicateClips(p, [a.id]);
  assert.equal(d.start, 10); assert.notEqual(d.id, a.id);
  const cb = C.copyClips(p, [a.id, b.id]);
  const pasted = C.pasteClips(p, cb, 20, t0.id);
  assert.equal(pasted.length, 2);
  assert.equal(pasted[0].start, 20); assert.equal(pasted[1].start, 22);
  assert.equal(pasted[0].trackId, t0.id); assert.equal(pasted[1].trackId, t1.id);   // relative track layout preserved
  assert.ok(C.deleteClips(p, pasted.map((x) => x.id)));
  assert.equal(p.clips.length, 3);
});

test('undo / redo restore the exact timeline and coalesce no-op commits', () => {
  const { p, h } = proj();
  assert.equal(h.commit(), false);
  const c = C.addClip(p, asset(), p.tracks[0].id, 1); h.commit();
  C.moveClips(p, [c.id], 4, 0); h.commit();
  C.splitClip(p, c.id, 8); h.commit();
  assert.equal(p.clips.length, 2);
  assert.ok(h.undo()); assert.equal(p.clips.length, 1); assert.equal(p.clips[0].start, 5);
  assert.ok(h.undo()); assert.equal(p.clips[0].start, 1);
  assert.ok(h.undo()); assert.equal(p.clips.length, 0);
  assert.equal(h.undo(), false);
  assert.ok(h.redo() && h.redo() && h.redo()); assert.equal(p.clips.length, 2);
  h.undo();
  C.addTrack(p); h.commit();                                    // a new action discards the redo branch
  assert.equal(h.canRedo(), false);
});

test('heavy parameter changes rescale offset/duration so the audible part stays the same', () => {
  const { p } = proj();
  const c = C.addClip(p, asset(), p.tracks[0].id, 0);
  C.trimStart(p, c.id, 2); C.trimEnd(p, c.id, 6);             // audible 2..6 of 10 s
  C.setHeavyParam(c, 'speed', 2);                              // processed buffer is now 5 s long
  assert.equal(C.processedLength(c), 5);
  assert.ok(Math.abs(c.offset - 1) < 1e-9 && Math.abs(c.duration - 2) < 1e-9);
  C.setHeavyParam(c, 'loops', 3);
  assert.equal(C.processedLength(c), 15);
});

test('crossfade lengths come from overlaps on the same track only', () => {
  const { p } = proj();
  const [t0, t1] = p.tracks;
  const a = C.addClip(p, asset(), t0.id, 0);                    // 0..10
  const b = C.addClip(p, asset('a2'), t0.id, 8);                // 8..18 overlaps by 2
  const c = C.addClip(p, asset('a3'), t1.id, 8);                // other track: layered, not crossfaded
  const x = C.crossfades(p);
  assert.equal(x[a.id].fadeOut, 2); assert.equal(x[b.id].fadeIn, 2); assert.equal(x[c.id], undefined);
  p.settings.crossfade = false;
  assert.deepEqual(C.crossfades(p), {});
});

test('solo overrides mute; muted tracks are inaudible', () => {
  const { p } = proj();
  p.tracks[0].mute = true;
  assert.deepEqual([...C.audibleTrackIds(p)], [p.tracks[1].id]);
  p.tracks[0].solo = true;
  assert.deepEqual([...C.audibleTrackIds(p)], [p.tracks[0].id]);
});

test('snapping: clip edges, playhead, grid, and disabled', () => {
  const { p } = proj();
  const a = C.addClip(p, asset(), p.tracks[0].id, 5);          // edges at 5 and 15
  assert.deepEqual(C.snapTime(p, 5.05, { pxPerSec: 100 }), { t: 5, snapped: true });
  assert.deepEqual(C.snapTime(p, 14.95, { pxPerSec: 100 }), { t: 15, snapped: true });
  assert.deepEqual(C.snapTime(p, 9, { pxPerSec: 100, playhead: 9.04 }), { t: 9.04, snapped: true });
  assert.equal(C.snapTime(p, 9, { pxPerSec: 100 }).snapped, false);
  assert.deepEqual(C.snapTime(p, 5.05, { pxPerSec: 100, exclude: new Set([a.id]) }), { t: 5.05, snapped: false });
  assert.equal(C.snapTime(p, 5.05, { pxPerSec: 100, enabled: false }).t, 5.05);
  assert.equal(C.snapTime(p, 2.51, { pxPerSec: 100, grid: 0.5 }).t, 2.5);
});

test('ducking gain falls during speech with attack/release, and recovers', () => {
  const rate = 100, n = 600;
  const act = new Float32Array(n).fill(-80);
  for (let i = 200; i < 400; i++) act[i] = -20;                // speech from 2 s to 4 s
  const g = C.duckGain(act, rate, { amountDb: -12, thresholdDb: -40, attack: 0.05, release: 0.3, hold: 0.1 });
  assert.ok(g[100] > 0.99);
  assert.ok(g[300] < 0.27 && g[300] > 0.24, 'about -12 dB (0.251) while speaking: ' + g[300]);
  assert.ok(g[205] > g[260], 'smooth attack, not a step');
  assert.ok(g[590] > 0.97, 'recovered after the release time');
});

test('silence regions keep only the audible pieces with correct timeline positions', () => {
  const { p } = proj();
  const c = C.addClip(p, asset(), p.tracks[0].id, 3);
  const out = C.keepRegions(p, c.id, [[0, 2], [4, 7], [9.995, 10]]);
  assert.equal(out.length, 2);                                  // the 5 ms sliver is dropped
  assert.equal(out[0].start, 3); assert.equal(out[1].start, 7); assert.equal(out[1].offset, 4); assert.equal(out[1].duration, 3);
  assert.equal(p.clips.length, 2);
});

test('keyboard shortcuts: matching, configurability, and text-field guard', () => {
  const m = C.DEFAULT_SHORTCUTS;
  const ev = (key, o = {}) => ({ key, ctrlKey: !!o.ctrl, shiftKey: !!o.shift, altKey: false, metaKey: false });
  assert.equal(C.actionFor(ev(' '), m), 'playPause');
  assert.equal(C.actionFor(ev('z', { ctrl: true }), m), 'undo');
  assert.equal(C.actionFor(ev('Z', { ctrl: true, shift: true }), m), 'redo');
  assert.equal(C.actionFor(ev('s'), m), 'split');
  assert.equal(C.actionFor(ev('s', { ctrl: true }), m), 'save');
  assert.equal(C.actionFor(ev('Delete'), m), 'delete');
  assert.equal(C.actionFor(ev('ArrowLeft'), m), 'left');
  assert.equal(C.actionFor(ev('q'), m), null);
  assert.equal(C.actionFor(ev('s'), { ...m, split: 'X', save: 'Ctrl+S' }), null);
  assert.equal(C.actionFor(ev('x'), { ...m, split: 'X' }), 'split');                 // rebinding works
  assert.equal(C.isTextTarget({ tagName: 'INPUT', type: 'text' }), true);
  assert.equal(C.isTextTarget({ tagName: 'INPUT', type: 'range' }), false);
  assert.equal(C.isTextTarget({ tagName: 'DIV', isContentEditable: true }), true);
  assert.equal(C.isTextTarget({ tagName: 'DIV' }), false);
});

test('removing a track removes its clips and ducking references; the last track is protected', () => {
  const { p } = proj();
  const [t0, t1] = p.tracks;
  C.addClip(p, asset(), t1.id, 0);
  t0.duck = { on: true, sources: [t1.id], amountDb: -12 };
  assert.ok(C.removeTrack(p, t1.id));
  assert.equal(p.clips.length, 0); assert.deepEqual(t0.duck.sources, []);
  assert.equal(C.removeTrack(p, t0.id), false);
});
