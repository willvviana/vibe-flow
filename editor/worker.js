/* DSP worker: runs the heavy, buffer-changing operations off the UI thread (see editor/dsp.js processClip). */
importScripts('dsp.js');
self.onmessage = (e) => {
  const { id, chans, sr, srcStart, srcDur, params } = e.data;
  try {
    const out = self.EditorDSP.processClip(chans, sr, srcStart, srcDur, params);
    self.postMessage({ id, ok: true, chans: out }, out.map((c) => c.buffer));
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
