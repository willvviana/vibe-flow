"""Compare ways of joining a MusicGen continuation to the earlier audio, using objective seam metrics.

Metrics (computed around the join point):
  click   = largest sample-to-sample step within +-10 ms, relative to the 99.9th percentile step of the whole clip
            (>>1 means an audible click/discontinuity)
  jump_db = RMS of the 150 ms after the join minus the 150 ms before it (large |value| = abrupt loudness change)
  skip_ms = how much musical time the join removed (0 for a time-aligned join)
Methods:  hard (append new audio, no fade) | old (fade end of old into start of new; skips time) | aligned (current).
This measures continuity, not musicality: it cannot tell whether a result sounds good."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import audio, generate, registry  # noqa: E402

sr = 32000
spec = registry.get("musicgen-small")
proc, model = generate._load(spec, "cpu", lambda **k: None)
torch.set_num_threads(generate._threads())
c = {"guidance": 3.0, "temperature": 1.0}
nop = lambda **k: None  # noqa: E731


def metrics(x, seam):
    w = int(0.010 * sr)
    d = np.abs(np.diff(x))
    ref = np.percentile(d, 99.9) + 1e-9
    click = float(d[max(seam - w, 0): seam + w].max() / ref)
    r = lambda s: np.sqrt(np.mean(s ** 2) + 1e-12)  # noqa: E731
    n = int(0.15 * sr)
    jump = 20 * np.log10(r(x[seam: seam + n]) / r(x[seam - n: seam]))
    return {"click": round(click, 2), "jump_db": round(float(jump), 2)}


results = []
for seed, prompt in [(3, "funk groove, instrumental"), (8, "ambient piano, calm, instrumental"), (21, "synthwave, driving bass, instrumental")]:
    torch.manual_seed(seed)
    a = generate._infer(proc, model, "cpu", prompt, 8, sr, c, nop, 0, 0)
    ctx = audio.tail(a, sr, 4)
    out = generate._infer(proc, model, "cpu", prompt, 4, sr, c, nop, 0, 0, ref=ctx)
    step = int(4 * sr)
    hard = np.concatenate([a, out[len(ctx):][:step]])
    old = audio.crossfade_concat(a, out[len(ctx):][: step + int(0.1 * sr)], sr, 0.1)
    n_ov = int(0.25 * sr)
    aligned = audio.crossfade_concat(a, out[len(ctx) - n_ov:][: n_ov + step], sr, n_ov / sr)
    row = {"seed": seed}
    for name, x, seam in (("hard", hard, len(a)), ("old", old, len(a) - int(0.1 * sr)), ("aligned", aligned, len(a) - n_ov // 2)):
        row[name] = {**metrics(x, seam), "len_s": round(len(x) / sr, 3), "analysis": audio.analyze(x)["clipped_samples"]}
    results.append(row)
    print(json.dumps(row), flush=True)
