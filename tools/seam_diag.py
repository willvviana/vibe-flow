"""Diagnose how MusicGen continuation output relates to its audio prompt (does it include / re-decode the prompt?)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import audio, generate, registry  # noqa: E402

spec = registry.get("musicgen-small")
proc, model = generate._load(spec, "cpu", lambda **k: None)
sr = 32000
c = {"guidance": 3.0, "temperature": 1.0}
torch.manual_seed(3)
base = generate._infer(proc, model, "cpu", "funk groove, instrumental", 6, sr, c, lambda **k: None, 0, 0)
ctx = audio.tail(base, sr, 4)
out = generate._infer(proc, model, "cpu", "funk groove, instrumental", 3, sr, c, lambda **k: None, 0, 0, ref=ctx)
print("ctx_s", len(ctx) / sr, "out_s", round(len(out) / sr, 2), "expected_if_includes_prompt", 4 + 3, "expected_if_new_only", 3)
# does out[:len(ctx)] resemble ctx? correlation at best lag within +-50 ms
n = min(len(ctx), len(out))
a, b = ctx[:n], out[:n]
best = max(((np.corrcoef(a[max(0, -l):n - max(0, l)], b[max(0, l):n - max(0, -l)])[0, 1], l) for l in range(-1600, 1601, 8)))
print("prompt-vs-output-start correlation %.3f at lag %d samples" % best)
