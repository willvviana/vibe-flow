"""Controlled, blinded A/B for the MusicGen default: A (current default) vs D, many seeds.

  python tools/experiments/musicgen_ab.py <out_dir> <key_file> [seeds=1-8] [seconds=15]

Everything except the two settings under test is identical: same text content (genres/moods/instruments/BPM), duration,
model, thread count, seed list. A = keyword prompt, T 1.0, top-k 250, guidance 3.0.
D = sentence prompt, T 0.8, top-k 250, guidance 5.0 (exactly the configs used for the earlier samples/musicgen clips).
Clips are written with shuffled neutral names (01.wav ...) so listening is blind; the mapping goes to <key_file>.
Also reports determinism (re-generating a seed must give identical audio), per-clip time and peak memory."""
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import psutil
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eval_audio as ev  # noqa: E402
from engine import generate, prompting, registry  # noqa: E402

out_dir, key_file = sys.argv[1], sys.argv[2]
a, _, b = (sys.argv[3] if len(sys.argv) > 3 else "1-8").partition("-")
seeds = list(range(int(a), int(b or a) + 1))
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 15
os.makedirs(out_dir, exist_ok=True)

CONFIGS = {
    "A": ("keywords", dict(temperature=1.0, top_k=250, guidance=3.0)),
    "D": ("sentence", dict(temperature=0.8, top_k=250, guidance=5.0)),
}
spec = registry.get("musicgen-small")
proc, model = generate._load(spec, "cpu", lambda **k: None)
torch.set_num_threads(generate._threads())
base = prompting.validate({"prompt": "", "genres": ["pop", "folk"], "moods": ["calm", "warm"], "bpm": 72, "duration": secs,
                           "instruments": ["acoustic guitar", "bass", "percussion"]})
jobs = [(cfg, s) for s in seeds for cfg in CONFIGS]
random.Random(2026).shuffle(jobs)                 # blind order and names
key, rows = {}, []
p = psutil.Process()
for n, (cfg, seed) in enumerate(jobs, 1):
    style, c = CONFIGS[cfg]
    text = prompting.build_prompt(base, spec.id, style)
    torch.manual_seed(seed)
    t0 = time.time()
    wav = generate._infer(proc, model, "cpu", text, secs, 32000, c, lambda **k: None, 0, 0)
    gen_s = time.time() - t0
    wav = wav * min(1.0, 0.99 / max(float(np.abs(wav).max()), 1e-6))     # same level-safety rule as the app
    name = f"{n:02d}.wav"
    path = os.path.join(out_dir, name)
    sf.write(path, wav.astype("float32"), 32000, subtype="PCM_16")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    r = ev.evaluate(path, asr=False)
    key[name] = {"config": cfg, "seed": seed, "prompt": text, "sha": sha}
    mi = p.memory_info()
    rows.append({"clip": name, "config": cfg, "seed": seed, "gen_s": round(gen_s, 1), "rtf": round(gen_s / secs, 2),
                 "peak_ram_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2), **r["rhythm"],
                 "rms_dbfs": r["technical"]["rms_dbfs"], "silent": r["technical"]["silent_frame_ratio"],
                 "clipped": r["technical"]["clipped_samples"]})
    print(json.dumps(rows[-1]), flush=True)
json.dump({"key": key, "rows": rows, "seconds": secs, "configs": {k: {"style": v[0], **v[1]} for k, v in CONFIGS.items()}},
          open(key_file, "w"), indent=1)
