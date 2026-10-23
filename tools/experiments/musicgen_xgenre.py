"""Cross-genre blinded comparison of MusicGen profiles A ("baseline") and D ("d").

  python tools/experiments/musicgen_xgenre.py <out_dir> <key_file> [seeds=11,12,13] [seconds=15]

Uses the PRODUCTION path for every clip: prompting.validate(settings incl. "profile") -> build_prompt(c, ..., c["prompt_style"])
-> generate._infer(c) with the profile's temperature / top-k / guidance. Per genre the *structured inputs* (genres, moods,
instruments, BPM) are identical for A and D; the prompt wording differs only because the profile defines the prompt style
(that is part of what is being compared). Same seed, duration, model, threads, model load.

Clips are written as shuffled neutral names g01.wav ... ; the full record (profile, prompt, seed, duration, time, parameters,
automated measures) goes to <key_file>, which must be kept away from the listener."""
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
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "11,12,13").split(",")]
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 15
os.makedirs(out_dir, exist_ok=True)

GENRES = {
    "folk":      dict(genres=["pop", "folk"], moods=["calm", "warm"], bpm=72, instruments=["acoustic guitar", "bass", "percussion"]),
    "rock":      dict(genres=["rock"], moods=["energetic"], bpm=120, instruments=["electric guitar", "bass", "drums"]),
    "synthwave": dict(genres=["synthwave", "electronic"], moods=["dreamy", "energetic"], bpm=100, instruments=["synthesizer", "bass", "drums"]),
    "hiphop_rnb": dict(genres=["hip hop", "r&b"], moods=["mellow"], bpm=88, instruments=["piano", "bass", "drums"]),
    "jazz":      dict(genres=["jazz"], moods=["mellow"], bpm=120, instruments=["piano", "bass", "drums", "saxophone"]),
    "cinematic": dict(genres=["cinematic"], moods=["epic", "uplifting"], bpm=90, instruments=["strings", "brass", "percussion"]),
}
PROFILES = {"A": "baseline", "D": "d"}

spec = registry.get("musicgen-small")
proc, model = generate._load(spec, "cpu", lambda **k: None)
torch.set_num_threads(generate._threads())
jobs = [(g, s, p) for g in GENRES for s in seeds for p in PROFILES]
random.Random(20261022).shuffle(jobs)            # blind order and names
proc_ps = psutil.Process()
key, rows = {}, []
for n, (genre, seed, label) in enumerate(jobs, 1):
    c = prompting.validate({"prompt": "", "duration": secs, "profile": PROFILES[label], **GENRES[genre]})
    text = prompting.build_prompt(c, spec.id, c["prompt_style"])
    torch.manual_seed(seed)
    t0 = time.time()
    wav = generate._infer(proc, model, "cpu", text, secs, 32000, c, lambda **k: None, 0, 0)
    gen_s = time.time() - t0
    peak = float(np.abs(wav).max())
    wav = wav * min(1.0, 0.99 / max(peak, 1e-6))          # same level-safety rule as the app
    name = f"g{n:02d}.wav"
    path = os.path.join(out_dir, name)
    sf.write(path, wav.astype("float32"), 32000, subtype="PCM_16")
    r = ev.evaluate(path, asr=False)
    mi = proc_ps.memory_info()
    rec = {"genre": genre, "seed": seed, "profile_label": label, "profile": PROFILES[label], "prompt": text, "duration_s": secs,
           "model": spec.id, "temperature": c["temperature"], "top_k": c["top_k"], "guidance": c["guidance"], "prompt_style": c["prompt_style"],
           "requested_bpm": GENRES[genre]["bpm"], "gen_s": round(gen_s, 1), "rtf": round(gen_s / secs, 2),
           "peak_ram_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2), "sha": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
           "median_bpm": r["rhythm"].get("median_bpm"), "bpm_iqr": r["rhythm"].get("bpm_iqr"), "pulse": r["rhythm"].get("pulse_strength"),
           "rms_dbfs": r["technical"]["rms_dbfs"], "silent": r["technical"]["silent_frame_ratio"], "clipped_before_scaling": bool(peak > 0.99)}
    key[name] = rec
    rows.append({"clip": name, **rec})
    print(json.dumps({"clip": name, "genre": genre, "seed": seed, "p": label, "gen_s": rec["gen_s"]}), flush=True)
    json.dump({"key": key, "seconds": secs, "seeds": seeds, "genres": GENRES, "profiles": PROFILES}, open(key_file, "w"), indent=1)
