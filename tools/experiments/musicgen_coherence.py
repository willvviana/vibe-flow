"""A/B experiment for the instrumental engine: prompt style x sampling on the acceptance prompt (MusicGen Small only;
larger MusicGen checkpoints don't fit this machine: medium ships only pickle .bin weights, 7.5 GB).

  python tools/experiments/musicgen_coherence.py <out_dir> [seconds=15]
Writes <out_dir>/<name>.wav and prints one JSON line per clip with objective proxies (see tools/eval_audio.py).
Listening is still required: the proxies cannot say which clip sounds like a band."""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eval_audio as ev  # noqa: E402
from engine import audio, generate, prompting, registry  # noqa: E402

out_dir = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 15
os.makedirs(out_dir, exist_ok=True)
spec = registry.get("musicgen-small")
proc, model = generate._load(spec, "cpu", lambda **k: None)
torch.set_num_threads(generate._threads())
base = prompting.validate({"prompt": "", "genres": ["pop", "folk"], "moods": ["calm", "warm"], "bpm": 72, "duration": secs,
                           "instruments": ["acoustic guitar", "bass", "percussion"]})
configs = [
    ("A_keywords_t1.0", "keywords", dict(temperature=1.0, top_k=250, guidance=3.0)),
    ("B_sentence_t1.0", "sentence", dict(temperature=1.0, top_k=250, guidance=3.0)),
    ("C_sentence_t0.8_k100", "sentence", dict(temperature=0.8, top_k=100, guidance=3.0)),
    ("D_sentence_t0.8_g5", "sentence", dict(temperature=0.8, top_k=250, guidance=5.0)),
]
for name, style, c in configs:
    for seed in (1, 2):
        torch.manual_seed(seed)
        text = prompting.build_prompt(base, spec.id, style)
        wav = generate._infer(proc, model, "cpu", text, secs, 32000, c, lambda **k: None, 0, 0)
        p = os.path.join(out_dir, f"{name}_s{seed}.wav")
        sf.write(p, np.clip(wav * min(1, 0.99 / max(np.abs(wav).max(), 1e-6)), -1, 1), 32000)
        r = ev.evaluate(p, asr=False)
        print(json.dumps({"clip": os.path.basename(p), "prompt": text, "rhythm": r["rhythm"],
                          "silent": r["technical"]["silent_frame_ratio"], "rms": r["technical"]["rms_dbfs"]}), flush=True)
