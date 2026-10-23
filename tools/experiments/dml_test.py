"""Isolated experiment: can MusicGen Small run on the AMD GPU through DirectML? (not part of the app)"""
import glob
import os
import sys
import time

import torch
import torch_directml
from transformers import AutoProcessor, MusicgenForConditionalGeneration

print("torch", torch.__version__, "devices", [torch_directml.device_name(i) for i in range(torch_directml.device_count())], flush=True)
snap = glob.glob(r"C:\dev\vibe-flow\models\models--facebook--musicgen-small\snapshots\*")[0]
dev = torch_directml.device()
proc = AutoProcessor.from_pretrained(snap)
model = MusicgenForConditionalGeneration.from_pretrained(snap).eval()
inp = proc(text=["lo-fi hip hop, mellow, piano, instrumental"], padding=True, return_tensors="pt")
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 2
for name, d in (("dml", dev), ("cpu", torch.device("cpu"))):
    try:
        m = model.to(d)
        torch.manual_seed(1)
        t = time.time()
        with torch.no_grad():
            out = m.generate(**{k: v.to(d) for k, v in inp.items()}, do_sample=True, guidance_scale=3.0, max_new_tokens=int(secs * 50))
        dt = time.time() - t
        a = out[0, 0].float().cpu().numpy()
        print(name, "ok", round(dt, 1), "s for", round(len(a) / 32000, 2), "s audio; rtf", round(dt / (len(a) / 32000), 2),
              "finite", bool(torch.isfinite(torch.tensor(a)).all()), "rms", round(float((a ** 2).mean() ** .5), 4), flush=True)
    except Exception as e:  # noqa: BLE001
        print(name, "FAILED:", type(e).__name__, str(e)[:300], flush=True)
