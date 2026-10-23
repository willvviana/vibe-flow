"""Benchmark MusicGen Small configurations on this machine.

Usage: python tools/bench.py <label> [--threads N] [--quant] [--seconds S] [--runs R]
Prints one JSON line. Each configuration should be run in a fresh process so memory numbers are honest.
Measures: model load time, first-run and warm generation time, real-time factor, process CPU utilisation
and peak working-set RAM.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psutil  # noqa: E402
import torch  # noqa: E402

from engine import generate, models, registry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--quant", action="store_true", help="dynamic int8 quantization of Linear layers")
ap.add_argument("--seconds", type=float, default=5)
ap.add_argument("--runs", type=int, default=2)
a = ap.parse_args()

spec = registry.get("musicgen-small")
torch.set_num_threads(a.threads)
proc = psutil.Process()

t0 = time.time()
processor, model = generate._load(spec, "cpu", lambda **k: None)
if a.quant:
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
load_s = time.time() - t0

c = {"guidance": 3.0, "temperature": 1.0}
runs = []
for i in range(a.runs):
    torch.manual_seed(11)
    cpu0, w0 = proc.cpu_times(), time.time()
    wav = generate._infer(processor, model, "cpu", "lo-fi hip hop, mellow, piano, instrumental", a.seconds, 32000, c,
                          lambda **k: None, 0, 0)
    wall = time.time() - w0
    cpu1 = proc.cpu_times()
    cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    runs.append({"gen_s": round(wall, 1), "rtf": round(wall / (len(wav) / 32000), 2),
                 "cpu_cores_avg": round(cpu_s / wall, 1), "audio_s": round(len(wav) / 32000, 2)})
mi = proc.memory_info()
print(json.dumps({"label": a.label, "threads": a.threads, "quant": a.quant, "load_s": round(load_s, 1), "runs": runs,
                  "peak_ram_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)}))
