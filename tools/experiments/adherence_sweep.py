"""Measurable sweep of the adherence control on one real sample (NO listening pack is produced; outputs stay in .sandbox).

  python tools/experiments/adherence_sweep.py <sample.wav> <out_json> [levels=0,10,25,50,75,100] [seed=5]
Runs the full production path (engines -> sample_compose.run, real MusicGen) once per level with identical settings, seed and sample, and records
what the percentage changed in the audio:  bars retained, similarity of the result to the sample (waveform correlation, chroma similarity),
rhythmic overlap with the sample, tempo/key error of the added layers, sample integrity, clipping, time and memory."""
import json
import os
import sys
import time

import numpy as np
import psutil
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ["VIBEFLOW_LIBRARY"] = os.path.join(ROOT, ".sandbox", "adapt", "lib")
os.makedirs(os.environ["VIBEFLOW_LIBRARY"], exist_ok=True)
from engine import engines, library, sample_analysis as sa  # noqa: E402

sample, out_json = sys.argv[1], sys.argv[2]
levels = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,10,25,50,75,100").split(",")]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 5
src, sr0 = sf.read(sample, dtype="float32", always_2d=True)
rows, ps = [], psutil.Process()
for p in levels:
    t0 = time.time()
    meta = engines.get().generate({"settings": {"mode": "adapt", "reference": sample, "adherence": p, "instruments": ["piano", "strings"], "genres": [], "moods": [],
                                                  "seed": seed, "prompt": "", "layer_level_db": -6}}, lambda **k: None, keep_loaded=True)
    wav, sr = sf.read(library.paths(meta["id"])[1], dtype="float32", always_2d=True)
    n = min(len(wav), len(src))
    c_wave = float(np.corrcoef(wav[:n].mean(1), src[:n].mean(1))[0, 1])
    Cw, Cs = sa.chroma(sa.resample(wav.mean(1), sr))[0].mean(1), sa.chroma(sa.resample(src.mean(1), sr0))[0].mean(1)
    ev, pl = meta["evaluation"], meta["plan"]
    mi = ps.memory_info()
    rows.append({"adherence": p, "bars_retained": sum(pl["retained_bars"]), "bars_total": len(pl["retained_bars"]), "context_bars": pl["context_bars"],
                 "tempo_hint": pl["tempo_hint"], "key_hint": pl["key_hint"], "align_strength": pl["align_strength"], "prompt": pl.get("prompt"),
                 "waveform_corr_with_sample": round(c_wave, 3), "chroma_sim_to_sample": round(float(np.corrcoef(Cw, Cs)[0, 1]), 3),
                 "layer_tempo_error_bpm": ev["layer_tempo_error_bpm"], "layer_chroma_sim": ev["layer_chroma_similarity_to_source"],
                 "rhythmic_overlap": ev["rhythmic_duplication_onset_corr"], "integrity_residual": ev["source_integrity_max_residual"],
                 "clipped": ev["clipped_samples"], "alignment": ev["alignment"], "gen_s": meta["timing"]["generate_s"], "total_s": round(time.time() - t0, 1),
                 "peak_ram_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2), "track": meta["id"]})
    print(json.dumps({k: rows[-1][k] for k in ("adherence", "bars_retained", "waveform_corr_with_sample", "chroma_sim_to_sample", "layer_tempo_error_bpm", "total_s")}), flush=True)
    json.dump({"sample": sample, "analysis": meta["analysis"], "rows": rows}, open(out_json, "w"), indent=1)
