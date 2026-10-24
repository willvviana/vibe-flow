"""Does a selected instrument survive audio conditioning? (Listening Pack finding: at Medium and above the added layers follow the sample's
timbre and the selected instrument drops from CLAP rank ~1 to ~23 of 36.) Tests whether a higher text guidance scale restores it.

  python tools/experiments/instrument_influence.py <sample.wav> <out.json> [instrument=piano] [levels=medium,high] [guidance=5,10] [seeds=5,6]
Production path (engines -> sample_compose.run, real MusicGen); outputs stay in .sandbox. Measures the CLAP rank of the instrument in the
ADDED layers and the adherence index of the result, so a guidance change that helps the instrument but hurts adherence is visible."""
import json
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ["VIBEFLOW_LIBRARY"] = os.path.join(ROOT, ".sandbox", "listening", "influence_lib")
from engine import adherence_eval as ae, engines, library, listening  # noqa: E402

sample, out_json = sys.argv[1], sys.argv[2]
instr = sys.argv[3] if len(sys.argv) > 3 else "piano"
levels = (sys.argv[4] if len(sys.argv) > 4 else "medium,high").split(",")
guid = [float(g) for g in (sys.argv[5] if len(sys.argv) > 5 else "5,10").split(",")]
seeds = [int(s) for s in (sys.argv[6] if len(sys.argv) > 6 else "5,6").split(",")]
ref, sr = sf.read(sample, dtype="float32", always_2d=True, frames=20 * sf.info(sample).samplerate)
clap = listening._clap_fn()
rows = []
for lv in levels:
    for g in guid:
        for seed in seeds:
            meta = engines.get().generate({"settings": {"mode": "adapt", "reference": sample, "adherence_level": lv, "instruments": [instr], "genres": [], "moods": [],
                                                         "seed": seed, "prompt": "", "layer_level_db": -6, "duration": 20, "guidance": g}}, lambda **k: None, keep_loaded=True)
            w = library.paths(meta["id"])[1]
            mix, _ = sf.read(w, dtype="float32", always_2d=True)
            lay, _ = sf.read(w[:-4] + ".layers.wav", dtype="float32", always_2d=True)
            rk = listening.tag_ranks(lay.mean(1), sr, [instr])[instr]
            rows.append({"level": lv, "guidance": g, "seed": seed, "rank_in_layers": rk["rank"], "score": rk["score"],
                         "mix_index": ae.compare(ref.mean(1), mix.mean(1), sr, clap)["adherence_index"],
                         "layers_index": ae.compare(ref.mean(1), lay.mean(1), sr, clap)["adherence_index"], "prompt": meta.get("model_prompt")})
            print(json.dumps(rows[-1]), flush=True)
            json.dump({"sample": sample, "instrument": instr, "reference_rank": listening.tag_ranks(ref.mean(1), sr, [instr])[instr], "rows": rows},
                      open(out_json, "w"), indent=1)
for lv in levels:
    for g in guid:
        r = [x for x in rows if x["level"] == lv and x["guidance"] == g]
        print(lv, g, "rank", np.mean([x["rank_in_layers"] for x in r]), "mix index", round(np.mean([x["mix_index"] for x in r]), 3))
