"""Re-measure an existing Listening Pack from its saved audio with the current metrics (engine/adherence_eval.py) and checks
(engine/listening.summarise). No generation: the audio is unchanged, only the measurements are recomputed.

  python tools/remeasure_pack.py <run_dir> [<run_dir> ...]
The previous report is kept as report.v1.json (once). Vocal comparisons and CLAP instrument ranks are copied over unchanged."""
import json
import os
import shutil
import sys

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import adherence_eval as ae, listening  # noqa: E402

for run_dir in sys.argv[1:]:
    rp = os.path.join(run_dir, "report.json")
    if not os.path.exists(os.path.join(run_dir, "report.v1.json")):
        shutil.copyfile(rp, os.path.join(run_dir, "report.v1.json"))
    rep = json.load(open(rp, encoding="utf-8"))
    clap = listening._clap_fn() if rep.get("clap") else None
    levels = list(rep["levels"])
    for src in rep["sources"]:
        ref, sr = sf.read(os.path.join(run_dir, src["ref"]), dtype="float32", always_2d=True)
        for r in src["runs"]:
            mix, _ = sf.read(os.path.join(run_dir, r["file"]), dtype="float32", always_2d=True)
            lay, _ = sf.read(os.path.join(run_dir, r["layers_file"]), dtype="float32", always_2d=True)
            r["mix"], r["layers"] = ae.compare(ref.mean(1), mix.mean(1), sr, clap), ae.compare(ref.mean(1), lay.mean(1), sr, clap)
        src["summary"], src["checks"] = listening.summarise(src, levels)
    rep["overall"] = listening.overall(rep)
    rep["remeasured"] = "metrics v2: tempo periodicity similarity in the index; strict judged by bars kept + bit-exact copy"
    json.dump(rep, open(rp, "w", encoding="utf-8"), indent=1)
    print(run_dir, json.dumps(rep["overall"]), flush=True)
