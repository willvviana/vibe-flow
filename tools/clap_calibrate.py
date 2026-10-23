"""Build the CLAP label calibration (engine/sampling.calibrate) from a background corpus that is disjoint from the recognition tests.

  python tools/clap_calibrate.py
Corpus: samples/musicgen, samples/musicgen_ab, samples/ace, samples/ace_warble, samples/ace_warble_r3 (MusicGen + ACE-Step clips, mostly
folk/pop with and without singing). NOT used: samples/musicgen_combos, samples/musicgen_xgenre and .sandbox/listening/inst, which are the
labelled test sets of tools/tag_eval.py. Writes models/clap_calibration.json; delete it to go back to uncalibrated rankings.

RESULT (docs/LISTENING.md): tested and NOT shipped. It lowered single-instrument top-1 from 62.5 % to 37.5 % and single-genre top-3
from 53 % to 47 %, with small gains on 3-instrument mixes; the app therefore runs without a calibration file (the code path stays inert)."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import sampling  # noqa: E402

CORPUS = ["musicgen", "musicgen_ab", "ace", "ace_warble", "ace_warble_r3"]
paths = sorted(p for d in CORPUS for p in glob.glob(os.path.join(ROOT, "samples", d, "*.wav")))
res = sampling.calibrate(paths)
print(json.dumps({"clips": res["clips"], "windows": res["windows"], "file": os.path.relpath(sampling.CALIB_PATH, ROOT)}))
