"""Is the AudioSet instrument tagger a usable proxy? Check it on clips whose instruments we know (the 36 cross-genre clips).

For every clip and every instrument in the vocabulary: positive = instrument was requested for that clip, negative = it was not.
Reports the AUC (probability that a requested instrument scores higher than a non-requested one), overall and per instrument.
AUC ~0.5 = useless; > 0.8 = usable as a rough proxy. Run from the project root:  python tools/experiments/validate_tagger.py"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import instrument_tags as it  # noqa: E402

key = json.load(open(os.path.join(ROOT, ".sandbox", "xg", "key.json")))
genres = key["genres"]
vocab = sorted({i for g in genres.values() for i in g["instruments"]} | {"saxophone", "flute", "harp", "organ", "trumpet", "cello", "violin"})
rows = []
for name, v in sorted(key["key"].items()):
    req = set(genres[v["genre"]]["instruments"])
    pres = it.presence(os.path.join(ROOT, ".sandbox", "xg", "blind", name), vocab)
    for ins in vocab:
        rows.append((ins, ins in req, pres[ins] if pres[ins] is not None else 0.0))
    print(name, v["genre"], {k: pres[k] for k in sorted(req)}, flush=True)


def auc(pos, neg):
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


print("OVERALL AUC", round(auc([r[2] for r in rows if r[1]], [r[2] for r in rows if not r[1]]), 3))
for ins in vocab:
    a = auc([r[2] for r in rows if r[0] == ins and r[1]], [r[2] for r in rows if r[0] == ins and not r[1]])
    print(f"  {ins:16} AUC {a if a is None else round(a, 2)}  mean p requested {np.mean([r[2] for r in rows if r[0]==ins and r[1]] or [0]):.2f} not requested {np.mean([r[2] for r in rows if r[0]==ins and not r[1]] or [0]):.2f}")
