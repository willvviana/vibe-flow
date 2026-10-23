"""Score the mood-override listening round (tools/experiments/mood_override.py). Ratings are for MOOD clips only.

  python tools/score_mood.py <ratings.csv|txt> <key.json>

ratings columns: clip,style_kept,mood_felt[,notes]
  style_kept 1-5: same genre, instruments, tempo and groove as the neutral reference of its set (5 = clearly the same music)
  mood_felt  1-5: clearly feels like the requested mood
Pairs: the same (base, mood, seed) rendered with the current prompt vs the layered prompt; difference = layered - current.
Uncertainty uses the BASE as the cluster (4 bases), bootstrapping bases then (mood, seed) pairs inside them; with only 4 clusters
the intervals are wide and a "significant" result is hard to get - that is intentional.
PRE-REGISTERED RULE (written before any clip existed):
  ADOPT layered   mean style_kept difference >= +0.5  AND  mean mood_felt difference >= -0.5  AND  bootstrap 90 % lower bound of the style_kept difference > 0
  REJECT layered  style_kept difference <= 0  OR  mood_felt difference <= -1.0   (the layered prompt hides the mood or does not protect the style)
  otherwise       INCONCLUSIVE (no change to the default)"""
import csv
import json
import random
import sys
from collections import defaultdict

import numpy as np


def load(ratings_path, key_path):
    key = json.load(open(key_path))["key"]
    pairs = defaultdict(dict)
    for r in csv.DictReader(open(ratings_path, encoding="utf-8")):
        clip = r["clip"].strip() + ("" if r["clip"].strip().endswith(".wav") else ".wav")
        if not (r.get("style_kept") or "").strip() or clip not in key or key[clip]["mood"] == "none":
            continue
        k = key[clip]
        pairs[(k["genre"], k["mood"], k["seed"])][k["profile_label"]] = (float(r["style_kept"]), float((r.get("mood_felt") or "nan") or "nan"))
    return {p: v for p, v in pairs.items() if "current" in v and "layered" in v}


def analyse(pairs, n_boot=20000, seed=1):
    by_base = defaultdict(list)
    for (base, mood, s), v in pairs.items():
        by_base[base].append((v["layered"][0] - v["current"][0], v["layered"][1] - v["current"][1]))
    all_d = np.array([d for lst in by_base.values() for d in lst], dtype=float)
    style, mood = float(np.nanmean(all_d[:, 0])), float(np.nanmean(all_d[:, 1]))
    rng, boots = random.Random(seed), []
    names = list(by_base)
    for _ in range(n_boot):
        vals = []
        for b in (rng.choice(names) for _ in names):
            lst = by_base[b]
            vals.append(np.mean([lst[rng.randrange(len(lst))][0] for _ in lst]))
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [5, 95])
    if style <= 0 or mood <= -1.0:
        verdict = "REJECT layered"
    elif style >= 0.5 and mood >= -0.5 and lo > 0:
        verdict = "ADOPT layered"
    else:
        verdict = "INCONCLUSIVE"
    return {"n_pairs": len(all_d), "style_diff": style, "mood_diff": mood, "style_ci90": (float(lo), float(hi)),
            "per_base": {b: (float(np.mean([d[0] for d in v])), float(np.nanmean([d[1] for d in v])), len(v)) for b, v in by_base.items()}, "verdict": verdict}


if __name__ == "__main__":
    r = analyse(load(sys.argv[1], sys.argv[2]))
    print(f"pairs {r['n_pairs']}")
    for b, (s, m, n) in r["per_base"].items():
        print(f"  {b:11} style_kept diff {s:+.2f}  mood_felt diff {m:+.2f}  (n={n})")
    print(f"overall: style_kept {r['style_diff']:+.2f} (90% bootstrap [{r['style_ci90'][0]:+.2f}, {r['style_ci90'][1]:+.2f}]), mood_felt {r['mood_diff']:+.2f}")
    print("VERDICT:", r["verdict"])
