"""Unblind and analyse the cross-genre A vs D listening round.

  python tools/score_xgenre.py <ratings.csv> <key.json> [--base=LABEL --test=LABEL]   (defaults A and D; roles print as A=base, D=test)

ratings.csv columns: clip,score,ship[,notes]   (score 1-5; ship y/n). Also accepts band_score / convincing.
Pairs are (genre, seed): the same seed and structured inputs, rendered with profile A and with profile D.

Uncertainty treats GENRE as the cluster, because the three seeds of one genre share a prompt and are not independent:
  * cluster bootstrap: resample genres (with replacement), then seeds within each, 20,000 times -> 90 % / 95 % interval
  * exact genre-level sign-flip test (all 2^6 sign patterns of the per-genre mean differences)
  * pair-level sign-flip test is printed only for comparison; it ignores clustering and is therefore anti-conservative
PRE-REGISTERED DECISION RULE (written before any ratings existed; see docs/EVALUATION.md):
  KEEP D           mean paired difference >= +0.5  AND  D worse in at most 1 genre (genre difference <= -0.33)  AND  bootstrap 90 % lower bound > 0
  REVERT TO A      mean difference <= 0  OR  D worse in >= 3 genres
  GENRE-SPECIFIC   >= 2 genres with difference >= +0.67 AND >= 2 genres with difference <= -0.67
  otherwise        INCONCLUSIVE: do not claim D is better; gather more seeds"""
import csv
import itertools
import json
import random
import sys
from collections import defaultdict

import numpy as np


def load(ratings_path, key_path, base="A", test="D"):
    """base/test = the profile_label values to treat as the reference and the candidate (roles are reported as A and D)."""
    key = json.load(open(key_path))["key"]
    rows = list(csv.DictReader(open(ratings_path, encoding="utf-8")))
    score = {}
    ship = {}
    for r in rows:
        clip = r["clip"].strip()
        if not clip.endswith(".wav"):
            clip += ".wav"
        s = (r.get("score") or r.get("band_score") or "").strip()
        if s:
            score[clip] = float(s)
            ship[clip] = (r.get("ship") or r.get("convincing") or "").strip().lower().startswith("y")
    pairs = defaultdict(dict)                    # (genre, seed) -> {"A": (score, ship), "D": ...}
    for clip, s in score.items():
        k = key[clip]
        role = "A" if k["profile_label"] == base else "D" if k["profile_label"] == test else None
        if role:
            pairs[(k["genre"], k["seed"])][role] = (s, ship[clip])
    return {p: v for p, v in pairs.items() if "A" in v and "D" in v}, key


def analyse(pairs, n_boot=20000, rng_seed=1):
    by_genre = defaultdict(list)
    for (g, s), v in sorted(pairs.items()):
        by_genre[g].append((s, v["A"][0], v["D"][0], v["A"][1], v["D"][1]))
    gstats = {}
    for g, lst in by_genre.items():
        a = np.array([x[1] for x in lst]); d = np.array([x[2] for x in lst])
        gstats[g] = {"n": len(lst), "A": a.mean(), "D": d.mean(), "diff": (d - a).mean(),
                     "shipA": sum(x[3] for x in lst), "shipD": sum(x[4] for x in lst)}
    diffs = np.array([x[2] - x[1] for lst in by_genre.values() for x in lst])
    overall = float(diffs.mean())
    genre_means = np.array([g["diff"] for g in gstats.values()])
    rng = random.Random(rng_seed)
    names = list(by_genre)
    boots = []
    for _ in range(n_boot):
        tot = []
        for g in (rng.choice(names) for _ in names):                          # resample genres...
            lst = by_genre[g]
            picks = [lst[rng.randrange(len(lst))] for _ in lst]               # ...then seeds within the genre
            tot.append(np.mean([p[2] - p[1] for p in picks]))
        boots.append(np.mean(tot))
    lo90, hi90 = np.percentile(boots, [5, 95]); lo95, hi95 = np.percentile(boots, [2.5, 97.5])
    obs = abs(genre_means.mean())
    flips = [abs(np.mean(genre_means * np.array(sg))) for sg in itertools.product([1, -1], repeat=len(genre_means))]
    p_genre = sum(f >= obs - 1e-12 for f in flips) / len(flips)
    pair_flips = np.array(list(itertools.product([1, -1], repeat=len(diffs))), dtype=np.int8)
    p_pair = float((np.abs(pair_flips @ diffs / len(diffs)) >= abs(overall) - 1e-12).mean())
    worse = [g for g, s in gstats.items() if s["diff"] <= -0.33]
    better_big = [g for g, s in gstats.items() if s["diff"] >= 0.67]
    worse_big = [g for g, s in gstats.items() if s["diff"] <= -0.67]
    if len(better_big) >= 2 and len(worse_big) >= 2:     # a real split is checked first: it can coexist with a mean near 0
        verdict = "GENRE-SPECIFIC"
    elif overall <= 0 or len(worse) >= 3:
        verdict = "REVERT TO A"
    elif overall >= 0.5 and len(worse) <= 1 and lo90 > 0:
        verdict = "KEEP D"
    else:
        verdict = "INCONCLUSIVE"
    return {"genres": gstats, "overall": overall, "ci90": (lo90, hi90), "ci95": (lo95, hi95), "p_genre": p_genre, "p_pair": p_pair,
            "n_pairs": len(diffs), "worse": worse, "verdict": verdict,
            "wins": int((diffs > 0).sum()), "losses": int((diffs < 0).sum()), "ties": int((diffs == 0).sum()),
            "ship": (sum(s["shipA"] for s in gstats.values()), sum(s["shipD"] for s in gstats.values()))}


def heard_fraction(ratings_path, key_path):
    """Optional column 'heard' = how many of the requested instruments you could hear in the clip. Returns {label: (mean fraction, n)}."""
    key = json.load(open(key_path))["key"]
    out = defaultdict(list)
    for r in csv.DictReader(open(ratings_path, encoding="utf-8")):
        clip = r["clip"].strip() + ("" if r["clip"].strip().endswith(".wav") else ".wav")
        h = (r.get("heard") or "").strip()
        if h and clip in key and key[clip].get("selection"):
            out[key[clip]["profile_label"]].append(min(float(h) / len(key[clip]["selection"]["instruments"]), 1.0))
    return {k: (sum(v) / len(v), len(v)) for k, v in out.items()}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a.lstrip("-").split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    pairs, _ = load(args[0], args[1], opt.get("base", "A"), opt.get("test", "D"))
    r = analyse(pairs)
    print(f"{'genre':12} n  meanA meanD  diff(D-A)  ship A/D")
    for g, s in r["genres"].items():
        print(f"{g:12} {s['n']}  {s['A']:5.2f} {s['D']:5.2f}  {s['diff']:+7.2f}   {s['shipA']}/{s['shipD']} of {s['n']}")
    print(f"\npairs {r['n_pairs']}: D wins {r['wins']}, ties {r['ties']}, A wins {r['losses']}; ship A {r['ship'][0]} vs D {r['ship'][1]}")
    print(f"mean paired difference (D-A) = {r['overall']:+.2f}; cluster-bootstrap 90% [{r['ci90'][0]:+.2f}, {r['ci90'][1]:+.2f}], 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
    print(f"genre-level sign-flip p = {r['p_genre']:.3f} (6 clusters: smallest possible two-sided p is 0.031); pair-level p = {r['p_pair']:.4f} (ignores clustering, anti-conservative)")
    print(f"genres where D is worse (<= -0.33): {r['worse'] or 'none'}")
    print(f"VERDICT under the pre-registered rule: {r['verdict']}")
    hf = heard_fraction(args[0], args[1])
    if hf:
        print("instruments heard (fraction of requested):", {k: f"{v[0]:.2f} (n={v[1]})" for k, v in hf.items()})


if __name__ == "__main__":
    main()
