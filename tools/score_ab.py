"""Unblind a listening round.

  python tools/score_ab.py <ratings.csv> <key.json>

ratings.csv columns: clip,score,convincing,notes  (score 1-5 where 5 = sounds like one band playing a real piece;
convincing = y/n for "I would be happy to ship this"). Prints per-config means, convincing rate, and the paired
(same-seed) A-vs-D comparison with a sign test, so a single lucky seed cannot masquerade as a trend."""
import csv
import json
import math
import sys
from collections import defaultdict

ratings = {r["clip"].strip(): r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r.get("score", "").strip()}
key = json.load(open(sys.argv[2]))["key"]
per = defaultdict(list)
by_seed = defaultdict(dict)
for clip, r in ratings.items():
    k = key[clip]
    s = float(r["score"])
    per[k["config"]].append((s, r.get("convincing", "").strip().lower().startswith("y")))
    by_seed[k["seed"]][k["config"]] = s
print("config  n  mean_score  convincing")
for cfg, v in sorted(per.items()):
    print(f"{cfg:6} {len(v):2}  {sum(s for s, _ in v) / len(v):9.2f}  {sum(c for _, c in v)}/{len(v)}")
wins = {"A": 0, "D": 0}
ties = 0
for seed, d in sorted(by_seed.items()):
    if "A" in d and "D" in d:
        print(f"seed {seed}: A={d['A']:.0f} D={d['D']:.0f}")
        if d["A"] == d["D"]:
            ties += 1
        else:
            wins["A" if d["A"] > d["D"] else "D"] += 1
n = wins["A"] + wins["D"]
if n:
    k = max(wins.values())
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)     # two-sided sign test
    print(f"paired wins A {wins['A']} / D {wins['D']} / ties {ties}; two-sided sign-test p = {p:.3f} (n={n})")
    print("D is only justified as the default if it wins most pairs AND p is small; one good seed is not a trend.")
