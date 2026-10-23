"""Instrument and genre recognition test for the Sample page's tagger (LAION-CLAP zero-shot, engine/sampling.py), on audio whose content is known.

  python tools/tag_eval.py gen  <out_dir>            # 8 single-instrument clips (MusicGen, production prompt path, 10 s, seed 3)
  python tools/tag_eval.py eval <out_dir> <out.json> # all tests below

Ground truth and its limits (stated in the result):
  single     8 clips prompted as ONE instrument. MusicGen may add parts it was not asked for, so the label is "what was requested".
  pairs      every pair of the single clips summed at equal loudness (28 mixes): BOTH instruments are known to be present.
  triples    12 random triples, same construction.
  combos     the 24 multi-instrument clips in samples/musicgen_combos (3 requested instruments each, key in .sandbox/mc/key.json).
  genres     the 36 clips in samples/musicgen_xgenre (6 genre prompts, 4 of them two-genre prompts, key in .sandbox/xg/key.json).
  genre mix  pairs of clips of different genres, (a) summed (overlapping) and (b) one after the other (sequence): both genres present.
Scores: is the true label at rank 1 / within the top 3 / within the top 5 (the Sample page shows 5 genres and 7 instruments)?
Every test runs in three modes: raw (the original ranking), calibrated (per-label bias removed, tools/clap_calibrate.py), and
calibrated+windows (also the strongest 10 s window). Neither calibrated mode beat raw; the app ships raw."""
import itertools
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import listening, sampling  # noqa: E402

INSTR = ["piano", "acoustic guitar", "electric guitar", "drum kit", "violin", "saxophone", "flute", "trumpet"]
GENRE_TRUTH = {"folk": ["folk", "pop"], "rock": ["rock"], "synthwave": ["synthwave", "electronic dance music"], "hiphop_rnb": ["hip hop", "R&B"],
               "jazz": ["jazz"], "cinematic": ["orchestral film score"]}


def gen(out):
    import torch
    from engine import generate, prompting, registry
    os.makedirs(out, exist_ok=True)
    spec = registry.get("musicgen-small")
    proc, model = generate._load(spec, "cpu", lambda **k: None)
    torch.set_num_threads(generate._threads())
    for ins in INSTR:
        c = prompting.validate({"prompt": f"solo {ins}, a single {ins} playing alone, no other instruments", "duration": 10, "instruments": [ins]})
        text = prompting.build_prompt(c, spec.id, c["prompt_style"])
        torch.manual_seed(3)
        w = generate._infer(proc, model, "cpu", text, 10, 32000, c, lambda **k: None, 0, 0)
        w = w * min(1.0, 0.99 / max(float(np.abs(w).max()), 1e-6))
        sf.write(os.path.join(out, ins.replace(" ", "_") + ".wav"), w.astype("float32"), 32000, subtype="PCM_16")
        print("generated", ins, "|", text, flush=True)


def load(path):
    return sampling.load_audio(path, 48000, mono=True)[:, 0]


def norm(x):
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.1


MODE = {"m": "raw"}       # "raw" = the shipped ranking, "calibrated" = bias removed, "calibrated+windows" = + strongest window (both rejected)


def ranks(x, key):
    emb, wins = sampling._embeddings(x)
    vocab = sampling.INSTRUMENTS if key == "instrument" else sampling.GENRES
    m = MODE["m"]
    r = sampling._rank(emb, key, len(vocab), wins if m == "calibrated+windows" else None, calibrated=m != "raw")
    return [e["label"] for e in r]


def pos(order, labels, key):
    vocab = sampling.INSTRUMENTS if key == "instrument" else sampling.GENRES
    out = []
    for l in labels:
        m = listening._vocab_match(l, vocab) if key == "instrument" else (l if l in vocab else None)
        out.append(order.index(m) + 1 if m in order else None)
    return out


def summary(rows, field="ranks"):
    """rows: [{"ranks": [rank per true label]}] -> share of labels at 1 / <=3 / <=5, and share of rows with ALL labels in the top 5."""
    flat = [r for row in rows for r in row[field] if r is not None]
    return {"labels": len(flat), "top1": round(np.mean([r == 1 for r in flat]), 3), "top3": round(np.mean([r <= 3 for r in flat]), 3),
            "top5": round(np.mean([r <= 5 for r in flat]), 3), "all_in_top5": round(np.mean([all(r is not None and r <= 5 for r in row[field]) for row in rows]), 3),
            "n": len(rows)}


def evaluate(inst_dir, out_json):
    res = {}
    rng = random.Random(7)
    single = {i: norm(load(os.path.join(inst_dir, i.replace(" ", "_") + ".wav"))) for i in INSTR}
    rows = []
    for i, x in single.items():
        o = ranks(x, "instrument")
        rows.append({"truth": [i], "ranks": pos(o, [i], "instrument"), "top3": o[:3]})
    res["instrument_single"] = {"summary": summary(rows), "rows": rows}
    rows = []
    for a, b in itertools.combinations(INSTR, 2):
        o = ranks(single[a] + single[b], "instrument")
        rows.append({"truth": [a, b], "ranks": pos(o, [a, b], "instrument"), "top5": o[:5]})
    res["instrument_pairs"] = {"summary": summary(rows), "rows": rows}
    rows = []
    for t in rng.sample(list(itertools.combinations(INSTR, 3)), 12):
        o = ranks(sum(single[i] for i in t), "instrument")
        rows.append({"truth": list(t), "ranks": pos(o, list(t), "instrument"), "top5": o[:5]})
    res["instrument_triples"] = {"summary": summary(rows), "rows": rows}
    mc = json.load(open(os.path.join(ROOT, ".sandbox", "mc", "key.json"), encoding="utf-8"))["key"]
    rows = []
    for name, k in sorted(mc.items()):
        o = ranks(load(os.path.join(ROOT, "samples", "musicgen_combos", name)), "instrument")
        want = k["selection"]["instruments"]
        rows.append({"clip": name, "truth": want, "ranks": pos(o, want, "instrument"), "top7": o[:7]})
    s = summary(rows)
    flat = [r for row in rows for r in row["ranks"] if r is not None]
    s["top7"] = round(np.mean([r <= 7 for r in flat]), 3)
    res["instrument_combos"] = {"summary": s, "rows": rows, "note": "labels = requested instruments; MusicGen does not always play all of them"}
    xg = json.load(open(os.path.join(ROOT, ".sandbox", "xg", "key.json"), encoding="utf-8"))["key"]
    clips = {}
    rows = []
    for name, k in sorted(xg.items()):
        x = load(os.path.join(ROOT, "samples", "musicgen_xgenre", name))
        clips.setdefault(k["genre"], []).append(x)
        o = ranks(x, "genre")
        truth = GENRE_TRUTH[k["genre"]]
        p = pos(o, truth, "genre")
        rows.append({"clip": name, "genre": k["genre"], "truth": truth, "ranks": p, "best": min(r for r in p if r), "top5": o[:5]})
    any_ = [r["best"] for r in rows]
    res["genre_single"] = {"summary": {**summary(rows), "any_true_top1": round(np.mean([b == 1 for b in any_]), 3), "any_true_top3": round(np.mean([b <= 3 for b in any_]), 3)},
                           "per_genre": {g: {"any_top1": round(np.mean([r["best"] == 1 for r in rows if r["genre"] == g]), 3),
                                             "any_top3": round(np.mean([r["best"] <= 3 for r in rows if r["genre"] == g]), 3)} for g in GENRE_TRUTH},
                           "rows": rows}
    over, seq = [], []
    for ga, gb in itertools.combinations(GENRE_TRUTH, 2):
        a, b = norm(clips[ga][0]), norm(clips[gb][0])
        n = min(len(a), len(b))
        for mode, x, out in (("overlap", a[:n] + b[:n], over), ("sequence", np.concatenate([a, b]), seq)):
            o = ranks(x, "genre")
            pa, pb = pos(o, GENRE_TRUTH[ga], "genre"), pos(o, GENRE_TRUTH[gb], "genre")
            out.append({"genres": [ga, gb], "ranks": [min(r for r in pa if r), min(r for r in pb if r)], "top5": o[:5]})
    res["genre_mixed_overlap"] = {"summary": summary(over), "rows": over}
    res["genre_mixed_sequence"] = {"summary": summary(seq), "rows": seq}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(k, json.dumps(v["summary"]))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2])
    else:
        out = {}
        for m in ("raw", "calibrated", "calibrated+windows"):
            MODE["m"] = m
            print("== mode", m, flush=True)
            out[m] = evaluate(sys.argv[2], sys.argv[3].replace(".json", f".{m.replace('+', '_')}.json"))
        with open(sys.argv[3], "w", encoding="utf-8") as f:
            json.dump({m: {k: v["summary"] for k, v in r.items()} for m, r in out.items()}, f, indent=1)
