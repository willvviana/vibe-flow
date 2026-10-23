"""Mood-override experiment: does changing ONLY the mood keep genre, instruments, tempo and groove intact?   (NOT YET RUN)

  python tools/experiments/mood_override.py --dry-run                       # print every prompt and the clip count, generate nothing
  python tools/experiments/mood_override.py <out_dir> <key_file> [--seeds=31,32] [--seconds=15]     # generate (only when asked to)

Design
  4 contrasting bases, each rendered neutral (no mood) and with two moods that pull AGAINST the base:
    metal       fast, heavy        + sad, calm            (energetic music with sad / calm moods)
    disco/funk  upbeat, danceable  + dark, tense          (dark moods on an upbeat genre)
    ambient     slow, spacious     + energetic, epic      (calm music with intense moods)
    rock        intense guitars    + calm, dreamy         (calm mood on intense instrumentation)
  Two prompt strategies, same sampling (profile D: T 0.8, top-k 250, guidance 5.0), same seed, duration and model:
    current = prompt_style "sentence" (shipped default: mood adjectives lead the sentence: "a sad metal instrumental ...")
    layered = prompt_style "layered"  (genre > instruments > tempo > mood > production; engine/layered.py)
  => per (base, seed, strategy): 1 neutral reference + 2 mood clips.  4 x 2 seeds x 2 strategies x 3 = 48 clips (~75 s each, ~1 h).
Listener task (blind, shuffled names; the neutral reference of each set is named in the sheet but not its strategy):
  for each MOOD clip: style_kept 1-5 ("same genre, instruments, tempo and groove as its neutral reference?") and
  mood_felt 1-5 ("does it clearly feel like the requested mood?").  Scored by tools/score_mood.py with a pre-registered rule.
Automated checks (support only, never proof): median-tempo drift vs the neutral reference, instrument-presence drift (AudioSet tagger)."""
import hashlib
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import prompting, registry  # noqa: E402

BASES = {
    "metal": dict(genres=["metal"], bpm=160, instruments=["electric guitar", "bass", "drums"], moods=[["sad"], ["calm"]]),
    "disco_funk": dict(genres=["disco", "funk"], bpm=118, instruments=["electric guitar", "bass", "drums", "brass"], moods=[["dark"], ["tense"]]),
    "ambient": dict(genres=["ambient"], bpm=60, instruments=["piano", "strings"], moods=[["energetic"], ["epic"]]),
    "rock": dict(genres=["rock"], bpm=130, instruments=["electric guitar", "bass", "drums"], moods=[["calm"], ["dreamy"]]),
}
STYLES = {"current": "sentence", "layered": "layered"}


def conditions(seeds, secs):
    out = []
    for base, b in BASES.items():
        for seed in seeds:
            for label, style in STYLES.items():
                for mood in [[]] + b["moods"]:
                    s = {"prompt": "", "duration": secs, "profile": "d", "prompt_style": style, "genres": b["genres"], "bpm": b["bpm"],
                         "instruments": b["instruments"], "moods": mood}
                    c = prompting.validate(s)
                    out.append({"base": base, "seed": seed, "label": label, "mood": "+".join(mood) or "none", "c": c,
                                "prompt": prompting.build_prompt(c, "musicgen-small", c["prompt_style"])})
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) if "=" in a else (a[2:], True) for a in sys.argv[1:] if a.startswith("--"))
    seeds = [int(x) for x in str(opt.get("seeds", "31,32")).split(",")]
    secs = float(opt.get("seconds", 15))
    conds = conditions(seeds, secs)
    if opt.get("dry-run"):
        for k in conds:
            if k["seed"] == seeds[0]:
                print(f"[{k['base']:10} {k['label']:8} {k['mood']:9}] {k['prompt']}")
        print(f"\n{len(conds)} clips would be generated (seeds {seeds}, {secs:.0f} s each)")
        sys.exit(0)
    import numpy as np
    import psutil
    import soundfile as sf
    import torch
    from engine import generate
    out_dir, key_file = args
    os.makedirs(out_dir, exist_ok=True)
    spec = registry.get("musicgen-small")
    proc, model = generate._load(spec, "cpu", lambda **k: None)
    torch.set_num_threads(generate._threads())
    order = list(range(len(conds)))
    random.Random(31415).shuffle(order)
    key, ps = {}, psutil.Process()
    for n, idx in enumerate(order, 1):
        k = conds[idx]
        torch.manual_seed(k["seed"])
        t0 = time.time()
        wav = generate._infer(proc, model, "cpu", k["prompt"], secs, 32000, k["c"], lambda **kk: None, 0, 0)
        gen_s = time.time() - t0
        wav = wav * min(1.0, 0.99 / max(float(np.abs(wav).max()), 1e-6))
        name = f"z{n:02d}.wav"
        path = os.path.join(out_dir, name)
        sf.write(path, wav.astype("float32"), 32000, subtype="PCM_16")
        mi = ps.memory_info()
        key[name] = {"genre": k["base"], "seed": k["seed"], "profile_label": k["label"], "mood": k["mood"], "prompt": k["prompt"], "duration_s": secs,
                     "gen_s": round(gen_s, 1), "temperature": k["c"]["temperature"], "top_k": k["c"]["top_k"], "guidance": k["c"]["guidance"],
                     "prompt_style": k["c"]["prompt_style"], "selection": {"genres": k["c"]["genres"], "instruments": k["c"]["instruments"], "bpm": k["c"]["bpm"], "moods": k["c"]["moods"]},
                     "peak_ram_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2), "sha": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]}
        print(json.dumps({"clip": name, "base": k["base"], "mood": k["mood"], "label": k["label"], "gen_s": round(gen_s, 1)}), flush=True)
        json.dump({"key": key, "seeds": seeds, "seconds": secs, "bases": BASES, "styles": STYLES}, open(key_file, "w"), indent=1)
