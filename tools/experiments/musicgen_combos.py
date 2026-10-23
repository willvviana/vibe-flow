"""Multi-instrument / multi-genre / multi-mood test: current default prompt vs the experimental arrangement prompt.

  python tools/experiments/musicgen_combos.py <out_dir> <key_file> [seeds=21,22] [seconds=15]

Both strategies use profile "d" sampling (T 0.8, top-k 250, guidance 5.0), same seed, duration and model: ONLY the prompt differs.
  current     = prompt_style "sentence" (shipped default)      arrangement = prompt_style "arrangement" (engine/arrangement.py)
Six selection combinations vary the number of instruments (2-5), genres (1-3) and moods (1-3), include a conflicting mood pair and an
instrument that needs adapting. Clips get shuffled neutral names; the key (labels, prompts, adaptations, times) stays outside samples/."""
import hashlib, json, os, random, sys, time

import numpy as np
import psutil
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from engine import arrangement, generate, prompting, registry  # noqa: E402

out_dir, key_file = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "21,22").split(",")]
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 15
os.makedirs(out_dir, exist_ok=True)
COMBOS = {
    "c1_folk_5instr": dict(genres=["folk"], moods=["calm"], bpm=76, instruments=["acoustic guitar", "bass", "percussion", "violin", "flute"]),
    "c2_rockblues_dark": dict(genres=["rock", "blues"], moods=["energetic", "dark"], bpm=110, instruments=["electric guitar", "bass", "drums"]),
    "c3_synth_cinematic": dict(genres=["synthwave", "cinematic", "electronic"], moods=["dreamy", "epic"], bpm=100, instruments=["synthesizer", "bass", "drums", "strings"]),
    "c4_jazz_rnb_3moods": dict(genres=["jazz", "r&b"], moods=["mellow", "romantic", "sad"], bpm=84, instruments=["piano", "bass", "drums", "saxophone"]),
    "c5_classical_conflict": dict(genres=["classical", "cinematic"], moods=["calm", "energetic"], bpm=90, instruments=["piano", "cello", "drums"]),
    "c6_funk_hiphop_5instr": dict(genres=["funk", "hip hop"], moods=["happy", "energetic"], bpm=100, instruments=["electric guitar", "bass", "drums", "brass", "synthesizer"]),
}
STYLES = {"current": "sentence", "arrangement": "arrangement"}
spec = registry.get("musicgen-small")
proc, model = generate._load(spec, "cpu", lambda **k: None)
torch.set_num_threads(generate._threads())
jobs = [(c, s, lab) for c in COMBOS for s in seeds for lab in STYLES]
random.Random(77).shuffle(jobs)
ps, key = psutil.Process(), {}
for n, (combo, seed, label) in enumerate(jobs, 1):
    c = prompting.validate({"prompt": "", "duration": secs, "profile": "d", "prompt_style": STYLES[label], **COMBOS[combo]})
    text = prompting.build_prompt(c, spec.id, c["prompt_style"])
    info = arrangement.describe(c) if label == "arrangement" else {"adaptations": [], "notes": []}
    torch.manual_seed(seed)
    t0 = time.time()
    wav = generate._infer(proc, model, "cpu", text, secs, 32000, c, lambda **k: None, 0, 0)
    gen_s = time.time() - t0
    wav = wav * min(1.0, 0.99 / max(float(np.abs(wav).max()), 1e-6))
    name = f"m{n:02d}.wav"
    path = os.path.join(out_dir, name)
    sf.write(path, wav.astype("float32"), 32000, subtype="PCM_16")
    mi = ps.memory_info()
    key[name] = {"genre": combo, "seed": seed, "profile_label": label, "prompt": text, "duration_s": secs, "gen_s": round(gen_s, 1),
                 "temperature": c["temperature"], "top_k": c["top_k"], "guidance": c["guidance"], "prompt_style": c["prompt_style"],
                 "selection": COMBOS[combo], "adaptations": info["adaptations"], "notes": info["notes"],
                 "peak_ram_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2), "sha": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]}
    print(json.dumps({"clip": name, "combo": combo, "seed": seed, "label": label, "gen_s": round(gen_s, 1)}), flush=True)
    json.dump({"key": key, "seconds": secs, "seeds": seeds, "combos": COMBOS, "styles": STYLES}, open(key_file, "w"), indent=1)
