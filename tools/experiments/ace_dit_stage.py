"""NOTE (found later): GenerationConfig.use_random_seed defaults to True, so runs made before this fix ignored the "seed" field.
Seeds are now honoured (use_random_seed=False) and the seed actually used is reported as seed_used.

ACE-Step stage 2 (run with the ACE sandbox venv): render a list of configurations with the 2B turbo DiT, no LM.

  python ace_dit_stage.py <runs.json> <out_dir>
runs.json = [{"name":..., "caption":..., "lyrics":..., "duration":..., "codes_json": optional path from stage 1,
              "steps": 8, "seed": 7, "instrumental": false}, ...]
The DiT is initialised once (about 2 min) and reused for every run. Prints one RESULT line per run."""
import json
import os
import sys
import threading
import time

import psutil

os.environ.setdefault("ACESTEP_INIT_LLM", "false")
sys.path.insert(0, os.getcwd())
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402

runs = json.load(open(sys.argv[1], encoding="utf-8"))
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
peak = [0.0]


def watch():
    p = psutil.Process()
    while True:
        peak[0] = max(peak[0], p.memory_info().rss / 2**30)
        time.sleep(0.5)


threading.Thread(target=watch, daemon=True).start()
t0 = time.time()
dit = AceStepHandler()
dit.initialize_service(project_root=os.getcwd(), config_path=os.environ.get("ACE_DIT", "acestep-v15-turbo"), device="cpu")
print("INIT", round(time.time() - t0, 1), "s", flush=True)
for r in runs:
    kw = dict(caption=r["caption"], lyrics=r.get("lyrics", ""), instrumental=r.get("instrumental", False), vocal_language="en",
              duration=float(r["duration"]), inference_steps=r.get("steps", 8), seed=r.get("seed", 7), guidance_scale=r.get("guidance", 7.0), shift=r.get("shift", 1.0), thinking=False,
              enable_normalization=False, use_cot_caption=False, use_cot_metas=False, use_cot_language=False)
    if r.get("codes_json"):
        c = json.load(open(r["codes_json"], encoding="utf-8"))
        kw["audio_codes"] = c["audio_codes"] or ""
        md = c.get("metadata") or {}
        if md.get("bpm"):
            kw["bpm"] = int(float(md["bpm"]))
        if md.get("keyscale"):
            kw["keyscale"] = md["keyscale"]
        if md.get("timesignature"):
            kw["timesignature"] = str(md["timesignature"])
    t1 = time.time()
    res = generate_music(dit, LLMHandler(), GenerationParams(**kw), GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False, seeds=[int(r.get("seed", 7))]), save_dir=out_dir)
    dt = time.time() - t1
    paths = [a["path"] for a in (res.audios or [])]
    if paths:
        dest = os.path.join(out_dir, r["name"] + ".wav")
        os.replace(paths[0], dest)
        paths = [dest]
    print("RESULT", json.dumps({"name": r["name"], "success": res.success, "error": getattr(res, "error", None), "gen_s": round(dt, 1),
                                "rtf": round(dt / float(r["duration"]), 2), "seed": r.get("seed", 7), "steps": r.get("steps", 8), "guidance": r.get("guidance", 7.0), "shift": r.get("shift", 1.0),
                                "seed_used": [a.get("params", {}).get("seed") for a in (res.audios or [])], "peak_rss_gb": round(peak[0], 2), "files": paths}), flush=True)
