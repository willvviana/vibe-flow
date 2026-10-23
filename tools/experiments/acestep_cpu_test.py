"""Isolated experiment (run with the ACE-Step sandbox venv, NOT the app venv):
can ACE-Step 1.5 (2B turbo DiT, LM disabled) generate a short sung clip on this CPU?

  cd .sandbox/ace && uv run python ../../tools/experiments/acestep_cpu_test.py 10
"""
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

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
peak = [0.0]


def watch():
    p = psutil.Process()
    while True:
        peak[0] = max(peak[0], p.memory_info().rss / 2**30)
        time.sleep(0.5)


threading.Thread(target=watch, daemon=True).start()
t0 = time.time()
dit = AceStepHandler()
msg = dit.initialize_service(project_root=os.getcwd(), config_path="acestep-v15-turbo", device="cpu")
print("INIT", round(time.time() - t0, 1), "s", msg if isinstance(msg, str) else msg[0][:200], "peak_rss_gb", round(peak[0], 2), flush=True)
params = GenerationParams(caption="warm acoustic pop ballad, female vocals, soft guitar", lyrics="[Verse]\nStreetlights hum a quiet tune\nI'm walking home beneath the moon",
                          vocal_language="en", duration=secs, inference_steps=8, seed=7, thinking=False)
t1 = time.time()
res = generate_music(dit, LLMHandler(), params, GenerationConfig(batch_size=1, audio_format="wav"), save_dir=os.path.join(os.getcwd(), "out"))
dt = time.time() - t1
print("RESULT success", res.success, "error", getattr(res, "error", None), "gen_s", round(dt, 1), "rtf", round(dt / secs, 2),
      "peak_rss_gb", round(peak[0], 2), [a["path"] for a in (res.audios or [])], flush=True)
