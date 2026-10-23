"""ACE-Step stage 1 (run with the ACE sandbox venv): the 5 Hz language model plans the song.

It returns metadata (BPM, key, duration, caption, language) and *audio semantic codes* - the structural
scaffold (melody contour, phrasing, vocal placement) that the DiT then renders. Run in its own process so the LM and
the 2B DiT are never in RAM at the same time (16 GB machine).

  python ace_lm_stage.py <out.json> <duration_s> <caption> <lyrics_file> [lm_model=acestep-5Hz-lm-1.7B]
"""
import json
import os
import sys
import time

import psutil

sys.path.insert(0, os.getcwd())
from acestep.llm_inference import LLMHandler  # noqa: E402

out, duration, caption, lyrics_file = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
lm_name = sys.argv[5] if len(sys.argv) > 5 else "acestep-5Hz-lm-1.7B"
lyrics = open(lyrics_file, encoding="utf-8").read()
t0 = time.time()
lm = LLMHandler()
msg, ok = lm.initialize(checkpoint_dir=os.path.join(os.getcwd(), "checkpoints"), lm_model_path=lm_name, backend="pt", device="cpu")
print("LM_INIT", ok, round(time.time() - t0, 1), "s", flush=True)
if not ok:
    print(msg)
    sys.exit(1)
t1 = time.time()
res = lm.generate_with_stop_condition(caption=caption, lyrics=lyrics, infer_type="llm_dit", temperature=0.85, cfg_scale=2.0,
                                      negative_prompt="NO USER INPUT", top_k=None, top_p=None, target_duration=duration,
                                      user_metadata={"duration": int(duration)}, use_cot_caption=False, use_cot_language=False,
                                      use_cot_metas=True, use_constrained_decoding=True, batch_size=1, seeds=[7])
dt = time.time() - t1
print("LM_RESULT success", res.get("success"), "gen_s", round(dt, 1), "peak_rss_gb", round(psutil.Process().memory_info().rss / 2**30, 2),
      "codes_len", len(res.get("audio_codes", "") or ""), "meta", res.get("metadata"), "err", res.get("error"), flush=True)
json.dump({"metadata": res.get("metadata"), "audio_codes": res.get("audio_codes"), "lm": lm_name, "lm_gen_s": dt},
          open(out, "w", encoding="utf-8"))
