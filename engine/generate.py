"""Generation pipeline: validate -> prompt -> load model -> infer -> post-process -> verify -> save."""
import gc
import math
import os
import time

import numpy as np

from . import audio, hardware, library, memory, models, prompting, registry

TOKENS_PER_SECOND = 50      # MusicGen's EnCodec frame rate
CONTEXT_SECONDS = 10        # audio prompt length for continuation / long-form chaining
SEGMENT_SECONDS = 30        # model's native max per call (includes the prompt)


class Cancelled(Exception):
    pass


def _peak_mem_gb():
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)


def _threads():
    """Physical cores: benchmarked equal to all logical threads (tools/bench.py) while leaving the machine responsive."""
    import psutil
    return psutil.cpu_count(logical=False) or os.cpu_count() or 4


_CACHE = {}   # (model id, device, fast) -> (processor, model); only used by the long-lived worker


def release_cache():
    """Drop cached models and give the memory back (idle timeout, or before loading a different model)."""
    _CACHE.clear()
    gc.collect()


def _load(spec, device, emit, fast=False, keep=False):
    key = (spec.id, device, fast)
    if key in _CACHE:
        emit(stage="load", pct=100, message=f"{spec.name} already loaded.")
        return _CACHE[key]
    if keep:
        release_cache()           # never hold two large models at once
    pm = _load_uncached(spec, device, emit, fast)
    if keep:
        _CACHE[key] = pm
    return pm


def _load_uncached(spec, device, emit, fast=False):
    import torch
    from transformers import AutoProcessor
    path = models.local_path(spec)
    if not path:
        raise RuntimeError(f"{spec.name} is not installed. Install it first (Models panel, or: python -m engine.cli install {spec.id}).")
    emit(stage="load", pct=0, message=f"Loading {spec.name} on {device}…")
    if spec.id == "musicgen-melody":
        from transformers import MusicgenMelodyForConditionalGeneration as Cls
    else:
        from transformers import MusicgenForConditionalGeneration as Cls
    try:
        proc = AutoProcessor.from_pretrained(path)
        model = Cls.from_pretrained(path, torch_dtype=torch.float32).to(device).eval()
    except Exception as e:  # truncated/corrupt weights surface here as safetensors or JSON errors
        raise RuntimeError(f"{spec.name} could not be loaded; its files may be damaged. Use Repair in the Models panel "
                           f"(or: python -m engine.cli repair {spec.id}).") from e
    if fast:
        if device != "cpu":
            raise RuntimeError("Fast mode (int8) is only available on CPU.")
        # Dynamic int8 quantisation of Linear layers: ~1.5x faster on this CPU (see README benchmarks),
        # needs extra RAM while converting, and changes the output slightly.
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    emit(stage="load", pct=100, message="Model loaded.")
    return proc, model


def _infer(proc, model, device, prompt, new_seconds, sr, c, emit, base_pct, span_pct, ref=None, label=""):
    """One model call. ref is a mono float32 array at sr used as audio prompt (or melody)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    n_new = max(int(math.ceil(new_seconds * TOKENS_PER_SECOND)), 25)

    class Progress(StoppingCriteria):
        def __init__(self):
            self.start = None

        def __call__(self, input_ids, scores, **kw):
            if self.start is None:
                self.start = input_ids.shape[-1]
            done = input_ids.shape[-1] - self.start
            emit(stage="generate", pct=base_pct + span_pct * min(done / n_new, 1), message=f"{label}Generating audio… {min(done, n_new)}/{n_new} steps")
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    kwargs = dict(text=[prompt], padding=True, return_tensors="pt")
    if ref is not None:
        kwargs.update(audio=ref, sampling_rate=sr)
    inputs = proc(**kwargs).to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=True, guidance_scale=c["guidance"], temperature=c["temperature"], top_k=c.get("top_k", 250),
                             max_new_tokens=n_new, stopping_criteria=StoppingCriteriaList([Progress()]))
    return out[0, 0].float().cpu().numpy()


def run(job: dict, emit=lambda **k: None, keep_loaded: bool = False) -> dict:
    """Execute a generation job. Returns the saved track metadata. Raises on failure.

    keep_loaded=True (worker mode) keeps the model in memory for the next job instead of reloading it."""
    t_start = time.time()
    warnings = []
    c = prompting.validate(job["settings"])
    task = c["mode"]
    spec = registry.pick_model(task, job.get("model"))
    hw = hardware.detect()
    device = hardware.choose_device(hw, spec)
    warnings += hardware.resource_warnings(hw, spec, device)
    # Refuse up front (clear message) rather than letting Windows thrash or kill the process mid-generation.
    warnings += memory.preflight(spec, hw, device, already_loaded=(spec.id, device, c["fast"]) in _CACHE)
    if task == "melody" and c["duration"] > spec.max_segment_s:
        raise prompting.ValidationError(f"Melody mode supports up to {spec.max_segment_s}s.")
    sr = spec.sample_rate

    ref_info = None
    ref_audio = None
    if task != "text":
        ref_info = audio.validate_reference(c["reference"])
        emit(stage="preprocess", pct=0, message="Preparing reference audio…")
        ref_audio = audio.load_mono(c["reference"], sr)

    text = prompting.build_prompt(c, spec.id, c["prompt_style"])
    seed = c["seed"] if c["seed"] is not None else int.from_bytes(os.urandom(3), "big")
    import torch
    torch.set_num_threads(_threads())
    torch.manual_seed(seed)

    proc = model = None
    try:
        t_load = time.time()
        proc, model = _load(spec, device, emit, fast=c["fast"], keep=keep_loaded)
        load_s = time.time() - t_load
        t_gen = time.time()
        total = c["duration"]

        if task == "melody":
            wav = _infer(proc, model, device, text, total, sr, c, emit, 0, 90, ref=audio.tail(ref_audio, sr, 30))
        else:
            if task == "continue":
                ctx = audio.tail(ref_audio, sr, CONTEXT_SECONDS)
                wav = ctx.copy()
                need = total
            else:
                first = min(total, SEGMENT_SECONDS)
                wav = _infer(proc, model, device, text, first, sr, c, emit, 0, 90 * first / total, label="Segment 1: ")
                need = total - first
            done_s = 0.0
            seams = []                 # times (s) where a continuation was joined, for quality analysis
            seg = 2
            while need > 0.05:
                ctx = audio.tail(wav, sr, CONTEXT_SECONDS)
                step = min(need, SEGMENT_SECONDS - CONTEXT_SECONDS)
                span = 90 * step / total
                base = 90 * (total - need) / total if task == "text" else 90 * done_s / total
                out = _infer(proc, model, device, text, step, sr, c, emit, base, span, ref=ctx, label=f"Segment {seg}: ")
                # MusicGen returns [re-decoded prompt | new audio] (verified: tools/seam_diag.py, lag 0, corr ~0.84).
                # So crossfade over the *time-aligned* prompt tail (original vs re-decoded: same music) rather than
                # blending the end of the old audio with the start of the new, which would skip time.
                n_ov = min(int(0.25 * sr), len(ctx))
                new = out[len(ctx) - n_ov:]
                if len(new) - n_ov < int(0.5 * step * sr):      # model returned too little new audio
                    raise RuntimeError("The model returned too little audio for a continuation step.")
                seams.append(round((len(wav) - n_ov) / sr, 3))
                wav = audio.crossfade_concat(wav, new[: n_ov + int(step * sr)], sr, fade_s=n_ov / sr)
                need -= step
                done_s += step
                seg += 1
        gen_s = time.time() - t_gen

        emit(stage="postprocess", pct=92, message="Analyzing and saving…")
        expected = (len(wav) / sr) if task == "continue" else total
        if task == "text" or task == "melody":
            wav = wav[: int(total * sr)]
            expected = len(wav) / sr
        raw_peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if raw_peak > 0.99:
            # The decoder can overshoot full scale; writing that to PCM would hard-clip. A small uniform gain
            # reduction keeps the waveform intact (peak limited to -0.1 dBFS).
            wav = (wav * (0.99 / raw_peak)).astype(np.float32)
            warnings.append(f"Level reduced by {20 * np.log10(raw_peak / 0.99):.1f} dB to avoid clipping.")
        stats = audio.analyze(wav)
        if stats["silent"]:
            raise RuntimeError("The model produced silent audio.")
        if stats["clipped_samples"]:
            warnings.append(f"{stats['clipped_samples']} samples are at full scale (possible clipping).")

        track_id = library.new_id()
        _, wav_path = library.paths(track_id)
        audio.write_audio(wav_path, wav, sr, "wav")           # original, never modified afterwards
        check = audio.verify_file(wav_path, expected)
        exported = None
        if c["format"] != "wav":
            emit(stage="export", pct=96, message=f"Exporting {c['format'].upper()}…")
            exported = library.export(track_id, c["format"])
            audio.verify_file(exported, expected, tolerance=0.2)

        meta = {
            "id": track_id,
            "title": (c["prompt"] or ", ".join(c["genres"]) or "Untitled")[:60],
            "status": "complete",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            # keep what the user asked for, but never persist server-side file paths
            "original_settings": {**job["settings"], **({"reference": job.get("reference_name") or os.path.basename(c["reference"])} if c["reference"] else {})},
            "settings": {k: v for k, v in c.items() if k != "reference"},
            "model_prompt": text,
            "controls": prompting.control_report(c),
            "model": {"id": spec.id, "name": spec.name, "repo": spec.repo, "license": spec.license},
            "device": device,
            "fast_mode": c["fast"],
            "seed": seed,
            "reference": ref_info and {**ref_info, "name": job.get("reference_name") or os.path.basename(c["reference"]), "workflow": task},
            "audio": {**check, "analysis": stats, "seams_s": seams if task != "melody" else [], "files": {"wav": track_id + ".wav",
                      **({c["format"]: os.path.basename(exported)} if exported else {})}},
            "timing": {"load_s": round(load_s, 1), "generate_s": round(gen_s, 1),
                       "total_s": round(time.time() - t_start, 1),
                       "real_time_factor": round(gen_s / max(check["duration"], 0.01), 2),
                       "peak_memory_gb": _peak_mem_gb()},
            "warnings": warnings,
        }
        library.save_meta(track_id, meta)
        emit(stage="done", pct=100, message="Done.")
        return meta
    finally:
        del model, proc      # (the cache, if any, still references the model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
