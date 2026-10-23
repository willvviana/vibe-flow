"""Sample-adaptive composition (mode "adapt"): new instrumental layers built around an uploaded sample, with a Loose<->Close control.

Pipeline (each step is a separate function so it can be tested without the model):
  1 load + validate the sample            engine.audio.validate_reference / native-rate decode (mono analysis copy at 22.05 kHz)
  2 analyse                               engine.sample_analysis (tempo, beats/bars, key, chords, energy, bands, sections, optional vocals)
  3 plan from the adherence percentage    engine.adherence.controls (retained bars, hints, alignment, audio-conditioning length, carve depth)
  4 generate the NEW layers               MusicGen Small: text (selected instruments/genre/moods + the sample's tempo/key per the plan)
                                          and, from p >= 25 %, the sample's first bars as an AUDIO PROMPT (continuation) so the layers
                                          continue in the sample's own tempo/key/timbre. No vocals ever: instrumental engine.
  5 align + harmonise + carve             tempo (phase vocoder, <= +-10 %), downbeat phase (onset cross-correlation), key (pitch shift <= 4 semitones),
                                          spectral carving where the sample is strong. Strengths come from the plan. Skipped when not needed.
  6 mix                                   output = sample * retained-bar mask  +  layers * gain(user level under retained bars, 0 dB elsewhere).
                                          The sample is COPIED, never regenerated or filtered; at 100 % the whole sample is present.
  7 evaluate                              clipping, tempo error, key match, rhythmic duplication (onset-envelope correlation), source integrity
                                          (exact residual), spectral overlap before/after carving. Returned in the metadata.
  8 save                                  mix + `<id>.source.wav` (the sample window) + `<id>.layers.wav` (the added layers) so the result
                                          can be compared with the original and the layers can be heard alone.

HONEST LIMITS (also returned as warnings): the model cannot condition on a sample AND add parts that are guaranteed to be in time and in key; steps 5-6
therefore correct tempo/phase/key with DSP, which can leave artefacts; the layers may still sound like a continuation rather than an accompaniment;
melody/motifs are not analysed; vocals are not available; the output has the sample's sample rate and channel layout (the layers are mono, centred)."""
import math
import os
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import adherence, audio, dsp, generate, hardware, library, memory, prompting, registry
from . import sample_analysis as sa

MAX_WINDOW_S = 30.0          # MusicGen's native limit per call (audio prompt + new audio)


def load_source(path, max_s=MAX_WINDOW_S):
    """Decode at the sample's native rate and channel layout (<= 2 ch), cut to the working window. Returns (x [n, ch], sr)."""
    info = audio.validate_reference(path)
    x, sr = sf.read(path, dtype="float32", always_2d=True, frames=int(max_s * info["sample_rate"]))
    x = x[:, :2]
    if not np.isfinite(x).all() or float(np.abs(x).max()) < 1e-4:
        raise audio.AudioError("The sample is silent or invalid.")
    return x, sr, info


def mono(x):
    return x.mean(axis=1)


def _resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def build_text_settings(c, a, plan):
    """Fold the plan's hints into the user's settings (a copy) so the shipped prompt builders write them into the text."""
    s = dict(c)
    s["bpm"] = int(round(a["bpm"])) if plan["tempo_hint"] >= 0.5 else c.get("bpm")
    s["key"] = a["key"]["name"] if plan["key_hint"] >= 0.5 and a["key"]["margin"] >= 0.02 else c.get("key")
    if s["key"] not in prompting.KEYS:
        s["key"] = c.get("key") if c.get("key") in prompting.KEYS else None
    return s


def align_layer(layer, sr, a_src, plan, src_mono):
    """Tempo, phase and key alignment of the generated layers to the sample. Returns (layer, report). `layer` and `src_mono` are at `sr`."""
    rep = {"tempo_before_bpm": None, "tempo_after_bpm": None, "stretch_rate": 1.0, "shift_ms": 0.0, "semitones": 0, "skipped": []}
    ctx = plan["context_bars"] > 0           # layers were generated as a continuation of the sample: expected to be in its tempo/phase already
    l22 = sa.resample(layer, sr)
    bpm_l, _, pulse_l = sa.tempo_and_beats(l22)
    rep["tempo_before_bpm"] = round(bpm_l, 2)
    if plan["align_strength"] > 0 and pulse_l > 0.1 and bpm_l > 0 and a_src["tempo_confidence"] > 0.1:
        ratio = a_src["bpm"] / bpm_l
        ratio = min((ratio, ratio * 2, ratio / 2), key=lambda r: abs(math.log(r)))      # fold octave errors
        rate = ratio ** plan["align_strength"]
        if ctx and abs(rate - 1) < 0.03:
            rep["skipped"].append("tempo: the continuation already follows the sample's tempo (within 3 %)")
        elif 0.9 <= rate <= 1.1:
            layer = dsp.time_stretch(layer, rate)
            rep["stretch_rate"] = round(rate, 4)
        else:
            rep["skipped"].append(f"tempo: needs x{rate:.2f}, outside the +-10 % the phase vocoder handles well")
    # phase: cross-correlate onset envelopes within +-half a bar
    n = min(len(layer), len(src_mono))
    la, sr_ = sa.resample(layer[:n], sr), sa.resample(src_mono[:n], sr)
    ea, fps = sa.onset_envelope(la)
    eb, _ = sa.onset_envelope(sr_)
    m = min(len(ea), len(eb))
    if plan["align_strength"] >= 0.5 and m > 8 and not ctx:
        max_lag = int(fps * a_src["bar_seconds"] * 0.5)
        best, best_lag = -1e9, 0
        for lag in range(-max_lag, max_lag + 1):
            u, v = (ea[:m - lag], eb[lag:m]) if lag >= 0 else (ea[-lag:m], eb[:m + lag])
            if len(u) > 8:
                s = float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 and v.std() > 0 else -1
                if s > best:
                    best, best_lag = s, lag
        shift = int(best_lag / fps * sr)
        layer = dsp.shift_samples(layer, shift)
        rep["shift_ms"] = round(1000.0 * shift / sr, 1)
    bpm_after, _, _ = sa.tempo_and_beats(sa.resample(layer, sr))
    rep["tempo_after_bpm"] = round(bpm_after, 2)
    # key: choose the semitone shift (-4..+4) whose pitch-class profile best matches the sample's, if it is clearly better than no shift.
    # (Comparing profiles directly avoids relative major/minor naming mistakes that a tonic-name comparison would make.)
    if plan["key_hint"] >= 0.5:
        cl = sa.chroma(sa.resample(layer, sr))[0].mean(axis=1)
        cs = sa.chroma(sa.resample(src_mono[: len(layer)], sr))[0].mean(axis=1)
        score = {d: float(np.corrcoef(np.roll(cl, d), cs)[0, 1]) for d in range(-4, 5)}
        best = max(score, key=score.get)
        if best != 0 and score[best] - score[0] >= 0.05:
            layer = dsp.pitch_shift(layer, best)
            rep["semitones"] = int(best)
        elif best != 0:
            rep["skipped"].append("key: no shift clearly improves the match with the sample")
    return layer, rep


def evaluate(src_mono, layer_final, mix, sr, a_src, source_part, layer_part, carve, align):
    """Measurable checks of the result (not a musicality judgement)."""
    l22, s22 = sa.resample(layer_final, sr), sa.resample(src_mono[: len(layer_final)], sr)
    ea, _ = sa.onset_envelope(l22)
    eb, _ = sa.onset_envelope(s22)
    m = min(len(ea), len(eb))
    dup = float(np.corrcoef(ea[:m], eb[:m])[0, 1]) if m > 8 and ea[:m].std() > 0 and eb[:m].std() > 0 else None
    Cl, _ = sa.chroma(l22)
    kl = sa.estimate_key(Cl.mean(axis=1))
    chroma_sim = float(np.corrcoef(Cl.mean(axis=1), sa.chroma(s22)[0].mean(axis=1))[0, 1])
    bl, _, _ = sa.tempo_and_beats(l22)
    err = min(abs(bl - a_src["bpm"]), abs(bl * 2 - a_src["bpm"]), abs(bl / 2 - a_src["bpm"]))
    return {"clipped_samples": int((np.abs(mix) >= 0.999).sum()), "peak_dbfs": round(20 * math.log10(float(np.abs(mix).max()) + 1e-9), 2),
            "layer_tempo_error_bpm": round(err, 2), "layer_key": kl["name"], "layer_chroma_similarity_to_source": round(chroma_sim, 3),
            "rhythmic_duplication_onset_corr": None if dup is None else round(dup, 3),
            "source_integrity_max_residual": float(np.abs(mix - layer_part - source_part).max()),
            "spectral_overlap_before_carve": round(carve[0], 4), "spectral_overlap_after_carve": round(carve[1], 4), "alignment": align}


def mix_down(x_src, layer, sr, a_src, retained, level_db, p):
    """Sample copy under the retained-bar mask + layers. x_src [n, ch], layer [n] at the same sr. Returns (mix, source_part, layer_part, gain)."""
    n = x_src.shape[0]
    if all(retained):
        mask = np.ones(n, dtype=np.float32)                              # the whole sample, bit-for-bit
    else:
        mask = dsp.bar_gain_envelope(n, sr, a_src["bar_starts_s"], a_src["bar_seconds"], retained)
    lin = 10 ** (level_db / 20)
    gain = (1 - mask) * 1.0 + mask * lin                                 # full level where the sample is absent, user level under it
    source_part = x_src * mask[:, None]
    layer_part = (layer[:n] * gain)[:, None] * np.ones((1, x_src.shape[1]), dtype=np.float32)
    mix = source_part + layer_part
    peak = float(np.abs(mix).max())
    if peak > 0.99:                                                      # peak safety: reduce the layers first, then (if still needed) everything
        lo = float(np.abs(source_part).max())
        if lo < 0.99 and float(np.abs(layer_part).max()) > 0:
            lp = float(np.abs(layer_part).max())
            scale = min(1.0, max(0.0, (0.99 - lo) / lp))
            layer_part = layer_part * scale
            mix = source_part + layer_part
        peak = float(np.abs(mix).max())
        if peak > 0.99:
            k = 0.99 / peak
            mix, source_part, layer_part = mix * k, source_part * k, layer_part * k
    return mix.astype(np.float32), source_part.astype(np.float32), layer_part.astype(np.float32), gain


def compose(x_src, sr_src, settings, generate_layer, emit=lambda **k: None):
    """The model-independent core: analysis -> plan -> layers -> align -> mix -> evaluate. `generate_layer(text_settings, plan, a, ctx, new_s)`
    returns the new layer audio at MusicGen's rate (mono float32). Returns a dict with the mix, parts, analysis, plan and evaluation."""
    p = settings["adherence"]
    src_m = mono(x_src)
    emit(stage="analyse", pct=3, message="Analysing the sample…")
    a = sa.analyse(src_m, sr_src, settings.get("vocals_probability"))
    if a["n_bars"] == 0:                                    # no usable beat grid: fall back to fixed 2-second segments
        a["bar_seconds"] = 2.0
        a["bar_starts_s"] = [float(t) for t in np.arange(0, a["duration_s"] - 1.0, 2.0)]
        a["n_bars"] = len(a["bar_starts_s"])
        a["grid_fallback"] = True
    plan = adherence.controls(p, a["n_bars"])
    retained = adherence.select_retained(a["n_bars"], plan["retain_fraction"])
    plan["retained_bars"] = [int(r) for r in retained]
    n = x_src.shape[0]
    ctx = None
    ctx_end_s = 0.0
    if plan["context_bars"] > 0 and a["n_bars"] >= 2:
        ctx_bars = min(plan["context_bars"], a["n_bars"] - 1)
        ctx_end_s = a["bar_starts_s"][0] + ctx_bars * a["bar_seconds"]
        plan["context_bars"] = ctx_bars
    else:
        plan["context_bars"] = 0
    new_s = n / sr_src - ctx_end_s
    text_settings = build_text_settings(settings, a, plan)
    emit(stage="generate", pct=8, message="Generating the new layers…")
    layer32 = generate_layer(text_settings, plan, a, mono(x_src[: int(ctx_end_s * sr_src)]) if ctx_end_s else None, new_s)
    layer = _resample(layer32, 32000, sr_src)
    if ctx_end_s:
        layer = np.concatenate([np.zeros(int(round(ctx_end_s * sr_src)), dtype=np.float32), layer])
    layer = layer[:n] if len(layer) >= n else np.pad(layer, (0, n - len(layer)))
    emit(stage="align", pct=80, message="Aligning tempo, timing and key…")
    layer, align = align_layer(layer, sr_src, a, plan, src_m)
    layer = layer[:n] if len(layer) >= n else np.pad(layer, (0, n - len(layer)))
    carve = (0.0, 0.0)
    if plan["retain_fraction"] > 0:
        layer, b, af = dsp.spectral_carve(layer, src_m, sr_src, depth_db=plan["carve_depth_db"])
        carve = (b, af)
    emit(stage="mix", pct=90, message="Mixing…")
    mix, source_part, layer_part, gain = mix_down(x_src, layer, sr_src, a, retained, settings["layer_level_db"], p)
    ev = evaluate(src_m, layer, mix, sr_src, a, source_part, layer_part, carve, align)
    return {"mix": mix, "source": x_src, "layers": layer_part, "analysis": a, "plan": plan, "text_settings": text_settings, "evaluation": ev}


def run(job, emit=lambda **k: None, keep_loaded=False):
    """Execute an adapt job (called through the MusicGen engine). Saves mix / source / layers to the library and returns metadata."""
    t_start = time.time()
    c = prompting.validate(job["settings"])
    if c["mode"] != "adapt":
        raise prompting.ValidationError("Not an adapt job.")
    spec = registry.pick_model("adapt")
    hw = hardware.detect()
    device = hardware.choose_device(hw, spec)
    warnings = list(hardware.resource_warnings(hw, spec, device))
    warnings += memory.preflight(spec, hw, device, already_loaded=(spec.id, device, c["fast"]) in generate._CACHE)
    emit(stage="preprocess", pct=1, message="Reading the sample…")
    x_src, sr_src, info = load_source(c["reference"], min(c["duration"] if job["settings"].get("duration") else MAX_WINDOW_S, MAX_WINDOW_S))
    c["vocals_probability"] = sa.vocals_probability(mono(x_src), sr_src)
    if c["vocals_probability"] is not None and c["vocals_probability"] > 0.5:
        warnings.append("The sample appears to contain singing or speech; it is kept as is and no vocals are generated or duplicated.")
    seed = c["seed"] if c["seed"] is not None else int.from_bytes(os.urandom(3), "big")
    import torch
    torch.set_num_threads(generate._threads())
    torch.manual_seed(seed)
    proc = model = None
    try:
        t0 = time.time()
        proc, model = generate._load(spec, device, emit, fast=c["fast"], keep=keep_loaded)
        load_s = time.time() - t0

        def gen_layer(text_settings, plan, a, ctx, new_s):
            prompt = prompting.build_prompt(text_settings, spec.id, c["prompt_style"])
            plan["prompt"] = prompt
            return generate._infer(proc, model, device, prompt, new_s, spec.sample_rate, c, emit, 10, 65,
                                   ref=sa.resample(ctx, sr_src, spec.sample_rate) if ctx is not None else None, label="Layers: ")[-int(new_s * spec.sample_rate):]

        t1 = time.time()
        res = compose(x_src, sr_src, c, gen_layer, emit)
        gen_s = time.time() - t1
    finally:
        del model, proc
        generate.gc.collect()
    ev, plan, a = res["evaluation"], res["plan"], res["analysis"]
    if ev["clipped_samples"]:
        warnings.append(f"{ev['clipped_samples']} samples at full scale.")
    warnings += ["The new layers are mono and centred; the sample keeps its own sample rate and channels.",
                 "Melody/motifs are not analysed and vocals are not available (instrumental only)."]
    warnings += [f"Not applied: {s}" for s in ev["alignment"]["skipped"]]
    emit(stage="export", pct=94, message="Saving…")
    track_id = library.new_id()
    _, wav_path = library.paths(track_id)
    audio.write_audio(wav_path, res["mix"][:, 0] if res["mix"].shape[1] == 1 else res["mix"], sr_src, "wav")
    base = wav_path[:-4]
    audio.write_audio(base + ".source.wav", res["source"][:, 0] if res["source"].shape[1] == 1 else res["source"], sr_src, "wav")
    audio.write_audio(base + ".layers.wav", res["layers"][:, 0] if res["layers"].shape[1] == 1 else res["layers"], sr_src, "wav")
    check = audio.verify_file(wav_path, res["mix"].shape[0] / sr_src)
    stats = audio.analyze(mono(res["mix"]))
    meta = {
        "id": track_id, "title": (c["prompt"] or "Adapted sample")[:60], "status": "complete", "kind": "adapt",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "original_settings": {**job["settings"], **({"reference": job.get("reference_name") or os.path.basename(c["reference"])})},
        "settings": {k: v for k, v in c.items() if k != "reference"},
        "model_prompt": plan.get("prompt"), "vocals": False,
        "model": {"id": spec.id, "name": spec.name, "repo": spec.repo, "license": spec.license}, "device": device, "seed": seed,
        "reference": {**info, "name": job.get("reference_name") or os.path.basename(c["reference"]), "workflow": "adapt"},
        "analysis": a, "plan": plan, "evaluation": ev,
        "audio": {**check, "analysis": stats, "files": {"wav": track_id + ".wav", "source": track_id + ".source.wav", "layers": track_id + ".layers.wav"}},
        "controls": {"enforced": ["adherence-driven retained bars (literal copy of the sample)", "tempo/phase/key alignment strength", "layer level", "seed", "instrumental output"],
                     "approximate": ["genre", "instruments", "mood", "how well the model's layers fit the sample musically"],
                     "note": "The percentage changes retained bars, text hints, alignment and audio conditioning; it is not a volume."},
        "timing": {"load_s": round(load_s, 1), "generate_s": round(gen_s, 1), "total_s": round(time.time() - t_start, 1)},
        "warnings": warnings,
    }
    library.save_meta(track_id, meta)
    emit(stage="done", pct=100, message="Done.")
    return meta
