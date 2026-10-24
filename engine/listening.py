"""Listening Pack: run the real adapt pipeline at every adherence level on one or more samples, measure each result against its
reference, check that the levels behave as promised, and write a blind pack for a human listening test.

  run(spec, emit)    spec = {"sources": [{"label", "kind": "full"|"instrumental"|"vocals", "path", "origin"}], "levels": ["low", ...],
                             "seeds": [5, 6], "instruments": [...], "genres": [...], "moods": [...], "layer_level_db": -6, "seconds": 20,
                             "out_dir": <run dir>}
  score(run_dir, ratings)   unblinds the human ratings and relates them to the levels.

Files in <run dir>:  report.json (everything measured), lib/ (the generated tracks incl. .source/.layers parts, kept out of the user's
library), ref/<source>.wav (reference windows), blind/<code>.wav (mixes under neutral names), key.json (code -> source/level/seed; the
server does not reveal it before ratings are saved), ratings.json + rating_summary in report.json after scoring.

What is checked automatically (per source; each check reports its numbers, PASS/FAIL and why):
  monotonic      the mix's adherence_index rises with the level (ties within 0.01 tolerated)
  distinct       adjacent levels differ by more than twice the seed-to-seed spread (and by >= 0.02)
  consistent     seed-to-seed spread of the index at each level <= 0.05
  mechanism      low keeps no bars; strict keeps all bars and copies them bit-exactly
  integrity      retained audio is a bit-exact copy (residual <= 1e-5)
  quality        no clipped samples; no click above max(8, 2x the reference's click score); level within +-6 dB of the reference
  vocals         (references with vocals) vocal pitch-contour agreement rises with the level
  instruments    (if instruments were selected) CLAP rank of the selected instruments in the ADDED layers vs. in the reference
Nothing here is a musical judgement: the blind pack exists because these measures cannot hear."""
import json
import os
import random
import shutil
import string
import time

import numpy as np
import soundfile as sf

from . import adherence, adherence_eval as ae, sample_analysis as sa, vocal_layers

LEVEL_ORDER = list(adherence.LEVELS)


def _clap_fn():
    """CLAP audio embedding (unit vector) or None if the tagging model is not installed."""
    from . import sampling, sampling_models
    if not sampling_models.clap_ready():
        return None

    def emb(x, sr):
        e, _ = sampling._audio_embedding(sa.resample(np.asarray(x, np.float32), sr, 48000), windows=3)
        return e[0].numpy()
    return emb


def _vocab_match(name, vocab):
    n = name.lower().strip()
    alias = {"strings": "string section", "drums": "drum kit", "bass": "bass guitar", "synth": "synthesizer pads", "synthesizer": "synthesizer pads",
             "guitar": "acoustic guitar", "brass": "brass section", "sax": "saxophone", "keys": "electric piano", "vocals": "female vocals"}
    n = alias.get(n, n)
    if n in vocab:
        return n
    hits = [v for v in vocab if n in v or v in n]
    return hits[0] if hits else None


def tag_ranks(x, sr, wanted, key="instrument"):
    """Rank (1 = best) and relative score of each wanted label among CLAP's candidates for `key` ('instrument' or 'genre')."""
    from . import sampling
    vocab = sampling.INSTRUMENTS if key == "instrument" else sampling.GENRES
    emb, _ = sampling._audio_embedding(sa.resample(np.asarray(x, np.float32), sr, 48000), windows=3)
    ranked = sampling._rank(emb, key, len(vocab))
    pos = {r["label"]: (i + 1, r["score"]) for i, r in enumerate(ranked)}
    out = {}
    for w in wanted:
        lab = _vocab_match(w, vocab)
        if lab:
            out[w] = {"label": lab, "rank": pos[lab][0], "score": pos[lab][1], "of": len(vocab)}
    return out


def _mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def _mean_sd(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None
    return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)


def _code(rng, used):
    while True:
        c = "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + str(rng.randint(10, 99))
        if c not in used:
            used.add(c)
            return c


def run(spec, emit=lambda **k: None):
    out_dir = spec["out_dir"]
    lib = os.path.join(out_dir, "lib")
    for d in (lib, os.path.join(out_dir, "ref"), os.path.join(out_dir, "blind"), os.path.join(out_dir, "stems")):
        os.makedirs(d, exist_ok=True)
    prev_lib = os.environ.get("VIBEFLOW_LIBRARY")
    os.environ["VIBEFLOW_LIBRARY"] = lib                     # generated tracks stay inside the run, not in "Your Songs"
    from . import engines, library, sampling, sampling_models
    levels = [l for l in LEVEL_ORDER if l in (spec.get("levels") or LEVEL_ORDER)]
    seeds = [int(s) for s in (spec.get("seeds") or [5])]
    secs = float(spec.get("seconds") or 20)
    sel_instr, sel_genres = list(spec.get("instruments") or []), list(spec.get("genres") or [])
    clap = _clap_fn()
    stems_ok = sampling_models.demucs_ready()
    rng = random.Random(spec.get("blind_seed") or time.time())
    used, key, blind = set(), {}, []
    report = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": spec.get("name") or "Listening pack", "levels": {l: adherence.LEVELS[l] for l in levels},
              "seeds": seeds, "seconds": secs, "instruments": sel_instr, "genres": sel_genres, "moods": spec.get("moods") or [],
              "layer_level_db": spec.get("layer_level_db", -6), "clap": clap is not None, "stems_model": stems_ok, "sources": [], "status": "running"}
    total = len(spec["sources"]) * len(levels) * len(seeds)
    done = 0
    t_start = time.time()

    def save():
        with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)

    try:
        for si, src in enumerate(spec["sources"]):
            sid = f"s{si + 1}"
            x, sr = sf.read(src["path"], dtype="float32", always_2d=True, frames=int(secs * sf.info(src["path"]).samplerate))
            x = x[:, :2]
            ref_path = os.path.join(out_dir, "ref", f"{sid}.wav")
            sf.write(ref_path, x, sr, subtype="FLOAT")
            ref_m = _mono(x)
            entry = {"id": sid, "label": src.get("label") or os.path.basename(src["path"]), "kind": src.get("kind", "full"), "origin": src.get("origin"),
                     "ref": f"ref/{sid}.wav", "sample_rate": sr, "duration_s": round(len(x) / sr, 2), "runs": [], "reference": {"quality": ae.quality(x, sr)}}
            # reference vocals (for vocal comparisons): the stem itself for a voice sample, Demucs for a full mix
            ref_vox = None
            if entry["kind"] == "vocals":
                ref_vox = x
            elif entry["kind"] == "full" and stems_ok:
                emit(stage="stems", pct=round(100 * done / total), message=f"{entry['label']}: separating the reference…")
                st = sampling.separate(ref_path, os.path.join(out_dir, "stems", sid))
                if sampling.vocal_presence(st):
                    ref_vox = sampling.load_audio(os.path.join(out_dir, "stems", sid, "vocals.wav"), sr)
                    inst = sampling.load_audio(os.path.join(out_dir, "stems", sid, "instrumental.wav"), sr)
                    _, _, entry["reference"]["vocal_layers"] = vocal_layers.analyse(ref_vox, inst, sr)
            entry["reference"]["has_vocals"] = ref_vox is not None
            if ref_vox is not None:
                entry["reference"]["vocals"] = ae.vocal_metrics(_mono(ref_vox), sr)
            if sel_instr and clap is not None:
                entry["reference"]["instrument_ranks"] = tag_ranks(ref_m, sr, sel_instr)
            if sel_genres and clap is not None:
                entry["reference"]["genre_ranks"] = tag_ranks(ref_m, sr, sel_genres, "genre")
            report["sources"].append(entry)
            for lv in levels:
                for seed in seeds:
                    label = f"{entry['label']} · {lv} · seed {seed}"

                    def em(**k):
                        frac = (done + (k.get("pct") or 0) / 100.0) / total
                        el = time.time() - t_start
                        emit(stage=k.get("stage", "generate"), pct=round(100 * frac, 1), message=f"{label}: {k.get('message', '')}",
                             eta_s=round(el / frac * (1 - frac)) if frac > 0.02 else None)
                    job = {"settings": {"mode": "adapt", "reference": ref_path, "adherence_level": lv, "instruments": sel_instr, "genres": sel_genres,
                                        "moods": spec.get("moods") or [], "seed": seed, "prompt": spec.get("prompt", ""),
                                        "layer_level_db": spec.get("layer_level_db", -6), "duration": int(min(30, round(secs)))},
                           "reference_name": entry["label"]}
                    meta = engines.get().generate(job, em, keep_loaded=True)
                    wav = library.paths(meta["id"])[1]
                    mix, _ = sf.read(wav, dtype="float32", always_2d=True)
                    lay, _ = sf.read(wav[:-4] + ".layers.wav", dtype="float32", always_2d=True)
                    em(stage="measure", pct=96, message="measuring…")
                    r = {"level": lv, "adherence": meta["plan"]["adherence"], "seed": seed, "track": meta["id"],
                         "file": f"lib/{meta['id']}.wav", "layers_file": f"lib/{meta['id']}.layers.wav",
                         "bars_kept": int(sum(meta["plan"]["retained_bars"])), "bars_total": len(meta["plan"]["retained_bars"]),
                         "context_bars": meta["plan"]["context_bars"], "prompt": meta.get("model_prompt"),
                         "integrity_residual": meta["evaluation"]["source_integrity_max_residual"],
                         "alignment": meta["evaluation"]["alignment"], "generate_s": meta["timing"]["generate_s"],
                         "tempo_retry": meta["evaluation"].get("tempo_retry"), "layer_level_match_db": meta["evaluation"].get("layer_level_match_db"),
                         "mix": ae.compare(ref_m, _mono(mix), sr, clap), "layers": ae.compare(ref_m, _mono(lay), sr, clap)}
                    if ref_vox is not None and stems_ok:
                        sd = os.path.join(out_dir, "stems", f"{sid}_{lv}_{seed}")
                        sampling.separate(wav, sd)
                        ov = sampling.load_audio(os.path.join(sd, "vocals.wav"), sr)
                        r["vocals"] = {**ae.vocal_compare(_mono(ref_vox), _mono(ov), sr), **{"out_" + k: v for k, v in ae.vocal_metrics(_mono(ov), sr).items()}}
                        shutil.rmtree(sd, ignore_errors=True)
                    if sel_instr and clap is not None:
                        r["instrument_ranks_layers"] = tag_ranks(_mono(lay), sr, sel_instr)
                    if sel_genres and clap is not None:
                        r["genre_ranks_mix"] = tag_ranks(_mono(mix), sr, sel_genres, "genre")
                    code = _code(rng, used)
                    shutil.copyfile(wav, os.path.join(out_dir, "blind", code + ".wav"))
                    key[code] = {"source": sid, "level": lv, "seed": seed, "track": meta["id"]}
                    blind.append({"code": code, "source": sid})
                    entry["runs"].append(r)
                    done += 1
                    save()
            entry["summary"], entry["checks"] = summarise(entry, levels)
            save()
        rng.shuffle(blind)
        report["blind"] = blind
        report["overall"] = overall(report)
        report["status"] = "complete"
        report["total_s"] = round(time.time() - t_start, 1)
        with open(os.path.join(out_dir, "key.json"), "w", encoding="utf-8") as f:
            json.dump(key, f, indent=1)
        save()
        emit(stage="done", pct=100, message="Listening pack ready.")
        return report
    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)[:300]
        save()
        raise
    finally:
        if prev_lib is None:
            os.environ.pop("VIBEFLOW_LIBRARY", None)
        else:
            os.environ["VIBEFLOW_LIBRARY"] = prev_lib


DIMS = [("adherence_index", "Adherence index"), ("tempo_sim", "Tempo (periodicity)"), ("rhythm_corr", "Rhythm"), ("chroma_frame_sim", "Harmony / melody"),
        ("pitch_class_agree", "Dominant pitch class"), ("spectral_profile_sim", "Instrumentation (spectrum)"), ("clap_sim", "Instrumentation (CLAP)"),
        ("structure_sim", "Structure"), ("dynamics_corr", "Dynamics"), ("ref_presence", "Reference present")]


def summarise(entry, levels):
    runs = entry["runs"]
    by = {lv: [r for r in runs if r["level"] == lv] for lv in levels}
    summ = {}
    for lv, rs in by.items():
        s = {"adherence": rs[0]["adherence"] if rs else None, "bars_kept": rs[0]["bars_kept"] if rs else None, "bars_total": rs[0]["bars_total"] if rs else None}
        for part in ("mix", "layers"):
            s[part] = {k: _mean_sd([r[part].get(k) for r in rs]) for k, _ in DIMS}
            s[part]["tempo_error_bpm"] = _mean_sd([r[part].get("tempo_error_bpm") for r in rs])
        s["quality"] = {k: _mean_sd([r["mix"]["quality"][k] for r in rs]) for k in ("clipped", "click_score", "level_diff_db", "peak_dbfs")}
        if any("vocals" in r for r in rs):
            s["vocals"] = {k: _mean_sd([r["vocals"].get(k) for r in rs if "vocals" in r]) for k in ("pitch_contour_agree", "vocal_envelope_corr", "ref_voiced_kept")}
        if any("instrument_ranks_layers" in r for r in rs):
            s["instrument_ranks_layers"] = {w: _mean_sd([r["instrument_ranks_layers"][w]["rank"] for r in rs if w in r["instrument_ranks_layers"]])
                                            for w in rs[0]["instrument_ranks_layers"]}
        summ[lv] = s
    checks = []
    idx = [summ[lv]["mix"]["adherence_index"][0] for lv in levels]
    sds = [summ[lv]["mix"]["adherence_index"][1] or 0.0 for lv in levels]
    mono_ok = all(b >= a - 0.01 for a, b in zip(idx, idx[1:]))
    checks.append({"id": "monotonic", "pass": mono_ok, "detail": "mix adherence index by level: " + ", ".join(f"{lv} {v:.3f}" for lv, v in zip(levels, idx))})
    for (la, a, sa_), (lb, b, sb) in zip(zip(levels, idx, sds), list(zip(levels, idx, sds))[1:]):
        need = max(0.02, 2 * max(sa_, sb))
        checks.append({"id": f"distinct {la}->{lb}", "pass": (b - a) > need, "detail": f"difference {b - a:+.3f}, needs > {need:.3f} (2x seed spread, min 0.02)"})
    worst = max(sds) if sds else 0
    checks.append({"id": "consistent", "pass": worst <= 0.05, "detail": f"largest seed-to-seed SD of the index: {worst:.3f} (limit 0.05)",
                   "skipped": len(runs) == len(levels)})
    if "low" in summ:
        checks.append({"id": "mechanism low", "pass": summ["low"]["bars_kept"] == 0, "detail": f"low keeps {summ['low']['bars_kept']} of {summ['low']['bars_total']} bars (expected 0)"})
    resid = max(r["integrity_residual"] for r in runs)
    if "strict" in summ:                 # all bars kept + the copied part is bit-exact. ref_presence is shown only: it is a projection,
        pres = summ["strict"]["mix"]["ref_presence"][0]   # pulled away from 1 whenever the added layers correlate with the sample
        sres = max(r["integrity_residual"] for r in runs if r["level"] == "strict")
        ok = summ["strict"]["bars_kept"] == summ["strict"]["bars_total"] and sres <= 1e-5
        checks.append({"id": "mechanism strict", "pass": ok, "detail": f"strict keeps {summ['strict']['bars_kept']} of {summ['strict']['bars_total']} bars, copied exactly (residual {sres:.1e}); reference projection {pres}"})
    checks.append({"id": "integrity", "pass": resid <= 1e-5, "detail": f"largest residual between the retained audio and the original: {resid:.2e}"})
    ref_click = entry["reference"]["quality"]["click_score"]
    lim = max(8.0, 2 * ref_click)
    bad = [f"{r['level']}/seed {r['seed']}: " + ", ".join(p for p, c in (("clipped", r["mix"]["quality"]["clipped"] > 0), (f"click {r['mix']['quality']['click_score']}", r["mix"]["quality"]["click_score"] > lim),
                                                                         (f"level {r['mix']['quality']['level_diff_db']} dB", abs(r["mix"]["quality"]["level_diff_db"]) > 6)) if c)
           for r in runs if r["mix"]["quality"]["clipped"] > 0 or r["mix"]["quality"]["click_score"] > lim or abs(r["mix"]["quality"]["level_diff_db"]) > 6]
    checks.append({"id": "quality", "pass": not bad, "detail": ("; ".join(bad) if bad else f"no clipping, clicks below {lim:.1f} (reference {ref_click}), level within 6 dB")})
    if any("vocals" in summ[lv] for lv in levels):
        va = [summ[lv].get("vocals", {}).get("pitch_contour_agree", (None,))[0] for lv in levels]
        vv = [v for v in va if v is not None]
        checks.append({"id": "vocals follow level", "pass": len(vv) >= 2 and all(b >= a - 0.02 for a, b in zip(vv, vv[1:])),
                       "detail": "vocal pitch-contour agreement by level: " + ", ".join(f"{lv} {v}" for lv, v in zip(levels, va))})
    if any("instrument_ranks_layers" in summ[lv] for lv in levels):
        ref_r = entry["reference"].get("instrument_ranks", {})
        parts = []
        for w, rr in ref_r.items():
            lay = [summ[lv].get("instrument_ranks_layers", {}).get(w, (None,))[0] for lv in levels]
            parts.append(f"{w}: reference rank {rr['rank']}/{rr['of']}, added layers " + ", ".join(f"{lv} {v}" for lv, v in zip(levels, lay)))
        checks.append({"id": "instruments (info)", "pass": None, "detail": "; ".join(parts)})
    return summ, checks


def overall(report):
    """Across sources: does each level keep the same ordering on every sample type?"""
    order = {}
    for e in report["sources"]:
        idx = [e["summary"][lv]["mix"]["adherence_index"][0] for lv in report["levels"]]
        order[e["label"]] = idx
    same = all(all(b >= a - 0.01 for a, b in zip(v, v[1:])) for v in order.values())
    fails = [f"{e['label']}: {c['id']}" for e in report["sources"] for c in e["checks"] if c["pass"] is False]
    return {"ordering_by_source": order, "levels_consistent_across_sources": same, "failed_checks": fails}


def score(run_dir, ratings):
    """ratings: {code: {"closeness": 1-5, "quality": 1-5, "guess": "low"|...|null}}. Writes ratings.json and returns the summary."""
    with open(os.path.join(run_dir, "key.json"), encoding="utf-8") as f:
        key = json.load(f)
    rows = []
    for code, r in ratings.items():
        if code in key:
            rows.append({**key[code], "code": code, **{k: r.get(k) for k in ("closeness", "quality", "guess")}})
    with open(os.path.join(run_dir, "ratings.json"), "w", encoding="utf-8") as f:
        json.dump({"ratings": ratings, "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=1)
    by = {}
    for r in rows:
        by.setdefault(r["level"], []).append(r)
    per = {lv: {"n": len(v), "closeness": _mean_sd([x["closeness"] for x in v if x["closeness"]])[0],
                "quality": _mean_sd([x["quality"] for x in v if x["quality"]])[0]} for lv, v in by.items()}
    rated = [r for r in rows if r["closeness"]]
    rho = None
    if len(rated) >= 4:
        from scipy.stats import spearmanr
        rho = spearmanr([LEVEL_ORDER.index(r["level"]) for r in rated], [r["closeness"] for r in rated])
        rho = {"rho": round(float(rho.correlation), 3), "p": round(float(rho.pvalue), 4), "n": len(rated)}
    guesses = [r for r in rows if r.get("guess")]
    acc = round(sum(r["guess"] == r["level"] for r in guesses) / len(guesses), 3) if guesses else None
    summary = {"per_level": {lv: per[lv] for lv in LEVEL_ORDER if lv in per}, "closeness_vs_level_spearman": rho, "level_guess_accuracy": acc,
               "chance_guess_accuracy": round(1 / len({k['level'] for k in key.values()}), 3), "rows": rows}
    rp = os.path.join(run_dir, "report.json")
    with open(rp, encoding="utf-8") as f:
        rep = json.load(f)
    rep["rating_summary"] = summary
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return summary
