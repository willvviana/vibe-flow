"""Adherence levels, adherence metrics, lead/backing vocal split and the Listening Pack checks (no model needed; signals with known answers)."""
import json
import os

import numpy as np
import pytest

from engine import adherence, adherence_eval as ae, listening, prompting, vocal_layers as vl

SR = 32000


def tone(f, s=4.0, a=0.3, sr=SR):
    t = np.arange(int(s * sr)) / sr
    return (a * np.sin(2 * np.pi * f * t) + 0.3 * a * np.sin(2 * np.pi * 2 * f * t)).astype(np.float32)


def clicks(bpm, s=12.0, sr=SR):
    x = np.zeros(int(s * sr), np.float32)
    for t in np.arange(0, s, 60 / bpm):
        i = int(t * sr)
        x[i:i + 400] += np.hanning(400) * np.sin(np.arange(400) * 0.5)
    return x


def si_sdr(est, ref):
    a = np.dot(est, ref) / np.dot(ref, ref)
    e = est - a * ref
    return 10 * np.log10(np.dot(a * ref, a * ref) / np.dot(e, e))


# ------------------------------------------------------------------ levels
def test_named_levels_map_to_fixed_percentages_and_are_ordered():
    assert list(adherence.LEVELS) == ["low", "medium", "high", "strict"]
    vals = list(adherence.LEVELS.values())
    assert vals == sorted(vals) and vals[-1] == 100.0
    assert adherence.level_value("High") == 75.0 and adherence.level_name(50) == "medium" and adherence.level_name(60) is None


def test_levels_differ_in_mechanism_not_only_in_number():
    c = {k: adherence.controls(v, 16) for k, v in adherence.LEVELS.items()}
    kept = {k: sum(adherence.select_retained(16, c[k]["retain_fraction"])) for k in c}
    assert kept["low"] == 0 and 0 < kept["medium"] < kept["high"] < kept["strict"] == 16
    assert c["low"]["context_bars"] == 0 and c["medium"]["context_bars"] >= 1            # audio conditioning starts at medium
    assert c["low"]["key_hint"] < 0.5 <= c["medium"]["key_hint"]                         # key follows the sample from medium
    assert c["low"]["tempo_hint"] >= 0.5                                                 # tempo always follows


def test_validate_accepts_level_names_and_rejects_unknown():
    base = {"prompt": "x", "mode": "adapt", "reference": "r.wav"}
    v = prompting.validate({**base, "adherence_level": "strict"})
    assert v["adherence"] == 100.0 and v["adherence_level"] == "strict"
    v = prompting.validate({**base, "adherence": 60})
    assert v["adherence"] == 60.0 and v["adherence_level"] is None
    v = prompting.validate({**base, "adherence": 10, "adherence_level": "medium"})       # the named level wins
    assert v["adherence"] == 50.0
    with pytest.raises(prompting.ValidationError):
        prompting.validate({**base, "adherence_level": "extreme"})


# ------------------------------------------------------------------ adherence metrics
def test_identical_audio_scores_full_adherence_and_presence():
    rng = np.random.default_rng(1)
    x = clicks(100) + tone(220, 12) * 0.5 + rng.standard_normal(int(12 * SR)).astype(np.float32) * 0.01
    r = ae.compare(x, x, SR)
    assert r["adherence_index"] > 0.97 and abs(r["ref_presence"] - 1) < 1e-6 and r["tempo_error_bpm"] < 0.5 and r["key_match"] == 1.0


def test_metrics_separate_related_from_unrelated_audio():
    ref = clicks(100) + tone(220, 12) * 0.5
    same_beat_other_note = clicks(100) + tone(311, 12) * 0.5
    noise = (np.random.default_rng(2).standard_normal(int(12 * SR)) * 0.1).astype(np.float32)
    a, b = ae.compare(ref, same_beat_other_note, SR), ae.compare(ref, noise, SR)
    assert a["rhythm_corr"] > 0.9 > 0.3 > b["rhythm_corr"]
    assert a["adherence_index"] > b["adherence_index"]
    assert ae.compare(ref, ref * 0.5 + same_beat_other_note * 0.5, SR)["ref_presence"] == pytest.approx(0.5 + 0.5 * a["ref_presence"], abs=0.02)


def test_tempo_error_is_measured_and_octave_folded():
    assert ae.compare(clicks(100), clicks(120), SR)["tempo_error_bpm"] > 15
    assert ae.compare(clicks(100), clicks(200), SR)["tempo_error_bpm"] < 2              # double time is the same tempo


def test_tempo_periodicity_is_continuous_and_level_independent():
    ref = clicks(100) + tone(220, 12) * 0.2
    same, other = ae.compare(ref, clicks(100) * 0.9 + tone(330, 12) * 0.2, SR), ae.compare(ref, clicks(133) + tone(220, 12) * 0.2, SR)
    assert same["tempo_sim"] > 0.8 > other["tempo_sim"]
    quieter = ae.compare(ref, (clicks(100) * 0.9 + tone(330, 12) * 0.2) * 10 ** (-1 / 20), SR)
    assert quieter["tempo_sim"] == pytest.approx(same["tempo_sim"], abs=1e-3)       # a 1 dB change does not move it


def test_quality_flags_clipping_and_clicks():
    x = tone(220) * 0.5
    y = x.copy()
    y[SR] = 1.0
    y[SR + 1] = -1.0
    assert ae.quality(x, SR)["click_score"] < 3
    q = ae.quality(y, SR)
    assert q["clipped"] == 2 and q["click_score"] > 10


@pytest.mark.parametrize("f", [110.0, 196.0, 440.0, 659.25])
def test_f0_tracker_is_within_a_few_cents(f):
    f0, _ = ae.f0_track(tone(f), SR)
    v = f0[f0 > 0]
    assert len(v) > 0.9 * len(f0)
    assert abs(1200 * np.log2(np.median(v) / f)) < 5


def test_vocal_compare_tells_same_from_different_melody():
    a, b = tone(220), tone(247)
    assert ae.vocal_compare(a, a, SR)["pitch_contour_agree"] == 1.0
    assert ae.vocal_compare(a, b, SR)["pitch_contour_agree"] < 0.05
    assert ae.vocal_compare(a, tone(440), SR)["pitch_contour_agree"] == 1.0            # an octave double is the same melody


# ------------------------------------------------------------------ lead / backing split
def stereo(x, p):
    th = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)], 1).astype(np.float32)


def test_split_recovers_centre_lead_and_adds_back_exactly():
    lead = tone(220, a=0.3)
    back = stereo(tone(277.18, a=0.12), -0.9) + stereo(tone(329.63, a=0.12), 0.9)
    v = stereo(lead, 0) + back
    L, B, info = vl.split(v, SR)
    assert np.abs(L + B - v).max() < 1e-5                                               # nothing is lost or invented
    assert si_sdr(L.mean(1), stereo(lead, 0).mean(1)) > 10
    assert 0.05 < info["backing_share"] < 0.5


def test_mono_vocals_have_no_backing_layer():
    L, B, info = vl.split(np.repeat(tone(220)[:, None], 2, 1), SR)
    assert not info["stereo"] and info["backing_share"] == 0 and not B.any()


def test_harmony_finds_the_planted_intervals():
    lead = tone(220, 6, a=0.3)                                                           # A3
    back = stereo(tone(277.18, 6, a=0.15), -0.9) + stereo(tone(329.63, 6, a=0.15), 0.9)  # C#4 (major 3rd) left, E4 (5th) right
    L, B, _ = vl.split(stereo(lead, 0) + back, SR)
    got = {i["semitones"] for i in vl.harmony(L.mean(1), B, SR)["intervals"]}
    assert {4, 7} <= got


# ------------------------------------------------------------------ pack checks and scoring
def _run(level, idx, seed, kept, total, presence=0.1, resid=0.0, click=2.0):
    m = {"adherence_index": idx, "ref_presence": presence, "quality": {"clipped": 0, "click_score": click, "level_diff_db": 0.0, "peak_dbfs": -1.0}}
    m.update({k: idx for k, _ in listening.DIMS if k not in m})
    m["tempo_error_bpm"] = 1.0
    return {"level": level, "adherence": adherence.LEVELS[level], "seed": seed, "bars_kept": kept, "bars_total": total, "integrity_residual": resid,
            "mix": m, "layers": dict(m)}


def _entry(runs):
    return {"runs": runs, "reference": {"quality": {"click_score": 2.0}}}


def test_checks_pass_for_well_behaved_levels():
    runs = [_run("low", 0.30, 1, 0, 8), _run("low", 0.31, 2, 0, 8), _run("medium", 0.60, 1, 3, 8), _run("medium", 0.62, 2, 3, 8),
            _run("high", 0.75, 1, 6, 8), _run("high", 0.76, 2, 6, 8), _run("strict", 0.98, 1, 8, 8, 1.0), _run("strict", 0.98, 2, 8, 8, 1.0)]
    _, checks = listening.summarise(_entry(runs), listening.LEVEL_ORDER)
    assert all(c["pass"] for c in checks if c["pass"] is not None), checks


def test_checks_catch_overlapping_levels_inconsistency_and_artefacts():
    runs = [_run("low", 0.30, 1, 0, 8), _run("low", 0.50, 2, 0, 8), _run("medium", 0.45, 1, 3, 8), _run("medium", 0.47, 2, 3, 8, click=40.0),
            _run("high", 0.75, 1, 6, 8, resid=0.01), _run("high", 0.76, 2, 6, 8), _run("strict", 0.98, 1, 7, 8, 1.0), _run("strict", 0.98, 2, 7, 8, 1.0)]
    _, checks = listening.summarise(_entry(runs), listening.LEVEL_ORDER)
    failed = {c["id"] for c in checks if c["pass"] is False}
    assert {"distinct low->medium", "consistent", "integrity", "quality", "mechanism strict"} <= failed


def test_score_unblinds_ratings(tmp_path):
    key = {"AB10": {"source": "s1", "level": "low", "seed": 1}, "CD20": {"source": "s1", "level": "medium", "seed": 1},
           "EF30": {"source": "s1", "level": "high", "seed": 1}, "GH40": {"source": "s1", "level": "strict", "seed": 1}}
    (tmp_path / "key.json").write_text(json.dumps(key))
    (tmp_path / "report.json").write_text(json.dumps({"status": "complete"}))
    ratings = {"AB10": {"closeness": 1, "quality": 4, "guess": "low"}, "CD20": {"closeness": 2, "quality": 4, "guess": "high"},
               "EF30": {"closeness": 4, "quality": 3, "guess": "high"}, "GH40": {"closeness": 5, "quality": 5, "guess": "strict"}}
    s = listening.score(str(tmp_path), ratings)
    assert s["closeness_vs_level_spearman"]["rho"] == 1.0 and s["level_guess_accuracy"] == 0.75
    assert json.loads((tmp_path / "report.json").read_text())["rating_summary"]["per_level"]["strict"]["closeness"] == 5
    assert os.path.exists(tmp_path / "ratings.json")
