import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import adherence, prompting, sample_compose as sc  # noqa: E402
from tests.test_adapt_dsp import synth_song  # noqa: E402


def src_stereo(bpm=120):
    x, sr = synth_song(bpm)
    return np.stack([x, 0.9 * x], axis=1).astype(np.float32), sr


def fake_layer(bpm=110, bars=("C", "F", "G", "C", "C", "F", "G", "C")):
    """Stands in for MusicGen: a different-tempo, same-key synthetic 'arrangement' at 32 kHz."""
    def gen(text_settings, plan, a, ctx, new_s):
        x, sr = synth_song(bpm, bars, sr=32000)
        n = int(new_s * 32000)
        x = np.pad(x, (0, max(0, n - len(x))))[:n]
        return (0.5 * x).astype(np.float32)
    return gen


def settings(p, **kw):
    return {"adherence": p, "layer_level_db": -6.0, "genres": [], "instruments": ["piano"], "moods": [], "bpm": None, "key": None, "prompt": "x", **kw}


def corr(a, b):
    a, b = a.mean(axis=1), b.mean(axis=1)
    return float(np.corrcoef(a, b)[0, 1])


def test_controls_are_monotone_continuous_and_not_a_volume():
    prev = None
    for p in range(0, 101, 5):
        c = adherence.controls(p, 8)
        keys = ("retain_fraction", "tempo_hint", "align_strength", "key_hint", "context_bars", "carve_depth_db")
        if prev:
            for k in keys:
                assert c[k] >= prev[k] - 1e-9, (p, k)
            for k, step in (("retain_fraction", 0.2), ("align_strength", 0.2), ("key_hint", 0.2), ("carve_depth_db", 0.5)):
                assert c[k] - prev[k] <= step, f"jump in {k} at {p}"          # gradual, no abrupt mode switches
        prev = c
    t = adherence.table()
    assert t[0]["retain_fraction"] == 0 and t[100]["retain_fraction"] == 1.0
    assert t[100]["context_bars"] >= 1 and t[10]["context_bars"] == 0
    assert "level" not in " ".join(t[10])                                    # the percentage never touches output level


def test_retained_bars_grow_with_adherence():
    counts = [sum(adherence.select_retained(16, adherence.controls(p)["retain_fraction"])) for p in range(0, 101, 10)]
    assert counts == sorted(counts) and counts[0] == 0 and counts[-1] == 16


def test_100_percent_keeps_the_whole_sample_exactly_and_adds_layers_under_it():
    x, sr = src_stereo()
    r = sc.compose(x, sr, settings(100), fake_layer())
    assert np.array_equal(r["plan"]["retained_bars"], [1] * r["analysis"]["n_bars"])
    assert r["evaluation"]["source_integrity_max_residual"] < 1e-6
    # subtract the added layers: what remains is the original sample, sample for sample
    assert np.allclose(r["mix"] - r["layers"], x, atol=1e-6)
    assert float(np.abs(r["layers"]).max()) > 0                              # layers really were added


def test_0_percent_contains_no_source_audio():
    x, sr = src_stereo()
    r = sc.compose(x, sr, settings(0), fake_layer())
    assert sum(r["plan"]["retained_bars"]) == 0
    assert np.allclose(r["mix"], r["layers"], atol=1e-6)                      # everything is new music
    assert r["plan"]["context_bars"] == 0


def test_source_presence_and_tempo_alignment_change_gradually_with_p():
    x, sr = src_stereo(120)
    ps = [0, 10, 25, 50, 75, 100]
    res = {p: sc.compose(x, sr, settings(p), fake_layer(110)) for p in ps}
    sim = [corr(res[p]["mix"], x) for p in ps]
    assert all(b >= a - 0.1 for a, b in zip(sim, sim[1:])), sim              # closer to the sample as p rises (chance-level noise below 0.1 tolerated)
    assert sim[3] > sim[1] + 0.3 and sim[5] > sim[3]                         # clear steps once bars of the sample are kept
    assert sim[-1] > 0.6 and sim[0] < 0.3
    err = [res[p]["evaluation"]["layer_tempo_error_bpm"] for p in ps]
    assert err[-1] <= err[0] + 0.5                                            # alignment never makes the tempo worse
    assert err[-1] < 4.0                                                      # and at high p the layers sit on the sample's tempo
    assert res[0]["evaluation"]["alignment"]["stretch_rate"] == 1.0
    assert 1.0 < res[50]["evaluation"]["alignment"]["stretch_rate"] <= 1.1 or res[50]["plan"]["context_bars"] > 0


def test_layer_level_is_a_separate_control_from_adherence():
    x, sr = src_stereo()
    x = (x * 0.4).astype(np.float32)                                         # headroom, so peak safety does not equalise the two mixes
    quiet = sc.compose(x, sr, settings(100, layer_level_db=-18.0), fake_layer())
    loud = sc.compose(x, sr, settings(100, layer_level_db=-3.0), fake_layer())
    assert np.abs(quiet["layers"]).max() < np.abs(loud["layers"]).max()
    assert np.array_equal(quiet["plan"]["retained_bars"], loud["plan"]["retained_bars"])   # same adherence, different mix level


def test_peak_safety_reduces_layers_before_touching_the_sample():
    x, sr = src_stereo()
    x = (x / np.abs(x).max() * 0.85).astype(np.float32)
    r = sc.compose(x, sr, settings(100, layer_level_db=0.0), lambda *a: np.ones(int(a[4] * 32000), dtype=np.float32) * 0.8)
    assert np.abs(r["mix"]).max() <= 0.9901
    assert np.allclose(r["mix"] - r["layers"], x, atol=1e-5)                  # the sample itself was not scaled


def test_validation_adapt_mode_and_no_vocals():
    ok = prompting.validate({"prompt": "x", "mode": "adapt", "reference": "r.wav", "adherence": "40", "layer_level_db": -9})
    assert ok["adherence"] == 40.0 and ok["layer_level_db"] == -9.0
    for bad in ({"adherence": None}, {"adherence": 101}, {"adherence": "lots"}, {"layer_level_db": 3}):
        with pytest.raises(prompting.ValidationError):
            prompting.validate({"prompt": "x", "mode": "adapt", "reference": "r.wav", "adherence": 50, **bad})
    with pytest.raises(prompting.ValidationError, match="Music with Vocals is not available"):
        prompting.validate({"prompt": "x", "mode": "adapt", "reference": "r.wav", "adherence": 50, "vocals": True})
    assert prompting.validate({"prompt": "x"})["adherence"] is None           # other modes are unaffected


def test_plan_hints_are_written_into_the_text_settings():
    x, sr = src_stereo(120)
    r = sc.compose(x, sr, settings(100), fake_layer())
    ts = r["text_settings"]
    assert abs(ts["bpm"] - r["analysis"]["bpm"]) < 1.0 and ts["key"] == r["analysis"]["key"]["name"]
    r0 = sc.compose(x, sr, settings(0), fake_layer())
    assert r0["text_settings"]["bpm"] is None and r0["text_settings"]["key"] is None     # 0 %: the sample's tempo/key are not imposed
