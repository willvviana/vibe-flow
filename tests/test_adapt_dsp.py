import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import dsp, sample_analysis as sa  # noqa: E402

SR = 22050
NOTE_HZ = {"C": 261.63, "E": 329.63, "G": 392.0, "F": 349.23, "A": 440.0, "B": 493.88, "D": 293.66}


def synth_song(bpm=120, bars=("C", "F", "G", "C", "C", "F", "G", "C"), sr=SR):
    """Triads on a 4/4 grid with a low kick on every beat and a hat on the off-beats: a stand-in for real music."""
    beat = 60 / bpm
    n = int(len(bars) * 4 * beat * sr)
    x = np.zeros(n, dtype=np.float32)
    triad = {"C": "CEG", "F": "FAC", "G": "GBD"}
    for b, ch in enumerate(bars):
        t = np.arange(int(4 * beat * sr)) / sr
        env = np.minimum(1, t * 20) * np.exp(-t * 0.8)
        for note in triad[ch]:
            f = NOTE_HZ.get(note, 523.25)
            x[int(b * 4 * beat * sr): int(b * 4 * beat * sr) + len(t)] += 0.12 * env * np.sin(2 * np.pi * f * t)
    for k in range(len(bars) * 4):
        i = int(k * beat * sr)
        kt = np.arange(int(0.12 * sr)) / sr
        x[i: i + len(kt)] += 0.8 * np.sin(2 * np.pi * 60 * kt) * np.exp(-kt * 30)
        j = int((k + 0.5) * beat * sr)
        if j + 800 < n:
            x[j: j + 800] += 0.15 * np.random.default_rng(k).standard_normal(800) * np.exp(-np.arange(800) / 150)
    return x, sr


def test_analysis_finds_tempo_key_and_chords():
    x, sr = synth_song(120)
    a = sa.analyse(x, sr)
    assert abs(a["bpm"] - 120) <= 3 or abs(a["bpm"] * 2 - 120) <= 3 or abs(a["bpm"] / 2 - 120) <= 3
    assert a["key"]["name"] == "C major", a["key"]
    assert a["n_bars"] >= 6
    assert a["chords_per_bar"][:4] == ["C", "F", "G", "C"], a["chords_per_bar"]
    assert a["vocals_probability"] is None                      # unknown unless the tagger is cached and passed in
    assert a["bands"]["low"] > 0.0 and abs(sum(a["bands"].values()) - 1) < 1e-6
    assert "melody/motifs" in a["not_analysed"]                 # the analysis states what it does NOT do


def test_tempo_is_not_fooled_by_a_different_bpm():
    x, sr = synth_song(96)
    bpm = sa.analyse(x, sr)["bpm"]
    assert min(abs(bpm - 96), abs(bpm * 2 - 96), abs(bpm / 2 - 96)) <= 3


def _peak_hz(y, sr):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / sr)[int(np.argmax(s))]


def test_time_stretch_changes_length_not_pitch():
    sr = 22050
    t = np.arange(sr * 2) / sr
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = dsp.time_stretch(x, 1.1)
    assert abs(len(y) / len(x) - 1 / 1.1) < 0.02
    assert abs(_peak_hz(y[2000:-2000], sr) - 440) < 6
    assert np.array_equal(dsp.time_stretch(x, 1.0), x)


def test_pitch_shift_moves_pitch_keeps_length():
    sr = 22050
    t = np.arange(sr * 2) / sr
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = dsp.pitch_shift(x, 2)
    assert len(y) == len(x)
    assert abs(_peak_hz(y[3000:-3000], sr) / 440 - 2 ** (2 / 12)) < 0.02


def test_spectral_carve_reduces_overlap_with_the_source():
    sr = 22050
    t = np.arange(sr * 3) / sr
    src = (np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    layer = (np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    y, before, after = dsp.spectral_carve(layer, src, sr, depth_db=9)
    assert len(y) == len(layer) and after < before
    # the carved layer is quieter at 500 Hz (where the source lives) than at 3 kHz
    s = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / sr)
    assert s[np.argmin(abs(f - 500))] < s[np.argmin(abs(f - 3000))]


def test_bar_envelope_is_exactly_one_when_everything_is_retained():
    m = dsp.bar_gain_envelope(44100, 22050, [0.0, 1.0], 1.0, [True, True])
    assert m.min() == 1.0 and m.max() == 1.0
    m2 = dsp.bar_gain_envelope(44100, 22050, [0.0, 1.0], 1.0, [True, False])
    assert m2[:20000].min() == 1.0 and m2[-1000:].max() == 0.0     # retained bar is fully on, the dropped bar fully off
