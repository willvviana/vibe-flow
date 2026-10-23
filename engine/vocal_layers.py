"""Lead vs. backing vocals, from the (Demucs) vocal stem, plus the vocal measurements of each layer.

Method (no extra model; classic stereo source separation): in a mix, the lead vocal is almost always panned to the centre and backing
vocals / doubles / harmonies are spread to the sides. For every STFT bin of the stereo vocal stem we compute
  similarity  psi = 2|L R*| / (|L|^2 + |R|^2)          (1 = identical in both channels, i.e. centre-panned and coherent)
  balance     b   = 1 - | |L| - |R| | / (|L| + |R|)     (1 = equal level left and right)
and a soft lead mask  m = (psi * b)^k ;  lead = m * mid,  backing = stereo stem - lead (so lead + backing add back up exactly).

HONEST LIMITS: backing vocals that are panned to the centre together with the lead (common in sparse mixes) end up in the lead layer;
a mono recording has no backing layer at all (reported as such, not guessed); reverb tails of the lead are partly decorrelated and leak
into the backing layer. Separation quality is measured on a constructed mix with known parts (tools/vocal_split_eval.py)."""
import math

import numpy as np
from scipy.signal import istft, stft

from . import adherence_eval as ae
from . import sample_analysis as sa

N_FFT = 2048
HOP = 512


def split(v, sr, k=2.0):
    """v: stereo vocal stem [n, 2]. Returns (lead [n, 2], backing [n, 2], info)."""
    v = np.asarray(v, np.float32)
    if v.ndim == 1 or v.shape[1] == 1 or float(np.abs(v[:, 0] - v[:, 1]).max()) < 1e-6:
        mono = v if v.ndim == 1 else v[:, 0]
        lead = np.repeat(mono.reshape(-1, 1), 2, axis=1)
        return lead, np.zeros_like(lead), {"stereo": False, "backing_share": 0.0,
                                           "note": "Mono vocal: no stereo information, so no backing layer can be separated."}
    _, _, L = stft(v[:, 0], fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
    _, _, R = stft(v[:, 1], fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
    aL, aR = np.abs(L), np.abs(R)
    psi = 2 * np.abs(L * np.conj(R)) / (aL ** 2 + aR ** 2 + 1e-12)
    bal = 1 - np.abs(aL - aR) / (aL + aR + 1e-12)
    m = np.clip(psi * bal, 0, 1) ** k
    M = 0.5 * (L + R) * m
    _, lead_m = istft(M, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
    n = v.shape[0]
    lead_m = lead_m[:n] if len(lead_m) >= n else np.pad(lead_m, (0, n - len(lead_m)))
    lead = np.repeat(lead_m.astype(np.float32).reshape(-1, 1), 2, axis=1)
    back = (v - lead).astype(np.float32)
    e_l, e_b = float((lead ** 2).sum()), float((back ** 2).sum())
    return lead, back, {"stereo": True, "backing_share": round(e_b / (e_l + e_b + 1e-12), 4)}


def _onsets(x, sr):
    env, fps = sa.onset_envelope(sa.resample(x, sr))
    if len(env) < 5 or env.max() <= 0:
        return np.array([])
    thr = env.mean() + 1.5 * env.std()
    pk = [i for i in range(1, len(env) - 1) if env[i] > thr and env[i] >= env[i - 1] and env[i] >= env[i + 1]]
    return np.array(pk) / fps


def timing_ms(vocal, inst, sr):
    """Median distance (ms) of vocal onsets from the nearest 8th note of the instrumental's beat grid, the same figure expected by chance
    (onsets unrelated to the grid), and the grid's pulse strength. (None, None, pulse) if there is no grid or no onset."""
    bpm, beats, pulse = sa.tempo_and_beats(sa.resample(inst, sr))
    on = _onsets(vocal, sr)
    if bpm <= 0 or len(beats) < 4 or not len(on):
        return None, None, round(float(pulse), 3)
    period = 60.0 / bpm / 2
    grid = np.arange(beats[0] - 4 * period, beats[-1] + 4 * period, period)
    d = np.abs(on[:, None] - grid[None, :]).min(axis=1)
    # onsets with no relation to the grid would land on average a quarter of the grid spacing away (median of a uniform distance)
    return round(1000 * float(np.median(d)), 1), round(1000 * period / 4, 1), round(float(pulse), 3)


def harmony(lead, back, sr):
    """Which intervals the backing layer sings relative to the lead: per 10 ms frame where the lead is voiced, the backing layer's
    12-bin chroma is rotated so the lead's pitch class is 0, then averaged. A single-pitch tracker cannot be used on the backing layer:
    on a two-voice harmony (e.g. a 3rd and a 5th) it locks onto the chord's common root and reports a unison.
    The backing chroma is taken from the SIDE signal (L-R) when stereo, where the centre-panned lead cancels."""
    fl, _ = ae.f0_track(lead, sr)
    side = 0.5 * (back[:, 0] - back[:, 1]) if back.ndim == 2 else back
    s22 = sa.resample(side, sr)
    C, t = sa.chroma(s22)
    C = C / (C.sum(axis=0, keepdims=True) + 1e-9)
    loud = C.sum(0) > 0
    e = np.sqrt(np.convolve(s22 ** 2, np.ones(sa.HOP) / sa.HOP, mode="same"))[(t * sa.SR).astype(int).clip(0, len(s22) - 1)]
    active = e > 0.1 * np.percentile(e, 95)
    names = ["unison/octave", "minor 2nd", "major 2nd", "minor 3rd", "major 3rd", "4th", "tritone", "5th", "minor 6th", "major 6th", "minor 7th", "major 7th"]
    acc, n = np.zeros(12), 0
    for i, ti in enumerate(t):
        j = int(round(ti / 0.01))
        if j < len(fl) and fl[j] > 0 and active[i] and loud[i]:
            pc = int(round(69 + 12 * np.log2(fl[j] / 440.0))) % 12
            acc += np.roll(C[:, i], -pc)
            n += 1
    if n < 50:
        return {"frames": n, "intervals": []}
    h = acc / n
    h = (h - 1 / 12) / (h.max() - 1 / 12 + 1e-9)        # 0 = no more than a flat profile, 1 = the strongest interval
    top = [i for i in np.argsort(-h) if h[i] > 0.5][:3]
    return {"frames": n, "profile": [round(float(v), 3) for v in h], "intervals": [{"semitones": int(i), "name": names[i], "strength": round(float(h[i]), 3)} for i in top]}


def analyse(vocals, instrumental, sr, transcribe=None):
    """vocals / instrumental: [n, 2] stems at sr. Returns (lead, backing, report). `transcribe(mono16k)` -> segments with avg_logprob (optional)."""
    lead, back, info = split(vocals, sr)
    lm, bm, im = lead.mean(1), back.mean(1), instrumental.mean(1)
    db = lambda x: 20 * math.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9)  # noqa: E731
    rep = {**info, "lead": ae.vocal_metrics(lm, sr), "backing": ae.vocal_metrics(bm, sr) if info["stereo"] else None,
           "lead_rms_db": round(db(lm), 1), "backing_rms_db": round(db(bm), 1) if info["stereo"] else None,
           "lead_to_instrumental_db": round(db(lm) - db(im), 1)}
    rep["lead_timing_ms"], rep["timing_chance_ms"], rep["instrumental_pulse"] = timing_ms(lm, im, sr)
    if info["stereo"] and info["backing_share"] > 0.05:
        rep["backing_timing_ms"], _, _ = timing_ms(bm, im, sr)
        rep["harmony"] = harmony(lm, back, sr)
        if len(rep["harmony"]["intervals"]) >= 2:          # several backing voices at once: a single-pitch tracker would report the chord root
            rep["backing"] = {"voiced_ratio": None, "median_f0_hz": None, "intonation_cents": None, "pitch_steadiness_cents": None, "pitch_range_semitones": None,
                              "note": "Several backing voices sing at once (harmony); single-pitch measures are not meaningful and are not reported."}
        # layering: how much of the time the backing layer is active while the lead sings
        fl, _ = ae.f0_track(lm, sr)
        fb, rb = ae.f0_track(bm, sr)
        m = min(len(fl), len(fb))
        rep["backing_active_while_lead"] = round(float(((fb[:m] > 0) & (fl[:m] > 0)).sum() / max(1, (fl[:m] > 0).sum())), 3)
    if transcribe is not None:
        segs = transcribe(sa.resample(lm, sr, 16000))
        lp = [s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
        rep["lead_clarity"] = {"words": sum(len(s["text"].split()) for s in segs),
                               "avg_logprob": round(float(np.mean(lp)), 3) if lp else None,
                               "text": " ".join(s["text"] for s in segs)[:400]}
    return lead, back, rep
