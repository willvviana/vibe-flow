"""Musical analysis of an uploaded sample (numpy/scipy only, no extra model).

What is measured, and how reliable it is (be honest in the UI/docs about this):
  tempo + beat grid   onset-strength autocorrelation with a tempo prior, beat phase by comb search; bar = 4 beats ASSUMED (4/4).
                      Good on steady rhythmic material, unreliable on rubato/ambient music (confidence reports the pulse strength).
  key + mode          Krumhansl-Schmuckler profile correlation on a 12-bin chroma (assumes A=440 Hz tuning). Confidence = margin over 2nd best.
  chords per bar      best of 24 major/minor triad templates per bar. Coarse; says nothing about extensions or inversions.
  energy / dynamics   RMS per bar, spectral centroid, low/mid/high band share.
  sections            novelty between neighbouring 2-bar blocks (chroma + loudness); a rough guide, not a score-level structure.
  vocals              OPTIONAL: AudioSet tagger probability for 'Singing'/'Speech' if that model is already cached locally; else None (unknown).
NOT analysed: melody/motif extraction, instrument identification (only band-energy shares), lyrics."""
import math

import numpy as np
from scipy.signal import resample_poly, stft

SR = 22050
HOP = 512
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def resample(x, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def onset_envelope(x, sr=SR, hop=HOP):
    """Half-wave-rectified log-magnitude spectral flux, slow trend removed. Returns (envelope, frames per second)."""
    _, _, Z = stft(x, fs=sr, nperseg=2048, noverlap=2048 - hop)
    mag = np.log1p(30 * np.abs(Z))
    flux = np.maximum(np.diff(mag, axis=1), 0).sum(axis=0)
    flux = np.maximum(flux - np.convolve(flux, np.ones(21) / 21, mode="same"), 0)
    return flux, sr / hop


def tempo_and_beats(x, sr=SR, lo=60.0, hi=180.0):
    env, fps = onset_envelope(x, sr)
    if len(env) < 16 or env.std() < 1e-9:                  # silence / constant signal: there is no pulse to find
        return 0.0, np.array([]), 0.0
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lags = np.arange(max(int(math.ceil(fps * 60 / hi)), 2), min(len(ac), int(fps * 60 / lo) + 2))
    if len(lags) < 3:
        return 0.0, np.array([]), 0.0
    bpm_l = 60 * fps / lags
    ok = (bpm_l >= lo) & (bpm_l <= hi)
    prior = np.exp(-0.5 * (np.log2(bpm_l / 110) / 0.8) ** 2)
    score = np.where(ok, ac[lags] * prior, -1)
    best = int(np.argmax(score))
    period = lags[best]
    # refine the period with parabolic interpolation, then search the beat phase with a comb over the onset envelope
    if 1 <= best < len(lags) - 1:
        a, b, c = ac[lags[best] - 1], ac[lags[best]], ac[lags[best] + 1]
        denom = a - 2 * b + c
        period = lags[best] + (float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0)) if abs(denom) > 1e-12 else 0.0)   # at most one lag of refinement
    if period < 2:
        return 0.0, np.array([]), 0.0
    bpm = 60 * fps / period
    phases = np.arange(0, int(period))
    comb = [env[int(p):: max(int(round(period)), 1)].sum() for p in phases]
    phase = int(phases[int(np.argmax(comb))])
    beats = np.arange(phase, len(env), period) / fps
    pulse = float(ac[lags[best]])
    return float(bpm), beats, pulse


def chroma(x, sr=SR, hop=HOP):
    f, t, Z = stft(x, fs=sr, nperseg=4096, noverlap=4096 - hop)
    mag = np.abs(Z) ** 2
    sel = (f >= 55) & (f <= 2000)
    midi = 69 + 12 * np.log2(np.maximum(f[sel], 1e-6) / 440.0)
    pc = np.round(midi).astype(int) % 12
    C = np.zeros((12, mag.shape[1]))
    for k in range(12):
        C[k] = mag[sel][pc == k].sum(axis=0)
    C = np.log1p(C / (C.max() + 1e-12) * 100)
    return C, t


def estimate_key(chroma_mean):
    prof = []
    for tonic in range(12):
        prof.append((float(np.corrcoef(chroma_mean, np.roll(KS_MAJOR, tonic))[0, 1]), tonic, "major"))
        prof.append((float(np.corrcoef(chroma_mean, np.roll(KS_MINOR, tonic))[0, 1]), tonic, "minor"))
    prof.sort(reverse=True)
    best, second = prof[0], prof[1]
    return {"tonic": best[1], "name": f"{NOTE[best[1]]} {best[2]}", "mode": best[2], "score": round(best[0], 3), "margin": round(best[0] - second[0], 3)}


def chord_for(chroma_vec):
    best, name = -2, "?"
    for tonic in range(12):
        for quality, ivs in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            t = np.zeros(12)
            t[[(tonic + i) % 12 for i in ivs]] = 1
            s = float(np.corrcoef(chroma_vec, t)[0, 1]) if chroma_vec.std() > 0 else -2
            if s > best:
                best, name = s, NOTE[tonic] + quality
    return name


def analyse(x, sr_in, vocals_prob=None):
    """Full analysis of a mono float signal. Returns a plain dict (JSON-serialisable)."""
    x = resample(np.asarray(x, dtype=np.float32), sr_in)
    dur = len(x) / SR
    bpm, beats, pulse = tempo_and_beats(x)
    C, t = chroma(x)
    key = estimate_key(C.mean(axis=1))
    bar_s = 4 * 60.0 / bpm
    # bars start at the beat grid's best 4-beat phase: choose the offset (0-3 beats) with the most low-frequency onset energy
    env, fps = onset_envelope(x)
    best_off, best_val = 0, -1
    for off in range(4):
        idx = (beats[off::4] * fps).astype(int)
        v = float(env[np.clip(idx, 0, len(env) - 1)].sum())
        if v > best_val:
            best_off, best_val = off, v
    bar_starts = beats[best_off::4]
    bar_starts = bar_starts[bar_starts + bar_s <= dur + 0.25 * bar_s]
    chords, rms_bar = [], []
    for s in bar_starts:
        a, b = int(s * SR), int(min((s + bar_s) * SR, len(x)))
        rms_bar.append(float(np.sqrt(np.mean(x[a:b] ** 2) + 1e-12)) if b > a else 0.0)
        fa, fb = np.searchsorted(t, s), np.searchsorted(t, s + bar_s)
        v = C[:, fa:max(fb, fa + 1)].mean(axis=1)
        chords.append(chord_for(v))
    S = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    tot = S.sum() + 1e-12
    bands = {"low": float(S[fr < 250].sum() / tot), "mid": float(S[(fr >= 250) & (fr < 2000)].sum() / tot), "high": float(S[fr >= 2000].sum() / tot)}
    centroid = float((S * fr).sum() / tot)
    # sections: novelty between neighbouring 2-block windows of (chroma, loudness) per bar
    sections = []
    if len(bar_starts) >= 4:
        feats = []
        for s in bar_starts:
            fa, fb = np.searchsorted(t, s), np.searchsorted(t, s + bar_s)
            feats.append(np.append(C[:, fa:max(fb, fa + 1)].mean(axis=1) / (C.max() + 1e-9), np.log1p(rms_bar[len(feats)] * 20)))
        feats = np.array(feats)
        nov = [float(np.linalg.norm(feats[i:i + 2].mean(0) - feats[max(i - 2, 0):i].mean(0))) if i >= 2 else 0.0 for i in range(len(feats) - 1)]
        thr = np.mean(nov) + np.std(nov)
        cuts = [0] + [i for i in range(2, len(nov) - 1) if nov[i] > thr and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]] + [len(bar_starts)]
        sections = [[int(a), int(b)] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    energy = "low" if np.mean(rms_bar or [0]) < 0.03 else "high" if np.mean(rms_bar or [0]) > 0.12 else "medium"
    return {"duration_s": round(dur, 2), "bpm": round(bpm, 2), "tempo_confidence": round(pulse, 3), "beats_per_bar_assumed": 4,
            "bar_seconds": round(bar_s, 3), "bar_starts_s": [round(float(s), 3) for s in bar_starts], "n_bars": len(bar_starts),
            "key": key, "chords_per_bar": chords, "rms_per_bar": [round(r, 4) for r in rms_bar], "energy": energy,
            "bands": {k: round(v, 3) for k, v in bands.items()}, "spectral_centroid_hz": round(centroid, 1), "sections_bars": sections,
            "vocals_probability": vocals_prob, "not_analysed": ["melody/motifs", "instrument identity (band shares only)", "lyrics"]}


def vocals_probability(x, sr_in):
    """Singing/speech probability from the AudioSet tagger IF it is already in the local Hugging Face cache; else None (unknown)."""
    try:
        from transformers import ASTForAudioClassification, AutoFeatureExtractor
        import torch
        name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        fe = AutoFeatureExtractor.from_pretrained(name, local_files_only=True)
        m = ASTForAudioClassification.from_pretrained(name, local_files_only=True).eval()
    except Exception:
        return None
    x16 = resample(np.asarray(x, dtype=np.float32), sr_in, 16000)[: 16000 * 10]
    with torch.no_grad():
        p = torch.sigmoid(m(**fe(x16, sampling_rate=16000, return_tensors="pt")).logits[0]).numpy()
    ids = [i for i, l in m.config.id2label.items() if l in ("Singing", "Male singing", "Female singing", "Child singing", "Speech")]
    return round(float(p[ids].max()), 3) if ids else None
