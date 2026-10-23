"""How closely does a generated result follow its reference sample? Objective, signal-level measurements (numpy/scipy + optional CLAP).

Every number here is computed from the actual audio; none is a listening judgement. Each dimension of the request maps to one measure:

  tempo           tempo_sim            correlation of the onset autocorrelations over 30-240 BPM lags (same periodicities), used in the index
                  tempo_error_bpm      |BPM(out) - BPM(ref)|, octave errors folded (x2, /2); reported only: a single BPM estimate flips
                                       between metrical levels on ambiguous material, so it was too unstable to drive the index
  rhythm          rhythm_corr          correlation of the onset-strength envelopes, frame by frame (same time axis)
  melody/harmony  chroma_frame_sim     mean cosine of the 12-bin chroma per frame (only frames where both have energy)
                  pitch_class_agree    share of those frames whose strongest pitch class is the same
                  key_match            1 if the estimated keys are equal, 0.5 for relative major/minor, else 0
  instrumentation spectral_profile_sim correlation of the 40-band log-mel average spectrum (timbre / instrument balance)
                  clap_sim             cosine of LAION-CLAP audio embeddings (if the model is installed; None otherwise)
  structure       structure_sim        correlation of the two self-similarity matrices (2 Hz chroma+mel features)
                  dynamics_corr        correlation of the loudness envelopes (0.5 s RMS)
  presence        ref_presence         least-squares share of the reference waveform contained in the output (1 = all of it, in place)
  quality         clipped, peak_dbfs, silent_ratio, click_score (largest sample-to-sample jump vs. the 99.9th percentile),
                  level_diff_db (RMS out - RMS ref)

adherence_index = mean of (max(tempo_sim,0), max(rhythm_corr,0), chroma_frame_sim, spectral_profile_sim, max(structure_sim,0), clap_sim if any).
It is a summary for comparing settings on the SAME sample, not an absolute quality score.

Vocal measures (vocal_metrics) work on a vocal stem: f0 by a YIN difference function (80-1000 Hz), voiced share, median pitch,
intonation (median distance to the nearest equal-tempered semitone), pitch steadiness (median frame-to-frame change), and
vocal_compare() compares two vocal stems: envelope correlation and pitch-contour agreement (share of co-voiced frames within 50 cents)."""
import math

import numpy as np
from scipy.signal import stft

from . import sample_analysis as sa

SR = sa.SR


def _mel_fb(sr, n_fft, n_mels=40, fmin=40.0, fmax=None):
    fmax = fmax or sr / 2
    hz2m = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    m2hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = m2hz(np.linspace(hz2m(fmin), hz2m(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - lo) / (c - lo + 1e-9), (hi - bins) / (hi - c + 1e-9)), 0, None)
    return fb


_FB = {}


def logmel(x, sr=SR, n_fft=2048, hop=sa.HOP):
    if (sr, n_fft) not in _FB:
        _FB[(sr, n_fft)] = _mel_fb(sr, n_fft)
    _, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    return np.log10(_FB[(sr, n_fft)] @ (np.abs(Z) ** 2) + 1e-10), t


def _corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _r(v, d=3):
    return None if v is None else round(float(v), d)


def _pool(F, k):
    """Average feature frames in blocks of k (frames on axis 1)."""
    n = F.shape[1] // k
    return F[:, : n * k].reshape(F.shape[0], n, k).mean(axis=2) if n else F


def _ssm(F):
    F = F - F.mean(axis=1, keepdims=True)
    F = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-9)
    return F.T @ F


def _key_match(ka, kb):
    if ka["name"] == kb["name"]:
        return 1.0
    rel = (ka["tonic"] - kb["tonic"]) % 12          # relative major/minor share the same notes
    if ka["mode"] != kb["mode"] and ((ka["mode"] == "major" and rel == 3) or (ka["mode"] == "minor" and rel == 9)):
        return 0.5
    return 0.0


def quality(x, sr):
    """Artefact / level checks on one signal (any sample rate)."""
    m = x if x.ndim == 1 else x.mean(axis=1)
    d = np.abs(np.diff(m))
    p999 = float(np.percentile(d, 99.9)) if len(d) else 0.0
    frames = m[: len(m) // 1024 * 1024].reshape(-1, 1024) if len(m) >= 1024 else m[None]
    rms_f = np.sqrt((frames ** 2).mean(axis=1))
    return {"clipped": int((np.abs(x) >= 0.999).sum()), "peak_dbfs": _r(20 * math.log10(float(np.abs(x).max()) + 1e-9), 2),
            "rms_dbfs": _r(20 * math.log10(float(np.sqrt(np.mean(m ** 2))) + 1e-9), 2),
            "silent_ratio": _r(float((rms_f < 10 ** (-60 / 20)).mean())),
            "click_score": _r(float(d.max()) / (p999 + 1e-9) if len(d) else 0.0, 2)}


def tempo_profile_sim(ea, eb, fps, lo=30.0, hi=240.0):
    """Correlation of the onset-envelope autocorrelations over the lags of 30-240 BPM: do the two signals have the same periodicities?
    Continuous and robust where a single BPM estimate is not: on ambiguous material (triplet / 4:3 feel) the argmax BPM flips between
    metrical levels with a 1 dB level change (measured), which turned a 0-or-1 tempo score into the largest source of noise."""
    def acf(e):
        e = e - e.mean()
        a = np.correlate(e, e, "full")[len(e) - 1:]
        return a / (a[0] + 1e-12)
    lags = np.arange(max(1, int(fps * 60 / hi)), int(fps * 60 / lo) + 1)
    if len(ea) <= lags[-1] + 2 or len(eb) <= lags[-1] + 2:
        return None
    return _corr(acf(ea)[lags], acf(eb)[lags])


def compare(ref, out, sr, clap=None):
    """ref, out: mono float arrays at `sr` (cut to the shorter one). clap: optional callable(mono, sr) -> unit embedding."""
    n = min(len(ref), len(out))
    ref, out = np.asarray(ref[:n], np.float32), np.asarray(out[:n], np.float32)
    r22, o22 = sa.resample(ref, sr), sa.resample(out, sr)
    res = {}
    # tempo + rhythm
    br, _, pr = sa.tempo_and_beats(r22)
    bo, _, po = sa.tempo_and_beats(o22)
    err = min(abs(bo - br), abs(bo * 2 - br), abs(bo / 2 - br)) if br > 0 and bo > 0 else None
    er, fps = sa.onset_envelope(r22)
    eo, _ = sa.onset_envelope(o22)
    res.update(ref_bpm=_r(br, 1), out_bpm=_r(bo, 1), tempo_error_bpm=_r(err, 2), tempo_sim=_r(tempo_profile_sim(er, eo, fps)), out_pulse=_r(po))
    res["rhythm_corr"] = _r(_corr(er, eo))
    # melody / harmony
    Cr, _ = sa.chroma(r22)
    Co, _ = sa.chroma(o22)
    m = min(Cr.shape[1], Co.shape[1])
    Cr, Co = Cr[:, :m], Co[:, :m]
    act = (Cr.sum(0) > 0.1 * Cr.sum(0).max()) & (Co.sum(0) > 0.1 * Co.sum(0).max())
    if act.sum() >= 4:
        a, b = Cr[:, act], Co[:, act]
        cos = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)
        res["chroma_frame_sim"] = _r(cos.mean())
        res["pitch_class_agree"] = _r((a.argmax(0) == b.argmax(0)).mean())
    else:
        res["chroma_frame_sim"] = res["pitch_class_agree"] = None
    kr, ko = sa.estimate_key(Cr.mean(1)), sa.estimate_key(Co.mean(1))
    res.update(ref_key=kr["name"], out_key=ko["name"], key_match=_key_match(kr, ko))
    # instrumentation / timbre
    Mr, _ = logmel(r22)
    Mo, _ = logmel(o22)
    m = min(Mr.shape[1], Mo.shape[1])
    Mr, Mo = Mr[:, :m], Mo[:, :m]
    res["spectral_profile_sim"] = _r(_corr(Mr.mean(1), Mo.mean(1)))
    res["clap_sim"] = None
    if clap is not None:
        try:
            ea, eb = clap(ref, sr), clap(out, sr)
            res["clap_sim"] = _r(float(np.dot(ea, eb)))
        except Exception:  # noqa: BLE001 - CLAP is optional
            pass
    # structure: self-similarity of 2 Hz features (chroma + normalised mel)
    k = max(1, int(round(SR / sa.HOP / 2)))
    Fr = np.vstack([_pool(Cr, k), _pool((Mr - Mr.mean()) / (Mr.std() + 1e-9), k)])
    Fo = np.vstack([_pool(Co, k), _pool((Mo - Mo.mean()) / (Mo.std() + 1e-9), k)])
    if Fr.shape[1] >= 4:
        Sr, So = _ssm(Fr), _ssm(Fo)
        iu = np.triu_indices(Sr.shape[0], 1)
        res["structure_sim"] = _r(_corr(Sr[iu], So[iu]))
    else:
        res["structure_sim"] = None
    w = int(0.5 * sr)
    if n >= 4 * w:
        env = lambda s: 20 * np.log10(np.sqrt((s[: len(s) // w * w].reshape(-1, w) ** 2).mean(1)) + 1e-6)  # noqa: E731
        res["dynamics_corr"] = _r(_corr(env(ref), env(out)))
    else:
        res["dynamics_corr"] = None
    # presence of the literal reference waveform
    res["ref_presence"] = _r(float(np.dot(out, ref) / (np.dot(ref, ref) + 1e-12)))
    q = quality(out, sr)
    res["quality"] = {**q, "level_diff_db": _r(q["rms_dbfs"] - quality(ref, sr)["rms_dbfs"], 2)}
    parts = [max(res["tempo_sim"] or 0.0, 0.0), max(res["rhythm_corr"] or 0.0, 0.0), res["chroma_frame_sim"] or 0.0, max(res["spectral_profile_sim"] or 0.0, 0.0),
             max(res["structure_sim"] or 0.0, 0.0)] + ([max(res["clap_sim"], 0.0)] if res["clap_sim"] is not None else [])
    res["adherence_index"] = _r(float(np.mean(parts)))
    return res


# ---------------------------------------------------------------------------------------------------- vocals
F0_SR = 16000


def f0_track(x, sr, fmin=80.0, fmax=1000.0, frame_s=0.04, hop_s=0.01, thresh=0.15):
    """YIN (cumulative-mean-normalised difference) f0 per 10 ms. Returns (f0 Hz with 0 = unvoiced, rms per frame)."""
    x = sa.resample(np.asarray(x, np.float32), sr, F0_SR)
    N, H = int(frame_s * F0_SR), int(hop_s * F0_SR)
    if len(x) < N + 2:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(N)[None, :] + H * np.arange((len(x) - N) // H + 1)[:, None]
    fr = x[idx]
    rms = np.sqrt((fr ** 2).mean(1))
    # difference function via FFT autocorrelation: d(tau) = r(0)[0..N-tau] + r(0)[tau..N] - 2 r(tau)
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(fr, nfft)
    ac = np.fft.irfft(F * np.conj(F), nfft)[:, :N]
    e = np.cumsum(fr ** 2, axis=1)
    tmax, tmin = min(int(F0_SR / fmin), N - 1), max(2, int(F0_SR / fmax))
    taus = np.arange(1, tmax + 1)
    # d(tau) = sum_{j<N-tau} x_j^2 + sum_{j>=tau} x_j^2 - 2 r(tau)
    head = e[:, N - 1 - taus]
    tail = e[:, -1][:, None] - np.concatenate([np.zeros((len(fr), 1)), e], 1)[:, taus]
    d = head + tail - 2 * ac[:, taus]
    cmnd = d * taus[None, :] / (np.cumsum(d, axis=1) + 1e-12)
    f0 = np.zeros(len(fr))
    loud = rms > max(1e-4, 0.05 * float(np.percentile(rms, 95)))
    for i in np.flatnonzero(loud):
        c = cmnd[i, tmin - 1:]
        below = np.flatnonzero(c < thresh)
        if not len(below):
            continue
        j = below[0]
        while j + 1 < len(c) and c[j + 1] < c[j]:
            j += 1
        tau = j + tmin
        if 1 <= j < len(c) - 1:                       # parabolic refinement
            a, b, cc = c[j - 1], c[j], c[j + 1]
            den = a - 2 * b + cc
            tau = tau + (0.5 * (a - cc) / den if abs(den) > 1e-12 else 0)
        f0[i] = F0_SR / tau
    return f0, rms


def _cents(f):
    return 1200 * np.log2(f / 440.0)


def vocal_metrics(x, sr):
    """Pitch, intonation and activity of one (mostly monophonic) vocal signal."""
    f0, rms = f0_track(x, sr)
    v = f0 > 0
    out = {"voiced_ratio": _r(v.mean() if len(v) else 0.0), "median_f0_hz": None, "intonation_cents": None, "pitch_steadiness_cents": None,
           "pitch_range_semitones": None}
    if v.sum() >= 10:
        c = _cents(f0[v])
        out["median_f0_hz"] = _r(float(np.median(f0[v])), 1)
        out["intonation_cents"] = _r(float(np.median(np.abs(c - 100 * np.round(c / 100)))), 1)    # 0 = on the semitone grid, 50 = worst
        both = v[1:] & v[:-1]
        if both.sum() >= 5:
            dc = np.abs(np.diff(_cents(np.where(v, f0, 1.0))))[both]
            out["pitch_steadiness_cents"] = _r(float(np.median(dc)), 1)
        out["pitch_range_semitones"] = _r(float((np.percentile(c, 95) - np.percentile(c, 5)) / 100), 1)
    return out


def vocal_compare(ref_v, out_v, sr):
    """Two vocal stems (mono, same sr): loudness-envelope correlation and pitch-contour agreement."""
    n = min(len(ref_v), len(out_v))
    fa, ra = f0_track(ref_v[:n], sr)
    fb, rb = f0_track(out_v[:n], sr)
    m = min(len(fa), len(fb))
    both = (fa[:m] > 0) & (fb[:m] > 0)
    agree = None
    if both.sum() >= 10:
        dc = np.abs(_cents(fa[:m][both]) - _cents(fb[:m][both]))
        dc = np.minimum(dc, np.abs(dc - 1200))         # octave errors of the tracker do not count as a different melody
        agree = float((dc <= 50).mean())
    act_a = fa[:m] > 0
    return {"vocal_envelope_corr": _r(_corr(20 * np.log10(ra[:m] + 1e-6), 20 * np.log10(rb[:m] + 1e-6))),
            "pitch_contour_agree": _r(agree), "co_voiced_frames": int(both.sum()),
            "ref_voiced_kept": _r(float((both.sum() / act_a.sum())) if act_a.sum() else None)}
