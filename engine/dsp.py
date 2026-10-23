"""Signal-processing helpers for sample-adaptive composition: tempo/pitch alignment, spectral carving, bar-wise mixing."""
import math
import warnings

import numpy as np
from fractions import Fraction

from scipy.signal import istft, resample_poly, stft


def time_stretch(x, rate, n_fft=2048, hop=512):
    """Phase-vocoder time stretch. rate > 1 = faster/shorter, rate < 1 = slower/longer; pitch unchanged.
    Quality is acceptable for |rate-1| <= ~10 %; larger factors smear transients (callers clamp)."""
    if abs(rate - 1.0) < 1e-4:
        return x.astype(np.float32)
    # pad by one window on both sides so every output sample is covered by overlapping windows (an unpadded STFT is not invertible at the
    # edges and produces large spikes there); the padding is trimmed again below
    x = np.pad(x.astype(np.float32), (n_fft, n_fft))
    n_out = int(round((len(x) - 2 * n_fft) / rate))
    _, _, Z = stft(x, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    steps = np.arange(0, Z.shape[1] - 1, rate)
    omega = 2 * np.pi * hop * np.arange(Z.shape[0]) / n_fft
    phase = np.angle(Z[:, 0])
    out = np.empty((Z.shape[0], len(steps)), dtype=np.complex64)
    for i, s in enumerate(steps):
        k = int(s)
        frac = s - k
        mag = (1 - frac) * np.abs(Z[:, k]) + frac * np.abs(Z[:, k + 1])
        out[:, i] = mag * np.exp(1j * phase)
        d = np.angle(Z[:, k + 1]) - np.angle(Z[:, k]) - omega
        phase = phase + omega + (d + np.pi) % (2 * np.pi) - np.pi
    with warnings.catch_warnings():            # edge frames of an unpadded STFT trigger a harmless NOLA notice
        warnings.simplefilter("ignore")
        _, y = istft(out, nperseg=n_fft, noverlap=n_fft - hop, boundary=False)
    start = int(round(n_fft / rate))
    y = y[start: start + n_out]
    if len(y) < n_out:
        y = np.pad(y, (0, n_out - len(y)))
    return y.astype(np.float32)


def pitch_shift(x, semitones):
    """Shift pitch by stretching then resampling back to the original length (polyphase resampling: no FFT wrap-around artefacts)."""
    if semitones == 0:
        return x.astype(np.float32)
    f = Fraction(2 ** (semitones / 12)).limit_denominator(500)
    y = time_stretch(x, 1 / float(f))                         # longer by f, same pitch
    y = resample_poly(y, f.denominator, f.numerator).astype(np.float32)    # play f times faster: pitch x f, original duration
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[: len(x)].astype(np.float32)


def shift_samples(x, n):
    """Delay (n > 0) or advance (n < 0) by n samples, keeping the length (silence fills the gap)."""
    if n == 0:
        return x
    out = np.zeros_like(x)
    if n > 0:
        out[n:] = x[: len(x) - n]
    else:
        out[: len(x) + n] = x[-n:]
    return out


def spectral_carve(layer, source, sr, depth_db=6.0, n_fft=2048):
    """Attenuate the layer, by up to depth_db, in the frequency regions where the source is strongest, so new parts do not mask it.
    Returns (carved layer, overlap before, overlap after); overlap = sum of min(normalised source spectrum, normalised layer spectrum)."""
    def spec(a):
        s = np.abs(np.fft.rfft(a[: len(a)], n=1 << int(math.ceil(math.log2(max(len(a), 2)))))) ** 2
        k = len(s) // (n_fft // 2)
        return s[: k * (n_fft // 2)].reshape(k, n_fft // 2).sum(axis=1) if k else s
    ss, ls = spec(source), spec(layer)
    n = min(len(ss), len(ls))
    ss, ls = ss[:n] / (ss[:n].sum() + 1e-12), ls[:n] / (ls[:n].sum() + 1e-12)
    before = float(np.minimum(ss, ls).sum())
    prominence = np.clip((ss / (ss.max() + 1e-12)) ** 0.5, 0, 1)
    prominence = np.convolve(prominence, np.ones(3) / 3, mode="same")
    atten = 10 ** (-depth_db * prominence / 20)
    f, t, Z = stft(layer, fs=sr, nperseg=n_fft, noverlap=n_fft * 3 // 4)
    grid = np.linspace(0, 1, len(atten))
    gain = np.interp(np.linspace(0, 1, Z.shape[0]), grid, atten)
    _, y = istft(Z * gain[:, None], fs=sr, nperseg=n_fft, noverlap=n_fft * 3 // 4)
    y = y[: len(layer)]
    if len(y) < len(layer):
        y = np.pad(y, (0, len(layer) - len(y)))
    ls2 = spec(y)[:n]
    after = float(np.minimum(ss, ls2 / (ls2.sum() + 1e-12)).sum())
    return y.astype(np.float32), before, after


def bar_gain_envelope(n, sr, bar_starts, bar_len, retained, fade_s=0.03):
    """Per-sample 0/1 mask that is 1 inside retained bars, with short linear fades at region edges (none inside contiguous runs)."""
    m = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(bar_starts):
        if retained[i]:
            m[int(s * sr): min(int((s + bar_len) * sr), n)] = 1.0
    fade = max(int(fade_s * sr), 1)
    k = np.ones(fade) / fade
    if m.all():
        return m
    padded = np.pad(m, fade, mode="edge")                  # edge padding: no spurious fade at the very start/end
    return np.clip(np.convolve(padded, k, mode="same")[fade:-fade], 0, 1)
