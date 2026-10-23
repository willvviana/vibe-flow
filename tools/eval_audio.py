"""Objective audio evaluation harness (development tool, not part of the app).

IMPORTANT: these are *proxies*. They can show that something is broken (silence, clipping, no recognisable
words, unstable tempo) but they cannot show that music sounds good. A listening test is the real acceptance test.

  python tools/eval_audio.py <audio.wav> [--lyrics "line one\\nline two"] [--no-asr]

Metrics
  technical   duration, sample rate, channels, peak/RMS dBFS, clipped samples, silent-frame ratio
  rhythm      tempo estimate per 4 s window from the onset-strength autocorrelation, median BPM, spread (IQR),
              and pulse strength (peak height of the autocorrelation; ~0 = no steady pulse)
  lyrics      Whisper-small transcript and word-error-rate vs the supplied lyrics (needs the model, ~0.5 GB,
              downloaded on first use). Singing is harder than speech for ASR, so WER is a *lower bound on
              intelligibility evidence*: a low WER proves words are audible; a high WER does not prove they aren't.
Controls: tools/eval_controls.py checks the harness itself with spoken audio (should score well) and with an
instrumental (should not).
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mono(path, sr_out=None):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    ch = x.shape[1]
    m = x.mean(axis=1)
    if sr_out and sr != sr_out:
        g = math.gcd(sr, sr_out)
        m = resample_poly(m, sr_out // g, sr // g).astype(np.float32)
        sr = sr_out
    return m, sr, ch, x


def technical(x_all, sr):
    m = x_all.mean(axis=1)
    peak = float(np.max(np.abs(x_all))) or 1e-9
    rms = float(np.sqrt(np.mean(m ** 2))) or 1e-9
    frame = int(0.05 * sr)
    n = len(m) // frame
    fr = np.sqrt((m[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    return {"duration_s": round(len(m) / sr, 2), "sample_rate": sr, "channels": x_all.shape[1],
            "peak_dbfs": round(20 * math.log10(peak), 1), "rms_dbfs": round(20 * math.log10(rms), 1),
            "clipped_samples": int((np.abs(x_all) >= 0.999).sum()),
            "silent_frame_ratio": round(float((fr < 10 ** (-60 / 20)).mean()), 3)}


def onset_envelope(m, sr, hop=512):
    _, _, Z = stft(m, fs=sr, nperseg=2048, noverlap=2048 - hop)
    mag = np.log1p(30 * np.abs(Z))
    flux = np.maximum(np.diff(mag, axis=1), 0).sum(axis=0)
    flux = flux - np.convolve(flux, np.ones(21) / 21, mode="same")      # remove slow trend
    return np.maximum(flux, 0), sr / hop


def tempo_windows(m, sr, win_s=4.0, hop_s=2.0, lo=50, hi=200):
    env, fps = onset_envelope(m, sr)
    out = []
    w, h = int(win_s * fps), int(hop_s * fps)
    for s in range(0, max(len(env) - w, 1), h):
        seg = env[s: s + w]
        if len(seg) < w // 2 or seg.std() < 1e-6:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        ac = ac / (ac[0] + 1e-9)
        lags = np.arange(len(ac))
        bpm = 60 * fps / np.maximum(lags, 1)
        ok = (bpm >= lo) & (bpm <= hi)
        # log-Gaussian prior around 110 BPM resolves octave errors (a 100 BPM pulse also peaks at 50 BPM)
        prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / 110) / 0.8) ** 2)
        i = np.argmax(np.where(ok, ac * prior, -1))
        out.append((float(bpm[i]), float(ac[i])))
    return out


def rhythm(m, sr):
    t = tempo_windows(m, sr)
    if not t:
        return {"windows": 0}
    b = np.array([x[0] for x in t])
    s = np.array([x[1] for x in t])
    return {"windows": len(t), "median_bpm": round(float(np.median(b)), 1),
            "bpm_iqr": round(float(np.subtract(*np.percentile(b, [75, 25]))), 1),
            "pulse_strength": round(float(np.median(s)), 2)}


def modulation_proxy(m, sr, lo=1000, hi=5000, mod_lo=4.0, mod_hi=12.0):
    """UNVALIDATED proxy for 'warble': RMS (in natural-log amplitude units; 0.1 ~ +-10 %) of the 1-5 kHz band envelope
    after band-passing it to 4-12 Hz (vibrato/tremolo range). Steady sound ~0, a 50 % tremolo ~0.3. Natural vibrato
    also scores, so only compare clips that sing the same thing, and check against listening ratings before trusting it.
    RESULT: it separates a synthetic 7 Hz tremolo (0.35) from a steady tone (0.04) but scores every real clip ~0.33-0.37
    (speech syllables and drum hits live in the same 4-12 Hz range), so it does NOT discriminate warble in music. Not used."""
    _, _, Z = stft(m, fs=sr, nperseg=2048, noverlap=2048 - 512)
    f = np.linspace(0, sr / 2, Z.shape[0])
    env = np.log(np.abs(Z[(f >= lo) & (f <= hi)]).sum(axis=0) + 1e-6)
    S = np.fft.rfft(env - env.mean())
    fr = np.fft.rfftfreq(len(env), 512 / sr)
    S[(fr < mod_lo) | (fr > mod_hi)] = 0
    return round(float(np.sqrt(np.mean(np.fft.irfft(S, len(env)) ** 2))), 3)


def _norm_words(s):
    return re.findall(r"[a-z']+", s.lower())


def wer(ref, hyp):
    r, h = _norm_words(ref), _norm_words(hyp)
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return round(d[len(h)] / len(r), 2)


_asr = None


def transcribe(m, sr):
    global _asr
    if _asr is None:
        from transformers import pipeline
        _asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1)
    m16 = resample_poly(m, 16000 // math.gcd(sr, 16000), sr // math.gcd(sr, 16000)).astype(np.float32)
    r = _asr({"raw": m16, "sampling_rate": 16000}, generate_kwargs={"language": "en", "task": "transcribe"})
    return r["text"].strip()


def evaluate(path, lyrics=None, asr=True):
    m, sr, ch, x_all = load_mono(path)
    res = {"file": os.path.basename(path), "technical": technical(x_all, sr), "rhythm": rhythm(m, sr),
           "modulation_4_12hz": modulation_proxy(m, sr)}
    if asr:
        text = transcribe(m, sr)
        res["lyrics"] = {"transcript": text, "wer_vs_lyrics": wer(lyrics, text) if lyrics else None,
                         "words_heard": len(_norm_words(text))}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--lyrics")
    ap.add_argument("--lyrics-file", help="UTF-8 text file; [Section] tag lines are ignored")
    ap.add_argument("--no-asr", action="store_true")
    a = ap.parse_args()
    lyr = a.lyrics.replace("\\n", "\n") if a.lyrics else None
    if a.lyrics_file:
        lyr = "\n".join(l for l in open(a.lyrics_file, encoding="utf-8").read().splitlines() if l.strip() and not l.strip().startswith("["))
    print(json.dumps(evaluate(a.audio, lyr, not a.no_asr), indent=1))
