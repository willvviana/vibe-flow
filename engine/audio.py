"""Audio validation, preprocessing, analysis and export. Pure numpy/scipy/soundfile (no ffmpeg needed)."""
import math
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MAX_REF_MB = 25
MAX_REF_SECONDS = 600
REF_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


class AudioError(ValueError):
    pass


def validate_reference(path: str) -> dict:
    """Check an uploaded reference file without fully decoding it. Raises AudioError with a clear reason."""
    if not os.path.isfile(path):
        raise AudioError("Reference file not found.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in REF_EXTS:
        raise AudioError(f"Unsupported audio type '{ext}'. Use WAV, FLAC, MP3 or OGG.")
    size_mb = os.path.getsize(path) / 2**20
    if size_mb > MAX_REF_MB:
        raise AudioError(f"File is {size_mb:.1f} MB; the limit is {MAX_REF_MB} MB.")
    try:
        info = sf.info(path)
    except Exception:
        raise AudioError("Could not read this audio file. It may be corrupted or in an unsupported encoding.")
    if info.frames == 0 or info.samplerate == 0:
        raise AudioError("The audio file is empty.")
    if info.duration > MAX_REF_SECONDS:
        raise AudioError(f"Reference is {info.duration:.0f}s long; the limit is {MAX_REF_SECONDS}s.")
    if info.duration < 1:
        raise AudioError("Reference must be at least 1 second long.")
    return {"duration": round(info.duration, 2), "sample_rate": info.samplerate,
            "channels": info.channels, "format": info.format, "size_mb": round(size_mb, 2)}


def load_mono(path: str, target_sr: int) -> np.ndarray:
    """Decode, mix down to mono, resample, and scale down only if it would clip."""
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        raise AudioError("Could not decode this audio file.")
    x = x.mean(axis=1)
    if not np.isfinite(x).all():
        raise AudioError("Audio contains invalid samples.")
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-4:
        raise AudioError("The reference audio is silent.")
    if peak > 1.0:
        x = x / peak
    return x


def tail(x: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    n = int(seconds * sr)
    return x[-n:] if len(x) > n else x


def analyze(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    db = lambda v: round(20 * math.log10(v), 2) if v > 0 else None
    return {"peak_dbfs": db(peak), "rms_dbfs": db(rms),
            "clipped_samples": int(np.sum(np.abs(x) >= 0.999)), "silent": rms < 1e-4}


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -50.0) -> np.ndarray:
    """Return a trimmed copy; never modifies the input array."""
    thr = 10 ** (thresh_db / 20)
    idx = np.where(np.abs(x) > thr)[0]
    if idx.size == 0:
        return x.copy()
    pad = int(0.02 * sr)
    return x[max(idx[0] - pad, 0): idx[-1] + pad].copy()


def crossfade_concat(a: np.ndarray, b: np.ndarray, sr: int, fade_s: float = 0.1) -> np.ndarray:
    n = min(int(fade_s * sr), len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])
    w = np.linspace(0, 1, n, dtype=np.float32)
    mid = a[-n:] * (1 - w) + b[:n] * w
    return np.concatenate([a[:-n], mid, b[n:]])


def normalize_peak(x: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0:
        return x.copy()
    return (x * (10 ** (target_dbfs / 20) / peak)).astype(np.float32)


def available_formats() -> dict:
    fmts = sf.available_formats()
    return {"wav": "WAV" in fmts, "flac": "FLAC" in fmts, "mp3": "MP3" in fmts}


def write_audio(path: str, x: np.ndarray, sr: int, fmt: str):
    """Write mono/stereo float audio. Refuses to overwrite existing files."""
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite {os.path.basename(path)}")
    if not available_formats().get(fmt):
        raise AudioError(f"This system's audio library cannot encode {fmt.upper()}.")
    subtype = {"wav": "PCM_16", "flac": "PCM_16", "mp3": None}[fmt]
    # Write to a private temp name, then rename: a killed/cancelled job never leaves a half-written track
    # under its final name, and concurrent exports of the same file cannot corrupt each other.
    part = f"{path}.part{os.getpid()}"
    try:
        sf.write(part, np.clip(x, -1, 1), sr, format=fmt.upper(), subtype=subtype)
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite {os.path.basename(path)}")
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)


def verify_file(path: str, expected_s: float, tolerance: float = 0.15) -> dict:
    """Decode the written file and confirm it is real audio of roughly the expected length."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise AudioError("Output file is missing or empty.")
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    dur = len(x) / sr
    if not np.isfinite(x).all():
        raise AudioError("Output contains invalid samples.")
    if abs(dur - expected_s) > max(tolerance * expected_s, 0.5):
        raise AudioError(f"Output is {dur:.1f}s but {expected_s:.1f}s was expected.")
    if analyze(x)["silent"]:
        raise AudioError("Output is silent.")
    return {"duration": round(dur, 2), "sample_rate": sr, "channels": x.shape[1], "bytes": os.path.getsize(path)}
