"""Local music library: one original WAV + one JSON metadata file per track."""
import json
import os
import re
import secrets
import shutil
import time

from . import audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID_RE = re.compile(r"^[a-z0-9-]{6,40}$")


def library_dir() -> str:
    d = os.environ.get("VIBEFLOW_LIBRARY") or os.path.join(ROOT, "library")
    os.makedirs(d, exist_ok=True)
    return d


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def _check(track_id: str):
    if not ID_RE.match(track_id or ""):
        raise ValueError("Invalid track id.")


def paths(track_id: str):
    _check(track_id)
    d = library_dir()
    return os.path.join(d, track_id + ".json"), os.path.join(d, track_id + ".wav")


def save_meta(track_id: str, meta: dict, overwrite: bool = False):
    jp, _ = paths(track_id)
    if os.path.exists(jp) and not overwrite:
        raise FileExistsError("Track metadata already exists.")
    tmp = jp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, jp)


def get(track_id: str) -> dict:
    jp, _ = paths(track_id)
    if not os.path.exists(jp):
        raise FileNotFoundError("Track not found.")
    with open(jp, encoding="utf-8") as f:
        return json.load(f)


def list_tracks() -> list:
    out = []
    for n in sorted(os.listdir(library_dir()), reverse=True):
        if n.endswith(".json"):
            try:
                out.append(get(n[:-5]))
            except Exception:
                pass
    return out


def rename(track_id: str, title: str) -> dict:
    title = (title or "").strip()[:80]
    if not title:
        raise ValueError("Title cannot be empty.")
    m = get(track_id)
    m["title"] = title
    save_meta(track_id, m, overwrite=True)
    return m


def delete(track_id: str):
    jp, wp = paths(track_id)
    d = library_dir()
    for n in os.listdir(d):
        if n.startswith(track_id + ".") or n.startswith(track_id + "-"):
            os.remove(os.path.join(d, n))


def export(track_id: str, fmt: str, normalize: bool = False) -> str:
    """Create a derived copy in another format. The original WAV is never touched."""
    if fmt not in ("wav", "flac", "mp3"):
        raise ValueError("Unsupported format.")
    _, wp = paths(track_id)
    if not os.path.exists(wp):
        raise FileNotFoundError("Track audio not found.")
    if fmt == "wav" and not normalize:
        return wp
    suffix = "-norm" if normalize else ""
    out = os.path.join(library_dir(), f"{track_id}{suffix}.{fmt}")
    if os.path.exists(out):
        return out
    import soundfile as sf  # noqa: E402
    x, sr = sf.read(wp, dtype="float32", always_2d=False)
    if normalize:
        x = audio.normalize_peak(x)
    try:
        audio.write_audio(out, x, sr, fmt)
    except FileExistsError:   # another request created the same export first; that file is complete (atomic write)
        pass
    return out


def cleanup(min_age_s: float = 0) -> list:
    """Remove leftovers of killed jobs: partial writes and audio files that never got metadata.

    Files younger than min_age_s are kept so a job that is still writing is never touched."""
    d, now, removed = library_dir(), time.time(), []
    ids = {n[:-5] for n in os.listdir(d) if n.endswith(".json")}
    for n in os.listdir(d):
        p = os.path.join(d, n)
        if min_age_s > 0 and now - os.path.getmtime(p) < min_age_s:   # (age can be slightly negative on Windows)
            continue
        orphan = n.endswith((".wav", ".flac", ".mp3")) and not any(n.startswith(i) for i in ids)
        if ".part" in n or n.endswith(".tmp") or orphan:
            os.remove(p)
            removed.append(n)
    return removed


def import_wav(path: str, title: str) -> dict:
    """Register an externally rendered WAV (the editor's mix) as a library track. The file is copied, never modified."""
    import soundfile as sf
    info = sf.info(path)
    if info.frames == 0:
        raise ValueError("The audio file is empty.")
    tid = new_id()
    _, wp = paths(tid)
    shutil.copyfile(path, wp)
    x, _sr = sf.read(wp, dtype="float32", always_2d=True)
    meta = {
        "id": tid, "title": (title or "Edited mix").strip()[:80] or "Edited mix", "status": "complete", "kind": "editor",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "original_settings": {}, "settings": {}, "model_prompt": None,
        "controls": {"enforced": [], "approximate": []}, "model": {"id": "editor", "name": "Vibe Flow Editor", "repo": "", "license": ""},
        "device": "browser", "seed": None, "warnings": [], "timing": {},
        "audio": {"duration": round(info.duration, 2), "sample_rate": info.samplerate, "channels": info.channels, "bytes": os.path.getsize(wp),
                  "analysis": audio.analyze(x.mean(axis=1)), "files": {"wav": tid + ".wav"}},
    }
    save_meta(tid, meta)
    return meta
