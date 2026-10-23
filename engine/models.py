"""Model installation, integrity verification and (explicit, user-initiated) weight download.

Integrity model: after a download we hash every weight file against the SHA-256 published by Hugging Face
and write a small marker (sizes + mtimes). Later status checks only compare sizes/mtimes (cheap), so a
truncated or modified file is detected immediately; `verify` re-hashes everything on demand."""
import hashlib
import json
import os
import threading
import time

from . import registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("VIBEFLOW_MODELS") or os.path.join(ROOT, "models")
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model"]


def _marker(spec):
    return os.path.join(CACHE, spec.id + ".verified.json")


def local_path(spec):
    """Path of the downloaded snapshot, or None if not installed."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(spec.repo, cache_dir=CACHE, local_files_only=True, allow_patterns=ALLOW)
    except Exception:
        return None


def _weights(path):
    return sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))


def _fingerprint(path, files):
    out = {}
    for f in files:
        st = os.stat(os.path.join(path, f))      # follows the snapshot symlink to the real blob
        out[f] = [st.st_size, st.st_mtime_ns]
    return out


def state(spec):
    """'missing' | 'unverified' | 'ok' | 'damaged'."""
    p = local_path(spec)
    if not p or not _weights(p):
        return "missing", None
    try:
        with open(_marker(spec), encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return "unverified", p
    try:
        return ("ok" if saved == _fingerprint(p, _weights(p)) else "damaged"), p
    except OSError:
        return "damaged", p


def status():
    out = []
    for s in registry.MODELS.values():
        st, p = state(s)
        d = s.to_dict()
        # 'unverified' (downloaded before verification existed) is still usable; 'damaged' is not.
        d.update(installed=st in ("ok", "unverified"), integrity=st, path=None)
        out.append(d)
    return out


def _expected(spec):
    """{filename: (sha256, size)} for the weight files, from Hugging Face metadata."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(spec.repo, files_metadata=True)
    return {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings if s.rfilename.endswith(".safetensors")}


def _sha256(path, emit=None, label=""):
    h, done = hashlib.sha256(), 0
    total = os.path.getsize(path)
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
            done += len(chunk)
            if emit:
                emit(stage="verify", pct=min(done / total, 1) * 100, message=f"Verifying {label}…")
    return h.hexdigest()


def verify(model_id: str, emit=lambda **k: None) -> dict:
    """Hash all weights against the published SHA-256 and write the integrity marker. Raises if corrupted."""
    spec = registry.get(model_id)
    p = local_path(spec)
    if not p or not _weights(p):
        raise RuntimeError(f"{spec.name} is not installed.")
    expected = _expected(spec)
    for f in _weights(p):
        sha, size = expected.get(f, (None, None))
        real = os.path.join(p, f)
        if size is not None and os.path.getsize(real) != size:
            raise RuntimeError(f"{f} is incomplete or damaged (size mismatch). Use Repair.")
        if sha and _sha256(real, emit, f) != sha:
            raise RuntimeError(f"{f} failed its checksum. Use Repair.")
    with open(_marker(spec), "w", encoding="utf-8") as fh:
        json.dump(_fingerprint(p, _weights(p)), fh)
    return {"verified": True, "files": _weights(p)}


def _dir_size(root):
    total = 0
    for base, _, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total


def install(model_id: str, emit=lambda **k: None, force: bool = False):
    """Download weights then verify them. Only ever called from an explicit install/repair command.

    An interrupted download resumes (huggingface_hub keeps *.incomplete blobs). force=True re-downloads."""
    spec = registry.get(model_id)
    from huggingface_hub import snapshot_download
    os.makedirs(CACHE, exist_ok=True)
    expected_bytes = sum(sz or 0 for _, sz in _expected(spec).values()) or spec.download_gb * 2**30
    emit(stage="download", pct=0, message=f"Downloading {spec.name} (~{spec.download_gb} GB) from Hugging Face…")

    stop = threading.Event()
    model_dir = os.path.join(CACHE, "models--" + spec.repo.replace("/", "--"))

    def watch():                                   # real byte-based progress from what is on disk
        start = _dir_size(model_dir) if os.path.isdir(model_dir) else 0
        while not stop.wait(1.0):
            now = _dir_size(model_dir) if os.path.isdir(model_dir) else 0
            emit(stage="download", pct=min(now / expected_bytes, 0.99) * 90,
                 message=f"Downloading {spec.name}: {now / 2**30:.2f} / {expected_bytes / 2**30:.2f} GB")
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        p = snapshot_download(spec.repo, cache_dir=CACHE, allow_patterns=ALLOW, force_download=force)
    finally:
        stop.set()
    if not _weights(p):
        raise RuntimeError(f"{spec.repo} has no safetensors weights; refusing to load pickle checkpoints.")
    verify(model_id, lambda **k: emit(stage="verify", pct=90 + k["pct"] / 10, message=k["message"]))
    emit(stage="done", pct=100, message="Installed and verified.")
    return p


def repair(model_id: str, emit=lambda **k: None):
    """Remove the marker and re-download every file (fixes corrupted or truncated weights)."""
    spec = registry.get(model_id)
    try:
        os.remove(_marker(spec))
    except OSError:
        pass
    return install(model_id, emit, force=True)
