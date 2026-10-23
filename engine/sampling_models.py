"""Model locations and (explicit, user-initiated) downloads for the Sample page.

  stems          Demucs htdemucs (Meta, MIT)                 -> models/torch/hub/checkpoints
  transcription  faster-whisper 'small' (Systran, MIT)       -> models/whisper-small
  tagging        LAION CLAP htsat-unfused (Apache-2.0)       -> models/models--laion--clap-htsat-unfused
Everything runs locally on CPU (or CUDA if torch sees one). Nothing is uploaded."""
import os

from . import models

CACHE = models.CACHE
WHISPER_REPO = "Systran/faster-whisper-small"
WHISPER_DIR = os.path.join(CACHE, "whisper-small")
CLAP_REPO = "laion/clap-htsat-unfused"
TORCH_HOME = os.path.join(CACHE, "torch")
DEMUCS_NAME = "htdemucs"
os.environ.setdefault("TORCH_HOME", TORCH_HOME)


def whisper_ready():
    return os.path.exists(os.path.join(WHISPER_DIR, "model.bin"))


def clap_ready():
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(CLAP_REPO, cache_dir=CACHE, local_files_only=True, allow_patterns=["pytorch_model.bin", "config.json"])
        return os.path.exists(os.path.join(p, "pytorch_model.bin"))
    except Exception:
        return False


def demucs_ready():
    d = os.path.join(TORCH_HOME, "hub", "checkpoints")
    return os.path.isdir(d) and any(f.endswith(".th") for f in os.listdir(d))


def status():
    return {"stems": demucs_ready(), "transcription": whisper_ready(), "tagging": clap_ready()}


def install(which=("stems", "transcription", "tagging")):
    """Download what is missing. Returns the status afterwards."""
    from huggingface_hub import snapshot_download
    if "stems" in which and not demucs_ready():
        from demucs.pretrained import get_model
        get_model(DEMUCS_NAME)
    if "transcription" in which and not whisper_ready():
        snapshot_download(WHISPER_REPO, local_dir=WHISPER_DIR)
    if "tagging" in which and not clap_ready():
        snapshot_download(CLAP_REPO, cache_dir=CACHE, allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "merges.txt", "vocab.json"])
    return status()
