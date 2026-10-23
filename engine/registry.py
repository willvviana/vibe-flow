"""Model registry: what each model can and cannot do. Capabilities here are verified claims only."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    repo: str
    params_m: int
    download_gb: float
    ram_gb: float            # estimated peak RAM when loaded on CPU (fp32)
    sample_rate: int
    max_segment_s: int
    license: str
    commercial_ok: bool
    tasks: tuple = ()        # subset of: text, continue, melody, adapt
    vocals: bool = False
    lyrics: bool = False
    notes: str = ""
    measured_peak_gb: float = None   # peak working set measured on the dev machine (tools/bench.py); None = only estimated

    def to_dict(self):
        return asdict(self)


MODELS = {
    "musicgen-small": ModelSpec(
        id="musicgen-small", name="MusicGen Small", repo="facebook/musicgen-small",
        params_m=300, download_gb=2.2, ram_gb=4.4, measured_peak_gb=4.4, sample_rate=32000, max_segment_s=30,
        license="CC-BY-NC 4.0 (weights)", commercial_ok=False,
        tasks=("text", "continue", "adapt"),
        notes="Instrumental text-to-music and audio continuation. Mono 32 kHz output. No vocals or lyrics."),
    "musicgen-melody": ModelSpec(
        id="musicgen-melody", name="MusicGen Melody", repo="facebook/musicgen-melody",
        params_m=1500, download_gb=5.8, ram_gb=10, sample_rate=32000, max_segment_s=30,
        license="CC-BY-NC 4.0 (weights)", commercial_ok=False,
        tasks=("text", "melody"),
        notes="Text-to-music guided by the melody (chroma) of a reference clip. Heavy on CPU."),
}

# Evaluated but not installed. Reasons are specific to this machine / the models' documented requirements.
UNSUPPORTED = [
    {"id": "yue", "name": "YuE", "reason": "Full-song vocals with lyrics, but its published inference setup targets large NVIDIA GPUs (tens of GB VRAM). Not practical on CPU."},
    {"id": "ace-step", "name": "ACE-Step", "reason": "Supports lyrics and vocals but is designed for CUDA/MPS-class GPUs; CPU generation would take far too long. Candidate if an NVIDIA GPU is added."},
    {"id": "diffrhythm", "name": "DiffRhythm", "reason": "Vocal songs with lyrics; needs a CUDA GPU with several GB VRAM. Not usable here."},
    {"id": "stable-audio-open", "name": "Stable Audio Open", "reason": "Gated weights (Hugging Face login and licence acceptance) and aimed at short sound design clips; not installed."},
]


def get(model_id: str) -> ModelSpec:
    if model_id not in MODELS:
        raise ValueError(f"Unknown model '{model_id}'. Available: {', '.join(MODELS)}")
    return MODELS[model_id]


def pick_model(task: str, preferred=None) -> ModelSpec:
    """Choose a model for a task: text/continue -> small, melody -> melody."""
    if preferred:
        spec = get(preferred)
        if task not in spec.tasks:
            raise ValueError(f"{spec.name} does not support '{task}'. It supports: {', '.join(spec.tasks)}.")
        return spec
    for spec in MODELS.values():
        if task in spec.tasks:
            return spec
    raise ValueError(f"No model supports task '{task}'.")
