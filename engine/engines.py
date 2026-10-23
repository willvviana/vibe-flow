"""Engine interface: everything model-specific hides behind `Engine`.

The server/worker only talks to this interface, so adding a model means adding one class here plus a registry entry.
Only engines that pass the acceptance criteria are registered in ENGINES. ACE-Step is NOT registered: it failed
evaluation (see docs/EVALUATION.md); its experiment code lives in tools/experiments/ and is not imported by the app."""
from abc import ABC, abstractmethod

from . import generate, hardware, memory, registry


class Engine(ABC):
    id: str
    name: str

    @abstractmethod
    def capabilities(self) -> dict:
        """What this engine can really do: tasks, vocals, lyrics, max duration, sample rate."""

    @abstractmethod
    def preflight(self, settings: dict) -> list:
        """Raise memory.ResourceError if the job cannot safely run; otherwise return warnings."""

    @abstractmethod
    def generate(self, job: dict, emit, keep_loaded: bool = False) -> dict:
        """Run the job, save to the library, return track metadata."""


class MusicGenEngine(Engine):
    """Instrumental text-to-music and continuation (transformers MusicGen). Vocals/lyrics unsupported."""
    id, name = "musicgen", "MusicGen"

    def capabilities(self):
        return {"tasks": ["text", "continue", "melody", "adapt"], "vocals": False, "lyrics": False, "max_duration_s": 60,
                "models": [m.id for m in registry.MODELS.values()]}

    def preflight(self, settings):
        spec = registry.pick_model(settings.get("mode") or "text")
        return memory.preflight(spec, hardware.detect(), "cpu")

    def generate(self, job, emit, keep_loaded=False):
        if (job.get("settings") or {}).get("mode") == "adapt":      # sample-adaptive composition (engine/sample_compose.py)
            from . import sample_compose
            return sample_compose.run(job, emit, keep_loaded=keep_loaded)
        return generate.run(job, emit, keep_loaded=keep_loaded)


ENGINES = {e.id: e for e in (MusicGenEngine(),)}


def get(engine_id: str = "musicgen") -> Engine:
    if engine_id not in ENGINES:
        raise ValueError(f"Unknown engine '{engine_id}'. Available: {', '.join(ENGINES)}")
    return ENGINES[engine_id]
