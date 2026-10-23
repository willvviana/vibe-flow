"""Generation-quality profiles for MusicGen: named bundles of prompt style + sampling.

DEFAULT_PROFILE is the production default. It became "d" after a blinded 6-seed listening test (docs/EVALUATION.md, round 2):
D scored 4.5 vs 3.67 for the old default, 6/6 vs 3/6 "would ship", and never lost a seed. Evidence is one listener and one prompt family.
TO REVERT to the original behaviour: set DEFAULT_PROFILE = "baseline" below, or set the environment variable VIBEFLOW_PROFILE=baseline.
Users/API callers can pass settings["profile"]; explicit settings (temperature, top_k, guidance, prompt_style) always win."""
import os

PROFILES = {
    "baseline": {"prompt_style": "keywords", "temperature": 1.0, "top_k": 250, "guidance": 3.0},
    # 'D': sentence prompt, lower temperature, stronger text guidance (blind-test winner).
    "d": {"prompt_style": "sentence", "temperature": 0.8, "top_k": 250, "guidance": 5.0},
    # 'layered': D sampling + the layered prompt (mood as a tone layer after genre, instruments and tempo). Experimental, not default.
    "layered": {"prompt_style": "layered", "temperature": 0.8, "top_k": 250, "guidance": 5.0},
}
DEFAULT_PROFILE = "d"


def default_profile() -> str:
    name = os.environ.get("VIBEFLOW_PROFILE") or DEFAULT_PROFILE
    return name if name in PROFILES else DEFAULT_PROFILE


def resolve(name=None) -> tuple:
    """Return (profile name, parameter dict). Raises ValueError for an unknown name."""
    name = (name or default_profile()).lower()
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {', '.join(PROFILES)}")
    return name, PROFILES[name]
