"""Settings validation and model-specific prompt building.

Every setting is classified as 'enforced' (the code guarantees it) or 'approximate'
(only text conditioning; the model may ignore it)."""
import re

from . import profiles

GENRES = ["pop", "rock", "jazz", "electronic", "lo-fi", "hip hop", "r&b", "classical", "country", "folk",
          "blues", "funk", "disco", "ambient", "cinematic", "reggae", "metal", "house", "techno", "synthwave"]
INSTRUMENTS = ["piano", "acoustic guitar", "electric guitar", "bass", "drums", "strings", "violin", "cello",
               "synthesizer", "saxophone", "trumpet", "flute", "organ", "percussion", "brass", "harp"]
MOODS = ["happy", "sad", "energetic", "calm", "dark", "romantic", "epic", "mellow", "uplifting", "tense", "dreamy"]
KEYS = [f"{n} {m}" for n in ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"] for m in ("major", "minor")]
TIME_SIGS = ["4/4", "3/4", "6/8", "5/4", "7/8"]
MAX_PROMPT = 500


class ValidationError(ValueError):
    pass


def _list(v, label):
    out = []
    for x in (v or []):
        x = str(x).strip().lower()
        if x:
            out.append(x[:40])
    if len(out) > 8:
        raise ValidationError(f"Too many {label} (max 8).")
    return out


def validate(s: dict, max_duration: int = 60) -> dict:
    """Return a clean settings dict or raise ValidationError."""
    prompt = str(s.get("prompt") or "").strip()
    if not prompt and not s.get("genres") and not s.get("instruments"):
        raise ValidationError("Describe your music or pick a genre/instrument.")
    if len(prompt) > MAX_PROMPT:
        raise ValidationError(f"Prompt is too long (max {MAX_PROMPT} characters).")
    if s.get("lyrics") or s.get("vocals"):
        raise ValidationError("Music with Vocals is not available: the installed engine (MusicGen) is instrumental-only, and the only vocal model "
                              "evaluated (ACE-Step) is not integrated (warble, ~12 GB RAM). Remove lyrics/vocals.")
    try:
        duration = float(s.get("duration", 10))
    except (TypeError, ValueError):
        raise ValidationError("Duration must be a number of seconds.")
    if not 2 <= duration <= max_duration:
        raise ValidationError(f"Duration must be between 2 and {max_duration} seconds.")
    bpm = s.get("bpm")
    if bpm in ("", None):
        bpm = None
    else:
        try:
            bpm = int(bpm)
        except (TypeError, ValueError):
            raise ValidationError("BPM must be a whole number.")
        if not 40 <= bpm <= 220:
            raise ValidationError("BPM must be between 40 and 220.")
    key = s.get("key") or None
    if key and key not in KEYS:
        raise ValidationError(f"Unknown key '{key}'.")
    ts = s.get("time_signature") or None
    if ts and ts not in TIME_SIGS:
        raise ValidationError(f"Unknown time signature '{ts}'.")
    seed = s.get("seed")
    try:
        seed = None if seed in ("", None) else int(seed)
    except (TypeError, ValueError):
        raise ValidationError("Seed must be a whole number.")
    fmt = str(s.get("format") or "wav").lower()
    if fmt not in ("wav", "flac", "mp3"):
        raise ValidationError("Format must be wav, flac or mp3.")
    mode = str(s.get("mode") or "text")
    if mode not in ("text", "continue", "melody", "adapt"):
        raise ValidationError("Mode must be text, continue, melody or adapt.")
    if mode != "text" and not s.get("reference"):
        raise ValidationError("This mode needs a reference audio file.")
    try:
        profile_name, prof = profiles.resolve(s.get("profile"))
    except ValueError as e:
        raise ValidationError(str(e))
    try:
        temp = float(s.get("temperature", prof["temperature"]))
        guidance = float(s.get("guidance", prof["guidance"]))
    except (TypeError, ValueError):
        raise ValidationError("Temperature and guidance must be numbers.")
    if not 0.1 <= temp <= 1.5 or not 1 <= guidance <= 10:
        raise ValidationError("Temperature must be 0.1–1.5 and guidance 1–10.")
    style = str(s.get("prompt_style") or prof["prompt_style"])
    if style not in ("keywords", "sentence", "arrangement", "layered"):
        raise ValidationError("prompt_style must be 'keywords', 'sentence', 'arrangement' or 'layered'.")
    production = s.get("production") or None
    if production is not None:
        from . import layered
        if production not in layered.PRODUCTION:
            raise ValidationError(f"production must be one of: {', '.join(layered.PRODUCTION)}.")
    try:
        top_k = int(s.get("top_k", prof["top_k"]))
    except (TypeError, ValueError):
        raise ValidationError("top_k must be a whole number.")
    if not 10 <= top_k <= 500:
        raise ValidationError("top_k must be between 10 and 500.")
    adh = s.get("adherence")
    if mode == "adapt":
        if adh in (None, ""):
            raise ValidationError("Adapt mode needs an adherence value between 0 (loose) and 100 (close to the sample).")
        try:
            adh = float(adh)
        except (TypeError, ValueError):
            raise ValidationError("Adherence must be a number from 0 to 100.")
        if not 0 <= adh <= 100:
            raise ValidationError("Adherence must be between 0 and 100.")
    else:
        adh = None
    try:
        level = float(s.get("layer_level_db", -6.0))
    except (TypeError, ValueError):
        raise ValidationError("Layer level must be a number of dB.")
    if not -24 <= level <= 0:
        raise ValidationError("Layer level must be between -24 and 0 dB.")
    return {
        "adherence": adh, "layer_level_db": level,
        "prompt": prompt,
        "genres": _list(s.get("genres"), "genres"),
        "instruments": _list(s.get("instruments"), "instruments"),
        "moods": _list(s.get("moods"), "moods"),
        "bpm": bpm, "key": key, "time_signature": ts,
        "duration": duration, "seed": seed, "format": fmt, "mode": mode,
        "temperature": temp, "guidance": guidance, "profile": profile_name, "prompt_style": style, "production": production, "top_k": top_k, "fast": bool(s.get("fast")),
        "reference": s.get("reference") or None,
    }


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_prompt_sentence(c: dict) -> str:
    """Natural-sentence caption. A flat keyword list can make the model treat each instrument as its own layer;
    one sentence describing a single ensemble ('played together') matches how MusicGen's training captions read."""
    mood = (_join(c["moods"]) + " ") if c["moods"] else ""
    genre = (_join(c["genres"]) + " ") if c["genres"] else ""
    s = f"A {mood}{genre}instrumental piece"
    if c["instruments"]:
        s += f" played together by {_join(c['instruments'])}, with the parts following the same rhythm and chords"
    tail = []
    if c["bpm"]:
        tail.append(f"{c['bpm']} bpm")
    if c["key"]:
        tail.append(f"in {c['key']}")
    if c["time_signature"]:
        tail.append(f"{c['time_signature']} time")
    if tail:
        s += ", " + ", ".join(tail)
    out = (c["prompt"] + ". " if c["prompt"] else "") + s
    return re.sub(r"\s+", " ", out).strip()


def build_prompt(c: dict, model_id: str, style: str = "keywords") -> str:
    """MusicGen was trained on short descriptive captions: 'genre, mood, instruments, bpm, key'."""
    if style == "sentence":
        return build_prompt_sentence(c)
    if style == "arrangement":                      # experimental, frozen for the round-5 listening test, see engine/arrangement.py
        from . import arrangement
        return arrangement.describe(c)["text"]
    if style == "layered":                          # experimental: genre > instruments > tempo > mood > production, see engine/layered.py
        from . import layered
        return layered.describe(c)["text"]
    parts = []
    if c["genres"]:
        parts.append(", ".join(c["genres"]))
    if c["moods"]:
        parts.append(", ".join(c["moods"]))
    if c["instruments"]:
        parts.append("featuring " + ", ".join(c["instruments"]))
    if c["bpm"]:
        parts.append(f"{c['bpm']} bpm")
    if c["key"]:
        parts.append(f"in {c['key']}")
    if c["time_signature"]:
        parts.append(f"{c['time_signature']} time")
    parts.append("instrumental")
    out = ", ".join(([c["prompt"]] if c["prompt"] else []) + parts)
    return re.sub(r"\s+", " ", out).strip()


def control_report(c: dict) -> dict:
    """Honest summary of what is guaranteed versus only suggested to the model."""
    enforced = ["duration", "seed (reproducible on the same machine/software)", "instrumental output", "output format"]
    approx = [k for k, v in (("genre", c["genres"]), ("instruments", c["instruments"]), ("mood", c["moods"]),
                              ("tempo (BPM)", c["bpm"]), ("key", c["key"]), ("time signature", c["time_signature"])) if v]
    return {"enforced": enforced, "approximate": approx,
            "note": "Approximate settings are sent as text to the model, which may not follow them exactly."}
