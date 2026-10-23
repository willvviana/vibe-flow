"""Arrangement-aware prompt builder (EXPERIMENTAL, API-only: prompt_style="arrangement").

MusicGen is a single text-conditioned model: it has NO per-instrument or per-genre control. All this module can do is write a
text description that (a) names every selected instrument, genre and mood, (b) gives each instrument a plausible role,
(c) says which genre is primary and which are influences, (d) turns moods into musical characteristics, and (e) adapts roles
when an instrument does not suit a genre. Whether the model then *renders* them is a separate question that must be checked by
listening (and, as a weak proxy, tools/instrument_tags.py). Nothing here guarantees an instrument is audible.

Order matters: the first genre is the primary genre, later ones are influences; the first mood is the dominant mood."""

# What each instrument typically does in an arrangement.
ROLES = {
    "acoustic guitar": "strummed or fingerpicked chords",
    "electric guitar": "riffs and lead lines",
    "bass": "a steady bass line locked to the drums",
    "drums": "the main groove",
    "percussion": "light rhythmic accents",
    "piano": "chords and a simple melody",
    "strings": "sustained harmonies",
    "violin": "an expressive melody",
    "cello": "a warm low counter-melody",
    "synthesizer": "pads and arpeggios",
    "saxophone": "a smooth lead melody",
    "trumpet": "bright melodic phrases",
    "flute": "an airy melody",
    "organ": "sustained chords",
    "brass": "bold accents and swells",
    "harp": "gentle arpeggios",
}

# Defining feel of each genre (rhythm / groove / harmony) and how drums and bass behave in it.
GENRE = {
    "pop": ("catchy, polished, with a clear chorus-like hook", "a steady backbeat", "a simple supportive bass line"),
    "rock": ("driving and guitar-led", "a driving backbeat", "a punchy bass line following the guitar riff"),
    "jazz": ("swinging, with extended chords and improvised-sounding phrases", "brushed swing drums", "a walking bass line"),
    "electronic": ("synthetic and rhythmic", "a steady electronic beat", "a pulsing synth bass"),
    "lo-fi": ("dusty, relaxed and loop-based", "a soft, lazy beat", "a warm, mellow bass"),
    "hip hop": ("beat-driven with a strong groove", "a laid-back boom-bap beat", "a deep, rolling bass"),
    "r&b": ("smooth, soulful and groove-based", "a smooth, understated groove", "a round, melodic bass"),
    "classical": ("orchestral, with flowing melody and rich harmony", "soft timpani-like percussion", "a gentle low-string bass line"),
    "country": ("warm, story-like and twangy", "a simple shuffle beat", "a walking country bass"),
    "folk": ("organic, warm and acoustic", "light hand percussion", "a soft, round bass"),
    "blues": ("soulful, with a 12-bar feel and bent notes", "a slow shuffle", "a walking blues bass"),
    "funk": ("syncopated and tight", "a tight syncopated groove", "a slapping, syncopated bass"),
    "disco": ("danceable and upbeat", "a four-on-the-floor beat", "an octave-jumping bass"),
    "ambient": ("spacious, slow-moving and atmospheric", "very sparse, soft percussion", "a deep, sustained bass"),
    "cinematic": ("wide, dramatic and building", "big, powerful percussion", "a deep, sustained low end"),
    "reggae": ("laid-back with off-beat rhythm", "a one-drop beat", "a deep, melodic bass"),
    "metal": ("heavy and aggressive", "fast, pounding drums", "a thick, distorted bass"),
    "house": ("repetitive, hypnotic and danceable", "a four-on-the-floor kick", "a pumping bass"),
    "techno": ("relentless, mechanical and hypnotic", "a hard four-on-the-floor kick", "a driving synth bass"),
    "synthwave": ("retro, neon-lit and nostalgic", "a steady electronic beat", "a pulsing analog synth bass"),
}

# Musical meaning of moods (harmony, dynamics, tempo feel).
MOOD = {
    "calm": "soft dynamics and slow-moving harmony",
    "sad": "a minor key and slow phrasing",
    "energetic": "driving rhythm and strong dynamics",
    "dark": "a minor key and a low register",
    "romantic": "a warm, flowing melody",
    "epic": "building dynamics and a wide, full sound",
    "mellow": "a relaxed, smooth feel",
    "uplifting": "bright major harmony and a rising melody",
    "tense": "sparse, restless harmony",
    "dreamy": "airy, reverb-soaked textures",
    "happy": "a bright, bouncy major key",
}
# Mood pairs that pull in opposite directions: the first mood leads, the second is placed as a contrast inside it.
CONFLICTING_MOODS = {frozenset(p) for p in [("calm", "energetic"), ("happy", "sad"), ("dark", "uplifting"), ("calm", "tense"),
                                            ("mellow", "energetic"), ("happy", "dark"), ("sad", "energetic")]}
# Instruments that do not suit a genre as-is: (genre, instrument) -> adapted role. Reported, never silently dropped.
ADAPT = {
    ("classical", "drums"): "soft timpani-like percussion",
    ("classical", "electric guitar"): "a restrained clean guitar in the background",
    ("classical", "synthesizer"): "a subtle string-like synth pad",
    ("ambient", "drums"): "very sparse, soft percussion",
    ("ambient", "electric guitar"): "slow, reverb-washed guitar swells",
    ("folk", "drums"): "light brushed percussion",
    ("folk", "electric guitar"): "a gentle clean guitar",
    ("jazz", "synthesizer"): "a soft electric-piano-like pad",
    ("metal", "flute"): "a high, eerie counter-melody",
    ("cinematic", "drums"): "big, powerful percussion",
}
MAX_CLEAR_INSTRUMENTS = 5     # beyond this a 300 M-parameter text-conditioned model very likely drops some


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _role(instr, genres):
    """Role phrase for an instrument, adapted to the primary genre when needed. Returns (phrase, adapted?)."""
    primary = genres[0] if genres else None
    if primary and (primary, instr) in ADAPT:
        return ADAPT[(primary, instr)], True
    if instr == "drums" and primary in GENRE:
        return GENRE[primary][1], False
    if instr == "bass" and primary in GENRE:
        return GENRE[primary][2], False
    return ROLES.get(instr, "a supporting part"), False


def describe(c: dict) -> dict:
    """Return {"text": prompt, "adaptations": [...], "notes": [...]}. Every selected genre, mood and instrument appears in text."""
    genres, moods, instr = c["genres"], c["moods"], c["instruments"]
    adaptations, notes = [], []
    # --- genre: primary + influences
    if genres:
        head = genres[0]
        g = f"{head} piece" + (f" with {_join(genres[1:])} influences" if len(genres) > 1 else "")
        feel = GENRE.get(head, ("", "", ""))[0]
        g += f", {feel}" if feel else ""
    else:
        g = "instrumental piece"
    # --- mood: dominant first, conflicts expressed as contrast
    mood_txt = ""
    if moods:
        mood_txt = _join(moods) + " "
    tail = ""
    if moods:
        parts = [MOOD[m] for m in moods[:2] if m in MOOD]
        tail = f", featuring {_join(parts)}" if parts else ""
        for a in moods:
            for b in moods:
                if a != b and frozenset((a, b)) in CONFLICTING_MOODS and moods.index(a) < moods.index(b):
                    tail += f"; a {a} foundation with {b} moments"
                    adaptations.append(f"moods '{a}' and '{b}' pull in opposite directions: '{a}' leads, '{b}' appears as contrast")
    parts = [f"A {mood_txt}{g}{tail}"]
    if c.get("bpm"):
        parts.append(f"{c['bpm']} bpm")
    if c.get("key"):
        parts.append(f"in {c['key']}")
    if c.get("time_signature"):
        parts.append(f"{c['time_signature']} time")
    text = ", ".join(parts) + "."
    # --- instruments: each gets a role
    if instr:
        roles = []
        for i in instr:
            phrase, adapted = _role(i, genres)
            if adapted:
                adaptations.append(f"{i} adapted for {genres[0]}: {phrase}")
            roles.append(f"the {i} playing {phrase}")
        text += " With " + _join(roles) + "."
        text += " All instruments follow one steady groove and the same chords."
        if len(instr) > MAX_CLEAR_INSTRUMENTS:
            notes.append(f"{len(instr)} instruments requested: the model will probably not render all of them clearly.")
    if len(genres) > 3:
        notes.append("More than 3 genres usually blurs into no identifiable style; the first is treated as primary.")
    if c.get("prompt"):
        text = c["prompt"].rstrip(".") + ". " + text
    return {"text": " ".join(text.split()), "adaptations": adaptations, "notes": notes}
