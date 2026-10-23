"""Layered prompt builder (EXPERIMENTAL, prompt_style="layered"): mood is a *tone layer on top of* the music, not its definition.

Priority order written into the text, from most to least important (earlier = stronger framing):
  1 genre   2 instruments (+ roles)   3 rhythm / tempo   4 mood   5 production
Rules enforced here (and unit-tested):
  * the text opens with the genre, never with a mood word; mood comes AFTER genre, instruments and tempo, and is short;
  * moods are expressed through tempo-neutral means (melodic phrasing, harmonic colour, dynamics, texture) and never
    through slower/faster, major/minor "switches of style", or new instruments;
  * the genre descriptions contain no instrument names, so nothing the user did not select is introduced;
  * when the mood points the other way from the genre's energy (sad + metal, energetic + ambient) the text says the genre's
    energy and groove stay in place and the mood shows in melody and harmony.
HONEST LIMIT: MusicGen takes ONE text prompt. There is no independent conditioning weight for genre vs mood, so this is a
framing of the text, not a weighted control. Whether the model obeys it is a listening question (tools/experiments/mood_override.py).
Kept separate from engine/arrangement.py on purpose: that module produced the clips of the pending round-5 listening test and
must keep reproducing them."""

# Genre identity: feel / groove phrase, plus drum and bass behaviour. No instrument names other than the drum/bass role itself.
GENRE = {
    "pop": ("catchy and polished, with a clear hook", "a steady backbeat", "a simple supportive line"),
    "rock": ("driving and powerful", "a driving backbeat", "a punchy line"),
    "jazz": ("swinging, with extended chords and improvised-sounding phrases", "brushed swing patterns", "a walking line"),
    "electronic": ("synthetic and rhythmic", "a steady electronic beat", "a pulsing line"),
    "lo-fi": ("dusty, relaxed and loop-based", "a soft, lazy beat", "a warm, mellow line"),
    "hip hop": ("beat-driven with a strong groove", "a laid-back boom-bap beat", "a deep, rolling line"),
    "r&b": ("smooth and groove-based", "a smooth, understated groove", "a round, melodic line"),
    "classical": ("flowing, with rich harmony and clear melodic lines", "soft timpani-like accents", "a gentle low line"),
    "country": ("warm and story-like", "a simple shuffle", "a walking line"),
    "folk": ("organic and warm", "light, gentle patterns", "a soft, round line"),
    "blues": ("soulful, with a 12-bar feel", "a slow shuffle", "a walking line"),
    "funk": ("syncopated and tight", "a tight syncopated groove", "a syncopated line"),
    "disco": ("danceable and upbeat", "a four-on-the-floor beat", "an octave-jumping line"),
    "ambient": ("spacious and slow-moving", "very sparse, soft patterns", "a deep, sustained line"),
    "cinematic": ("wide, dramatic and building", "big, powerful hits", "a deep, sustained low end"),
    "reggae": ("laid-back, with off-beat rhythm", "a one-drop beat", "a deep, melodic line"),
    "metal": ("heavy and aggressive", "fast, pounding patterns", "a thick, heavy line"),
    "house": ("repetitive and danceable", "a four-on-the-floor kick pattern", "a pumping line"),
    "techno": ("relentless and hypnotic", "a hard four-on-the-floor pattern", "a driving line"),
    "synthwave": ("retro and nostalgic", "a steady electronic beat", "a pulsing line"),
}
ROLES = {
    "acoustic guitar": "strummed or fingerpicked chords", "electric guitar": "riffs and lead lines", "bass": None, "drums": None,
    "percussion": "light rhythmic accents", "piano": "chords and a simple melody", "strings": "sustained harmonies",
    "violin": "an expressive melody", "cello": "a warm low counter-melody", "synthesizer": "pads and arpeggios",
    "saxophone": "a lead melody", "trumpet": "bright melodic phrases", "flute": "an airy melody", "organ": "sustained chords",
    "brass": "bold accents and swells", "harp": "gentle arpeggios",
}
# Mood = tone + tempo-neutral technique. Nothing here mentions speed, key switches, genres or instruments.
MOOD = {
    "calm": (["serene", "composed"], "gentle phrasing and smooth melodic contours"),
    "sad": (["wistful", "melancholic"], "falling melodic lines and a touch of minor harmonic colour"),
    "energetic": (["animated", "spirited"], "lively phrasing and accented dynamics"),
    "dark": (["brooding", "shadowed"], "darker harmonic colour and emphasis on the low register"),
    "romantic": (["tender", "heartfelt"], "lyrical, expressive phrasing"),
    "epic": (["grand", "triumphant"], "broad, rising phrases and big dynamic swells"),
    "mellow": (["smooth", "gentle"], "relaxed phrasing"),
    "uplifting": (["hopeful", "optimistic"], "brighter harmonic colour and rising melodic lines"),
    "tense": (["anxious", "suspenseful"], "unresolved harmonic tension"),
    "dreamy": (["hazy", "floating"], "spacious reverb and floating textures"),
    "happy": (["joyful", "warm"], "bright harmonic colour and playful phrasing"),
}
HIGH_ENERGY_GENRES = {"rock", "metal", "funk", "disco", "techno", "house", "hip hop", "electronic", "synthwave", "pop"}
LOW_ENERGY_GENRES = {"ambient", "classical", "lo-fi", "folk", "blues", "jazz", "reggae"}
LOW_ENERGY_MOODS = {"sad", "calm", "dark", "mellow", "romantic", "dreamy", "tense"}
HIGH_ENERGY_MOODS = {"energetic", "epic", "happy", "uplifting"}
PRODUCTION = {
    "warm": "a warm, rounded mix", "spacious": "a spacious mix with natural reverb", "dry": "a close, dry mix",
    "lo-fi": "a lo-fi, tape-like texture", "polished": "a polished, clean mix", "live": "a live-room feel",
}
MAX_MOOD_WORDS = 30      # the mood layer must stay a small part of the prompt


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def mood_layer(moods, primary_genre):
    """The emotional-tone sentence (<= MAX_MOOD_WORDS words so mood stays a small part of the prompt).
    Dominant mood first; secondary moods are 'undertones'. If the full wording is too long, detail is dropped in this order:
    second technique, then undertone adjectives, then all adjectives. Every mood NAME is always kept."""
    known = [m for m in moods if m in MOOD]
    names = known[0] + (f" with {_join(known[1:])} undertones" if len(known) > 1 else "")
    g = primary_genre or "the"
    if primary_genre in HIGH_ENERGY_GENRES and any(m in LOW_ENERGY_MOODS for m in moods) and not any(m in HIGH_ENERGY_MOODS for m in moods):
        keep = f"while keeping the full energy and drive of {g} music"
    elif primary_genre in LOW_ENERGY_GENRES and any(m in HIGH_ENERGY_MOODS for m in moods) and not any(m in LOW_ENERGY_MOODS for m in moods):
        keep = f"within the restrained character of {g} music"
    else:
        keep = f"while keeping the {g} groove, tempo and instruments".replace("the the", "the")
    variants = [
        (MOOD[known[0]][0] + [MOOD[m][0][0] for m in known[1:]], known[:2]),      # full
        (MOOD[known[0]][0] + [MOOD[m][0][0] for m in known[1:]], known[:1]),      # one technique
        (MOOD[known[0]][0], known[:1]),                                           # dominant mood detail only
        ([], known[:1]),                                                          # names + one technique + protection clause
    ]
    for adjectives, tech_moods in variants:
        adj = f"{_join(adjectives)}, " if adjectives else ""
        text = f"Emotional tone: {names}, {adj}shown through {_join([MOOD[m][1] for m in tech_moods])}, {keep}."
        if len(text.split()) <= MAX_MOOD_WORDS:
            return text
    return text


def _a(word):
    return "An" if word[:1].lower() in "aeiou" else "A"


def describe(c: dict) -> dict:
    """Return {"text": ..., "layers": {"genre": ..., "instruments": ..., "rhythm": ..., "mood": ..., "production": ...}, "notes": [...]}."""
    genres, moods, instr = c["genres"], c["moods"], c["instruments"]
    primary = genres[0] if genres else None
    notes = []
    # 1 genre
    if primary:
        g = f"{_a(primary)} {primary} instrumental piece" + (f" with {_join(genres[1:])} influences" if len(genres) > 1 else "")
        feel = GENRE.get(primary, ("",))[0]
        g += f", {feel}." if feel else "."
    else:
        g = "An instrumental piece."
    # 2 instruments
    roles = []
    for i in instr:
        phrase = ROLES.get(i, "a supporting part")
        if phrase is None:                                      # drums / bass follow the primary genre's behaviour
            idx = 1 if i == "drums" else 2
            phrase = GENRE[primary][idx] if primary in GENRE else ("a steady beat" if i == "drums" else "a steady bass line")
        roles.append(f"the {i} playing {phrase}" if ROLES.get(i) else f"the {i} providing {phrase}")
    ins = ("With " + _join(roles) + ", all following one steady groove and the same chords.") if roles else ""
    if len(instr) > 5:
        notes.append(f"{len(instr)} instruments requested: the model will probably not render all of them clearly.")
    # 3 rhythm and tempo
    bits = []
    if c.get("bpm"):
        bits.append(f"{c['bpm']} bpm")
    if c.get("time_signature"):
        bits.append(f"{c['time_signature']} time")
    if c.get("key"):
        bits.append(f"in {c['key']}")
    rhythm = (", ".join(bits) + ".") if bits else ""
    # 4 mood (a layer ON TOP of the above) and 5 production
    mood = mood_layer(moods, primary) if moods else ""
    prod = ""
    if c.get("production"):
        prod = f"Production: {PRODUCTION[c['production']]}."
    parts = [p for p in (g, ins, rhythm, mood, prod) if p]
    text = " ".join(parts)
    if c.get("prompt"):
        text = c["prompt"].rstrip(".") + ". " + text
    return {"text": " ".join(text.split()), "layers": {"genre": g, "instruments": ins, "rhythm": rhythm, "mood": mood, "production": prod}, "notes": notes}
