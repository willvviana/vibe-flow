"""Adherence: map the user-facing 0-100 "Loose <-> Close to sample" setting to the controls that actually change generation.

The percentage is NOT a volume and NOT MusicGen's guidance scale. It drives six measurable controls (all continuous in p):

  retain_fraction   share of the sample's bars kept as the literal original audio (phrase-aligned, evenly spread). p=100 -> 1.0
                    (the whole sample, untouched). Source material is never regenerated or altered; retained = copied.
  tempo_hint        the sample's BPM is written into the generation text (weight >= 0.5 -> included)
  align_strength    how much of the tempo/phase gap between generated layers and the sample grid is corrected by DSP (0..1)
  key_hint          the sample's key is written into the text, and layers are transposed to it when >= 0.5 (small shifts only)
  context_bars      bars of the SAMPLE passed to MusicGen as an AUDIO PROMPT (continuation conditioning); the new layers then start
                    after the context, in the sample's own tempo, key and timbre. 0 below p = 25.
  layer_level_db    (user control, separate from p) level of the generated layers under retained bars; unretained bars use 0 dB.

Anchor behaviour (see table()):  0-10 % new music, tempo only; 25 % + key & groove hints; 50 % audio-conditioned, ~1/3 of bars kept;
75 % most bars kept, strong alignment; 100 % every bar kept untouched with layers added underneath.
Deliberately simple, monotone and smooth, so "intermediate settings behave predictably"."""


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def controls(p: float, n_bars: int = 8) -> dict:
    p = _clip(float(p), 0.0, 100.0)
    return {
        "adherence": p,
        "retain_fraction": round((p / 100.0) ** 1.5, 4),
        "tempo_hint": round(_clip(p / 10.0), 3),
        "align_strength": round(_clip((p - 5.0) / 45.0), 3),
        "key_hint": round(_clip((p - 10.0) / 30.0), 3),
        "context_bars": 0 if p < 25 else max(1, int(round(1 + (p - 25.0) / 75.0 * max(0.0, min(n_bars * 0.25, 4.0) - 1)))),
        "carve_depth_db": round(3.0 + 6.0 * (p / 100.0) ** 1.5, 2),
    }


def table(points=(0, 10, 25, 50, 75, 100), n_bars: int = 8):
    return {p: controls(p, n_bars) for p in points}


def select_retained(n_bars: int, retain_fraction: float):
    """Which bars are kept: phrase units (2 bars when the sample has >= 8 bars, else 1), evenly spread, always incl. the first unit
    when anything is kept. Returns a list of booleans per bar. Deterministic and nested-ish: raising p only adds units."""
    if n_bars <= 0:
        return []
    if retain_fraction >= 0.9999:
        return [True] * n_bars
    unit = 2 if n_bars >= 8 else 1
    units = -(-n_bars // unit)
    k = int(round(retain_fraction * units))
    keep_units = set()
    if k > 0:
        keep_units = {int(round(i * (units - 1) / max(k - 1, 1))) for i in range(k)} if k > 1 else {0}
        i = 0
        while len(keep_units) < k:             # rounding collisions: top up with the nearest free units
            if i not in keep_units:
                keep_units.add(i)
            i += 1
    return [(b // unit) in keep_units for b in range(n_bars)]
