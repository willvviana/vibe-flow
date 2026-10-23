"""Instrument-presence PROXY using an AudioSet tagger (MIT/ast-finetuned-audioset-10-10-0.4593, BSD-3, 330 MB safetensors).

Development/evaluation tool only (not imported by the app). It asks the tagger how likely each selected instrument is audible.
It is NOT proof of anything musical: AudioSet labels overlap (a "guitar" label fires for most plucked strings), it hears
genre context, and it says nothing about roles, timing or coherence. Use it to *support* listening, never to replace it.

  python tools/instrument_tags.py <clip.wav> [instrument ...]

`presence(path, instruments)` -> {instrument: max probability over 10 s windows}. `validate()` (below, via
tools/experiments/validate_tagger.py) checks on clips whose instruments are known that requested instruments score higher than
non-requested ones (AUC), which is the only evidence that this proxy means anything."""
import math
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
# AudioSet label names that count as evidence for each instrument selection (max over the group is used).
LABELS = {
    "acoustic guitar": ["Acoustic guitar", "Guitar", "Steel guitar, slide guitar", "Plucked string instrument"],
    "electric guitar": ["Electric guitar", "Guitar", "Distortion"],
    "bass": ["Bass guitar", "Double bass", "Bass drum"] ,
    "drums": ["Drum kit", "Drum", "Snare drum", "Bass drum", "Hi-hat", "Cymbal"],
    "percussion": ["Percussion", "Tambourine", "Maraca", "Shaker", "Drum", "Hi-hat", "Cymbal"],
    "piano": ["Piano", "Keyboard (musical)", "Electric piano"],
    "strings": ["String section", "Violin, fiddle", "Cello", "Bowed string instrument"],
    "violin": ["Violin, fiddle", "Bowed string instrument", "String section"],
    "cello": ["Cello", "Bowed string instrument", "String section"],
    "synthesizer": ["Synthesizer", "Electronic organ", "Electronic music"],
    "saxophone": ["Saxophone", "Wind instrument, woodwind instrument"],
    "trumpet": ["Trumpet", "Brass instrument"],
    "flute": ["Flute", "Wind instrument, woodwind instrument"],
    "organ": ["Organ", "Electronic organ", "Hammond organ"],
    "brass": ["Brass instrument", "Trumpet", "Trombone", "French horn"],
    "harp": ["Harp", "Plucked string instrument"],
}
_pipe = None


def _model():
    global _pipe
    if _pipe is None:
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
        import torch
        fe = AutoFeatureExtractor.from_pretrained(MODEL)
        m = ASTForAudioClassification.from_pretrained(MODEL).eval()
        _pipe = (fe, m, torch)
    return _pipe


def probabilities(path):
    """Per-window sigmoid probabilities for all 527 AudioSet classes: array [windows, 527] and the label list."""
    fe, m, torch = _model()
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    win, hop = 16000 * 10, 16000 * 5
    starts = list(range(0, max(len(x) - win, 0) + 1, hop)) or [0]
    out = []
    for s in starts:
        inp = fe(x[s: s + win], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out.append(torch.sigmoid(m(**inp).logits[0]).numpy())
    labels = [m.config.id2label[i] for i in range(len(m.config.id2label))]
    return np.array(out), labels


def presence(path, instruments):
    probs, labels = probabilities(path)
    idx = {l: i for i, l in enumerate(labels)}
    res = {}
    for ins in instruments:
        cols = [idx[l] for l in LABELS.get(ins, []) if l in idx]
        res[ins] = round(float(probs[:, cols].max()), 3) if cols else None
    return res


if __name__ == "__main__":
    print(presence(sys.argv[1], sys.argv[2:] or list(LABELS)))
