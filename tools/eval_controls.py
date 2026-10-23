"""Sanity-check the evaluation harness with known signals before trusting it on model output.

Positive control (lyrics): Windows SAPI speaks the two test lyric lines -> ASR WER should be low.
Negative control (lyrics): an instrumental MusicGen clip -> WER should be ~1.0 or no words.
Rhythm controls: a click track at 100 BPM -> median ~100 BPM, strong pulse; white noise -> weak pulse.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_audio as ev  # noqa: E402

LYRICS = "Streetlights hum a quiet tune\nI'm walking home beneath the moon"
tmp = tempfile.mkdtemp()
out = {}

# --- rhythm controls
sr = 32000
clicks = np.zeros(sr * 12, dtype=np.float32)
for t in np.arange(0, 12, 60 / 100):
    i = int(t * sr)
    clicks[i: i + 400] = np.hanning(400) * np.sin(np.arange(400) * 0.5)
out["click_100bpm"] = ev.rhythm(clicks, sr)
out["white_noise"] = ev.rhythm((np.random.default_rng(0).standard_normal(sr * 12) * 0.1).astype(np.float32), sr)

# --- spoken lyrics (positive control)
wav = os.path.join(tmp, "spoken.wav")
txt = os.path.join(tmp, "lyrics.txt")
open(txt, "w", encoding="utf-8").write(LYRICS.replace(chr(10), ". "))   # file avoids PowerShell quote escaping
ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
      f"$s.SetOutputToWaveFile('{wav}'); $s.Speak((Get-Content -Raw '{txt}')); $s.Dispose()")
subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
m, srs, _, _ = ev.load_mono(wav)
t = ev.transcribe(m, srs)
out["spoken_lyrics"] = {"transcript": t, "wer": ev.wer(LYRICS, t)}

# --- instrumental (negative control): any MusicGen clip passed as argv[1]
if len(sys.argv) > 1:
    m, srs, _, _ = ev.load_mono(sys.argv[1])
    t = ev.transcribe(m, srs)
    out["instrumental"] = {"transcript": t, "wer": ev.wer(LYRICS, t)}
print(json.dumps(out, indent=1))
