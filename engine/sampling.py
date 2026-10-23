"""The Sample page's engine: stem separation (Demucs), transcription (faster-whisper) and tagging (CLAP).

Real models only; nothing here fabricates a result. Honest limits (also shown in docs/SAMPLE.md):
  stems          Demucs htdemucs separates drums/bass/other/vocals. 'Instrumental' = drums+bass+other, 'Voice' = the vocal stem.
                 Separation is never perfect: faint vocal remnants stay in the instrumental and drum/hi-hat bleed in the vocals.
                 We measure and report how well vocal + instrumental add back up to the original (reconstruction residual).
  transcription  faster-whisper 'small' on CPU. Language is detected automatically per recording (and per live window).
                 NO speaker identification: there is no diarization model installed (pyannote needs a gated licence).
  tags           CLAP zero-shot. Scores are relative (a softmax over the candidate labels), NOT calibrated probabilities.
  tempo/key      the classical estimators in sample_analysis.py (reliable on steady 4/4 music, shaky otherwise)."""
import math
import os
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import sampling_models as sm

STEM_SR = 44100
_cache = {}


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path, sr, mono=False):
    x, in_sr = sf.read(path, dtype="float32", always_2d=True)
    if mono:
        x = x.mean(axis=1, keepdims=True)
    elif x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if in_sr != sr:
        g = math.gcd(int(in_sr), int(sr))
        x = resample_poly(x, sr // g, in_sr // g, axis=0).astype(np.float32)
    return x


def db(v):
    return 20 * math.log10(max(float(v), 1e-9))


# ---------------------------------------------------------------------------------------------- waveform peaks
def write_peaks(wav_path, json_path, bins=1600):
    """Min/max envelope of a (possibly very long) file in N bins, streamed in blocks so memory stays flat. Used for waveform drawing."""
    import json
    info = sf.info(wav_path)
    n = info.frames
    mn = np.zeros(bins, dtype=np.float32)
    mx = np.zeros(bins, dtype=np.float32)
    pos = 0
    block = 1 << 20
    with sf.SoundFile(wav_path) as f:
        while pos < n:
            x = f.read(min(block, n - pos), dtype="float32", always_2d=True)
            if not len(x):
                break
            m = x.mean(axis=1)
            idx = ((np.arange(pos, pos + len(m), dtype=np.int64) * bins) // n).astype(np.int64)
            starts = np.concatenate(([0], np.flatnonzero(np.diff(idx)) + 1))
            bi = idx[starts]
            mn[bi] = np.minimum(mn[bi], np.minimum.reduceat(m, starts))
            mx[bi] = np.maximum(mx[bi], np.maximum.reduceat(m, starts))
            pos += len(m)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"bins": bins, "duration": round(n / info.samplerate, 3), "min": [round(float(v), 4) for v in mn], "max": [round(float(v), 4) for v in mx]}, fh)


# ---------------------------------------------------------------------------------------------- stems
def _demucs():
    if "demucs" not in _cache:
        from demucs.pretrained import get_model
        m = get_model(sm.DEMUCS_NAME)
        m.eval()
        _cache["demucs"] = m
    return _cache["demucs"]


def separate(src, out_dir, progress=lambda **k: None, block_s=40.0, overlap_s=3.0):
    """Write vocals.wav and instrumental.wav (24-bit, 44.1 kHz stereo) next to each other in out_dir.
    The audio is processed in overlapping blocks (cross-faded) so long recordings use bounded memory and report progress."""
    import torch
    from demucs.apply import apply_model
    if not sm.demucs_ready():
        raise RuntimeError("The Demucs model is not installed. Run: npm run install-sampling")
    model = _demucs()
    sr = model.samplerate
    x = load_audio(src, sr)                                   # (T, 2)
    n = x.shape[0]
    if n < sr:
        raise ValueError("The recording is shorter than one second.")
    mono = x.mean(axis=1)
    mu, sd = float(mono.mean()), float(mono.std()) + 1e-8       # global normalisation, same for every block (as Demucs' own CLI does)
    dev = device()
    blk, ov = int(block_s * sr), int(overlap_s * sr)
    step = blk - ov
    starts = list(range(0, max(n - ov, 1), step)) or [0]
    acc = np.zeros((4, 2, n), dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    progress(stage="separate", fraction=0.0, eta_s=round(n / sr * 0.6, 1))      # CPU runs at roughly 0.4-0.6x the audio length
    with torch.no_grad():
        for i, s in enumerate(starts):
            e = min(s + blk, n)
            seg = torch.from_numpy(((x[s:e] - mu) / sd).T.copy())[None]
            y = apply_model(model, seg, device=dev, shifts=1, split=True, overlap=0.25, progress=False, num_workers=0)[0]
            y = (y.cpu().numpy() * sd + mu).astype(np.float32)   # (4, 2, L)
            L = e - s
            w = np.ones(L, dtype=np.float32)
            if s > 0:
                r = min(ov, L)
                w[:r] = np.linspace(0, 1, r, endpoint=False, dtype=np.float32)
            if e < n:
                r = min(ov, L)
                w[L - r:] = np.minimum(w[L - r:], np.linspace(1, 0, r, endpoint=False, dtype=np.float32) + 1e-3)
            acc[:, :, s:e] += y * w
            wsum[s:e] += w
            done = (i + 1) / len(starts)
            progress(stage="separate", fraction=round(done, 3), eta_s=round((time.time() - t0) / done * (1 - done), 1))
    acc /= np.maximum(wsum, 1e-6)
    names = list(model.sources)
    vocals = acc[names.index("vocals")].T
    inst = sum(acc[i] for i, nm in enumerate(names) if nm != "vocals").T
    orig = x
    peak = max(np.abs(vocals).max(), np.abs(inst).max(), 1e-9)
    gain = 0.99 / peak if peak > 0.99 else 1.0                   # only ever scale DOWN, and the same for both stems, to avoid clipping
    # how well do the two stems add back up to the original? (residual in dB relative to the original)
    resid = orig - (vocals + inst)
    recon = db(np.sqrt(np.mean(resid ** 2))) - db(np.sqrt(np.mean(orig ** 2)))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, "vocals.wav"), vocals * gain, sr, subtype="PCM_24")
    sf.write(os.path.join(out_dir, "instrumental.wav"), inst * gain, sr, subtype="PCM_24")
    return {"model": sm.DEMUCS_NAME, "device": dev, "sample_rate": sr, "gain": round(gain, 4), "seconds": round(time.time() - t0, 1),
            "reconstruction_residual_db": round(recon, 1),
            "vocals_rms_db": round(db(np.sqrt(np.mean(vocals ** 2))), 1), "instrumental_rms_db": round(db(np.sqrt(np.mean(inst ** 2))), 1),
            "vocal_energy_share": round(float(np.mean(vocals ** 2) / (np.mean(vocals ** 2) + np.mean(inst ** 2) + 1e-12)), 4)}


# ---------------------------------------------------------------------------------------------- transcription
def _whisper(threads=4):
    key = ("whisper", threads)
    if key not in _cache:
        if not sm.whisper_ready():
            raise RuntimeError("The Whisper model is not installed. Run: npm run install-sampling")
        from faster_whisper import WhisperModel
        _cache[key] = WhisperModel(sm.WHISPER_DIR, device="cpu", compute_type="int8", cpu_threads=threads)
    return _cache[key]


def transcribe(x16k, language=None, offset=0.0, threads=4, vad=True, prompt=None):
    """x16k: mono float32 at 16 kHz. Returns {'language','language_probability','segments':[{start,end,text}]} (times shifted by offset)."""
    model = _whisper(threads)
    segs, info = model.transcribe(x16k, language=language, vad_filter=vad, beam_size=1 if threads and vad == "live" else 5,
                                  condition_on_previous_text=False, initial_prompt=prompt,
                                  vad_parameters={"min_silence_duration_ms": 400})
    out = [{"start": round(offset + s.start, 2), "end": round(offset + s.end, 2), "text": s.text.strip()} for s in segs if s.text.strip()]
    return {"language": info.language, "language_probability": round(float(info.language_probability), 3), "segments": out}


def transcribe_file(path, language=None, progress=lambda **k: None, chunk_s=30 * 60):
    x = load_audio(path, 16000, mono=True)[:, 0]
    out = {"language": None, "language_probability": 0.0, "segments": []}
    n = len(x)
    for s in range(0, n, chunk_s * 16000):
        r = transcribe(x[s:s + chunk_s * 16000], language=language, offset=s / 16000)
        out["segments"] += r["segments"]
        if out["language"] is None:
            out["language"], out["language_probability"] = r["language"], r["language_probability"]
        progress(stage="transcribe", fraction=round(min(1, (s + chunk_s * 16000) / n), 3))
    return out


# ---------------------------------------------------------------------------------------------- tagging
GENRES = ["pop", "rock", "hip hop", "rap", "R&B", "soul", "funk", "jazz", "blues", "country", "folk", "reggae", "heavy metal", "punk rock",
          "indie rock", "alternative rock", "classical", "orchestral film score", "ambient", "electronic dance music", "house", "techno", "trance",
          "drum and bass", "dubstep", "lo-fi", "synthwave", "disco", "latin", "reggaeton", "afrobeats", "bossa nova", "gospel", "K-pop", "trap",
          "new age", "singer-songwriter acoustic", "world music", "opera", "chillout", "experimental", "children's music"]
SUBGENRES = ["deep house", "tech house", "progressive house", "acid techno", "melodic techno", "psytrance", "liquid drum and bass", "future bass",
             "boom bap hip hop", "trap hip hop", "lo-fi hip hop", "drill", "neo soul", "smooth jazz", "bebop", "jazz fusion", "delta blues",
             "country rock", "bluegrass", "indie folk", "post-rock", "shoegaze", "emo", "pop punk", "synth-pop", "electropop", "dance pop",
             "hard rock", "classic rock", "progressive rock", "death metal", "power metal", "baroque", "romantic era classical", "minimalism",
             "cinematic ambient", "dark ambient", "downtempo", "trip hop", "vaporwave", "city pop", "dancehall", "dub", "salsa", "bachata", "samba",
             "flamenco", "afro house", "highlife", "bollywood", "gospel choir", "a cappella", "chiptune", "EDM festival anthem", "ballad"]
INSTRUMENTS = ["acoustic guitar", "electric guitar", "distorted electric guitar", "bass guitar", "synth bass", "piano", "electric piano", "organ",
               "synthesizer pads", "synthesizer lead", "drum kit", "drum machine", "percussion", "hand drums", "violin", "viola", "cello",
               "string section", "double bass", "brass section", "trumpet", "trombone", "saxophone", "clarinet", "flute", "harp", "accordion",
               "banjo", "mandolin", "ukulele", "choir", "male vocals", "female vocals", "bells and mallets", "turntable scratching", "sitar"]
MOODS = ["happy", "sad", "energetic", "calm", "relaxing", "dark", "aggressive", "romantic", "melancholic", "uplifting", "dreamy", "tense",
         "epic", "playful", "nostalgic", "mysterious", "peaceful", "angry", "hopeful", "groovy", "intimate", "triumphant", "sensual", "eerie",
         "festive", "serious", "lonely", "confident"]


def _clap():
    if "clap" not in _cache:
        if not sm.clap_ready():
            raise RuntimeError("The CLAP tagging model is not installed. Run: npm run install-sampling")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import ClapModel, ClapProcessor
        p = snapshot_download(sm.CLAP_REPO, cache_dir=sm.CACHE, local_files_only=True)
        model = ClapModel.from_pretrained(p).eval()
        proc = ClapProcessor.from_pretrained(p)
        texts = {}
        with torch.no_grad():
            for key, labels, tmpl in (("genre", GENRES, "this is a recording of {} music"), ("subgenre", SUBGENRES, "this is {} music"),
                                      ("instrument", INSTRUMENTS, "the sound of {} playing in a song"), ("mood", MOODS, "{} music")):
                inp = proc(text=[tmpl.format(l) for l in labels], return_tensors="pt", padding=True)
                e = model.get_text_features(**inp)
                texts[key] = (labels, torch.nn.functional.normalize(e, dim=-1))
        _cache["clap"] = (model, proc, texts)
    return _cache["clap"]


def _audio_embedding(x48, windows=6, win_s=10):
    import torch
    model, proc, _ = _clap()
    n, w = len(x48), win_s * 48000
    if n <= w:
        starts = [0]
    else:                                                     # evenly spread windows, skipping near-silent ones
        starts = [int(s) for s in np.linspace(0, n - w, windows)]
    chunks = [x48[s:s + w] for s in starts]
    loud = [c for c in chunks if np.sqrt(np.mean(c ** 2)) > 1e-3] or chunks[:1]
    with torch.no_grad():
        inp = proc(audios=[np.asarray(c, dtype=np.float32) for c in loud], sampling_rate=48000, return_tensors="pt")
        e = model.get_audio_features(**inp)
        e = torch.nn.functional.normalize(e, dim=-1).mean(0, keepdim=True)
        return torch.nn.functional.normalize(e, dim=-1), len(loud)


def _rank(emb, key, top, scale=None):
    import torch
    model, _, texts = _clap()
    labels, te = texts[key]
    logits = (emb @ te.T)[0] * float(model.logit_scale_a.exp())
    p = torch.softmax(logits, dim=0).numpy()
    order = np.argsort(-p)[:top]
    return [{"label": labels[i], "score": round(float(p[i]), 3)} for i in order]


def tag(full_path, instrumental_path=None, progress=lambda **k: None):
    """Genre / subgenre / mood from the full mix, instruments from the instrumental stem (cleaner), plus tempo and key."""
    from . import sample_analysis
    full48 = load_audio(full_path, 48000, mono=True)[:, 0]
    emb, used = _audio_embedding(full48)
    out = {"genres": _rank(emb, "genre", 5), "subgenres": _rank(emb, "subgenre", 4), "moods": _rank(emb, "mood", 4),
           "model": "laion/clap-htsat-unfused (zero-shot; scores are relative rankings, not calibrated probabilities)", "windows_used": used}
    if instrumental_path:
        ie, _ = _audio_embedding(load_audio(instrumental_path, 48000, mono=True)[:, 0])
        inst = _rank(ie, "instrument", 7)
    else:
        inst = _rank(emb, "instrument", 7)
    out["instruments"] = inst
    x22 = sample_analysis.resample(full48, 48000)
    try:
        a = sample_analysis.analyse(x22, sample_analysis.SR)
        out["bpm"] = {"value": round(float(a["bpm"]), 1), "confidence": round(float(a["tempo_confidence"]), 2)}
        out["key"] = {"value": a["key"]["name"], "confidence": round(float(min(1.0, a["key"]["margin"] * 5)), 2)}
    except Exception as e:  # noqa: BLE001 - tempo/key are best-effort and must not fail the tagging
        out["bpm"] = out["key"] = None
        out["analysis_error"] = str(e)[:200]
    progress(stage="tag", fraction=1.0)
    return out


def vocal_presence(stem_meta):
    """Judge vocal presence from the separated vocal stem: its loudness and share of the total energy."""
    return bool(stem_meta["vocals_rms_db"] > -40 and stem_meta["vocal_energy_share"] > 0.02)


def process(full_path, out_dir, progress=lambda **k: None):
    """The whole post-recording pipeline. Returns a dict with stems/tags/transcript/vocals (written into meta.json by the caller)."""
    t0 = time.time()
    stems = separate(full_path, out_dir, progress)
    has_vocals = vocal_presence(stems)
    tags = tag(full_path, os.path.join(out_dir, "instrumental.wav"), progress)
    transcript = None
    if has_vocals:
        transcript = transcribe_file(os.path.join(out_dir, "vocals.wav"), progress=progress)
    for v, f in (("full", os.path.basename(full_path)), ("instrumental", "instrumental.wav"), ("voice", "vocals.wav")):
        write_peaks(os.path.join(out_dir, f), os.path.join(out_dir, f"peaks.{v}.json"))
    progress(stage="done", fraction=1.0)
    return {"stems": stems, "tags": tags, "vocals": {"present": has_vocals, "language": transcript["language"] if transcript else None,
                                                    "language_probability": transcript["language_probability"] if transcript else None},
            "final_transcript": transcript, "processing_seconds": round(time.time() - t0, 1)}
