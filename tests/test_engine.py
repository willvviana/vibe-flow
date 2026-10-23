import json
import os
import sys
import time

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import audio, hardware, library, models, prompting, registry  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_library(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEFLOW_LIBRARY", str(tmp_path / "lib"))


def tone(path, seconds=3, sr=32000, channels=1, amp=0.5, ext=None):
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    sf.write(str(path), x, sr)
    return str(path)


# ---- prompt / settings validation -------------------------------------------------
def test_validate_ok_and_defaults():
    c = prompting.validate({"prompt": "calm piano", "duration": 8})
    assert c["mode"] == "text" and c["format"] == "wav" and c["seed"] is None


@pytest.mark.parametrize("bad", [
    {}, {"prompt": "x" * 501}, {"prompt": "a", "duration": 1}, {"prompt": "a", "duration": 999},
    {"prompt": "a", "bpm": 10}, {"prompt": "a", "key": "H minor"}, {"prompt": "a", "format": "ogg"},
    {"prompt": "a", "mode": "continue"}, {"prompt": "a", "lyrics": "la la"}, {"prompt": "a", "vocals": True},
    {"prompt": "a", "temperature": 9},
])
def test_validate_rejects(bad):
    with pytest.raises(prompting.ValidationError):
        prompting.validate(bad)


def test_prompt_template_includes_controls():
    c = prompting.validate({"prompt": "night drive", "genres": ["synthwave"], "instruments": ["synthesizer", "drums"],
                            "moods": ["dreamy"], "bpm": 100, "key": "A minor", "time_signature": "4/4"})
    p = prompting.build_prompt(c, "musicgen-small")
    for part in ["night drive", "synthwave", "dreamy", "synthesizer", "drums", "100 bpm", "A minor", "4/4", "instrumental"]:
        assert part in p


def test_control_report_is_honest():
    c = prompting.validate({"prompt": "x", "genres": ["jazz"], "bpm": 90})
    r = prompting.control_report(c)
    assert "genre" in r["approximate"] and "tempo (BPM)" in r["approximate"]
    assert "genre" not in " ".join(r["enforced"])


# ---- registry / capabilities / devices --------------------------------------------
def test_capabilities():
    assert registry.pick_model("text").id == "musicgen-small"
    assert registry.pick_model("melody").id == "musicgen-melody"
    with pytest.raises(ValueError):
        registry.pick_model("melody", "musicgen-small")
    assert not any(m.vocals or m.lyrics for m in registry.MODELS.values())


def test_cpu_fallback_when_no_cuda_or_too_little_vram():
    spec = registry.get("musicgen-small")
    assert hardware.choose_device({"cuda": False}, spec) == "cpu"
    assert hardware.choose_device({"cuda": True, "vram_gb": 2}, spec) == "cpu"
    assert hardware.choose_device({"cuda": True, "vram_gb": 8}, spec) == "cuda"


def test_memory_preflight_uses_measured_peak():
    from engine import memory
    spec = registry.get("musicgen-small")            # measured peak 4.4 GB
    assert memory.check(spec, 12)[0] == "ok"
    assert memory.check(spec, 5)[0] == "warn"        # fits, but < peak + 2 GB headroom
    level, msg = memory.check(spec, 2)
    assert level == "refuse" and "Close other applications" in msg and "measured" in msg
    assert memory.check(spec, 0.1, already_loaded=True)[0] == "ok"   # warm worker: nothing new to allocate
    assert memory.check(registry.get("musicgen-melody"), 5)[0] == "refuse"   # estimated need, padded


def test_dependency_and_model_status():
    hw = hardware.detect()
    assert hw["torch"] and hw["device"] in ("cpu", "cuda")
    st = models.status()
    assert {m["id"] for m in st} == set(registry.MODELS)


# ---- audio validation / preprocessing ---------------------------------------------
def test_reference_validation(tmp_path):
    good = tone(tmp_path / "a.wav", 3, 44100, 2)
    info = audio.validate_reference(good)
    assert info["channels"] == 2 and info["sample_rate"] == 44100
    bad = tmp_path / "b.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(audio.AudioError):
        audio.validate_reference(str(bad))
    txt = tmp_path / "c.txt"
    txt.write_text("x")
    with pytest.raises(audio.AudioError):
        audio.validate_reference(str(txt))
    with pytest.raises(audio.AudioError):
        audio.validate_reference(str(tone(tmp_path / "short.wav", 0.3)))
    with pytest.raises(audio.AudioError):
        audio.validate_reference(str(tmp_path / "missing.wav"))


def test_load_mono_resamples_and_does_not_clip(tmp_path):
    p = tone(tmp_path / "s.wav", 2, 44100, 2, amp=0.9)
    x = audio.load_mono(p, 32000)
    assert x.ndim == 1 and abs(len(x) - 64000) < 5 and np.max(np.abs(x)) <= 1.0
    sf.write(str(tmp_path / "z.wav"), np.zeros(32000 * 2, dtype=np.float32), 32000)
    with pytest.raises(audio.AudioError):
        audio.load_mono(str(tmp_path / "z.wav"), 32000)


def test_analysis_and_helpers():
    x = np.array([0, 1.0, -1.0, 0.1], dtype=np.float32)
    assert audio.analyze(x)["clipped_samples"] == 2
    a = np.ones(1000, dtype=np.float32); b = np.ones(1000, dtype=np.float32)
    assert len(audio.crossfade_concat(a, b, 1000, 0.1)) == 1900
    y = audio.normalize_peak(np.array([0.1, -0.2], dtype=np.float32))
    assert abs(np.max(np.abs(y)) - 10 ** (-1 / 20)) < 1e-4
    src = np.concatenate([np.zeros(500), np.ones(500) * 0.5, np.zeros(500)]).astype(np.float32)
    before = src.copy()
    assert len(audio.trim_silence(src, 1000)) < len(src)
    assert np.array_equal(src, before)          # input untouched


# ---- output integrity, library, metadata ------------------------------------------
def test_write_verify_and_no_overwrite(tmp_path):
    x = (0.3 * np.sin(np.arange(32000 * 2) / 20)).astype(np.float32)
    p = str(tmp_path / "o.wav")
    audio.write_audio(p, x, 32000, "wav")
    assert audio.verify_file(p, 2.0)["duration"] == 2.0
    with pytest.raises(FileExistsError):
        audio.write_audio(p, x, 32000, "wav")
    with pytest.raises(audio.AudioError):
        audio.verify_file(p, 10.0)               # wrong duration
    sf.write(str(tmp_path / "s.wav"), np.zeros(64000, dtype=np.float32), 32000)
    with pytest.raises(audio.AudioError):
        audio.verify_file(str(tmp_path / "s.wav"), 2.0)  # silent


def test_library_roundtrip_export_rename_delete():
    tid = library.new_id()
    _, wp = library.paths(tid)
    x = (0.3 * np.sin(np.arange(32000 * 2) / 20)).astype(np.float32)
    audio.write_audio(wp, x, 32000, "wav")
    library.save_meta(tid, {"id": tid, "title": "t", "settings": {"prompt": "p"}})
    with pytest.raises(FileExistsError):
        library.save_meta(tid, {"id": tid})
    assert library.get(tid)["settings"]["prompt"] == "p"
    assert library.rename(tid, "New name")["title"] == "New name"
    for fmt in ("flac", "mp3"):
        f = library.export(tid, fmt)
        assert os.path.getsize(f) > 0 and sf.info(f).frames > 0
    orig_bytes = open(wp, "rb").read()
    library.export(tid, "wav", normalize=True)
    assert open(wp, "rb").read() == orig_bytes   # original preserved
    assert [t["id"] for t in library.list_tracks()] == [tid]
    library.delete(tid)
    assert library.list_tracks() == [] and not os.path.exists(wp)


def test_library_rejects_path_traversal():
    for bad in ("../x", "a/b", "", "..\\x"):
        with pytest.raises(ValueError):
            library.paths(bad)


def test_failed_generation_without_model(monkeypatch):
    from engine import generate
    monkeypatch.setattr(models, "local_path", lambda spec: None)
    with pytest.raises(RuntimeError, match="not installed"):
        generate.run({"settings": {"prompt": "x", "duration": 2}})
    assert library.list_tracks() == []


# ---- real inference smoke test ----------------------------------------------------
def _installed(mid):
    return any(m["id"] == mid and m["installed"] for m in models.status())


@pytest.mark.skipif(not _installed("musicgen-small"), reason="musicgen-small weights not installed")
def test_smoke_real_generation_and_continuation(tmp_path):
    from engine import generate
    events = []
    meta = generate.run({"settings": {"prompt": "gentle piano melody", "genres": ["ambient"], "duration": 4,
                                      "seed": 7, "format": "flac"}}, lambda **k: events.append(k))
    assert meta["status"] == "complete" and abs(meta["audio"]["duration"] - 4) < 0.6
    assert os.path.getsize(library.paths(meta["id"])[1]) > 0
    assert meta["timing"]["real_time_factor"] > 0 and meta["timing"]["peak_memory_gb"] > 0
    assert any(e["stage"] == "generate" for e in events) and events[-1]["stage"] == "done"
    print("\nTIMING", json.dumps(meta["timing"]))

    ref = tone(tmp_path / "ref.wav", 3, 44100, 2, amp=0.3)
    cont = generate.run({"settings": {"prompt": "continue with soft strings", "duration": 3, "seed": 1,
                                      "mode": "continue", "reference": ref}})
    assert cont["reference"]["workflow"] == "continue" and cont["audio"]["duration"] > 3
    print("CONTINUE", json.dumps(cont["timing"]))


@pytest.mark.skipif(not _installed("musicgen-small"), reason="musicgen-small weights not installed")
def test_smoke_fast_mode_int8_is_valid_and_valid_audio():
    """Opt-in int8 path must still produce finite, non-silent audio of the right length and be flagged in metadata."""
    from engine import generate
    s = {"prompt": "calm piano", "duration": 3, "seed": 4}
    base = generate.run({"settings": s})
    fast = generate.run({"settings": {**s, "fast": True}})
    assert fast["fast_mode"] is True and base["fast_mode"] is False
    assert abs(fast["audio"]["duration"] - 3) < 0.5 and not fast["audio"]["analysis"]["silent"]
    # sanity only: loudness in a plausible range. int8 changes the sampled music, so outputs are not comparable
    # sample-by-sample and this is NOT a listening test.
    assert -50 < fast["audio"]["analysis"]["rms_dbfs"] < -3
    print("\nFAST", fast["timing"]["generate_s"], "vs", base["timing"]["generate_s"])
