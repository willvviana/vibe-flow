import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import audio, cli, deps, library, models, registry  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_library(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEFLOW_LIBRARY", str(tmp_path / "lib"))


def test_atomic_write_leaves_no_partials(tmp_path, monkeypatch):
    x = (0.2 * np.sin(np.arange(16000) / 9)).astype(np.float32)
    p = str(tmp_path / "a.wav")
    audio.write_audio(p, x, 16000, "wav")
    assert [f for f in os.listdir(tmp_path) if ".part" in f] == []
    import soundfile as sf
    monkeypatch.setattr(sf, "write", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        audio.write_audio(str(tmp_path / "b.wav"), x, 16000, "wav")
    assert [f for f in os.listdir(tmp_path) if "b.wav" in f] == []        # failed write leaves nothing behind


def test_cleanup_removes_only_orphans_and_partials():
    d = library.library_dir()
    keep = library.new_id()
    for n in (keep + ".json", keep + ".wav", keep + ".flac", "orphan.wav", "x.wav.part123", "meta.json.tmp"):
        open(os.path.join(d, n), "wb").write(b"x")
    assert sorted(library.cleanup(0)) == ["meta.json.tmp", "orphan.wav", "x.wav.part123"]
    assert sorted(os.listdir(d)) == sorted([keep + ".json", keep + ".wav", keep + ".flac"])


def test_cleanup_spares_recent_files():
    d = library.library_dir()
    open(os.path.join(d, "fresh.wav.part1"), "wb").write(b"x")
    assert library.cleanup(60) == []


def test_error_messages_do_not_leak_paths():
    msg = cli.scrub(f"Cannot open {cli._ROOT}\\library\\x.wav and C:\\Users\\someone\\secret\\file.txt")
    assert cli._ROOT not in msg and "someone" not in msg and "secret" not in msg


def test_dependency_check_reports_ok_and_problems(monkeypatch, tmp_path):
    assert deps.check()["ok"]
    req = tmp_path / "r.txt"
    req.write_text("numpy==0.0.1\nnot-a-real-package==1.0\n")
    monkeypatch.setattr(deps, "REQ", str(req))
    r = deps.check()
    assert not r["ok"] and len(r["problems"]) == 2 and "npm run setup" in r["problems"][0]


def test_model_integrity_states(tmp_path, monkeypatch):
    """missing -> unverified -> ok -> damaged (truncated file) using a fake snapshot."""
    spec = registry.get("musicgen-small")
    snap = tmp_path / "snap"
    snap.mkdir()
    w = snap / "model.safetensors"
    w.write_bytes(b"0" * 1000)
    monkeypatch.setattr(models, "CACHE", str(tmp_path))
    monkeypatch.setattr(models, "local_path", lambda s: str(snap))
    assert models.state(spec)[0] == "unverified"
    json.dump(models._fingerprint(str(snap), ["model.safetensors"]), open(models._marker(spec), "w"))
    assert models.state(spec)[0] == "ok"
    w.write_bytes(b"0" * 500)                                    # simulate an interrupted/truncated download
    assert models.state(spec)[0] == "damaged"
    assert next(m for m in models.status() if m["id"] == spec.id)["installed"] is False


def test_corrupt_weights_give_actionable_error(tmp_path, monkeypatch):
    from engine import generate
    bad = tmp_path / "snap"
    bad.mkdir()
    (bad / "config.json").write_text("{}")
    (bad / "model.safetensors").write_bytes(b"garbage")
    monkeypatch.setattr(models, "local_path", lambda s: str(bad))
    with pytest.raises(RuntimeError, match="Repair"):
        generate._load(registry.get("musicgen-small"), "cpu", lambda **k: None)


def test_metadata_never_stores_server_paths(tmp_path, monkeypatch):
    from engine import generate
    monkeypatch.setattr(generate, "_load", lambda *a, **k: (None, None))
    def fake(*a, ref=None, secs=None, **k):    # like MusicGen: continuation output = [prompt | new audio]
        new = (0.3 * np.sin(np.arange(32000 * 3) / 13)).astype(np.float32)
        return new if ref is None else np.concatenate([ref, new])
    monkeypatch.setattr(generate, "_infer", fake)
    import soundfile as sf
    ref = tmp_path / "my ref.wav"
    sf.write(str(ref), (0.3 * np.sin(np.arange(32000 * 3) / 13)).astype(np.float32), 32000)
    meta = generate.run({"settings": {"prompt": "x", "duration": 3, "mode": "continue", "reference": str(ref)}})
    assert str(tmp_path) not in json.dumps(meta)
    assert meta["original_settings"]["reference"] == "my ref.wav"


def test_overshoot_is_scaled_not_clipped(monkeypatch):
    from engine import generate
    monkeypatch.setattr(generate, "_load", lambda *a, **k: (None, None))
    monkeypatch.setattr(generate, "_infer", lambda *a, **k: (1.4 * np.sin(np.arange(32000 * 3) / 13)).astype(np.float32))
    meta = generate.run({"settings": {"prompt": "x", "duration": 3}})
    assert meta["audio"]["analysis"]["clipped_samples"] == 0
    assert any("Level reduced" in w for w in meta["warnings"])


def test_long_form_chaining_is_time_aligned(monkeypatch):
    """Fake model: output = [prompt | continuation of a ramp]. A time-aligned join must be gapless and keep exact length."""
    from engine import generate
    sr = 32000
    state = {"t": 0}

    def fake_infer(proc, model, device, prompt, secs, sr_, c, emit, b, s, ref=None, label=""):
        tone = lambda a, n: (0.4 * np.sin(2 * np.pi * 220 * (np.arange(n) + a) / sr)).astype(np.float32)  # noqa: E731
        if ref is None:
            return tone(0, int(secs * sr))
        # continues the same sine phase-continuously after the prompt, prompt is returned slightly different
        return np.concatenate([ref * 0.98, tone(len(ref) + state["t"], int(secs * sr))])
    monkeypatch.setattr(generate, "_load", lambda *a, **k: (None, None))
    monkeypatch.setattr(generate, "_infer", fake_infer)
    meta = generate.run({"settings": {"prompt": "x", "duration": 55, "seed": 1}})
    assert abs(meta["audio"]["duration"] - 55) < 0.2
    assert len(meta["audio"]["seams_s"]) == 2 and meta["audio"]["seams_s"][0] > 29
    assert meta["audio"]["analysis"]["clipped_samples"] == 0


def test_fast_flag_is_validated_and_recorded():
    from engine import prompting
    assert prompting.validate({"prompt": "x", "fast": True})["fast"] is True
    assert prompting.validate({"prompt": "x"})["fast"] is False


def test_prompt_style_and_sampling_validation():
    from engine import prompting
    c = prompting.validate({"prompt": "x", "genres": ["pop"], "instruments": ["acoustic guitar", "bass"], "prompt_style": "sentence"})
    text = prompting.build_prompt(c, "musicgen-small", c["prompt_style"])
    assert "played together by acoustic guitar and bass" in text and "keywords" not in text
    assert prompting.validate({"prompt": "x"})["top_k"] == 250
    for bad in ({"prompt_style": "poem"}, {"top_k": 5}, {"top_k": "many"}):
        with pytest.raises(prompting.ValidationError):
            prompting.validate({"prompt": "x", **bad})


def test_engine_interface_is_honest_about_vocals():
    from engine import engines
    caps = engines.get().capabilities()
    assert caps["vocals"] is False and caps["lyrics"] is False
    assert set(engines.ENGINES) == {"musicgen"}          # ACE-Step failed evaluation: must not be registered
    with pytest.raises(ValueError):
        engines.get("ace-step")


def test_default_profile_is_d_and_baseline_is_a_one_step_revert(monkeypatch):
    """Pins the shipped defaults. D won the blind test; 'baseline' must keep reproducing the ORIGINAL behaviour exactly."""
    from engine import prompting, profiles
    monkeypatch.delenv("VIBEFLOW_PROFILE", raising=False)
    assert profiles.DEFAULT_PROFILE == "d"
    c = prompting.validate({"prompt": "x"})
    assert (c["profile"], c["prompt_style"], c["temperature"], c["top_k"], c["guidance"]) == ("d", "sentence", 0.8, 250, 5.0)
    monkeypatch.setenv("VIBEFLOW_PROFILE", "baseline")                       # revert switch #1: environment
    c = prompting.validate({"prompt": "x"})
    assert (c["profile"], c["prompt_style"], c["temperature"], c["top_k"], c["guidance"]) == ("baseline", "keywords", 1.0, 250, 3.0)
    monkeypatch.delenv("VIBEFLOW_PROFILE")
    assert prompting.validate({"prompt": "x", "profile": "baseline"})["prompt_style"] == "keywords"   # per request


def test_explicit_settings_win_and_bad_profile_falls_back(monkeypatch):
    from engine import prompting
    monkeypatch.delenv("VIBEFLOW_PROFILE", raising=False)
    assert prompting.validate({"prompt": "x", "temperature": 0.9})["temperature"] == 0.9
    monkeypatch.setenv("VIBEFLOW_PROFILE", "nonsense")
    assert prompting.validate({"prompt": "x"})["profile"] == "d"             # unknown env value -> shipped default
    with pytest.raises(prompting.ValidationError):
        prompting.validate({"prompt": "x", "profile": "nope"})


def test_score_ab_unblinds_and_applies_sign_test(tmp_path):
    """The decision tool must not call a trend from one lucky seed."""
    import subprocess
    key = {"key": {f"{i:02d}.wav": {"config": "A" if i % 2 else "D", "seed": (i + 1) // 2} for i in range(1, 9)}}
    (tmp_path / "key.json").write_text(json.dumps(key))
    rows = ["clip,score,convincing,notes"]
    for i in range(1, 9):
        cfg = key["key"][f"{i:02d}.wav"]["config"]
        seed = key["key"][f"{i:02d}.wav"]["seed"]
        score = 4 if (cfg == "D" and seed == 1) else 3          # D better on ONE seed only, ties elsewhere
        rows.append(f"{i:02d}.wav,{score},{'y' if score == 4 else 'n'},")
    (tmp_path / "r.csv").write_text("\n".join(rows))
    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "score_ab.py"),
                          str(tmp_path / "r.csv"), str(tmp_path / "key.json")], capture_output=True, text=True).stdout
    assert "paired wins A 0 / D 1 / ties 3" in out and "p = 1.000" in out


def _xgenre_fixture(tmp_path, diffs_by_genre):
    """Build a key + ratings where, per genre, D-A differences are given for seeds 11,12,13 (A fixed at 3)."""
    key, rows, n = {}, ["clip,score,ship"], 0
    for genre, diffs in diffs_by_genre.items():
        for seed, d in zip((11, 12, 13), diffs):
            for label, score in (("A", 3), ("D", 3 + d)):
                n += 1
                name = f"g{n:02d}.wav"
                key[name] = {"genre": genre, "seed": seed, "profile_label": label}
                rows.append(f"{name},{score},{'y' if score >= 4 else 'n'}")
    (tmp_path / "key.json").write_text(json.dumps({"key": key}))
    (tmp_path / "r.csv").write_text("\n".join(rows))
    return str(tmp_path / "r.csv"), str(tmp_path / "key.json")


def test_score_xgenre_rule_keep_revert_inconclusive(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("score_xgenre", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "score_xgenre.py"))
    sx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sx)
    g6 = ["folk", "rock", "synthwave", "hiphop_rnb", "jazz", "cinematic"]

    def verdict(diffs):
        p, k = _xgenre_fixture(tmp_path, dict(zip(g6, diffs)))
        pairs, _ = sx.load(p, k)
        return sx.analyse(pairs, n_boot=2000)

    keep = verdict([[1, 1, 1]] * 6)                                  # D better everywhere
    assert keep["verdict"] == "KEEP D" and keep["overall"] == 1.0 and keep["wins"] == 18
    assert abs(keep["p_genre"] - 2 / 64) < 1e-9                       # smallest possible genre-level p with 6 clusters
    assert verdict([[-1, -1, -1]] * 6)["verdict"] == "REVERT TO A"
    assert verdict([[0, 0, 0]] * 6)["verdict"] == "REVERT TO A"       # no improvement is not a reason to keep D
    split = verdict([[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [0, 0, 0], [0, 0, 0]])
    assert split["verdict"] == "GENRE-SPECIFIC"
    mild = verdict([[1, 0, 0]] * 6)                                   # small, noisy improvement
    assert mild["verdict"] == "INCONCLUSIVE"


def test_arrangement_prompt_preserves_every_selection_and_reports_adaptations():
    """Every selected genre, mood and instrument must appear in the conditioning text; adaptations are reported, never silent."""
    import itertools
    import random
    from engine import arrangement, prompting
    rng = random.Random(5)
    for _ in range(200):
        s = {"prompt": "", "duration": 15, "genres": rng.sample(list(arrangement.GENRE), rng.randint(1, 3)),
             "moods": rng.sample(list(arrangement.MOOD), rng.randint(1, 3)), "instruments": rng.sample(list(arrangement.ROLES), rng.randint(1, 6)),
             "style": None}
        s.pop("style")
        c = prompting.validate({**s, "prompt_style": "arrangement", "bpm": 100})
        d = arrangement.describe(c)
        low = d["text"].lower()
        for item in c["genres"] + c["moods"] + c["instruments"]:
            assert item in low, f"{item!r} missing from: {d['text']}"
        assert c["genres"][0] in low.split("influences")[0] or len(c["genres"]) == 1        # primary genre named first
        if len(c["instruments"]) > arrangement.MAX_CLEAR_INSTRUMENTS:
            assert d["notes"]                                                               # over-full requests are flagged
    c = prompting.validate({"prompt": "x", "genres": ["classical"], "instruments": ["drums", "piano"], "moods": ["calm", "energetic"], "prompt_style": "arrangement"})
    d = arrangement.describe(c)
    assert any("drums adapted for classical" in a for a in d["adaptations"]) and "timpani" in d["text"]
    assert any("opposite directions" in a for a in d["adaptations"])


def test_arrangement_is_opt_in_and_intensity_control_is_gone(monkeypatch):
    from engine import prompting
    monkeypatch.delenv("VIBEFLOW_PROFILE", raising=False)
    assert prompting.validate({"prompt": "x"})["prompt_style"] == "sentence"      # shipped default is untouched
    assert prompting.validate({"prompt": "x", "prompt_style": "arrangement"})["prompt_style"] == "arrangement"
    # MusicGen has no mood-weight conditioning, so a mood "intensity" setting must not exist (it would be a placebo)
    assert "intensity" not in prompting.validate({"prompt": "x", "intensity": 5})


def _layered(**kw):
    from engine import prompting
    return prompting.validate({"prompt": "", "duration": 15, "profile": "layered", **kw})


def test_layered_priority_order_genre_instruments_tempo_then_mood():
    """Mood must come after genre, instruments and tempo, and must not lead the text (the 'sad metal' defect of the sentence style)."""
    from engine import layered, prompting
    c = _layered(genres=["metal"], moods=["sad", "dark"], instruments=["electric guitar", "drums", "bass"], bpm=170)
    t = layered.describe(c)["text"]
    low = t.lower()
    assert low.startswith("a metal instrumental piece")
    first_mood = min(low.index(m) for m in ("sad", "dark"))
    assert low.index("electric guitar") < low.index("170 bpm") < first_mood
    # the shipped sentence style, for contrast, opens with the mood
    d = prompting.build_prompt(prompting.validate({"prompt": "", "genres": ["metal"], "moods": ["sad"], "instruments": ["drums"], "profile": "d"}), "m", "sentence")
    assert d.lower().startswith("a sad metal")


def test_layered_preserves_selections_adds_no_instruments_and_keeps_mood_small():
    import random
    from engine import layered
    rng = random.Random(9)
    nouns = {"guitar", "piano", "violin", "cello", "saxophone", "trumpet", "flute", "organ", "harp", "synth", "synthesizer", "strings", "brass", "percussion",
             "drums", "bass", "orchestra", "timpani"}
    for _ in range(300):
        c = _layered(genres=rng.sample(list(layered.GENRE), rng.randint(1, 3)), moods=rng.sample(list(layered.MOOD), rng.randint(1, 3)),
                     instruments=rng.sample(["acoustic guitar", "electric guitar", "bass", "drums", "piano", "saxophone", "flute", "organ"], rng.randint(1, 4)), bpm=100)
        d = layered.describe(c)
        low = d["text"].lower()
        for item in c["genres"] + c["moods"] + c["instruments"]:
            assert item in low, (item, d["text"])
        selected_words = {w for i in c["instruments"] for w in i.split()} | {"timpani"} if "classical" in c["genres"][:1] else {w for i in c["instruments"] for w in i.split()}
        import re
        for w in set(re.findall(r"[a-z]+", low)) & nouns:
            assert w in selected_words or w in ("drums", "bass") and w in " ".join(c["instruments"]), f"unselected instrument word {w!r} in: {d['text']}"
        assert len(d["layers"]["mood"].split()) <= layered.MAX_MOOD_WORDS
        mood_text = d["layers"]["mood"].lower()
        for banned in ("slow", "faster", "upbeat", "major key", "minor key", "tempo of"):
            assert banned not in mood_text, (banned, mood_text)


def test_layered_energy_clauses_protect_the_genre():
    from engine import layered
    assert "full energy and drive of metal" in layered.mood_layer(["sad"], "metal")        # high-energy genre + low-energy mood
    assert "restrained character of ambient" in layered.mood_layer(["energetic"], "ambient")  # low-energy genre + high-energy mood
    assert "groove, tempo and instruments" in layered.mood_layer(["happy"], "rock")


def test_production_setting_validated_and_layer_is_last():
    import pytest
    from engine import layered, prompting
    with pytest.raises(prompting.ValidationError):
        _layered(genres=["pop"], production="shiny")
    c = _layered(genres=["pop"], moods=["calm"], instruments=["piano"], production="warm")
    t = layered.describe(c)["text"]
    assert t.rstrip().endswith("a warm, rounded mix.") and t.index("Emotional tone") < t.index("Production")


def test_layered_is_not_the_default_profile(monkeypatch):
    from engine import prompting
    monkeypatch.delenv("VIBEFLOW_PROFILE", raising=False)
    assert prompting.validate({"prompt": "x"})["profile"] == "d"
    assert prompting.validate({"prompt": "x", "profile": "layered"})["prompt_style"] == "layered"


def test_score_mood_rule(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("score_mood", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "score_mood.py"))
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)

    def run(style_delta, mood_delta):
        key, rows, n = {}, ["clip,style_kept,mood_felt"], 0
        for base in ("metal", "disco_funk", "ambient", "rock"):
            for mood in ("sad", "calm"):
                for seed in (31, 32):
                    for label, (s, m) in (("current", (3, 4)), ("layered", (3 + style_delta, 4 + mood_delta))):
                        n += 1
                        key[f"z{n:02d}.wav"] = {"genre": base, "mood": mood, "seed": seed, "profile_label": label}
                        rows.append(f"z{n:02d},{s},{m}")
        (tmp_path / "k.json").write_text(json.dumps({"key": key}))
        (tmp_path / "r.csv").write_text("\n".join(rows))
        return sm.analyse(sm.load(str(tmp_path / "r.csv"), str(tmp_path / "k.json")), n_boot=1000)

    assert run(1, 0)["verdict"] == "ADOPT layered" and run(1, 0)["n_pairs"] == 16
    assert run(1, -1)["verdict"] == "REJECT layered"            # protects style but hides the mood
    assert run(0, 1)["verdict"] == "REJECT layered"             # no style protection
    assert run(1, -0.75)["verdict"] == "INCONCLUSIVE"           # mood fell between -0.5 and -1.0
