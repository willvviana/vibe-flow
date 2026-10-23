# Vibe Flow

Marketing site + a **local AI music engine**. Music is generated on your machine with pretrained open-source models. No cloud inference, no paid API.

## Install and run (Windows; needs Node 20+ and [uv](https://docs.astral.sh/uv/))

```bash
npm run setup            # Python 3.11 venv + pinned dependencies (CPU PyTorch)
npm run install-model    # MusicGen Small, ~2.2 GB from Hugging Face, SHA-256 verified. Only runs when you run it
npm run install-sampling # Sample page models: Demucs htdemucs, Whisper small, CLAP (~1.1 GB). Only runs when you run it
npm start                # http://localhost:3000
npm test                 # pytest (units + real-model smoke) then node --test (editor, DSP, HTTP integration with real models)
npm run test:fast        # offline unit tests only, ~6 s
npm run verify-model     # re-hash the model weights against the published checksum
```

The UI shows "Install…" with size and licence and asks for confirmation; nothing is downloaded implicitly. A damaged or truncated model is detected (size check on every status call, checksum on `verify`) and can be fixed with **Repair** in the Models panel. Interrupted downloads resume.

Environment variables: `PORT`, `VIBEFLOW_LIBRARY` (output folder, default `./library`), `VIBEFLOW_MODELS` (model cache, default `./models`), `VIBEFLOW_SAMPLES` (Editor samples, default `./userdata/samples`), `VIBEFLOW_RECORDINGS` (Sample page recordings, default `./userdata/recordings`), `PYTHON`.

Tests that play sound through the speakers are opt-in: `VIBEFLOW_TEST_LOOPBACK=1 node --test tests/sample.test.js`.

## Architecture

| Piece | File |
|---|---|
| Model registry, capabilities, licences, rejected models | `engine/registry.py` |
| Hardware detection, device choice, CPU fallback, RAM/VRAM warnings | `engine/hardware.py` |
| Settings validation, model-specific prompt, "exact vs approximate" report | `engine/prompting.py` |
| Reference-audio validation, resample, analysis, crossfade, atomic export | `engine/audio.py` |
| Model install state, checksum verification, repair | `engine/models.py` |
| Pinned-dependency check | `engine/deps.py` |
| Inference pipeline with progress events | `engine/generate.py` |
| Library (WAV + JSON metadata, rename/delete/export, orphan cleanup) | `engine/library.py` |
| JSON-lines CLI used by the server | `engine/cli.py` |
| Sample page: Demucs stems, Whisper transcription, CLAP tags, waveform peaks | `engine/sampling.py`, `engine/sampling_models.py` |
| Sample page: live-transcription worker; system-audio (WASAPI loopback) capture | `engine/sample_worker.py`, `engine/loopback.py` |
| Sample page backend (capture, live transcript, processing queue, library) | `recordings.js` |
| Sample page UI | `sample.html`, `sample.css`, `sample/app.js`, `sample/recorder-worklet.js` |
| Editor (model, DSP, Web Audio engine, storage, UI) | `editor.html`, `editor.css`, `editor/` |
| HTTP API, job orchestration, cancel, uploads, security checks | `server.js` |
| Benchmarks and experiments | `tools/` |

Each job runs in its own Python process: memory is fully released afterwards and cancelling is a clean kill. One heavy job at a time (409 otherwise), 30 min hard timeout. Trade-off: the model reloads for every job (6–20 s); a persistent worker would remove that but holds ~3–4 GB RAM while idle (see next milestone).

## Feature status

| Feature | Status |
|---|---|
| MusicGen Small text-to-music, continuation, seed/duration/format, library, export | **Implemented and tested** (real-model tests) |
| Warm worker (model stays loaded between jobs, idle exit), memory preflight | **Implemented and tested** |
| int8 "fast mode" | Implemented, sound quality **not evaluated by listening** |
| MusicGen quality profile **D** (sentence prompt, T 0.8, guidance 5) is the **default** | Implemented; chosen on folk evidence (blind 6 seeds); a 6-genre x 3-seed test was INCONCLUSIVE (mean +0.11, D worse in 2 genres), so D is **unproven outside folk**. Revert: `VIBEFLOW_PROFILE=baseline` |
| Melody guide (MusicGen Melody) | Implemented, **never run here** |
| **Vocals / lyrics (ACE-Step 1.5 sft + LM)** | **Experimental, not integrated.** Objective check says it sings the lyrics; musical quality unverified; needs ~11.5 GB RAM. See docs/EVALUATION.md |
| Editor (multi-track editing, DSP, export WAV/FLAC/MP3, send to Create) | **Implemented; 88 browser checks, Node tests**; DSP not judged by ear. No pitch correction. See docs/EDITOR.md |
| Sample page (system audio capture, live transcript, Demucs stems, tags, library, use in Create) | **Implemented; real-model tests**; separation/tag quality not judged by ear. See docs/SAMPLE.md |
| GPU acceleration on AMD RX 5700 XT | **Unsupported** (no working backend) |

Evaluation of musical coherence, vocal candidates, root causes and measurements: **docs/EVALUATION.md**. Listening pack: **samples/**.

Multi-instrument / multi-genre / multi-mood: every selection is always written into the prompt (tested). The role-aware `arrangement` prompt is **experimental, API-only, off by default**; MusicGen has no per-instrument control and cannot guarantee any instrument is audible (see docs/EVALUATION.md, round 5).

Mood handling: the shipped default still writes the mood *before* the genre ("a sad metal ..."), which can let the mood override the music. An experimental fix, `profile: "layered"` (or `VIBEFLOW_PROFILE=layered`), orders the prompt genre > instruments > tempo > mood > production and keeps mood a short tone layer. Unverified by listening; not the default. No mood-intensity control exists because MusicGen cannot weight conditioning (docs/EVALUATION.md, round 6).

**Adapt a sample** (create page, tab "Adapt a sample"): upload a sample and choose Loose <-> Close (0-100 %). The percentage controls how many bars of the sample are kept as the untouched original, the tempo/key hints, alignment and audio conditioning of the generated layers (not a volume). Instrumental only; vocals are not available. Mechanically measured and tested; **musical quality not yet evaluated by listening** (docs/EVALUATION.md, round 7).

**Editor** (`/editor`): multi-track browser audio editor with real DSP, live preview and export to WAV/FLAC/MP3, plus hand-off to Create as a sample. See docs/EDITOR.md for what is implemented, tested and not supported.

**Sample** (`/sample`): records everything the computer plays (Windows system-output loopback, started within milliseconds of pressing Start; browser tab/screen share and microphone as fallbacks), with live Whisper transcription. After recording it separates the instrumental and the voice with Demucs, tags genre/instruments/mood/BPM/key with CLAP, keeps a persistent library (Full, Instrumental, Voice players with waveforms), and hands any version to Create, which fills in its genre, instrument, mood, BPM and key fields. Install its models with `npm run install-sampling`. No speaker identification; separation and tags are imperfect. See docs/SAMPLE.md for the measured results and limits.

Design choices and trade-offs are listed in **decisions.md**.

## What is supported (verified)

- **MusicGen Small**: text → instrumental music, and continuation of an uploaded clip (the last 10 s are used as audio prompt; the output is that tail followed by the new audio). Up to 60 s by chaining 20 s continuation steps.
- **Exact controls:** duration, seed (identical WAV bytes on 3 repeated runs here), output format. **Approximate (text only):** genre, instruments, mood, BPM, key, time signature. Each track's details list which were which.
- **Fast mode (int8):** opt-in dynamic quantisation, ≈1.5–2× faster, +~2 GB peak RAM, changes the output. Not validated by listening.
- WAV/FLAC/MP3 export. The original WAV is never modified; exports and normalised copies are derived. Decoder overshoot is scaled down (never hard-clipped) and reported.

## Not supported

- **Vocals / lyrics**: MusicGen cannot sing; such requests are rejected. See "Vocal model evaluation".
- **Melody guide** (`musicgen-melody`, 5.8 GB, ~10 GB RAM): implemented, **never run on this machine**.
- **GPU**: the AMD RX 5700 XT is not used. DirectML was tried (see below) and fails.
- Voice cloning: not implemented.

## Benchmarks (Ryzen 7 3700X 8C/16T, 16 GB, CPU; `tools/bench.py`, 5 s clip, model already cached)

| Config | Load | Gen run 1 | Gen run 2 (warm) | RTF warm | Avg cores busy | Peak RAM |
|---|---|---|---|---|---|---|
| **Before:** 16 threads (all logical) | 7.3 s | 24.7 s | 23.0 s | 4.65 | 15.1 | 4.3 GB |
| 12 threads | 7.2 s | 24.1 s | 22.9 s | 4.64 | 11.8 | 4.3 GB |
| **After (default):** 8 threads (physical cores) | 7.1 s | 30.1 s | 22.7 s | 4.60 | 7.9 | 4.3 GB |
| 8 threads + int8 (`fast`) | 12.1 s | 21.8 s | 14.7 s | 2.98 | 7.8 | 6.2 GB |

Findings: thread count doesn't change speed (memory-bound), so the default now uses physical cores and leaves the machine responsive. Int8 gives ≈1.5× warm speed-up (1.9× on a 3 s clip in the test suite). Longer track: 35 s took 276 s total (load 9 s, generate 264 s, RTF 7.6, peak 4.3 GB), slower per second than short clips because each continuation step re-processes a 10 s prompt. All measured on this machine at one point in time with background load, so treat differences under ~10 % as noise.

## Audio continuity (`tools/seams.py`)

MusicGen returns `[re-decoded prompt | new audio]` aligned at lag 0 (correlation ≈ 0.84, `tools/seam_diag.py`). The previous join blended the *end of the old audio with the start of the new*, which skipped 0.1 s of music per seam. The join now crossfades 0.25 s over the time-aligned prompt tail, so length is exact. Objective results on 3 seeds: click metric < 1.0 for hard concat, old and aligned joins (no audible-click-level discontinuity by this metric), loudness steps at the join of ±1–5 dB for every method (the same as normal musical variation), old join 0.1 s shorter than the aligned one. **Conclusion:** the crossfade is not the dominant quality factor; these metrics cannot judge musicality, repetition or whether the seam "sounds right", and I did not listen to the output.

## Vocal model evaluation

| Candidate | Licence | Fits this machine? |
|---|---|---|
| **ACE-Step 1.5** (2B DiT turbo, ~6 GB without LM) | Code MIT; weights listed MIT on the Hugging Face card (check training-data/commercial terms yourself) | **Ran on CPU**, DiT-only: init 123 s, 10 s of 48 kHz stereo in 44 s (RTF 4.4), **peak 12.6 GB RAM** on a 16 GB machine. Output file valid and non-silent. Whether it actually sings the lyrics well was **not verified**; listen to `samples/acestep-vocal-test.wav` |
| YuE | Apache-2.0 | Documented setups target large GPUs; not tried |
| DiffRhythm | Apache-2.0 | Needs ≥8 GB CUDA VRAM; not tried |
| AMD GPU path | — | ROCm on Windows doesn't list gfx1010 (RX 5700 XT). DirectML + MusicGen: fails (`The parameter is incorrect`, `tools/experiments/dml_test.py`) |

Recommendation: ACE-Step 1.5 is the only realistic local vocal candidate here. Next milestone: listen to the sample; if acceptable, integrate as a second engine with a RAM preflight (refuse or warn below ~13 GB free) and the LM left off.

## Security and reliability

Server binds to 127.0.0.1; API requires a localhost `Host` and rejects cross-origin `Origin` (CSRF/DNS-rebinding); track ids are validated (`[a-z0-9-]`), formats whitelisted; uploads: extension + decode check, 25 MB / 10 min limits, 24 h expiry, filename sanitised; error messages are scrubbed of filesystem paths; temp job files and partial audio are removed after failure/cancel (`engine.cli cleanup` at start-up and after each generation); exports are written atomically; one job at a time.

## Known limitations

- CPU generation is slow (RTF ≈ 3–11); long tracks take minutes.
- MusicGen weights are **CC-BY-NC 4.0**: non-commercial only.
- Quality of generated music (and of the int8 mode) has not been evaluated by listening.
- Windows-first: `npm run setup` uses the Windows venv path.
- No CI. User data (`userdata/`, `models/`, `library/`, `uploads/`) is git-ignored.
