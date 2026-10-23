# Decisions

Short record of the choices that shaped Vibe Flow, why they were made, and what they cost. Newest areas last. Evidence lives in `docs/EVALUATION.md`, `docs/EDITOR.md` and `docs/SAMPLE.md`.

## Product and engine
| # | Decision | Why | Cost / revisit when |
|---|---|---|---|
| 1 | Site renamed **Vibe Flow**, canonical domain `www.vibeflow.app` | Requested | Update `canonical`, sitemap and OG tags if the domain changes |
| 2 | Generate music **locally** (MusicGen Small), no hosted API | No paid API, no cloud inference; hosted/third-party options were tried first and replaced | CPU only: RTF 3-11, minutes for long tracks. MusicGen weights are **CC-BY-NC 4.0** (non-commercial) |
| 3 | AMD RX 5700 XT is **not used** | ROCm on Windows lacks gfx1010; DirectML failed (`tools/experiments/dml_test.py`) | Revisit with a supported GPU or backend |
| 4 | Each job in its own Python process, then a **warm worker** with idle exit | Clean memory release and clean cancel; warm worker saves 6-20 s per job | Holds ~4 GB RAM while idle, so it quits after 300 s. Memory preflight refuses jobs when free RAM is too low |
| 5 | Models download only on explicit user action, SHA-256 verified | No implicit multi-GB downloads | Extra click on first run |
| 6 | Server binds to 127.0.0.1, Host/Origin guard, path scrubbing, whitelist of static files (`PUBLIC`) | Local server must not be reachable or driven by other web pages | Every new static file must be added to `PUBLIC` |
| 7 | Default MusicGen profile **D** (sentence prompt, T 0.8, guidance 5) | Won a blind 6-seed folk test | Cross-genre test (6 genres x 3 seeds) was **inconclusive**; D is unproven outside folk. `VIBEFLOW_PROFILE=baseline` reverts |
| 8 | **Vocals are not generated.** ACE-Step 1.5 stays experimental, not integrated | It sang the lyrics but quality was mixed (warble, one convincing seed) and it needs ~11.5-12.6 GB RAM | Integrate only after a listening pass that meets the criteria, with a RAM preflight |
| 9 | "Layered" mood prompt is experimental, off by default | Mood was overriding the music; the fix is built but **unverified by listening** | The 48-clip mood test waits for the user's go |
| 10 | **Adapt a sample** with 0-100 % adherence (bars kept as the untouched original + tempo/key hints + alignment + audio conditioning) | Gives one understandable control over a mechanism MusicGen cannot weight directly | Mechanically measured only; listening pack still to come |

## Editor (`/editor`)
| # | Decision | Why | Cost / revisit when |
|---|---|---|---|
| 11 | Non-destructive project model (clip references + instructions, audio stored once in IndexedDB) | Trim, split, undo and reopen never alter the source | Heavy edits (speed, pitch, reverse, noise reduction) are rendered in a Web Worker and cached |
| 12 | **One Web Audio graph builder** for live playback and offline export | What you hear is what is exported | Live limiter is compressor-based (compensated), the export uses an exact look-ahead limiter, so peaks can differ slightly |
| 13 | Phase vocoder with identity phase locking for time-stretch/pitch | Fixed a 2 dB level loss; level ratio now 1.000 at 0.5-2x | Not formant-preserving; transients smear at extreme rates; **never judged by ear** |
| 14 | MP3 via vendored lamejs (LGPL-3.0); FLAC via the local server (libsndfile) | No native browser encoders for these | Licence note in `docs/EDITOR.md` |
| 15 | **Not implemented:** pitch correction, true vocal/instrument isolation inside the Editor | No model/approach shipped for them; "voice/music only" exports use track roles | Isolation of mixed audio is done on the Sample page (Demucs) |
| 16 | After a split only the right piece stays selected; clicking inside a multi-selection narrows it | Users could not move one half on its own | |
| 17 | Header spans the full width on the Editor page; wraps instead of scrolling | 1080p layout and no overlap of controls | |

## Sample page (`/sample`)
| # | Decision | Why | Cost / revisit when |
|---|---|---|---|
| 18 | Stems with **Demucs htdemucs**, blockwise (40 s, 3 s cross-fade) | Proven separator, bounded memory, progress and ETA on CPU | ~0.45x real time here; residue/bleed remains (reported as a residual in dB) |
| 19 | Transcription with **faster-whisper small** (int8, CPU); final pass on the voice stem | Local, multilingual, auto language detection | No speaker labels (diarization models are gated); live text lags 10-20 s on this CPU |
| 20 | Tags with **CLAP zero-shot** + the existing tempo/key estimators | Genre/instrument/mood without training data | Scores are relative rankings, not probabilities; fixed vocabulary; instrument tags are guesses |
| 21 | Capture lossless 32-bit float; originals written bit-exact | "Keep the original untouched" | Larger files |
| 22 | **System audio via WASAPI loopback in a server-side process** (Windows), not a browser screen share | The browser path needs a share dialog and can only deliver tab audio on most platforms | Windows only; default output device only; browser share and microphone remain as fallbacks |
| 23 | **Warm capture process**: device open, nothing read until Start | Start begins recording within milliseconds (34 ms click to Recording in the page) without capturing anything beforehand | One idle process while the page is open; it exits after 300 s |
| 24 | Recording survives page close: server finalises after 90 s without a heartbeat, and recovers interrupted captures on restart | No lost recordings | |
| 25 | Create receives Full / Instrumental / Voice as a mono 32 kHz 16-bit reference (<= 10 min) and fills genre, instruments, mood, BPM and key from the edited tags | Fits Create's existing reference limits | A voice reference guides rhythm, key and tempo only; the generator stays instrumental |
| 26 | Static files are served with `Cache-Control: no-cache` | Stale scripts were served after edits | One revalidation per file |

## Listening Pack (`/listening`)
| # | Decision | Why | Cost / revisit when |
|---|---|---|---|
| 27 | Named adherence levels **Low 15 / Medium 50 / High 75 / Strict 100 %** on top of the existing control mapping; slider kept | Each step changes a mechanism (audio conditioning from Medium, key hint from Medium, bars kept 0 / ~1/3 / ~2/3 / all) | Fixed points; revisit if blind ratings show two levels sound the same |
| 28 | Measure every result twice: the **mix** and the **added layers only** | Copied bars make the mix look close at high levels; the layers show what the model itself follows | Two numbers to read instead of one |
| 29 | Blind pack with a server-side key that is never served; ratings unblind it | Measures cannot hear; the listener must not know the level | One listener and few clips: a direction, not proof |
| 30 | **Lead / backing split by stereo position** (inter-channel similarity mask on the Demucs voice stem), no extra model | No lead/backing separation model is installed; panning is the standard production cue | Centre-panned backing stays in Lead; mono voices have no backing layer (measured on known mixes, docs/LISTENING.md) |
| 31 | Harmony measured as a chroma profile relative to the lead's pitch, on the side signal | A single-pitch tracker on a two-voice harmony reports the chord root (seen in testing) | Coarse: intervals, not voices |
| 32 | Generated tracks of a pack stay inside the pack folder (`userdata/listening/<id>/lib`) | Test runs must not fill "Your Songs" | Delete a pack to free its audio |

## Working rules adopted
- Do not claim audio qualities that were not confirmed by listening; state what was measured and what was not.
- Do not start listening packs or large generation runs until the user says go.
- Keep pinned dependencies in `requirements.txt` (CPU PyTorch 2.5.1, numpy 1.26.4); install heavy extras with `--no-deps` plus a constraints file so PyTorch is never upgraded.
- Test suites: `npm test` (pytest + Node suites with real models); the opt-in `VIBEFLOW_TEST_LOOPBACK=1` test plays sound through the speakers.
