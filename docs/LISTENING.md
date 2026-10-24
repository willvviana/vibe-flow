# Listening Pack and adherence testing

`/listening.html` (header link **Listening**) builds a *listening pack*: the real adapt pipeline (MusicGen Small + `engine/sample_compose.py`) runs on one or more samples at every adherence level, each result is measured against its sample, automatic checks say whether the levels behave as promised, and a blind pack lets a person rate the results without knowing the level. Everything runs locally; nothing is simulated.

## Adherence levels
Create's *Adapt a sample* tab has four level buttons above the 0-100 % slider (the slider remains for in-between values). API: `"adherence_level": "low" | "medium" | "high" | "strict"` (wins over `adherence`). Each level is a fixed point of the existing control mapping (`engine/adherence.py`) chosen so neighbouring levels differ in a *mechanism*, not only a number:

| Level | % | Bars of the sample kept (literal copy) | Audio conditioning (sample's first bar(s) as MusicGen audio prompt) | Tempo / key written into the prompt | Alignment of the new layers |
|---|---|---|---|---|---|
| Low | 15 | none | no | tempo only | 22 % |
| Medium | 50 | ~1/3 | yes | tempo + key | full |
| High | 75 | ~2/3 | yes | tempo + key | full |
| Strict | 100 | all (bit-exact), layers added underneath | yes | tempo + key | full |

## What is measured (`engine/adherence_eval.py`)
Every number is computed from the generated audio and the reference window; none is a listening judgement.

| Dimension asked for | Measure |
|---|---|
| Tempo | BPM of result vs. reference (octave errors folded); tempo match = max(0, 1 − error/10 BPM) |
| Rhythm | correlation of the onset-strength envelopes on the same time axis |
| Melody / harmony | mean per-frame cosine of the 12-bin chroma; share of frames with the same strongest pitch class; key match |
| Instrumentation | correlation of the 40-band log-mel average spectrum; cosine of LAION-CLAP audio embeddings |
| Structure | correlation of the two self-similarity matrices (2 Hz chroma + mel features); loudness-envelope correlation |
| Vocal characteristics | Demucs vocal stem of the result vs. the reference's: share of co-voiced 10 ms frames whose pitch agrees within 50 cents (octave errors ignored), vocal loudness-envelope correlation, share of sung frames kept |
| Quality | clipped samples, click score (largest sample-to-sample jump ÷ 99.9th percentile), level vs. reference, silent share; bit-exact integrity of retained bars |

**Adherence index** = mean of tempo match, rhythm, harmony, spectrum and structure similarity (+ CLAP when installed). It compares settings on the *same* sample; it is not a quality score. Each result is measured twice: the **mix** (what the user gets) and the **added layers only** (what the model generated), so copied bars cannot hide how much the model itself follows the sample.

Harness controls (`tests/test_listening.py`, signals with known answers): identical audio → index > 0.97, presence 1.0, key match; same beat with a different note → rhythm > 0.9, noise → rhythm < 0.3; 100 vs 120 BPM → error > 15 BPM, 100 vs 200 BPM → < 2 (octave folding); pitch tracker within 5 cents on 110-659 Hz tones; an octave double counts as the same melody.

## Automatic checks per sample (`engine/listening.py`)
monotonic (index rises with the level), distinct (adjacent levels differ by more than twice the seed-to-seed spread, min 0.02), consistent (seed spread ≤ 0.05), mechanism (Low keeps no bars; Strict keeps all and contains the whole reference), integrity (retained audio bit-exact, residual ≤ 1e-5), quality (no clipping, no click above max(8, 2× the reference), level within ±6 dB), vocals follow the level (for references with vocals), instruments (CLAP rank of the selected instruments in the added layers vs. the reference; informational). Across samples: the level ordering must be the same on every sample type.

## Blind listening test
Each result is copied to `blind/<code>.wav` (random codes, shuffled). The page plays the reference and the coded clips; the listener rates *closeness* (1-5), *quality* (1-5) and may guess the level. The key (`key.json`) is never served by the API; saving ratings unblinds them: closeness and quality per level, Spearman correlation of closeness with the level, level-guess accuracy vs. chance. Rate before opening *Measured results*, which shows the levels.

## Vocal layers: lead vs. backing (`engine/vocal_layers.py`)
After Demucs, the Sample page now also splits the voice stem into **Lead** and **Backing** (new versions on the recording card, usable on Create and in packs) and reports per layer: pitch (median, range), intonation (median distance to the nearest semitone), steadiness, sung share, onset timing against the instrumental's 8th-note grid (with the figure expected by chance), harmony intervals of the backing above the lead, how often backing sings with the lead, and a clarity proxy (Whisper decoding confidence on the voice stem).

Method: per STFT bin, inter-channel similarity × level balance → a soft centre mask; lead = masked mid signal, backing = stem − lead (they add back up exactly). Harmony uses the backing's side signal (where the centre lead cancels) and a chroma profile relative to the lead's pitch class, because a single-pitch tracker on a two-voice harmony reports the chord root.

**Limits:** backing panned to the centre stays in Lead (shown below: scenario B); a mono voice has no backing layer (reported as such); reverb tails leak into Backing. Single-pitch statistics are not reported for a backing layer that sings several notes at once.

---

# Results (Ryzen 7 3700X, CPU only, 2026-10-23)

Samples, all through the real app: **"Listening test - sung clip"** recorded through the Sample page API (a 30 s sung ACE-Step clip streamed as PCM, processed by Demucs/CLAP/Whisper) and used as **Full**, **Instrumental** and **Voice**; **"Listening test - band"** saved as an Editor sample (instrumental). 20 s window, levels Low/Medium/High/Strict, seeds 5 and 6, selected instruments piano + strings. 32 generations per pack, about 56-70 minutes each.

## Adherence levels: final pack (shipped settings) vs. baseline
Mix adherence index, mean of 2 seeds (± seed spread in the final pack). Both packs measured with the final metrics (the baseline was re-measured from its saved audio, `tools/remeasure_pack.py`).

| Sample | | Low | Medium | High | Strict | Failed checks |
|---|---|---|---|---|---|---|
| Full mix (recording) | baseline | 0.31 | 0.53 | 0.76 | 0.99 | quality (−15.6 dB level drop) |
| | **final** | 0.32 ±0.02 | 0.52 ±0.04 | 0.72 ±0.05 | 0.95 ±0.01 | **none** |
| Instrumental (recording) | baseline | 0.29 | 0.51 | 0.69 | 0.81 | consistency (SD 0.058), quality |
| | **final** | 0.29 ±0.02 | 0.51 ±0.04 | 0.70 ±0.00 | 0.84 ±0.01 | **none** |
| Voice only (recording) | baseline | 0.16 | 0.33 | 0.53 | 0.72 | quality (−7.6 dB) |
| | **final** | 0.16 ±0.01 | 0.27 ±0.01 | 0.48 ±0.00 | 0.62 ±0.00 | **none** |
| Band (Editor sample) | baseline | 0.44 | 0.68 | 0.80 | 0.99 | none |
| | **final** | 0.44 ±0.02 | 0.67 ±0.03 | 0.79 ±0.03 | 0.96 ±0.01 | **none** |

- **Each level is measurably different and in the same order on every sample type** (full, instrumental, voice, Editor sample); adjacent levels differ by more than twice the seed spread; no clipping, no clicks; retained audio is bit-exact (residual ≤ 6e-8).
- **Vocals follow the level**: share of sung frames whose pitch matches the reference (Demucs voice of the result vs. of the reference) — full mix 0.03 / 0.64 / 0.99 / 1.00, voice-only 0.07 / 0.72 / 0.98 / 0.99 (Low / Medium / High / Strict). The engine generates no vocals; vocals in a result are the sample's own retained/continued voice.
- **What the levels change is mostly how much of the sample is kept.** The *added layers alone* get only slightly closer: full mix 0.32 / 0.44 / 0.43 / 0.52, band 0.44 / 0.49 / 0.47 / 0.61. Audio conditioning (from Medium) helps the layers once; more retained bars do not make the model follow the sample more.
- Strict scores a little lower than before (e.g. 0.99 → 0.95) because the added layers are now as loud as the sample (−6 dB under it), so more new material is audible. That is the intended effect of the loudness fix, not a regression of the copy (which is bit-exact).

### Problems found and fixed through the packs
| Found | Cause | Fix | Result |
|---|---|---|---|
| Results up to 15.6 dB quieter than the sample (3 of 4 samples) | MusicGen's output level varies by seed (−22 vs −33 dBFS); nothing matched it | Layers loudness-matched to the sample (±12 dB) before the layer level | worst level difference −15.6 → −4.8 dB |
| Whole take 7-13 dB quieter on a near-full-scale sample | Peak safety scaled the **entire** layer by (0.99 − sample peak) / layer peak for one coincident peak | Gain envelope on the layers only where the sum would clip (10 ms smoothing); sample untouched | level within ±5 dB; limiter active on ≤ 1.5 % of samples |
| Seed-to-seed swings of 0.2 in the index; levels "not distinct" | The tempo score was 0 or 1 from a single BPM estimate, which flips between metrical levels (78 vs 104 BPM, a 4:3 ratio) on ambiguous material, even with a 1 dB level change | Tempo in the index is now the similarity of the periodicity profiles; BPM error still reported | all levels distinct on every sample |
| "Strict" check failed although every bar was kept exactly | It used a projection that drops when the layers correlate with the sample | Strict = all bars kept + bit-exact copy; projection shown as info | — |
| Generated parts at 4:3 of the sample's tempo (band: 7 of 8 takes from Medium up) | Ambiguous pulse: that clip was planned at 79 BPM and another estimator reads it at 106 | A seed retry was built and tested: fixed 2 of 8, cost +28 % time. **Kept opt-in only** (`tempo_retry: true`) | needs listening: may be inaudible |

**Remaining artefact risk (not judged by ear):** where the sample itself is at full scale, the limiter silences the added layers for a few milliseconds at a time (at most ~1 % of a take, ~0.2 s total). The click check does not flag it.

## Lead / backing vocals: separation on known mixes (`tools/vocal_split_eval.py`)
Real audio, real Demucs, real split; only the mix is constructed so that the right answer is known: lead = Demucs voice of the sung clip (cleaned by a second Demucs pass; Demucs puts 100 % of it back into the voice stem), backing = the lead pitch-shifted +4 and +7 semitones, 25 ms late, −6 dB, band = another ACE-Step clip. SI-SDR in dB (higher is better; "unprocessed" = the mix itself as the estimate).

| Scenario | Lead | (whole voice stem as lead) | Backing | Backing share true / estimated | Instrumental | Harmony found |
|---|---|---|---|---|---|---|
| A: backing panned wide (usual) | **6.6** (mix: 1.3) | 3.8 | **4.8** (mix: −9.3) | 19.6 % / 18.5 % | 2.8 (mix: −0.3) | **5th, major 3rd** (the planted intervals) |
| B: backing panned centre | 3.9 | 3.9 | −24.3 (fails) | 19.5 % / 1.5 % | 3.3 | none reported |
| C: lead only | 7.2 | 6.9 | n/a | 0 % / 2.3 % | 3.5 | none reported |

- The split improves the lead by 2.8 dB over using the whole voice stem when the backing is panned, and finds the backing share within about 1 point. **It cannot separate centre-panned backing (B)**, as expected for a position-based method; it then reports almost no backing rather than guessing.
- The instrumental figure (~3 dB) is Demucs' own result on this mix (vocals ~4 dB louder than the band; part of the band is pulled into the voice stem). Not compared with other separators.
- Bugs found and fixed while testing: the harmony detector reported "unison" for a two-voice harmony (a single-pitch tracker locks onto the chord root), and the backing "median pitch" was the chord root too (now suppressed when several backing voices sing at once).
- On the real recording: backing 4 % of the voice energy (reported as little or no backing), lead median 304 Hz, 8.6 cents median distance from the semitone grid, onsets 51 ms from the 8th-note grid where unrelated onsets would be ~71 ms (some relation; sung onsets are soft and the clip's tempo confidence is 7 %), 28 words decoded with mean log-probability −0.24.

## Instrument and genre recognition (`tools/tag_eval.py`, the Sample page's CLAP tagger)
| Test | n | True label rank 1 | in top 3 | in top 5 | all true labels in top 5 |
|---|---|---|---|---|---|
| Single instrument (8 MusicGen "solo" clips) | 8 | 62.5 % | 87.5 % | 87.5 % | 87.5 % |
| Two instruments summed (all 28 pairs) | 28 | 17.9 % | 42.9 % | 50.0 % | **17.9 %** |
| Three instruments summed | 12 | 0 % | 13.9 % | 22.2 % | **0 %** |
| Requested instruments in the 24 multi-instrument clips | 24 | 7.3 % | 16.7 % | 24.0 % (top 7: 30.2 %) | 0 % |
| Genre (36 clips, 6 prompts, 4 of them two-genre) | 36 | any true genre: 25.0 % | 52.8 % | | |
| Two genres overlapping (summed) | 15 | | | 23.3 % | **0 %** |
| Two genres in sequence | 15 | | | 16.7 % | **0 %** |

- **Single instruments are recognised well; mixtures are not.** In a mix, the tagger reports one dominant instrument family and misses the rest: drum kit fell to rank 21-24 when paired with violin, flute or saxophone. Electric guitar alone was ranked 6th (saxophone first).
- **Genre is weak**: rock was never in the top 3 (0/6), folk 1/6; hip hop/R&B and cinematic 6/6. Mixed genres were never both found.
- Labels are what MusicGen was *asked* to play; it does not always comply, which lowers these numbers somewhat. The pair/triple mixes are exact by construction.
- **Fix tried and rejected:** per-label calibration against a disjoint background set (46 clips; `tools/clap_calibrate.py`), with and without the strongest-window score. It lowered single-instrument top-1 to 37.5 % and genre top-3 to 47 %, with small gains on triples (top 5 22 → 39 %). Not an improvement overall, so the app keeps the original ranking (the code stays inert without a calibration file). Better multi-label detection needs a different model (e.g. a multi-label instrument tagger); none is installed.

## Selected instruments vs. adherence level
CLAP rank of **piano** (1 = best of 36) in the *added layers*, mean of 2 seeds (baseline pack; the reference's own piano rank in brackets):

| Sample | Low | Medium | High | Strict |
|---|---|---|---|---|
| Full mix (21) | **1** | 14.5 | 14.5 | 12.5 |
| Instrumental (22) | 6 | 7 | 3.5 | 3.5 |
| Voice only (32) | 3 | **1** | 9 | 9 |
| Band, Editor (27) | 7.5 | 9.5 | 10 | 22 |

At Low (no audio conditioning) the text selection drives the layers. From Medium up, MusicGen continues the sample's own timbre and the selected instrument is followed only where the sample leaves room for it (voice-only, instrumental stem); on a full mix it is mostly overridden. This is a property of audio-prompt continuation, not a bug; the Create page now says that Low follows the selected instruments most closely. (Strings, the second selection, was weaker everywhere: rank 7-20.)

**Stronger text guidance (`tools/experiments/instrument_influence.py`, band sample, piano only):** guidance 10 instead of 5 made piano rank **1** with seed 5 at both Medium and High, but left it at 17 with seed 6; mean rank Medium 13.5 → 9, High 8.5 → 9; mix index unchanged within seed noise (0.60 → 0.63, 0.73 → 0.74). Two seeds: **inconclusive, default unchanged**. Worth a larger sweep (6-8 seeds) if instrument control at Medium/High matters to you.

Noticed in passing (not changed): with no genre or mood selected, the sentence prompt reads "A instrumental piece ...". Changing it alters every generation of the default profile, so it is left for a separate change.

### Also fixed along the way
- A job could hang forever when it reached the warm generation worker just as it idled out (the retry replaced the job's own resolve/reject before chaining them); found by the test suite as an intermittent 120 s timeout.
- The Node test files now run one at a time (`--test-concurrency=1`): run in parallel, the real-model pack test starved the live-transcription test of CPU.
- Lead/Backing waveforms returned 404 (the recordings route only allowed full/instrumental/voice).
- A pack with no instrument, genre or mood failed at its first generation; it is now refused up front with a clear message.

## What is still open
- **No listening has been done.** Every number above is a measurement. The blind pack of the final run (`/listening`, pack "Adherence levels - final") is ready to rate; until then it is unverified whether the levels *sound* different, whether Medium/High sound musical, and whether the 4:3 tempo takes or the short limiter gaps are audible.
- One listener, two seeds, four samples from two source songs: enough to find mechanical problems, not to generalise.
- Multi-instrument and genre tagging need a better model; the selected-instrument override at Medium+ needs a larger guidance sweep.

## Reproduce
- Listening page: `npm start`, open `/listening`, pick samples, *Build listening pack*. API: `POST /api/listening/run`, `GET /api/listening[/:id]`, `GET /api/listening/:id/file?path=…`, `POST /api/listening/:id/ratings`.
- `python tools/vocal_split_eval.py samples/ace/s2_sft_codes_30s_32steps.wav samples/ace/s5_sft_instrumental_30s_32steps.wav .sandbox/listening/vsplit`
- `python tools/tag_eval.py gen .sandbox/listening/inst` then `python tools/tag_eval.py eval .sandbox/listening/inst .sandbox/listening/tag_eval.json`
- `python tools/remeasure_pack.py userdata/listening/<id>` (re-measure saved packs after a metric change)
- Tests: `tests/test_listening.py` (levels, metrics with known answers, vocal split, checks, scoring), `tests/test_adapt_compose.py` (loudness match, local limiter, opt-in retry), `tests/server.test.js` (a real two-level pack over HTTP, blind files, hidden key, ratings).

