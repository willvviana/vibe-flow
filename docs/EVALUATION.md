# Vocal and coherence evaluation

Machine: Ryzen 7 3700X (8C/16T), 16 GB RAM, AMD RX 5700 XT (not usable by PyTorch on Windows), CPU inference only.

**Honesty note.** I cannot listen to audio. Every "passes" below is an *objective proxy* (see `tools/eval_audio.py`):
silence/clipping, a tempo/pulse estimate, and a Whisper-small transcript compared with the lyrics. A transcript that matches
the lyrics is real evidence that words are audible. It is **not** evidence that the singing, the arrangement or the mix
sounds good. The acceptance criteria for musicality, rhythm feel and naturalness are therefore **still open until you listen**
(`samples/README.md` says what to listen to). Your own listening verdict on the first ACE-Step clip (no singing, isolated
instruments) was confirmed by the proxies, which is why I trust them for what they do show.

## Harness validation (controls)

| Control | Result |
|---|---|
| Windows speech synthesiser reading the lyrics | Whisper transcript matches (WER 0.18, only "Streetlights" split in two) |
| MusicGen instrumental clip | transcript "you", WER 1.0 |
| 100 BPM click track | 100.0 BPM, pulse strength 0.66 |
| White noise | pulse strength 0.08, tempo spread 50 BPM |
| Your rejected clip (`acestep-vocal-test.wav`) | transcript "you", WER 1.0, **29 % of frames silent** |

## Root causes

**Why ACE-Step did not sing (identified and tested).** Two compounding configuration problems, not a model limit:

1. **Wrong DiT variant.** I used `acestep-v15-turbo` (8-step distilled). With the same lyrics, codes and seed, `acestep-v15-sft`
   (32 steps) is the first configuration whose output Whisper can transcribe as the lyrics.
2. **No language-model planning.** With `thinking=False` and no LM, the DiT gets no semantic audio codes (the song's melody /
   phrasing scaffold). Result: 29 % silent frames in the 10 s clip and no stable structure. LM codes cut silence to 1 % (10 s)
   and 6 % (30 s, which is silent intro/outro only) and stabilised tempo.
3. Minor: 10 s is at the model's documented minimum, and the docs recommend letting duration follow the lyrics; 30 s works.

Turbo with LM codes still did not sing (WER 1.0), so it is the **combination sft + LM codes** that matters.

**Why MusicGen sounds like isolated instruments.** Model capacity and conditioning, not post-processing:
MusicGen Small is 300 M parameters, mono, with a T5 text prompt only. Larger checkpoints don't fit this machine
(`musicgen-medium` ships only a 7.5 GB pickle `.bin`, which our loader refuses and which would need >15 GB RAM).
Tempo/key text in the prompt is only weakly followed. What I could change: prompt wording, sampling, and the
stitching of long tracks (previously skipped 0.1 s per join; fixed earlier). Evidence for the other levers is weak (below).

## Models and configurations evaluated

| # | Model / config | Vocals attempted | Lyrics recognised (Whisper WER; lower is better) | Silent frames | Gen time | Peak RAM |
|---|---|---|---|---|---|---|
| 1 | ACE-Step turbo, no LM, 10 s (the rejected clip) | yes | 1.00 ("you") | 29 % | 45 s (RTF 4.4) | 12.6 GB |
| 2 | turbo, no LM, 30 s, 4 lines | yes | 1.00 ("You") | 9 % | 96 s | 11.0 GB |
| 3 | turbo + LM codes, 10 s | yes | 1.00 (vocal-like sound, wrong words) | 1 % | 66 s | 11.3 GB |
| 4 | turbo + LM codes, 30 s | yes | 1.00 ("Yeah") | 6 % | 99 s | 11.2 GB |
| 5 | **sft + LM codes, 32 steps, 10 s** | yes | **0.42** (12 words heard) | 0 % | 110 s (RTF 11) | 11.3 GB |
| 6 | **sft + LM codes, 32 steps, 30 s, seed 7** | yes | **0.07** (all 4 lines) | 6 % | 275 s (RTF 9.2) | 11.3 GB |
| 7 | sft + LM codes, 32 steps, 30 s, seed 11 | yes | 0.15 | 6 % | 248 s | 11.5 GB |
| 8 | sft + LM codes, **16** steps, 30 s | yes | 0.26 | 6 % | 153 s (RTF 5.1) | 11.5 GB |
| 9 | sft + LM codes, 32 steps, **instrumental** 30 s (band only) | no | n/a | 6 % | 239 s (RTF 8.0) | 10.8 GB |
| 10 | MusicGen Small (the shipped engine) | no (instrumental only) | n/a | 0 % | RTF 3-5 | 4.4 GB |

LM stage (ACE-Step 5 Hz LM 1.7B, separate process so the 2 B DiT and the LM are never in RAM together): 30 s plan in 91 s (+34 s load),
peak 7.4 GB; 10 s plan 40 s, peak 8.0 GB. A full 30 s sung song is therefore about 7 minutes end to end
(LM 125 s + DiT load 25 s + DiT 250 s) on this machine.

Whisper transcripts for rows 6-8 (lyrics: *Streetlights hum a quiet tune / I am walking home beneath the moon / Hold me close, do not let me go / Carry me where the soft winds blow*):
- seed 7: "Street lights, hum a quiet tune I am walking home beneath the moon Hold me close, do not let me go Carry me where the soft winds blow"
- seed 11: "Street lights, I'm a quiet tune I am walking home beneath the moon Hold me close to not let me go Carry me where the soft winds blow"
- 16 steps: "Street lights, I'm a quiet tune I'm walking home beneath the moon Hold me close, then let me go Carry me where the soft winds blow"

ACE-Step instrumental (row 9): pulse strength 0.32, tempo estimate 106 BPM with a wide spread (IQR 28) against the LM-planned 79 BPM. That proxy hints the pulse may not be steady,
but onset-based tempo tracking is unreliable on sustained guitar/strings, so this is **a reason to listen closely, not a verdict**.

Other candidates (not run): YuE (Apache-2.0; published setups target large GPUs, 7 B stage-1 model), DiffRhythm (Apache-2.0, needs ≥8 GB CUDA VRAM),
Stable Audio Open (gated weights, not vocal). AMD GPU: ROCm on Windows does not list gfx1010; DirectML + MusicGen fails.

### MusicGen Small A/B (15 s, 2 seeds, requested 72 BPM; proxies only)

| Config | Median BPM seeds 1 / 2 | Pulse strength 1 / 2 |
|---|---|---|
| A keyword prompt, T 1.0 (current default) | 134 / 144 | 0.42 / 0.47 |
| B sentence prompt, T 1.0 | 139 / 150 | 0.40 / 0.27 |
| C sentence, T 0.8, top-k 100 | 144 / **72** | 0.45 / 0.55 |
| D sentence, T 0.8, guidance 5 | 96 / **72** | 0.41 / 0.57 |

(144 is 72 BPM counted at double rate.) Two seeds per row is far too little to call a winner: C/D look slightly more tempo-stable but this is within noise,
and I cannot tell by proxy whether they sound more like a band. **Defaults were left unchanged**; the options are available
(`prompt_style: "sentence"`, `temperature`, `top_k`, `guidance` in the API) and the clips are in `samples/musicgen/` for you to compare by ear.

## Acceptance criteria (ACE-Step 1.5, sft + LM codes, two-stage)

| # | Criterion | Status |
|---|---|---|
| 1 | Recognisable singing that follows the lyrics | **Objective evidence yes** (3 of 3 sft 30 s runs, WER 0.07-0.26; 10 s WER 0.42). Not yet confirmed by listening that it is *singing* rather than rhythmic speech |
| 2 | Instruments form one arrangement | **Unverified** (needs listening) |
| 3 | Shared rhythm / pulse | **Unverified.** Pulse proxy is weak (0.14-0.22 vs 0.35-0.4 for turbo clips), which may just be a voice-dominated onset signal; tempo estimates are stable (BPM 76-84 vs LM-planned 79) |
| 4 | Naturalness justifies cost | **Unverified** |
| 5 | No severe artifacts | Partly: no clipping (1 full-scale sample), no mid-song dropouts (silence only at start/end, ~1.9 s). Warble/harshness unverified |
| 6 | Reliability | Ran 11 generations over several processes without a crash or corrupt file; **not** yet exercised through the app's job/cancel/worker path |
| 7 | Performance | **Heavy.** 11.3-11.5 GB peak RAM on a 16 GB machine (only ~4-11 GB is typically free), 5-9x real time for the DiT plus the LM stage |
| 8 | Licensing | Code MIT; DiT, LM 0.6B and main repo listed MIT on Hugging Face; text encoder Qwen3-Embedding Apache-2.0. **Uncertain:** training-data provenance and the "commercial use" statement come from the model card, not a legal review. The 1.7B LM's own licence field is not shown on Hugging Face (it ships inside the MIT main repo) |

## Recommendation: **continue experimenting; do not integrate yet; keep the stable MusicGen instrumental engine**

1. Listen to `samples/ace/s2_...`, `s3_...`, `s1_...` (and `s5_...` instrumental). If the voice sings and the band feels unified, criteria 1-5 can pass.
2. If they pass, the remaining blocker is cost: ~11.5 GB peak RAM and ~7 minutes per 30 s song. Integration would need the two-stage worker, the RAM preflight
   (already in `engine/memory.py`) with a measured figure, and a clear "close other apps" message. A 16 GB machine without a GPU is at the edge.
3. If you want the best quality/time trade-off, the 16-step sft run (153 s, WER 0.26) is the faster setting to listen to first.

## What changed in the app in this round (instrumental engine kept working)

- `engine/memory.py`: preflight uses the *measured* MusicGen peak (4.4 GB) and refuses when free RAM is below 60 % of it, warns below peak + 2 GB headroom.
- `engine/worker.py` + server: model stays loaded between jobs (second job `load_s` < 1 s, tested), exits after 5 min idle (`VIBEFLOW_IDLE_S`) and frees ~4 GB. Cancel kills the worker.
- `engine/engines.py`: clean `Engine` interface; only MusicGen is registered; a test asserts ACE-Step is not.
- `prompting.py`: optional `prompt_style` and `top_k`; `generate.py` passes `top_k`.
- `library.cleanup`: fixed a real flake (age check on a future timestamp) found while testing.
- `tools/`: evaluation harness, controls, benchmark, seam analysis, ACE-Step experiment scripts (not imported by the app).

---

# Round 2: MusicGen D validation and ACE-Step warbling

## Correction to round 1 (my error)
ACE-Step's `GenerationConfig.use_random_seed` defaults to **True**, so the "seed 7 / seed 11" labels on the round-1 ACE-Step clips were
**not honoured**: those clips were different random draws. (The LM planning stage's seed was honoured.) What survives: the three sft + codes
30 s runs and the 10 s run were four independent draws that Whisper could all transcribe, so singing was not a one-seed fluke *in that sample*.
Round 2 uses `use_random_seed=False` and records `seed_used` (verified equal to the requested seed).

## Listening judgments you gave (not mine)
ACE-Step s1-s4: recognisable singing, lyrics followed, vocal mix good, **warbling artifacts**. s5 instrumental: cohesive arrangement.
MusicGen: D preferred over A-C, but only one of the two seeds was convincing. Which seed that was is **unknown to me**.

## Task 1: MusicGen D vs A

Configurations (identical text content: genres pop+folk, moods calm+warm, 72 BPM, acoustic guitar+bass+percussion; 15 s; MusicGen Small; 8 threads):

| | Prompt | Temperature | top-k | Guidance |
|---|---|---|---|---|
| A (current default) | keyword list | 1.0 | 250 | 3.0 |
| D | one sentence ("played together by ...") | 0.8 | 250 | 5.0 |

- Round-1 manifest: A/B/C/D, seeds 1 and 2 (`samples/musicgen/`).
- Round 2 (`tools/experiments/musicgen_ab.py`): A and D, **seeds 1-8** (16 clips), shuffled neutral names. Re-generating seeds 1 and 2 reproduced the round-1 clips
  **bit-for-bit** (checked), so results are deterministic and the round-1 files are exactly A/D seeds 1-2.
- Blind listening pack: `samples/musicgen_ab/` = seeds **3-8** (12 clips; seeds 1-2 excluded because you have already heard them unblinded).
  Rate with `ratings_template.csv`, then `python tools/score_ab.py ratings.csv .sandbox/ab/key.json` (the key stays outside `samples/`).

Automated measurements, 8 seeds each (proxies; **not** listening):

| Measure | A | D |
|---|---|---|
| Median tempo within 4 % of the requested 72 BPM or its double | 3 / 8 seeds | **7 / 8 seeds** |
| Tempo-window spread (IQR, BPM; lower = steadier) | 23.1 | **5.3** |
| Pulse strength (autocorrelation peak) | **0.48** | 0.44 |
| RMS level (dBFS, same peak scaling) | -18.1 | -21.8 (quieter, higher crest) |
| Silent-frame ratio / clipped samples | 0.000 / 0 | 0.003 / 0 |
| Generation time per 15 s clip (mean) | 72.2 s (RTF 4.8) | 70.8 s (RTF 4.7) |
| Peak RAM | 3.7 GB | 3.7 GB |

Interpretation: D follows the requested tempo much more often and keeps it steadier, which fits "D sounded better". But pulse strength is not higher and D is ~3.7 dB quieter, so the
proxies neither confirm nor refute "more like a band". Both configs run at the same speed and memory.

**Decision: the default stays A (`baseline`).** The only listening evidence is 2 seeds of which 1 convinced you; the proxies are weak; the blinded 6-pair test is waiting for your ratings.
D is now selectable as `profile: "d"` (API/settings, no UI control) and can become the default by changing one constant (`engine/profiles.py: DEFAULT_PROFILE`)
or the env var `VIBEFLOW_PROFILE=d`. Regression tests pin the default to the original values. Suggested rule once you have rated: adopt D only if it wins most of the 6 pairs *and* the sign test
is not trivially insignificant (`score_ab.py` prints both; with only 6 pairs a clean 6-0 gives p = 0.031, 5-1 gives p = 0.22).

## Task 2: ACE-Step warbling

Fixed conditions: sft DiT, LM codes from the 10 s plan (`codes10.json`, LM-planned 73 BPM C minor), same two lyric lines, same caption, 48 kHz stereo, CPU fp32, no normalisation. Seeds 7 and 8 (honoured).

| Config | Seed 7: Whisper WER / heard | Seed 8: Whisper WER / heard | Time (s) s7 / s8 | Peak RAM |
|---|---|---|---|---|
| W1 32 steps, guidance 7, shift 1 (what you heard) | 0.25 / "...beneath the" | 1.00 / "Music" | 104 / 93 | 11.4-11.7 GB |
| W2 **16** steps | 0.17 / full line | 1.00 / "Music" | 66 / 58 | 11.4-11.7 |
| W3 **50** steps (docs' recommended sft setting) | 0.17 / full line | 1.00 / "Music" | 140 / 133 | 11.7 |
| W4 50 steps, shift 3 | 0.17 / "...beneath" | 0.58 / partial words | 137 / 139 | 11.7 |
| W5 32 steps, guidance 4 | 0.50 / "humming of gratitude" | 1.00 / "Music" | 94 / 90 | 11.7 |

What this shows (automated):
- **Step count is not the driver of intelligibility**: 16 / 32 / 50 steps give about the same WER on seed 7, and all but shift 3 fail on seed 8. I therefore do **not** attribute anything to the step count.
- **Reliability is seed-dependent**: seed 7 sings in every config; seed 8 produced no recognisable words in 4 of 5. A "works on one seed" pattern, like MusicGen's.
- Lower guidance (4) hurt intelligibility on seed 7. Shift 3 was the only setting that produced words on seed 8 (one seed: not conclusive).
- Time scales roughly linearly with steps (16 steps about 60 s, 50 steps about 135 s per 10 s); memory is 11.4-11.7 GB regardless.
- My "warble" proxy (4-12 Hz band-envelope modulation) is **useless**: it separates a synthetic tremolo from a steady tone in a control, but scores every real clip about 0.3-0.4. I did not use it.
- **Warble itself has not been measured**, only your report of it. Plausible causes, none isolated yet: diffusion step/guidance/shift settings, the VAE decode / 48 kHz stereo
  codec, and the LM codes' quality. VAE tiling is *not* a candidate at 10 s (a single 256-frame tile).
- Blind pack: `samples/ace_warble/` (10 clips, w01-w10; key in `.sandbox/exp/warble_key.json`). Rate warble 1-5 and whether it sings; then I can tell whether steps / guidance / shift change *warble*, which I cannot hear.

## Is ACE-Step ready for integration work? **No.**
Singing, lyrics and mix are confirmed by you; but (1) warbling is unresolved and untested by listening across settings, (2) singing is not reliable across seeds (seed 8 failed in 4 of 5 configs),
(3) it needs about 11.7 GB peak RAM and 1-2.5 minutes per 10 s plus an LM stage, and memory safety was only shown in isolated runs, not through the app worker.
Next steps in order: rate `samples/ace_warble/`; if some config is clearly less warbly and still sings, run a 6-8 seed reliability sweep with it (and a seed-retry policy using the Whisper transcript as an automatic check);
only then build the two-stage worker with the RAM preflight.

## Tests this round
- `pytest -k "not smoke"`: 43 passed (new: default-profile regression, profile D + env revert, score_ab unblinding / sign test).
- Full `npm test`: see the result stated in the final report.
- Not run: any listening; ACE-Step through the app; the `profile` setting through the HTTP API (unit-tested at the validation layer only).

---

# Round 2b: your blind ratings

All numbers below that come from "score", "ships" or "sings" are **your listening**. ASR / tempo figures are automated.

## MusicGen A vs D (12 blind clips, seeds 3-8, one clip per config per seed)

| Seed | A score | D score |
|---|---|---|
| 3 | 3 | 5 |
| 4 | 5 | 5 |
| 5 | 4 | 4 |
| 6 | 3 | 5 |
| 7 | 4 | 4 |
| 8 | 3 | 4 |
| **mean** | **3.67** | **4.50** |
| "would ship" | 3 / 6 | **6 / 6** |

Paired by seed: D won 3, tied 3, **lost 0**. The formal two-sided sign test on the 3 non-tied pairs gives p = 0.25, so this is **not statistically significant**
(6 pairs, one listener, one prompt family), and I am saying so plainly.

**Decision: D becomes the default.** Reasons: D was never worse on any seed (the "one lucky seed" concern is answered: D was rated 4-5 on all six, A as low as 3);
D was preferred in round 1 as well; D "ships" 6/6; the tempo proxy agrees (7/8 vs 3/8); cost is identical (same speed, same memory). Risks, stated honestly: one listener, one prompt
(acoustic pop/folk, 72 BPM, 3 instruments), and the significance test does not clear 0.05. D has not been tested on other genres. **Revert in one step:** `VIBEFLOW_PROFILE=baseline`
or `DEFAULT_PROFILE = "baseline"` in `engine/profiles.py`; tests pin both the new default and the exact original values.

## ACE-Step warble ratings (10 blind clips)

| Clip | Setting (seed) | You: warble (1-5) | You: sings | ASR heard words? |
|---|---|---|---|---|
| w01 | 50 steps, shift 3 (8) | **5** | y | partly (WER 0.58) |
| w02 | 50 steps, shift 3 (7) | 4 | y | yes (0.17) |
| w03 | **16 steps** (7) | **3** | y | yes (0.17) |
| w09 | 50 steps (7) | 4 | y | yes (0.17) |
| w04 | 32 steps, guidance 4 (7) | 2 | **n** | yes (0.50) |
| w06 | 32 steps (7) | 2 | **n** | yes (0.25) |
| w05, w07, w08, w10 | 50 / g4 / 16 / 32 steps (all seed 8) | not rated | n | no ("Music") |

What this supports (n = 6 rated clips, so tentative):
1. **Singing is unreliable: 4 of 10 clips sing.** Seed 8 never sang except with shift 3.
2. **My speech-recognition check is not a singing detector.** It heard words in w04 and w06, which you judged not to sing (probably spoken/rapped delivery). Earlier statements that ASR "proves it sings" were too strong:
   it shows words are audible, not that they are sung.
3. **Warble appears tied to actually singing**: the two clips that did not sing have the *lowest* warble (2), the singing clips have 3-5. Among the clips that sing, the **16-step clip had the least warble (3)**
   versus 4-5 for 50-step clips. That is 1 vs 3 clips and a different seed than w01, so it is a lead, not a finding. I still cannot attribute warble to step count: seeds are confounded (the 16-step clip is seed 7, as are w02/w09).
4. Guidance 4 and the 32-step/guidance-7 seed-7 setting both failed to sing on this seed although the same 32-step setting sang for you on the round-1 clips (different random draws): another sign that
   singing depends on the draw more than on these parameters.

## Round 3 (ACE-Step, ready to rate): fast setting, new seeds, and a caption aimed at warble
Fixed: sft, 16 steps, guidance 7, shift 1, LM codes 10 s, same lyrics. Varied: seeds 7, 9, 10 and the caption (original vs "clear steady straight tone, no vibrato, no ad-libs").
- Time 58-73 s per 10 s clip; peak RAM 11.0-12.0 GB.
- ACE-Step with a fixed seed is **bit-identical across separate processes** (R3 original/seed 7 equals W2/seed 7, max difference 0.0).
- Blind pack `samples/ace_warble_r3/` (v1-v6; key in `.sandbox/exp/r3_key.json`). One clip repeats a clip you already rated (w03), a free consistency check on the ratings.
  ASR heard words in all six (WER 0.08-0.42), which as shown above does not mean it sings: your ratings decide.
- Question this answers: does the "no vibrato" caption reduce warble without losing singing, and does 16 steps sing reliably on 3 new seeds?

## Is ACE-Step ready for integration? **Still no.** Singing 4/10, warble 2-5, ~11-12 GB RAM, seed-dependent. Next gate: round-3 ratings.

## Tests
- `pytest -k "not smoke"`: 41 passed after switching the default (two profile tests rewritten to pin D as default and baseline as the revert).
- Full `npm test` (real-model smoke + HTTP, now exercising the D default): result given in the final message.

---

# Round 3 results (your ratings; ACE-Step, 16 steps, guidance 7, sft + LM codes, 10 s)

| Clip | Caption | Seed | You: warble | You: sings |
|---|---|---|---|---|
| v2 | original | 7 | 2 | y |
| v1 | original | 9 | 5 | y |
| v5 | original | 10 | 5 | y |
| v3 | "steady straight tone, no vibrato" | 7 | 4 | y |
| v6 | "steady straight tone, no vibrato" | 9 | 4 | y |
| v4 | "steady straight tone, no vibrato" | 10 | 5 | y |

- **Singing at 16 steps is reliable on these seeds: 6 of 6 sing** (versus 4 of 10 in the earlier matrix, which mixed settings and included seed 8). Seeds 7, 9, 10; one setting; so "reliable" is 3 seeds, not a sweep.
- **The "no vibrato" caption did not reduce warble**: mean 4.3 vs 4.0 for the original caption (it was worse on seed 7, 4 vs 2). No benefit shown.
- **Warble is seed-dependent and mostly high**: 2, 5, 5 with the original caption. The earlier hint that "16 steps warbles least" is **not supported**: it rested on one clip (seed 7, rated 3, then 2 here).
  Seed 7 is the only low-warble draw; seeds 9 and 10 are 5.
- Rating consistency check: the repeated clip (original, seed 7) was rated 3 then 2, both "sings".
- Performance: 58-73 s per 10 s clip, peak 11.0-12.0 GB.

Warble is therefore **not explained by step count or by the vibrato caption**. Still-untested candidates: guidance above 7, the VAE decode (a community alternative VAE ships in ACE-Step's registry and could be swapped on the same latents,
~0.3 GB), the LM codes, and the larger XL DiT (needs more RAM than this machine has). **ACE-Step remains not ready for integration** (warble 2-5 on every setting tried; ~12 GB peak RAM).

---

# Round 4: cross-genre MusicGen profile evaluation (A "baseline" vs D "d")

**Status: design and pre-registered decision rule written BEFORE any ratings existed. Results are further below.**

## Design
- 6 genres x 3 matched seeds x 2 profiles = **36 clips**, 15 s each, MusicGen Small, 8 threads, one model load. Seeds: 11, 12, 13 (new; none was used in earlier rounds).
- Per genre the *structured inputs* are identical for A and D; the prompt *wording* differs only because each profile defines its prompt style (keyword list for A, one sentence for D),
  which is part of what is compared. Sampling differs by profile: A = T 1.0, top-k 250, guidance 3.0; D = T 0.8, top-k 250, guidance 5.0.
- Every clip goes through the production path (`prompting.validate` with `profile` -> `build_prompt` -> `generate._infer`), so the profiles are tested exactly as shipped.

| Genre key | Genres | Moods | BPM | Instruments |
|---|---|---|---|---|
| folk | pop, folk | calm, warm | 72 | acoustic guitar, bass, percussion |
| rock | rock | energetic | 120 | electric guitar, bass, drums |
| synthwave | synthwave, electronic | dreamy, energetic | 100 | synthesizer, bass, drums |
| hiphop_rnb | hip hop, r&b | mellow | 88 | piano, bass, drums |
| jazz | jazz | mellow | 120 | piano, bass, drums, saxophone |
| cinematic | cinematic | epic, uplifting | 90 | strings, brass, percussion |

- Clips are named `g01.wav`...`g36.wav` in a seeded random order; the key (profile, prompt, seed, time, parameters, automated measures) is kept in `.sandbox/xg/key.json`, outside `samples/`.
- Listening instructions: rate each clip 1-5 for "one coherent band playing a real piece" (coordinated instruments, steady rhythm, natural, no obvious artifacts) and `ship` y/n.
  Rate each clip on its own; do not try to identify A or D.

## Analysis plan (`tools/score_xgenre.py`, unit-tested)
- Paired by (genre, seed): difference = score(D) - score(A). 18 pairs.
- **The three seeds of one genre share a prompt, so they are not independent.** Uncertainty therefore treats genre as the cluster: cluster bootstrap (resample genres, then seeds within them) for
  the mean difference, and an exact genre-level sign-flip test (6 clusters, so the smallest possible two-sided p is 0.031). A pair-level test is printed only for comparison and is anti-conservative.
- Caveats that will stay attached to any result: one listener, 6 prompts, MusicGen Small only, 15 s clips, ratings are subjective, A vs D differ in both prompt wording and sampling (cannot separate them).

## Pre-registered decision rule (fixed before listening)
- **KEEP D** as default if: mean paired difference >= +0.5, AND D is worse (genre difference <= -0.33) in at most 1 genre, AND the cluster-bootstrap 90 % lower bound > 0.
- **REVERT TO A** if: mean difference <= 0, OR D is worse in >= 3 genres.
- **GENRE-SPECIFIC profiles** if: >= 2 genres favour D by >= +0.67 AND >= 2 genres favour A by >= 0.67 (checked first).
- Otherwise **INCONCLUSIVE**: no claim that D is better; the default stays as it is (currently D, chosen in round 2b on folk only) but is flagged unproven for other genres, and more seeds are needed.
  Note the current default was set on folk evidence alone; this round is its first test elsewhere.

## Round 4: generation record (done; listening NOT yet done)
36 clips generated and verified present (`samples/musicgen_xgenre/g01.wav`...`g36.wav`, template `ratings_template.csv`). Full per-clip record
(profile, exact prompt, seed, duration, time, sampling parameters, SHA prefix, automated measures) is in `.sandbox/xg/key.json`.

Generation performance (automated, 18 clips per profile, 15 s clips, CPU, one loaded model):

| | A (baseline) | D |
|---|---|---|
| Mean generation time | 74.4 s (RTF 5.0) | 74.6 s (RTF 5.0) |
| Peak RAM | 4.3 GB | 4.3 GB |

Automated proxies (not listening; weak):

| | A | D |
|---|---|---|
| Median tempo within 4 % of requested BPM (or x2, /2) | 13 / 18 | 15 / 18 |
| Pulse strength (mean) | 0.43 | 0.57 |
| Tempo-window spread, IQR BPM (mean; lower = steadier) | 11.1 | 7.8 |
| Mean RMS level | -15.8 dBFS | -20.2 dBFS (4.4 dB quieter) |
| Clips whose raw peak exceeded 0.99 and had to be scaled down | 11 / 18 | 1 / 18 |
| Silent-frame ratio | 0.000 | 0.002 |

Per genre, tempo hits (A / D, of 3): folk 1/3, 3/3; rock 3/3, 3/3; synthwave 2/3, 1/3; hip-hop/R&B 3/3, 3/3; jazz 3/3, 3/3; cinematic 1/3, 2/3.

Notes: D is consistently quieter and louder clips tend to be rated higher, so loudness works *against* D in this comparison; clips are deliberately **not** level-matched so they are what the app produces.
A's higher raw peaks (decoder overshoot, scaled down by the app's level-safety rule) is a technical difference, not a verdict on sound. These proxies did not predict listening preference in earlier rounds well enough to count as evidence.

**Result: pending your ratings.** Send `clip,score,ship` for g01-g36; I will run `python tools/score_xgenre.py ratings.csv .sandbox/xg/key.json` and apply the pre-registered rule above without changing it.

## Round 4 results (your ratings of g01-g36; analysis by `tools/score_xgenre.py`, rule applied exactly as pre-registered)

| Genre | mean A | mean D | D - A | "ship" A / D (of 3) |
|---|---|---|---|---|
| folk | 3.67 | 4.33 | +0.67 | 3 / 3 |
| jazz | 2.67 | 3.33 | +0.67 | 2 / 2 |
| cinematic | 3.00 | 3.33 | +0.33 | 2 / 2 |
| rock | 2.67 | 2.67 | 0.00 | 1 / 2 |
| hip-hop / R&B | 4.33 | 4.00 | -0.33 | 3 / 2 |
| synthwave | 4.00 | 3.33 | -0.67 | 3 / 2 |

Per pair (genre, seed): D won 7, tied 6, A won 5 (18 pairs). Ship: A 14 of 18, D 13 of 18.
**Mean paired difference +0.11; cluster bootstrap 90 % interval [-0.61, +0.78], 95 % [-0.72, +0.89]; genre-level sign-flip p = 0.81** (pair-level p = 0.89, anti-conservative).
D worse (<= -0.33) in 2 genres (hip-hop/R&B, synthwave). **Verdict under the pre-registered rule: INCONCLUSIVE.**
Not genre-specific either (only one genre favours A by >= 0.67). Not revert (mean > 0, D worse in < 3 genres).

What this means (listening results are yours; the interpretation is mine):
- **D is not shown to be better than A across genres.** The folk advantage seen in round 2b (+0.83 on 6 seeds) reappears in folk here (+0.67; D never lost in 9 folk pairs overall) but does not generalise: the other five genres average about +0.0.
- **The biggest effect is the seed, not the profile.** Inside a genre both profiles swing from 1 to 5 (e.g. rock A: 2/5/1, D: 3/2/3; hip-hop seed 12: A 5, D 2; jazz seed 11: A 2, D 5).
  So "quality consistency across seeds" is the main weakness of MusicGen Small, whatever the profile.
- Cost is identical (74.5 s per 15 s clip, 4.3 GB), so there is no performance reason to prefer either.
- **Default decision:** the rule says an inconclusive result leaves the default as is, flagged. The default is D (set on folk evidence in round 2b). It is **unproven outside folk**; reverting
  (`VIBEFLOW_PROFILE=baseline`) is equally defensible. I did not change it.
- Limits: one listener, 6 prompts x 3 seeds, 15 s clips, MusicGen Small only; A and D differ in prompt wording *and* sampling; D is 4.4 dB quieter (works against D).

---

# Round 5: multi-instrument, multi-genre, multi-mood generation

## What the current model can and cannot do (architecture, not opinion)
MusicGen Small is one 300 M-parameter model conditioned on **one text string** (T5 encoder). It has **no per-instrument, per-genre or per-mood control**: it cannot be told "bass does X, piano does Y".
The only lever is how the selections are written into that string. So nothing below can *guarantee* that an instrument is audible, and I do not claim it. The honest limits:
- Every added instrument dilutes the text signal; with more than ~5 a 300 M model will very probably drop some. (The code warns above 5; this threshold is my judgement, not a measured limit.)
- Genre blending and mood blending happen implicitly inside one generation pass. The model cannot "start calm and become energetic"; contrasting moods can only be described as a foundation with moments of contrast.
- Roles ("bass locked to the drums") are only a description. Whether the generated bass actually locks to the drums is a listening question.
- Realistic alternatives if this proves insufficient (not done, not integrated): ACE-Step instrumental (you judged `s5` cohesive; ~11-12 GB RAM, minutes per clip, warble concerns apply to vocals only),
  or a larger MusicGen (does not fit this machine: medium ships only a 7.5 GB pickle).

## Implemented vs experimental
| Item | Status |
|---|---|
| Every selected genre, mood and instrument is written into the conditioning text (all 3 prompt styles) | **Implemented, unit-tested** (200 random selections, nothing dropped) |
| `prompt_style: "arrangement"`: roles per instrument, primary genre + influences (first genre = primary), mood characteristics, dominant first mood, conflicting-mood handling, genre-adapted roles (e.g. drums in classical), optional `intensity` 1-5 | **Experimental, API-only, off by default**; effect on sound **unverified until you listen** |
| Adaptations and over-full warnings are returned (`arrangement.describe` -> adaptations / notes), never silent | Implemented, tested |
| UI controls for primary/secondary genre order, mood weights, intensity | **Not added**: they would do nothing unless the arrangement prompt proves better, and I will not ship a control the evidence doesn't support |
| Per-instrument volume/role control, independent genre sections | **Not possible with this model** |
| Instrument-presence tagger (AudioSet AST, 330 MB, BSD-3, dev tool) | **Experimental evaluation proxy**, not in the app |

Defaults are unchanged: `prompt_style` stays "sentence" (profile D). Tests pin that.

## Tagger validation (`tools/experiments/validate_tagger.py`, on the 36 cross-genre clips with known instruments)
Overall AUC 0.785 (requested instruments score higher than non-requested in 78 % of comparisons). By instrument: synthesizer 1.00, piano 0.97, saxophone 0.92, acoustic guitar 0.91, electric guitar 0.86, drums 0.83
(**usable**); bass 0.68, brass 0.71, strings 0.65 (weak); **percussion 0.24 (worse than chance: it fires on drums)**; cello, flute, harp, organ, trumpet, violin: never requested in that set, **untested**.
Absolute probabilities are low (0.02-0.18 typical) even for present instruments, so only comparisons between clips are meaningful.

## Experiment: current prompt vs arrangement prompt (24 clips, `tools/experiments/musicgen_combos.py`)
Same sampling (profile D: T 0.8, top-k 250, guidance 5.0), seed, duration (15 s) and model; **only the prompt differs**. Six selection combinations x seeds 21, 22 x 2 prompts:

| Combo | Genres | Moods | Instruments |
|---|---|---|---|
| c1_folk_5instr | folk | calm | acoustic guitar, bass, percussion, violin, flute |
| c2_rockblues_dark | rock, blues | energetic, dark | electric guitar, bass, drums |
| c3_synth_cinematic | synthwave, cinematic, electronic | dreamy, epic | synthesizer, bass, drums, strings |
| c4_jazz_rnb_3moods | jazz, r&b | mellow, romantic, sad | piano, bass, drums, saxophone |
| c5_classical_conflict | classical, cinematic | calm, energetic (conflicting) | piano, cello, drums (adapted in classical) |
| c6_funk_hiphop_5instr | funk, hip hop | happy, energetic | electric guitar, bass, drums, brass, synthesizer |

Automated (proxy only): mean tagger presence of the *usable* requested instruments, arrangement minus current, over the 12 (combo, seed) pairs where at least one usable instrument was requested:
**+0.024 on average, arrangement higher in 8 of 12 pairs**; presence is low in both (mostly 0.01-0.30). That is **not** evidence of better coherence or of instruments being audible. Cost: 68.5 s (current) vs 69.6 s (arrangement) per clip, peak 4.3 GB.

Exact prompts, adaptations, times and presence scores per clip are in `.sandbox/mc/key.json`.

## Blind listening round (pending)
`samples/musicgen_combos/m01.wav`-`m24.wav` (shuffled, strategy hidden), `ratings_template.txt` and `requested_selections.txt` (what was *requested* for each clip, so you can check whether it is audible).
For each clip give: `score` 1-5 (one coherent band playing a real piece), `ship` y/n, and `heard` = how many of the requested instruments you can actually hear.
Analysis: `python tools/score_xgenre.py ratings.txt .sandbox/mc/key.json --base=current --test=arrangement`, same pre-registered rule as round 4
(keep/adopt only if mean >= +0.5, worse in at most 1 combo, bootstrap 90 % lower bound > 0; clusters = combos, so only 6), plus the fraction of requested instruments heard per strategy.
Until then **the arrangement prompt stays experimental and off**.

## Tests
- `pytest -k "not smoke"`: 44 passed (new: selections preserved over 200 random combinations, adaptations reported, arrangement opt-in, intensity validation).
- Full `npm test` result: see the final message.
- Not done: any listening; per-instrument audibility by ear; the UI.

---

# Round 6: mood as a tone layer, not the definition of the music

Your observation: the mood overrides the music instead of tinting it. **No clips were generated this round** (you will tell me when). Everything below is code, tests and design; nothing here has been listened to.

## Root cause (found by reading the prompts, not by guessing)
The shipped default (profile D, "sentence" style) builds the prompt as `A {mood words} {genre words} instrumental piece played together by {instruments} ... {bpm}`. Examples produced by the current code:
- `A sad metal instrumental piece played together by electric guitar, bass and drums, ... 170 bpm`
- `A dark and tense disco and funk instrumental piece ...`
- `A energetic ambient instrumental piece ...` (also a grammar slip)
The mood is an **adjective glued to the genre noun and placed first**, so for a text-conditioned model "sad metal" is a different genre than "metal" and tends to pull toward slow, sad-sounding music: mood is *defining* the music, exactly the failure you described.
The older "keywords" style puts moods second (`rock, energetic, featuring ...`) but still before the instruments and tempo. The experimental `arrangement` style (used for the pending round-5 clips) is worse in one respect: its mood descriptors include
"slow-moving harmony", "driving rhythm", "bright major harmony", i.e. tempo/groove-changing words, and it opens with the mood. It is therefore **superseded**; it stays in the code only so the round-5 clips remain reproducible.
I suggest you skip rating `samples/musicgen_combos/` unless you want it as evidence of the override; the question is now better asked with the layered prompt.

## What was implemented (experimental, off by default)
`prompt_style: "layered"` / `profile: "layered"` (D sampling + layered prompt), `engine/layered.py`:
1. **Order of the text = your hierarchy**: genre -> instruments (with roles) -> rhythm/tempo -> mood -> production. The text must open with the genre; mood comes after genre, instruments and BPM.
2. **Mood is tone, expressed with tempo-neutral means**: melodic phrasing, harmonic colour, dynamics, texture. Mood words never say slower/faster, never switch key/mode, never name instruments or genres (tested).
3. **Mood is a small part of the prompt** (<= 30 words; detail is trimmed in a fixed order, mood names are always kept). Secondary moods become "undertones" of the dominant first mood.
4. **Genre protection clause**: when the mood points against the genre's energy (sad/calm/dark on metal, rock, funk, disco, techno, hip hop...) the text says the genre keeps "the full energy and drive"; when an energetic/epic/happy mood meets a restrained genre (ambient, classical, folk, ...) it says "within the restrained character of ...". Otherwise: "while keeping the {genre} groove, tempo and instruments".
5. **No unselected instruments are introduced**: the genre descriptions contain no instrument names (tested over 300 random selections); drums/bass roles adapt to the primary genre without naming other instruments.
6. **Optional `production`** (warm, spacious, dry, lo-fi, polished, live) as the last layer; validated, API-only.
7. The earlier **`intensity` setting was removed**: with one text prompt it was only a word in the prompt, i.e. a placebo "weight". Exact example: for metal + sad the layered text is
`A metal instrumental piece, heavy and aggressive. With the electric guitar playing riffs and lead lines, the bass providing a thick, heavy line and the drums providing fast, pounding patterns, all following one steady groove and the same chords. 170 bpm. Emotional tone: sad, wistful and melancholic, shown through falling melodic lines and a touch of minor harmonic colour, while keeping the full energy and drive of metal music.`

Not changed: the shipped default (profile D) still uses the mood-first sentence. I did not switch it because this is an unverified fix and you asked me not to generate yet; you can try it immediately with `VIBEFLOW_PROFILE=layered` or `profile: "layered"`.
No UI control was added.

## Can mood be *weighted*? (the "weighted conditioning" question)
- **Not with the current pipeline.** MusicGen takes one text embedding; `guidance_scale` scales the whole prompt against an empty prompt, not genre against mood. A prompt that says "mostly genre, a little mood" is wording, not a weight. Hence **no mood-intensity control is exposed**.
- **A real mechanism exists in principle**: *composable classifier-free guidance*. Transformers already runs MusicGen with a doubled batch [conditional, unconditional]. A 3-row version [unconditional, genre+instruments+tempo only, same + mood] with
  `logits = u + g*(base - u) + w*(full - base)` gives a true mood weight `w` (w = 0 reproduces the neutral music exactly). It needs a patched `prepare_inputs_for_generation`, hand-built encoder outputs and a custom logits processor (~100 lines tied to the pinned transformers 4.46)
  and costs about 1.5x the compute per step. **Not built**: I have not verified that the embedding difference between "base" and "base + mood" prompts carries a usable mood direction, and testing that needs generation. It is the recommended next step *if* the layered prompt still lets mood override the music.
- Alternative models: ACE-Step takes a caption plus structured metadata (BPM, key) and was judged cohesive by you for instrumentals, but it is heavy (~11-12 GB) and its mood handling has not been examined.

## Test plan (ready, NOT run): `tools/experiments/mood_override.py`
4 contrasting bases, each neutral and with two moods that pull against it: metal (sad, calm), disco/funk (dark, tense), ambient (energetic, epic), rock (calm, dreamy). Same seed, duration, sampling and model; current prompt vs layered prompt; seeds 31, 32.
48 clips (about 1 hour; 16 neutral references + 32 mood clips). `--dry-run` prints every prompt and generates nothing (checked).
Listener rates each *mood* clip: `style_kept` (still the same genre/instruments/tempo/groove as its neutral reference?) and `mood_felt` (clearly the requested mood?).
**Pre-registered rule** (`tools/score_mood.py`, unit-tested): ADOPT layered if style_kept improves by >= +0.5, mood_felt does not drop by more than 0.5, and the 90 % bootstrap lower bound (clusters = 4 bases) is above 0; REJECT if style_kept does not improve or mood_felt drops by >= 1.0; otherwise INCONCLUSIVE.
Automated support, only after generation: tempo drift vs the neutral reference and instrument-presence drift (AudioSet tagger, usable for guitars, piano, saxophone, synthesizer, drums). They support the listening; they do not replace it.

## Tests this round
`pytest -k "not smoke"`: 50 passed (new: priority order and mood placement, no unselected instruments and mood <= 30 words over 300 random selections, energy-protection clauses, production validation, layered not default, mood-scoring rule).

---

# Round 7: sample-adaptive composition ("Adapt a sample", Loose <-> Close to sample)

**Status: implemented and measured mechanically. NOT yet listened to. No listening pack has been created; tell me when to build one.**
Code: `engine/sample_analysis.py`, `engine/adherence.py`, `engine/dsp.py`, `engine/sample_compose.py` (+ dispatch in `engine/engines.py`, `mode: "adapt"` in `prompting.validate`, `?part=source|layers` in `server.js`, an "Adapt a sample" tab in `create.html` / `studio.js`).

## What is and is not available
| Directive item | Status |
|---|---|
| Instrumental mode | **Implemented** (MusicGen is instrumental-only; verified `vocals: false`) |
| Music with Vocals | **Not available.** The only vocal model evaluated (ACE-Step) is not integrated (warble, ~12 GB RAM, seed-dependent singing). Requests with vocals are rejected with a clear message (tested over HTTP); the UI has no vocal control |
| 0-100 % Loose<->Close control, separate from layer level / instruments / vocals | **Implemented**; separate controls: adherence, instrument chips, "Added layers level" (-24..0 dB), seed, format |
| Sample analysis | **Implemented** (numpy/scipy, no new model): tempo + beat grid + bars (4/4 assumed), key/mode, chords per bar, energy/dynamics per bar, band shares, rough sections, optional singing/speech probability only if the AudioSet tagger is already cached. **Not analysed:** melody/motifs, instrument identity, lyrics |
| Source separation / inpainting / melody conditioning | **Not used.** No separation model is installed (would be a new download); MusicGen Small has no inpainting; MusicGen Melody (5.8 GB, ~10 GB RAM) was not installed |
| Audio conditioning | **Used**: from 25 % the sample's first bars are the audio prompt for MusicGen continuation, so the new layers continue in the sample's tempo/key/timbre |
| Compare with the original | **Implemented**: each result keeps the sample window and the added layers as separate WAVs (UI shows both players) |

## How the percentage changes the actual generation (`engine/adherence.py`)
It is not a volume and not MusicGen's guidance scale. It drives measurable controls, all continuous in p:

| p | bars of the sample kept (of a 7-bar sample) | tempo hint | tempo/phase alignment | key hint | audio prompt (bars) | carve depth |
|---|---|---|---|---|---|---|
| 0 | 0 | off | 0 | off | 0 | 3.0 dB |
| 10 | 0 | on | 0.11 | off | 0 | 3.2 dB |
| 25 | 1 | on | 0.44 | 0.5 | 1 | 3.8 dB |
| 50 | 2 | on | 1.0 | 1.0 | 1 | 5.1 dB |
| 75 | 5 | on | 1.0 | 1.0 | 2 | 6.9 dB |
| 100 | **7 (all)** | on | 1.0 | 1.0 | 2 | 9.0 dB |

Retained bars are a literal copy of the sample (phrase-aligned, evenly spread); the sample is never regenerated, filtered or time-stretched. Mix = sample x retained-bar mask + layers x gain (user level under retained bars, 0 dB elsewhere), peak safety reduces the layers first.
Layers go through tempo/phase alignment (phase vocoder, within +-10 %), key shift (-4..+4 semitones, chosen by pitch-class profile match), and spectral carving where the sample is strong. Short samples keep fewer bars at low p (rounding to whole phrases), so 10 % and sometimes 25 % keep no literal bars there: the sample then acts through tempo/key hints and audio conditioning only.

## Measured on one real sample (`tools/experiments/adherence_sweep.py`, g12.wav, 119 BPM, 7 bars; piano + strings; seed 5; real MusicGen Small; outputs not published)
| p | bars kept | waveform corr. with sample | chroma sim. to sample | layers' tempo error | total time |
|---|---|---|---|---|---|
| 0 | 0/7 | -0.006 | 0.50 | 27.5 BPM (unaligned) | 92 s |
| 10 | 0/7 | 0.008 | -0.17 | 1.1 BPM | 78 s |
| 25 | 1/7 | 0.082 | 0.89 | 1.1 BPM | 72 s |
| 50 | 2/7 | 0.282 | 0.94 | 1.1 BPM | 64 s |
| 75 | 5/7 | 0.724 | 0.99 | 0.6 BPM | 55 s |
| 100 | 7/7 | **0.981** | 0.995 | 0.6 BPM | 56 s |

- Closeness to the sample rises monotonically with p (waveform correlation -0.006 -> 0.98); lower settings give a different piece. The 0.98 (not 1.0) at 100 % is the added layers; subtracting them returns the sample **exactly** (integrity residual 1e-8, and an exact unit test).
- The tempo hint alone brought the layers from 27.5 BPM off (p = 0) to 1.1 BPM off (p = 10); the audio prompt kept them there.
- **Non-monotone detail:** the chroma similarity of the *whole mix* dips at p = 10 (-0.17) because no key information is given until 25 %, and the layers' own chroma similarity falls from 0.88 (p = 25) to 0.60 (p = 100): a longer audio prompt drifts away from the key as the continuation proceeds. The key hint was not even written into the text for this sample because its key estimate was ambiguous (margin < 0.02).
- No clipping, 4.3 GB peak RAM, no vocals. Rhythmic overlap between layers and sample was about -0.1 at every level (low duplication by that proxy).
- The DSP alignment (stretch, phase, key shift) was exercised on synthetic data in unit tests; on this real sample the model's continuation was already within 3 % of the tempo, so the DSP path barely fired. It is therefore **unproven on real model output**.

## Analysis validation
Tempo: on the 36 real MusicGen clips of round 4 (requested BPM known) the estimate was within 4 % in 30, off by an octave in 2, wrong in 4 (some of the 4 may be the model not following the requested BPM). Key and chord estimates have no ground truth here and are coarse (the sample above had an ambiguous key).

## Honest limits
- **Musical quality, "do the added instruments complement the sample", and whether the layers sound like an accompaniment rather than a continuation are unverified: nothing has been listened to.**
- Layers generated by audio prompting continue the *mix*, so they may imitate the sample's instruments (duplication) instead of complementing them; the onset-correlation proxy is low but is only a proxy.
- Layers are mono and centred; at 100 % the sample keeps its own sample rate and channels. Sample window is the first <= 30 s.
- 4/4 is assumed for bars; rubato or ambient material gives unreliable bar grids (then 2-second segments are used and a flag is set).
- Instrument selection is text conditioning only: no guarantee any instrument is audible.

## Tests
`tests/test_adapt_dsp.py` (analysis on synthetic songs, stretch/pitch/carve/envelope) and `tests/test_adapt_compose.py` (mapping monotone and gradual; retained bars grow; **100 % returns the original exactly after subtracting the layers**; 0 % contains no source; similarity rises with p; alignment never worsens tempo; layer level independent of p; peak safety does not scale the sample; no-vocals validation) are offline (fake generator).
Real-model HTTP test: upload, adapt at 100 %, vocals refused, `source`/`layers` parts downloadable, delete removes all three files.

---

# Round 8: Listening Pack (adherence levels, vocals, instruments and genres)

Full method and numbers: **docs/LISTENING.md**. In short (measurements only; no listening yet):
- Adherence levels Low / Medium / High / Strict (15 / 50 / 75 / 100 %) were run through the real adapt pipeline on four samples (full mix, instrumental stem, voice stem from a Sample-page recording; an Editor sample), 2 seeds each, three times (baseline, intermediate, final). Final pack: every automatic check passes, levels distinct and in the same order on every sample type.
- Fixed from the measurements: seed-dependent output level (loudness match), whole-take attenuation by the mixer's peak safety (now a local limiter on the layers), an unstable tempo score (now periodicity similarity), a flawed Strict check. A tempo-retry fix was built, measured (2 of 8 helped, +28 % time) and left opt-in.
- Lead / backing vocals: stereo-position split of the Demucs voice stem; on known mixes lead SI-SDR 6.6 dB, backing 4.8 dB, planted harmony found; centre-panned backing cannot be separated.
- CLAP tagging: single instruments 87.5 % top 3; mixtures poor (both of 2 instruments in top 5: 18 %; 3: 0 %); genre 53 % top 3, mixed genres never both. A label calibration was tried and rejected.
- Selected instruments drive the layers at Low; from Medium the sample's timbre mostly overrides them on a full mix. Guidance 10: inconclusive on 2 seeds.
