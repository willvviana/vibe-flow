# Listening pack

Everything here was generated locally. **I could not listen to any of it**, so the numbers in `docs/EVALUATION.md`
are only objective proxies (silence, clipping, tempo stability, and a speech-recognition check of whether the lyrics are
audible). Please listen and tell me what you hear.

Lyrics used (two-line test, and the four-line version for 30 s clips):

```
[Verse 1]  Streetlights hum a quiet tune / I am walking home beneath the moon
[Chorus]   Hold me close, do not let me go / Carry me where the soft winds blow
```
Caption: *acoustic pop ballad, female vocals, soft fingerpicked acoustic guitar, warm, intimate, slow tempo*

## ACE-Step 1.5 (`ace/`) - the vocal candidate

| File | Setup | What the ASR check heard |
|---|---|---|
| `acestep-vocal-test.wav` | turbo, no LM, 10 s (the clip you rejected) | "you" |
| `c3_dit_only_10s.wav` | turbo, no LM, 10 s | "you" |
| `c4_lm_codes_10s.wav` | turbo + LM codes, 10 s | "She likes how the queen likes you" (vocal-like sound, wrong words) |
| `c1_dit_only_30s.wav` | turbo, no LM, 30 s | "You" |
| `c2_lm_codes_30s.wav` | turbo + LM codes, 30 s | "Yeah" |
| `c5_instrumental_30s_codes.wav` | turbo + LM codes, instrumental, 30 s | (no lyrics) |
| `s1_sft_codes_10s_32steps.wav` | **sft + LM codes, 32 steps, 10 s** | "Street lights humming a quiet tune I am walking home beneath them" |
| `s2_sft_codes_30s_32steps.wav` | **sft + LM codes, 32 steps, 30 s** | all four lines recognised |
| `s3_sft_codes_30s_32steps_seed11.wav` | sft + codes, 30 s, another seed | all four lines recognised, minor word errors |
| `s4_sft_codes_30s_16steps_seed7.wav` | sft + codes, 16 steps (faster) | all four lines recognised, a few more errors |
| `s5_sft_instrumental_30s_32steps.wav` | sft + codes, **instrumental**, band only | (no lyrics) - judge whether the arrangement is one band |

Listen for: is it a *sung* melody or speech? Do the guitar, bass and drums feel like one band on one pulse? Any
warble, harsh noise, sudden cut-off? Does the voice sit inside the mix?

## MusicGen Small (`musicgen/`) - the instrumental engine

Same idea, 15 s, 2 seeds each, requested 72 BPM, acoustic guitar + bass + percussion:
`A_` = current keyword prompt, `B_` = sentence prompt, `C_` = sentence + temperature 0.8 + top-k 100,
`D_` = sentence + temperature 0.8 + guidance 5. Which ones sound like a band playing together?

## Round 2 packs (blind: file names carry no information)

- `musicgen_ab/` - 12 clips (A vs D, seeds 3-8, shuffled). Rate each 1-5 ("sounds like one band playing a real piece") and say if you would ship it
  in `ratings_template.csv`. I unblind with `tools/score_ab.py`.
- `ace_warble/` - 10 ACE-Step clips (5 settings x 2 seeds, shuffled). Rate warble 1-5 (5 = severe) and whether it sings. Some clips may not sing at all
  (the speech check heard no words in 4 of the 5 seed-8 clips).

Note: the round-1 ACE-Step clips above were generated with random seeds (my bug), so "seed 7/11" in earlier notes was wrong.

- `ace_warble_r3/` - 6 ACE-Step clips (16 steps, seeds 7/9/10, original vs "no vibrato" caption). Rate warble 1-5 and whether it sings; one clip repeats w03.

- `musicgen_xgenre/` - 36 blind clips (6 genres x 3 seeds x 2 profiles, 15 s). Rate each 1-5 (one coherent band playing a real piece) and ship y/n in `ratings_template.csv`. Do not try to identify the profile.

- `musicgen_combos/` - 24 blind clips (6 multi-instrument/genre/mood selections x 2 seeds x 2 prompt strategies, 15 s). `requested_selections.txt` lists what was requested per clip. Rate in `ratings_template.txt`: score 1-5, ship y/n, and `heard` = how many requested instruments you can hear.
