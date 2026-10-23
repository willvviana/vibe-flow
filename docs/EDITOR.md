# Vibe Flow Editor

Multi-track audio editor at `/editor` (`editor.html`). Runs in the browser (Web Audio API) with the local Vibe Flow server for the library, samples and FLAC encoding.
Everything is non-destructive: a project is a list of tracks and clip *references* with editing instructions; the original audio is stored once (IndexedDB) and never modified.

## Architecture
| File | Role |
|---|---|
| `editor/core.js` | project model, clip operations (split/trim/move/copy/duplicate/delete/silence regions), undo/redo history, snapping, crossfade overlaps, ducking envelope, shortcut matching. Pure functions, tested in Node |
| `editor/dsp.js` | real offline DSP: phase-vocoder time stretch (identity phase locking), pitch shift, spectral-subtraction noise reduction, silence detection, normalisation, **look-ahead brick-wall limiter**, reverse, loops, WAV encoder, reverb impulse response. Tested in Node |
| `editor/worker.js` | runs the heavy buffer-changing DSP off the UI thread |
| `editor/engine.js` | ONE Web Audio graph builder used for live playback and for offline export, so what you hear is what is rendered |
| `editor/storage.js` | IndexedDB: audio assets, projects, crash-recovery slot |
| `editor/ui.js`, `ui-panels.js`, `ui-export.js` | timeline, interactions, panels, dialogs, export, projects, shortcut settings |
| `editor/selftest.js` | development self-test that drives the real UI and verifies audio numerically (see Testing) |
| `vendor/lame.min.js` | lamejs 1.2.1 MP3 encoder (**LGPL-3.0**; vendored, runs in the browser) |

Server additions (`server.js`, `engine/`): `POST /api/editor/encode` (WAV to FLAC via libsndfile), `POST /api/editor/save` (render to Your Songs), `GET/POST/DELETE /api/samples`, `POST /api/samples/:id/use` (hand a saved sample to Create without re-uploading).

## What every control does (all implemented and exercised by tests)
- **Transport/playhead:** play/pause (position kept), click or drag on the ruler to scrub, snap to clip edges (toggle), Home/End, arrows; the needle follows the audio clock.
- **Timeline:** unlimited tracks; add/rename/mute/solo/lock/delete; track volume and pan; drag clips in time and between tracks; trim by edges; split at the playhead; copy/cut/paste/duplicate/delete; Shift+click multi-select (edits apply to all); overlap clips; automatic equal-power crossfade where clips on one track overlap; zoom (Ctrl+wheel, slider, Fit); scroll both ways; resizable library/timeline panels; region selection (Shift+drag on the ruler).
- **Clip properties:** volume, fade in/out, pan, mute, normalise, reverse, loop count, noise reduction, voice enhance, pitch (independent of speed), speed (with or without pitch), remove silence, 5-band EQ, compressor, limiter, reverb, delay/echo, filter (low/high/band-pass, notch). Effects update live while playing (toggling reverb/delay/voice-enhance/fades restarts the graph from the playhead).
- **Ducking:** "Under voice" slider (music tracks duck under tracks of type Voice) and per-track ducking with trigger tracks, reduction, threshold, attack and release. The envelope is computed from the actual audio, identically for preview and export.
- **Library:** Imports, Your Songs (from Create), Samples (from "Send to Create"), Recordings, Voices; preview, `+` button, drag-and-drop onto a lane; local file import (also drag files onto the timeline); microphone recording onto a Voice track.
- **Projects:** autosave (recovery slot), Save (Ctrl+S), open/delete projects, restore prompt after an interrupted session, warning before leaving with unsaved changes.
- **Export:** WAV 16/24/32-float, FLAC, MP3 (128-320 kbps); 44.1/48/96 kHz; complete mix, region, selected clips, each track separately, voice tracks only, music tracks only; Download / Save to Your Songs / Send to Create as Sample. Progress, error handling, peak and clipping report.
- **Send to Create:** renders, stores a WAV sample, saves the project, opens `create.html?sample=<id>`; Create loads it as the sample input (Adapt a sample tab) with no second upload. Also available from Create's "saved sample" dropdown.
- **Shortcuts (configurable in Settings):** Space, Ctrl+Z, Ctrl+Shift+Z, Ctrl+C/V/X, Ctrl+A, Delete, Ctrl+D, Ctrl+S, Ctrl+E, S, M, Home, End, arrows (Shift = 1 s), Ctrl+wheel, Shift+click. Ignored while typing in fields.

## Limits and things NOT implemented (stated plainly)
- **Pitch correction** (auto-tune to a key): not implemented.
- **Isolating vocals/instrumentals from a mixed recording:** not possible (no source-separation model). "Voice tracks only" / "Music tracks only" export the tracks you marked as Voice / Music.
- **Live preview vs export:** the preview uses WebAudio's compressor-based limiter (compensated for its automatic make-up gain, but not look-ahead); the export applies a true look-ahead brick-wall limiter at -1 dBFS. They can differ slightly on transient peaks.
- Engine rate is 48 kHz: 96 kHz exports are upsampled; 44.1 kHz exports are resampled by the browser. "Sample-accurate" here means times are kept as 64-bit seconds and the browser schedules with sub-sample precision; this was verified to millisecond tolerance, not sample-by-sample.
- Pitch shifting is not formant-preserving; the phase vocoder smears transients at extreme speeds (usable ~0.5x-2x); noise reduction can leave faint watery artefacts at high amounts. **Not judged by ear.**
- MP3 uses lamejs (LGPL-3.0). FLAC and the library need the local server (`npm start`). OGG/AAC export is not offered.
- Decoded audio and processed copies live in memory: very long projects need RAM. Undo history is per session (not stored). No marquee selection, no track reordering, desktop-first layout (limited touch support).
- Tested only in one Chromium build (the embedded browser). Firefox/Safari untested. Microphone recording was tested with a synthetic stream, not a physical microphone.

## Testing
- `node --test tests/editor-core.test.js tests/editor-dsp.test.js` (also part of `npm test`): model/history/snapping/ducking logic and the DSP (stretch keeps pitch and level, pitch shift, noise reduction, silence regions, limiter ceiling, WAV encoding).
- `tests/server.test.js`: editor endpoints (FLAC magic bytes, byte-exact save, samples CRUD, hand-off id accepted by Create's generate path, origin and traversal checks).
- Browser self-test (88 checks): open `/editor`, then `await new Promise(r=>{const s=document.createElement('script');s.src='editor/selftest.js';s.onload=r;document.head.appendChild(s)}); await EditorSelfTest.run()`.
  It imports synthetic audio, drives real pointer/keyboard events, renders the timeline offline and analyses the samples (positions, levels, fades, spectra, ducking depth, effects, undo/redo, save/reopen, MP3/FLAC/WAV export decoded and matched to the render, samples hand-off, recording path, panel resizing).
