"""Command line entry used by the Node server. Every command prints JSON lines on stdout.

  python -m engine.cli status
  python -m engine.cli install|repair|verify <model-id>
  python -m engine.cli generate <job.json>
  python -m engine.cli serve                 (long-lived worker; see engine/worker.py)
  python -m engine.cli probe <audio-file>
  python -m engine.cli convert <in.wav> <flac|wav> <out> [24] | import-track <wav> <title>
  python -m engine.cli sample-status | sample-install | sample-serve | sample-process <wav> <dir> | sample-pcm2wav ... | sample-reference <wav> <out>
  python -m engine.cli library | rename <id> <title> | delete <id> | export <id> <fmt> [--normalize] | cleanup [min-age-s]
"""
import json
import os
import re
import sys

from . import audio, deps, hardware, library, models, registry

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scrub(msg: str) -> str:
    """Never show filesystem layout to the browser: replace app/home/drive paths in messages."""
    msg = msg.replace(_ROOT, "<app>").replace(os.path.expanduser("~"), "~")
    return re.sub(r"[A-Za-z]:[\\/][^\s'\"]+", "<path>", msg)


def out(**kw):
    print(json.dumps(kw), flush=True)


def main(argv):
    if not argv:
        out(event="error", message="No command.")
        return 2
    cmd, args = argv[0], argv[1:]
    prog = lambda **k: out(event="progress", **k)  # noqa: E731
    try:
        if cmd == "status":
            out(event="result", hardware=hardware.detect(), models=models.status(), unsupported=registry.UNSUPPORTED,
                formats=audio.available_formats(), dependencies=deps.check())
        elif cmd in ("install", "repair"):
            (models.install if cmd == "install" else models.repair)(args[0], prog)
            out(event="result", ok=True)
        elif cmd == "verify":
            out(event="result", **models.verify(args[0], prog))
        elif cmd == "generate":
            from . import engines
            with open(args[0], encoding="utf-8") as f:
                job = json.load(f)
            out(event="result", track=engines.get(job.get("engine", "musicgen")).generate(job, prog))
        elif cmd == "serve":
            from . import worker
            return worker.serve()
        elif cmd == "probe":
            out(event="result", info=audio.validate_reference(args[0]))
        elif cmd == "library":
            out(event="result", tracks=library.list_tracks())
        elif cmd == "rename":
            out(event="result", track=library.rename(args[0], args[1]))
        elif cmd == "delete":
            library.delete(args[0])
            out(event="result", ok=True)
        elif cmd == "export":
            out(event="result", file=os.path.basename(library.export(args[0], args[1], normalize="--normalize" in args)))
        elif cmd == "convert":                      # editor export: WAV -> FLAC / WAV (server-side encoders via libsndfile)
            import soundfile as sf
            src, fmt, dst = args[0], args[1].lower(), args[2]
            if fmt not in ("flac", "wav"):
                raise ValueError("Unsupported format.")
            x, sr = sf.read(src, dtype="float32", always_2d=True)
            sf.write(dst, x, sr, format=fmt.upper(), subtype="PCM_24" if fmt == "flac" and len(args) > 3 and args[3] == "24" else "PCM_16")
            out(event="result", ok=True)
        elif cmd == "import-track":                 # editor "Save to Your Songs": register a rendered WAV in the library
            out(event="result", track=library.import_wav(args[0], args[1]))
        elif cmd == "sample-status":
            from . import sampling_models
            out(event="result", models=sampling_models.status())
        elif cmd == "sample-install":
            from . import sampling_models
            out(event="result", models=sampling_models.install())
        elif cmd == "sample-loopback-info":
            from . import loopback
            out(event="result", **loopback.info())
        elif cmd == "sample-loopback":
            from . import loopback
            loopback.record(args[0] if args else None)
        elif cmd == "sample-serve":
            from . import sample_worker
            return sample_worker.serve()
        elif cmd == "sample-pcm2wav":              # <pcm f32 interleaved> <sr> <channels> <out.wav>: finalise a live capture bit-exactly
            import numpy as np
            import soundfile as sf
            pcm, sr, ch, dst = args[0], int(args[1]), int(args[2]), args[3]
            with sf.SoundFile(dst, "w", samplerate=sr, channels=ch, subtype="FLOAT", format="WAV") as f, open(pcm, "rb") as src:
                step = 1 << 20
                frame = 4 * ch
                while True:
                    raw = src.read(step * frame)
                    if not raw:
                        break
                    f.write(np.frombuffer(raw[: len(raw) // frame * frame], dtype="<f4").reshape(-1, ch))
            from . import sampling
            sampling.write_peaks(dst, os.path.join(os.path.dirname(dst), "peaks.full.json"))
            info = sf.info(dst)
            out(event="result", duration=round(info.duration, 3), sample_rate=info.samplerate, channels=info.channels)
        elif cmd == "sample-process":              # <full.wav> <out_dir>: separate + tag + transcribe, writes <out_dir>/result.json
            from . import sampling
            res = sampling.process(args[0], args[1], prog)
            with open(os.path.join(args[1], "result.json"), "w", encoding="utf-8") as f:
                json.dump(res, f)
            out(event="result", ok=True)
        elif cmd == "sample-reference":            # <wav> <out.wav>: mono 32 kHz 16-bit copy that Create accepts as a reference (<= 600 s)
            from . import sampling
            import soundfile as sf
            x = sampling.load_audio(args[0], 32000, mono=True)
            if len(x) / 32000 > audio.MAX_REF_SECONDS:
                raise ValueError(f"Create accepts references up to {audio.MAX_REF_SECONDS // 60} minutes; this audio is {len(x) / 32000 / 60:.1f} minutes.")
            sf.write(args[1], x, 32000, subtype="PCM_16")
            out(event="result", ok=True)
        elif cmd == "listening-run":               # <spec.json>: Listening Pack (all adherence levels on each sample, measured; see engine/listening.py)
            from . import listening
            with open(args[0], encoding="utf-8") as f:
                spec = json.load(f)
            rep = listening.run(spec, prog)
            out(event="result", run=os.path.basename(spec["out_dir"]), overall=rep["overall"])
        elif cmd == "listening-score":             # <run_dir> <ratings.json>: unblind the human ratings of a pack
            from . import listening
            with open(args[1], encoding="utf-8") as f:
                ratings = json.load(f)
            out(event="result", summary=listening.score(args[0], ratings))
        elif cmd == "cleanup":
            out(event="result", removed=library.cleanup(float(args[0]) if args else 30))
        else:
            out(event="error", message=f"Unknown command '{cmd}'.")
            return 2
    except Exception as e:  # noqa: BLE001 - surface every failure as structured output
        out(event="error", kind=type(e).__name__, message=scrub(str(e)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
