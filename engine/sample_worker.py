"""Live-transcription worker for the Sample page (JSON lines on stdin/stdout), exits after VIBEFLOW_SAMPLE_IDLE_S idle seconds.

stdin  {"id": n, "cmd": "live", "pcm": "<path>", "sr": 48000, "ch": 2, "from": 12.0, "to": 30.0, "lang": null}
stdout {"event": "ready"} then {"event": "result", "id": n, "language", "language_probability", "segments": [{start,end,text}]}
       or {"event": "error", "id": n, "message"}
The PCM file is raw little-endian float32, interleaved, appended to by the server while the user records."""
import json
import math
import os
import queue
import sys
import threading

import numpy as np
from scipy.signal import resample_poly

IDLE_S = float(os.environ.get("VIBEFLOW_SAMPLE_IDLE_S", "240"))


def _out(**kw):
    print(json.dumps(kw), flush=True)


def read_window(path, sr, ch, a, b):
    frame = 4 * ch
    start, n = int(a * sr), int((b - a) * sr)
    with open(path, "rb") as f:
        f.seek(start * frame)
        raw = f.read(n * frame)
    x = np.frombuffer(raw[: len(raw) // frame * frame], dtype="<f4").reshape(-1, ch).mean(axis=1)
    g = math.gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def serve():
    from . import cli, sampling
    lines: "queue.Queue[str | None]" = queue.Queue()

    def reader():
        for line in sys.stdin:
            lines.put(line)
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()
    _out(event="ready")
    while True:
        try:
            line = lines.get(timeout=IDLE_S)
        except queue.Empty:
            _out(event="idle_exit")
            return 0
        if line is None:
            return 0
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            if req.get("cmd") != "live":
                raise ValueError("Unknown command.")
            x = read_window(req["pcm"], int(req["sr"]), int(req["ch"]), float(req["from"]), float(req["to"]))
            if len(x) < 8000 or float(np.sqrt(np.mean(x ** 2))) < 1e-4:        # under half a second, or digital silence
                _out(event="result", id=rid, language=None, language_probability=0.0, segments=[])
                continue
            r = sampling.transcribe(x, language=req.get("lang"), offset=float(req["from"]), threads=4, vad=True)
            _out(event="result", id=rid, **r)
        except Exception as e:  # noqa: BLE001 - a failed window must not kill the worker
            _out(event="error", id=rid, kind=type(e).__name__, message=cli.scrub(str(e)))
