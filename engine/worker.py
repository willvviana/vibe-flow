"""Long-lived generation worker: keeps the loaded model in RAM between jobs, exits after an idle period.

Protocol (JSON lines): stdin  {"cmd": "generate", "job": "<job.json path>"}
                       stdout {"event": "progress"|"result"|"error", ...}  (same events as `engine.cli generate`)
Trade-off: a warm worker holds ~4 GB of RAM while idle, so it quits after VIBEFLOW_IDLE_S seconds (default 300);
the server starts a new one on demand. Cancelling a job kills the worker (the next job reloads the model)."""
import json
import os
import queue
import sys
import threading

IDLE_S = float(os.environ.get("VIBEFLOW_IDLE_S", "300"))


def _out(**kw):
    print(json.dumps(kw), flush=True)


def serve():
    from . import cli, engines, generate
    lines: "queue.Queue[str | None]" = queue.Queue()

    def reader():                       # stdin is read on a thread so the main loop can time out when idle
        for line in sys.stdin:
            lines.put(line)
        lines.put(None)                 # parent closed the pipe
    threading.Thread(target=reader, daemon=True).start()

    _out(event="ready")
    while True:
        try:
            line = lines.get(timeout=IDLE_S)
        except queue.Empty:
            generate.release_cache()
            _out(event="idle_exit")
            return 0
        if line is None:
            return 0
        try:
            req = json.loads(line)
            if req.get("cmd") != "generate":
                raise ValueError("Unknown command.")
            with open(req["job"], encoding="utf-8") as f:
                job = json.load(f)
            meta = engines.get(job.get("engine", "musicgen")).generate(job, lambda **k: _out(event="progress", **k), keep_loaded=True)
            _out(event="result", track=meta)
        except Exception as e:  # noqa: BLE001 - a failed job must not take the worker (and its cached model) down
            _out(event="error", kind=type(e).__name__, message=cli.scrub(str(e)))
            generate.gc.collect()
