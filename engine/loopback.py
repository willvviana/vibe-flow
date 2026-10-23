"""System-output capture (WASAPI loopback, Windows): records everything the default playback device is playing, with no screen/tab share.

  python -m engine.cli sample-loopback-info            -> {"event":"result","available":bool,"device":..,"sample_rate":..,"reason":..}
  python -m engine.cli sample-loopback <out.pcm>       -> appends raw little-endian float32 stereo frames to <out.pcm>

stdin lines: pause | resume | stop.   stdout events: started {sr, ch, device} / level {pl, pr, clip, wave:[[min,max]..], frames, paused} / result {frames}.
WASAPI loopback delivers nothing while the device is silent, so silence is written to keep the recording on the real time line.
Only audio the machine is playing is captured (not the microphone). The data goes to the file the server named and nowhere else."""
import json
import queue
import sys
import threading
import time

import numpy as np


def _out(**kw):
    print(json.dumps(kw), flush=True)


def _default_loopback(p):
    import pyaudiowpatch as pa
    host = p.get_host_api_info_by_type(pa.paWASAPI)
    out = p.get_device_info_by_index(host["defaultOutputDevice"])
    if out.get("isLoopbackDevice"):
        return out
    for lb in p.get_loopback_device_info_generator():
        if out["name"] in lb["name"]:
            return lb
    raise RuntimeError("No loopback device matches the default playback device.")


def info():
    if sys.platform != "win32":
        return {"available": False, "reason": "System audio capture without a screen share is implemented for Windows (WASAPI loopback) only."}
    try:
        import pyaudiowpatch as pa
    except ImportError:
        return {"available": False, "reason": "PyAudioWPatch is not installed. Run: npm run setup"}
    p = pa.PyAudio()
    try:
        d = _default_loopback(p)
        return {"available": True, "device": d["name"].replace(" [Loopback]", ""), "sample_rate": int(d["defaultSampleRate"]), "channels": int(d["maxInputChannels"])}
    except Exception as e:  # noqa: BLE001 - report why instead of failing the page
        return {"available": False, "reason": f"No playback device could be opened for loopback capture ({e})."}
    finally:
        p.terminate()


def record(path=None, idle_s=300.0):
    """Warm start: open the device but do NOT capture. Print ready, then wait for "start <file>" on stdin (so recording begins within milliseconds
    of the user pressing Start). Nothing is read from the device until that command arrives; the process exits if it never comes."""
    import pyaudiowpatch as pa
    p = pa.PyAudio()
    d = _default_loopback(p)
    sr, ch_dev = int(d["defaultSampleRate"]), int(d["maxInputChannels"])
    q: "queue.Queue[bytes]" = queue.Queue()
    cmds: "queue.Queue[str]" = queue.Queue()

    def reader():
        for line in sys.stdin:
            cmds.put(line.strip())
        cmds.put("stop")                                      # the server went away: stop cleanly
    threading.Thread(target=reader, daemon=True).start()

    stream = p.open(format=pa.paFloat32, channels=ch_dev, rate=sr, input=True, input_device_index=d["index"], frames_per_buffer=1024, start=False,
                    stream_callback=lambda data, n, t, s: (q.put(data), (None, pa.paContinue))[1])
    device = d["name"].replace(" [Loopback]", "")
    _out(event="ready", sr=sr, ch=2, device=device)
    deadline = time.monotonic() + idle_s
    while path is None:
        try:
            c = cmds.get(timeout=max(0.1, deadline - time.monotonic()))
        except queue.Empty:
            c = "idle"
        if c.startswith("start "):
            path = c[6:].strip()
        elif c in ("stop", "idle"):
            stream.close()
            p.terminate()
            _out(event="idle_exit" if c == "idle" else "result", frames=0, sr=sr)
            return
    f = open(path, "ab")
    stream.start_stream()
    _out(event="started", sr=sr, ch=2, device=device)
    frames, paused, paused_at, paused_total = 0, False, 0.0, 0.0
    t0 = last_data = last_rep = time.monotonic()
    pl = pr = 0.0
    clip = False
    wave, wacc, wn = [], [0.0, 0.0], 0

    def stereo(raw):
        x = np.frombuffer(raw, dtype="<f4").reshape(-1, ch_dev)
        if ch_dev == 1:
            return np.repeat(x, 2, axis=1)
        return np.ascontiguousarray(x[:, :2])

    def write(x):
        nonlocal frames, pl, pr, clip, wacc, wn
        f.write(x.astype("<f4").tobytes())
        f.flush()
        frames += len(x)
        pl, pr = max(pl, float(np.abs(x[:, 0]).max())), max(pr, float(np.abs(x[:, 1]).max()))
        if pl >= 0.999 or pr >= 0.999:
            clip = True
        m = x.mean(axis=1)
        for i in range(0, len(m), 4096):
            seg = m[i:i + 4096]
            wave.append([round(float(seg.min()), 3), round(float(seg.max()), 3)])

    try:
        while True:
            try:
                c = cmds.get(timeout=0.04)
                if c == "stop":
                    break
                if c == "pause" and not paused:
                    paused, paused_at = True, time.monotonic()
                elif c == "resume" and paused:
                    paused = False
                    paused_total += time.monotonic() - paused_at
                    last_data = time.monotonic()
            except queue.Empty:
                pass
            got = False
            while True:
                try:
                    raw = q.get_nowait()
                except queue.Empty:
                    break
                got = True
                if not paused and raw:
                    write(stereo(raw))
            now = time.monotonic()
            if got:
                last_data = now
            elif not paused and now - last_data > 0.2:        # silent output: no callbacks arrive, so fill the gap with zeros
                expected = int((now - t0 - paused_total) * sr)
                if expected > frames:
                    write(np.zeros((expected - frames, 2), dtype="<f4"))
            if now - last_rep >= 0.12:
                _out(event="level", pl=round(pl, 4), pr=round(pr, 4), clip=clip, wave=wave, frames=frames, paused=paused)
                pl = pr = 0.0
                clip = False
                wave = []
                last_rep = now
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        f.close()
    _out(event="result", frames=frames, sr=sr)
