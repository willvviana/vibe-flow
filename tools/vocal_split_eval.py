"""Measure lead / backing / instrumental separation on mixes whose parts are KNOWN (real audio, real Demucs, real split; nothing simulated
in the measurement itself; only the mix is constructed so that the right answer exists).

  python tools/vocal_split_eval.py <sung.wav> <instrumental.wav> <out_dir>

Parts:   lead       = Demucs vocal stem of <sung.wav> (a sung ACE-Step clip), separated twice to remove band bleed, mono, centre-panned
         backing    = the lead pitch-shifted +4 and +7 semitones (a major-triad harmony), 25 ms late, -6 dB, panned left / right
         instrument = <instrumental.wav> (stereo)
Scenarios: A wide backing (normal studio practice), B backing panned to the centre (the method's known blind spot), C lead only (no backing).
Pipeline under test = the Sample page's own: sampling.separate (Demucs) -> vocal_layers.split.
Metric: SI-SDR in dB of each estimated part against its known part (higher is better; 0 dB = as much error as signal). For context the
'unprocessed' SI-SDR (the mix itself used as the estimate) is reported too, and the harmony detector must find intervals 4 and 7 in A."""
import json
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine import dsp, sampling, vocal_layers  # noqa: E402

SR = 44100


def si_sdr(est, ref):
    est, ref = est.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    t, e = a * ref, est - a * ref
    return round(10 * np.log10((np.dot(t, t) + 1e-12) / (np.dot(e, e) + 1e-12)), 2)


def pan(x, p):  # p in [-1, 1], constant power
    th = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)], axis=1).astype(np.float32)


def main(sung, inst_path, out):
    os.makedirs(out, exist_ok=True)
    lead_dir, lead2_dir = os.path.join(out, "lead_source"), os.path.join(out, "lead_source_pass2")
    if not os.path.exists(os.path.join(lead2_dir, "vocals.wav")):        # two Demucs passes: the second removes most band bleed from the lead
        sampling.separate(sung, lead_dir)
        sampling.separate(os.path.join(lead_dir, "vocals.wav"), lead2_dir)
    lead = sampling.load_audio(os.path.join(lead2_dir, "vocals.wav"), SR).mean(axis=1)
    inst = sampling.load_audio(inst_path, SR)
    n = min(len(lead), len(inst))
    lead, inst = lead[:n], inst[:n] * 0.7
    d = int(0.025 * SR)
    h4, h7 = dsp.shift_samples(dsp.pitch_shift(lead, 4), d) * 0.5, dsp.shift_samples(dsp.pitch_shift(lead, 7), d) * 0.5
    scen = {"A_wide_backing": (pan(lead, 0), pan(h4, -0.8) + pan(h7, 0.8)),
            "B_centre_backing": (pan(lead, 0), pan(h4, 0) + pan(h7, 0)),
            "C_lead_only": (pan(lead, 0), np.zeros((n, 2), np.float32))}
    rows = {}
    for name, (L, B) in scen.items():
        mix = L + B + inst
        pk = float(np.abs(mix).max())
        if pk > 0.99:
            L, B, I, mix = L * 0.99 / pk, B * 0.99 / pk, inst * 0.99 / pk, mix * 0.99 / pk
        else:
            I = inst
        d_ = os.path.join(out, name)
        os.makedirs(d_, exist_ok=True)
        sf.write(os.path.join(d_, "mix.wav"), mix, SR, subtype="FLOAT")
        st = sampling.separate(os.path.join(d_, "mix.wav"), d_)
        v = sampling.load_audio(os.path.join(d_, "vocals.wav"), SR)[:n] / st["gain"]
        ins = sampling.load_audio(os.path.join(d_, "instrumental.wav"), SR)[:n] / st["gain"]
        lead_e, back_e, info = vocal_layers.split(v, SR)
        _, _, rep = vocal_layers.analyse(v, ins, SR)
        sf.write(os.path.join(d_, "lead.wav"), lead_e, SR, subtype="FLOAT")
        sf.write(os.path.join(d_, "backing.wav"), back_e, SR, subtype="FLOAT")
        r = {"instrumental_si_sdr": si_sdr(ins, I), "instrumental_unprocessed": si_sdr(mix, I),
             "vocals_total_si_sdr": si_sdr(v, L + B), "lead_si_sdr": si_sdr(lead_e, L), "lead_unprocessed": si_sdr(mix, L),
             "lead_if_whole_vocal_stem": si_sdr(v, L), "backing_share_true": round(float((B ** 2).sum() / ((L ** 2).sum() + (B ** 2).sum() + 1e-12)), 4),
             "backing_share_est": info["backing_share"], "harmony": rep.get("harmony"), "lead_metrics": rep["lead"]}
        if (B ** 2).sum() > 0:
            r.update(backing_si_sdr=si_sdr(back_e, B), backing_unprocessed=si_sdr(mix, B))
        rows[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k not in ("lead_metrics",)}), flush=True)
    with open(os.path.join(out, "vocal_split_eval.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    main(*sys.argv[1:4])
