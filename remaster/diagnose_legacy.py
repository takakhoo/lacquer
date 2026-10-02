"""Measure why the thesis pipeline (Curriculum_Tokenize_Master) could not make music sound better.

  python -m remaster.diagnose_legacy --data data/raw/fma_medium --out docs/evidence/legacy

Each check reuses the original functions/parameter ranges from Curriculum_Tokenize_Master/demastering.py.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import fftconvolve, freqz, iirpeak, lfilter

from .data import list_tracks, load_audio, split_of
from .losses import log_spec_dist, si_sdr

SR = 44100
# ranges copied from demastering.py BASE
OLD = dict(eq=dict(fc=(500, 2000), Q=(0.9, 1.2), gain=(-0.8, 0.8)), gain=dict(db=(-1, 1)),
           echo=dict(delay=(0.1, 0.2), att=(0.2, 0.3)), reverb=dict(decay=(0.2, 0.4), ir=(0.05, 0.15)),
           comp=dict(thr=(-12, -8), ratio=(1.5, 2.5), makeup=(0, 0.5)))


def old_reverb(x, sr, decay, ir_len):
    t = np.linspace(0, ir_len / sr, ir_len)
    ir = np.exp(-decay * t)
    return fftconvolve(x, ir, mode="same").astype(np.float32), ir


def old_eq(x, sr, fc, Q, gain_db):
    b, a = iirpeak(fc / (sr / 2), Q)
    b = b * 10 ** (gain_db / 20)
    return lfilter(b, a, x), (b, a)


def metrics(deg, clean):
    d, c = torch.from_numpy(deg)[None, None].float(), torch.from_numpy(clean)[None, None].float()
    return dict(si_sdr=si_sdr(d, c).item(), lsd=log_spec_dist(d, c).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"][: a.tracks]
    clips = []
    for f in files:
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] >= 12 * SR:
            clips.append(x[:, 5 * SR: 11 * SR])
    rng = np.random.default_rng(0)
    res = {}

    # 1. the "reverb": a 50-150 ms one-sided exponential with decay 0.2-0.4 /s is an almost flat boxcar
    ir_len = int(0.1 * SR)
    _, ir = old_reverb(np.zeros(SR, dtype=np.float32), SR, 0.3, ir_len)
    w, h = freqz(ir, worN=8192, fs=SR)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(np.arange(ir_len) / SR * 1000, ir); ax[0].set_ylim(0, 1.1)
    ax[0].set(title="legacy 'reverb' impulse response (decay=0.3, 100 ms)", xlabel="ms", ylabel="amplitude")
    ax[1].semilogx(w[1:], 20 * np.log10(np.abs(h[1:]) + 1e-9)); ax[1].set(title="its frequency response", xlabel="Hz", ylabel="dB", xlim=(20, 20000))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "legacy_reverb_ir.png"), dpi=120); plt.close(fig)
    peaks, lp = [], []
    for x in clips:
        m = x.mean(0)
        y, _ = old_reverb(m, SR, rng.uniform(*OLD["reverb"]["decay"]), int(rng.uniform(*OLD["reverb"]["ir"]) * SR))
        peaks.append(float(np.abs(y).max() / (np.abs(m).max() + 1e-9)))
        spec = np.abs(np.fft.rfft(y)) ** 2
        fr = np.fft.rfftfreq(len(y), 1 / SR)
        lp.append(float(10 * np.log10(spec[fr > 1000].sum() / spec.sum() + 1e-12)))
    res["legacy_reverb"] = dict(dc_gain_db=float(20 * np.log10(ir.sum())), response_at_1khz_db=float(20 * np.log10(np.abs(h[np.argmin(np.abs(w - 1000))]))),
                                median_peak_growth_x=float(np.median(peaks)), median_energy_above_1khz_db=float(np.median(lp)),
                                note="output peaks far exceed 1.0, so sf.write to 16-bit PCM hard-clipped every 'reverb' example")

    # 2. gain / EQ ranges sit near the just-noticeable difference
    gains = rng.uniform(*OLD["gain"]["db"], 10000)
    res["legacy_gain"] = dict(range_db=OLD["gain"]["db"], mean_abs_db=float(np.abs(gains).mean()))
    res["legacy_eq"] = dict(range_db=OLD["eq"]["gain"], note="iirpeak is a band-PASS resonator, so apply_eq returned the band-passed signal scaled by <1 dB instead of a boost/cut")
    b, a_ = iirpeak(1000 / (SR / 2), 1.0)
    w2, h2 = freqz(b, a_, worN=8192, fs=SR)
    res["legacy_eq"]["response_db_at_100hz_1khz_10khz"] = [float(20 * np.log10(np.abs(h2[np.argmin(np.abs(w2 - f))]) + 1e-9)) for f in (100, 1000, 10000)]

    # 3. EnCodec round trip of CLEAN audio: the best the token model could ever output
    try:
        from encodec import EncodecModel
        codec = EncodecModel.encodec_model_48khz(); codec.set_target_bandwidth(24.0); codec.eval()
        import torchaudio
        rt, gain_tok = [], []
        for x in clips[:10]:
            xt = torchaudio.functional.resample(torch.from_numpy(x), SR, 48000)[None]
            with torch.no_grad():
                fr = codec.encode(xt)
                y = codec.decode(fr)[..., : xt.shape[-1]]
                fr2 = codec.encode(xt * 10 ** (1 / 20))
            rt.append(dict(si_sdr=si_sdr(y, xt).item(), lsd=log_spec_dist(y, xt).item()))
            same = np.mean([(f1[0] == f2[0]).float().mean().item() for f1, f2 in zip(fr, fr2)])
            gain_tok.append(float(same))
        res["encodec_ceiling"] = dict(si_sdr_db=float(np.mean([r["si_sdr"] for r in rt])), lsd_db=float(np.mean([r["lsd"] for r in rt])), n=len(rt))
        res["gain_in_tokens"] = dict(fraction_tokens_unchanged_after_plus_1db=float(np.mean(gain_tok)),
                                     note="the 48 kHz EnCodec normalizes each chunk and stores level in a separate scale, so tokens barely encode gain")
    except ImportError:
        res["encodec_ceiling"] = "encodec not installed"

    # 4. how far the legacy degradations move the signal, for comparison with the codec ceiling
    deg = {k: [] for k in ("gain", "eq", "echo")}
    for x in clips:
        m = x.mean(0)
        deg["gain"].append(metrics(m * 10 ** (rng.uniform(*OLD["gain"]["db"]) / 20), m))
        d = int(SR * rng.uniform(*OLD["echo"]["delay"])); e = m.copy(); e[d:] += m[:-d] * rng.uniform(*OLD["echo"]["att"])
        deg["echo"].append(metrics(e, m))
    res["legacy_degradation_distance"] = {k: dict(si_sdr_db=float(np.mean([r["si_sdr"] for r in v])), lsd_db=float(np.mean([r["lsd"] for r in v]))) for k, v in deg.items() if v}

    json.dump(res, open(os.path.join(a.out, "legacy_findings.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
