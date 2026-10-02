"""Capture what four limiters do to the same two seconds of an unmastered mix, for the limiter figure.

  python -m remaster.trace_limiters --musdb <musdb18hq/test> --xl <musdb_XL_ratio> --song "<name>" --out docs/evidence/loudness/limiter_trace.npz

Every system is driven to the loudness and true peak of the Ozone 9 Maximizer version (musdb-XL), as in E29.
Stored per system: the output, the smooth gain fitted in 10 ms windows, and the remainder.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf
from scipy import signal

from .analysis import SR, loudest_window, true_peak_db
from .evaluate_loudness import LIMITERS, align, to_target
from .master import loudness


def fit_gain(y, x, win=441, hop=110):
    w = np.hanning(win).astype(np.float64)
    num = signal.fftconvolve((y * x).sum(0).astype(np.float64), w, mode="same")[::hop]
    den = signal.fftconvolve((x * x).sum(0).astype(np.float64), w, mode="same")[::hop] + 1e-9
    return np.interp(np.arange(x.shape[1]), np.arange(len(num)) * hop, num / den)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--musdb", required=True)
    p.add_argument("--xl", required=True)
    p.add_argument("--song", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    x = sf.read(os.path.join(a.musdb, a.song, "mixture.wav"), dtype="float32", always_2d=True)[0].T
    r = np.load(os.path.join(a.xl, a.song + ".npy"))
    r = r.T if r.ndim == 2 and r.shape[0] != x.shape[0] else r
    m = min(x.shape[1], r.shape[-1])
    oz = (x[:, :m] * r[..., :m]).astype(np.float32)
    s0, s1 = loudest_window(x[:, :m], seconds=20.0)
    x, oz = x[:, s0:s1], oz[:, s0:s1]
    target, ceil = loudness(oz), 10 ** (true_peak_db(oz) / 20)
    outs = dict(ozone=oz)
    for name, key in (("ours", "v2_slow250_clip2"), ("matchering", "matchering"), ("clip", "clip")):
        outs[name] = to_target(x, LIMITERS[key], target, ceil, true_peak_safe=True)
    # two seconds around the largest input peak
    c = int(np.argmax(np.abs(x).max(0)))
    lo = int(np.clip(c - SR, 0, x.shape[1] - 2 * SR))
    save = dict(sr=SR, input=x[:, lo:lo + 2 * SR], peak_index=c - lo, target_lufs=target, ceiling_db=20 * np.log10(ceil))
    for name, y in outs.items():
        y, xa = align(y, x)
        g = fit_gain(y, xa)
        save[name] = y[:, lo:lo + 2 * SR]
        save[name + "_gain_db"] = 20 * np.log10(np.clip(g[lo:lo + 2 * SR], 1e-4, None)).astype(np.float32)
        save[name + "_residual"] = (y - g * xa)[:, lo:lo + 2 * SR].astype(np.float32)
    np.savez_compressed(a.out, **save)
    print("wrote", a.out, {k: (v.shape if hasattr(v, "shape") else v) for k, v in save.items() if k in ("input", "target_lufs", "ceiling_db")})


if __name__ == "__main__":
    main()
