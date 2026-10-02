"""Getting an unmastered mix to a loudness target: how much damage does each limiter do on the way?

  python -m remaster.evaluate_loudness --musdb data/raw/musdb18hq/test --out docs/evidence/loudness

Every system is driven until its output measures the target loudness, so the comparison is at equal loudness.
Reported: true-peak overshoot above the ceiling, SI-SDR and log-spectral distance against the input (level
matched), and the peak-to-loudness ratio that is left.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from .analysis import SR, loudness_stats, true_peak_db
from .degrade import match_level
from .losses import log_spec_dist, si_sdr
from .master import limiter, limiter_v2, loudness


def sys_clip(x, ceil):
    return np.clip(x, -ceil, ceil)


def sys_ours(x, ceil):
    return limiter(x, SR, ceiling_db=20 * np.log10(ceil))[0]


def sys_pedalboard(x, ceil):
    from pedalboard import Limiter, Pedalboard
    return Pedalboard([Limiter(threshold_db=20 * np.log10(ceil), release_ms=100.0)])(x, SR)


def sys_matchering(x, ceil):
    from matchering import Config
    from matchering.limiter import limit
    cfg = Config()
    return (limit((x / ceil).T.astype(np.float64), cfg).T * ceil).astype(np.float32)


def sys_alimiter(x, ceil):
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        sf.write(a, x.T, SR, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-af", f"alimiter=limit={ceil:.5f}:attack=5:release=50:level=disabled", "-c:a", "pcm_f32le", b], check=True)
        return sf.read(b, dtype="float32")[0].T


def sys_loudnorm(x, target, ceiling_db):
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        sf.write(a, x.T, SR, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-af", f"loudnorm=I={target}:TP={ceiling_db}:LRA=11", "-ar", str(SR), "-c:a", "pcm_f32le", b], check=True)
        return sf.read(b, dtype="float32")[0].T


def v2(clip_db, **kw):
    return lambda x, ceil: limiter_v2(x, SR, ceiling_db=20 * np.log10(ceil), clip_db=clip_db, **kw)[0]


LIMITERS = dict(clip=sys_clip, ffmpeg_alimiter=sys_alimiter, pedalboard=sys_pedalboard, matchering=sys_matchering, ours_v1=sys_ours,
                v2_clip0=v2(0.0), v2_clip1=v2(1.0), v2_clip2=v2(2.0), v2_clip3=v2(3.0),
                v2_noslow_clip0=v2(0.0, slow_ms=0), v2_noslow_clip1=v2(1.0, slow_ms=0), v2_noslow_clip2=v2(2.0, slow_ms=0),
                v2_fast_clip1=v2(1.0, slow_ms=0, fast_ms=60.0, lookahead_ms=5.0), v2_fast_clip2=v2(2.0, slow_ms=0, fast_ms=60.0, lookahead_ms=5.0),
                v2_slow250_clip2=v2(2.0, slow_ms=250.0, fast_ms=25.0))


def to_target(x, fn, target, ceil, rounds=8, true_peak_safe=False):
    """Raise the drive into the limiter until the output measures the target loudness.

    With true_peak_safe, the limiter's own ceiling is also lowered until the 4x-oversampled peak of its output
    respects the requested ceiling, so that all systems are compared at equal loudness and equal true peak.
    """
    drive = target - loudness(x)
    y, c = x, ceil
    for _ in range(rounds):
        y = fn(x * 10 ** (drive / 20), c)
        err = target - loudness(y)
        over = true_peak_db(y) - 20 * np.log10(ceil)
        if abs(err) < 0.1 and (not true_peak_safe or over < 0.05):
            break
        drive += err
        if true_peak_safe and over > 0.05:
            c *= 10 ** (-over / 20)
    return y


def align(y, x, max_lag=4096):
    """Remove a processing delay: shift y by the lag that maximizes its correlation with x."""
    a, b = y.mean(0)[: 10 * SR], x.mean(0)[: 10 * SR]
    n = min(len(a), len(b))
    c = signal.correlate(a[:n], b[:n], mode="full", method="fft")
    lag = int(np.argmax(c[n - 1 - max_lag:n + max_lag]) - max_lag)
    if lag > 0:
        y = y[:, lag:]
    elif lag < 0:
        y = np.pad(y, ((0, 0), (-lag, 0)))
    n = min(x.shape[1], y.shape[1])
    return y[:, :n], x[:, :n]


def gain_and_distortion(y, x, win=441, hop=110):
    """Split what a limiter did into a smooth gain (10 ms windows) and what that gain cannot explain.

    Returns (distortion relative to the signal in dB, spread of the gain in dB). Clipping scores badly on the
    first, a slow pumping limiter on the second.
    """
    w = np.hanning(win).astype(np.float64)
    num = signal.fftconvolve((y * x).sum(0).astype(np.float64), w, mode="same")[::hop]
    den = signal.fftconvolve((x * x).sum(0).astype(np.float64), w, mode="same")[::hop] + 1e-9
    g = np.interp(np.arange(x.shape[1]), np.arange(len(num)) * hop, num / den)
    r = y - g * x
    active = den > den.max() * 1e-4
    gdb = 20 * np.log10(np.clip(num / den, 1e-3, None))[active]
    return float(10 * np.log10((r ** 2).sum() / ((g * x) ** 2).sum() + 1e-12)), float(np.percentile(gdb, 95) - np.percentile(gdb, 5))


def measure(y, x, target, ceiling_db):
    tp, lufs = true_peak_db(y), loudness(y)
    y, x = align(y, x)
    a, b = torch.from_numpy(match_level(y, -20.0))[None], torch.from_numpy(match_level(x, -20.0))[None]
    dist, pump = gain_and_distortion(y, x)
    return dict(lufs=lufs, lufs_err=abs(lufs - target), overshoot_db=max(0.0, tp - ceiling_db), sisdr=si_sdr(a, b).item(), lsd=log_spec_dist(a, b).item(),
                plr=tp - lufs, distortion_db=dist, gain_spread_db=pump)


def one(job):
    d, targets, ceiling, seconds, tp_safe = job
    ceil, n, rows = 10 ** (ceiling / 20), int(seconds * SR), []
    x = sf.read(os.path.join(d, "mixture.wav"), dtype="float32", always_2d=True)[0].T
    st = loudness_stats(x)["short_term"]
    start = int(np.clip(np.argmax(st) * 0.1 - seconds / 2 + 1.5, 0, max(0, x.shape[1] / SR - seconds)) * SR)
    x = x[:, start:start + n]
    for target in targets:
        row = dict(song=os.path.basename(d.rstrip("/")), target=target, input_lufs=loudness(x), input_plr=true_peak_db(x) - loudness(x), systems={})
        for name, fn in LIMITERS.items():
            row["systems"][name] = measure(to_target(x, fn, target, ceil, true_peak_safe=tp_safe), x, target, ceiling)
        row["systems"]["ffmpeg_loudnorm"] = measure(sys_loudnorm(x, target, ceiling), x, target, ceiling)
        rows.append(row)
    return rows


def one_xl(job):
    """Same excerpt, but the target is what a commercial limiter did: iZotope Ozone 9 Maximizer, from musdb-XL.

    Every other system is driven to the loudness of the Ozone output and held to its true peak.
    """
    d, xl_dir, seconds = job
    n = int(seconds * SR)
    name = os.path.basename(d.rstrip("/"))
    x = sf.read(os.path.join(d, "mixture.wav"), dtype="float32", always_2d=True)[0].T
    ratio = np.load(os.path.join(xl_dir, name + ".npy"))
    ratio = ratio.T if ratio.ndim == 2 and ratio.shape[0] != x.shape[0] else ratio
    m = min(x.shape[1], ratio.shape[-1])
    oz = (x[:, :m] * ratio[..., :m]).astype(np.float32)
    st = loudness_stats(x)["short_term"]
    start = int(np.clip(np.argmax(st) * 0.1 - seconds / 2 + 1.5, 0, max(0, m / SR - seconds)) * SR)
    x, oz = x[:, start:start + n], oz[:, start:start + n]
    target, ceiling = loudness(oz), true_peak_db(oz)
    ceil = 10 ** (ceiling / 20)
    row = dict(song=name, target=target, ceiling_db=ceiling, input_lufs=loudness(x), input_plr=true_peak_db(x) - loudness(x), systems={})
    row["systems"]["ozone9_maximizer"] = measure(oz, x, target, ceiling)
    for k in ("matchering", "ffmpeg_alimiter", "v2_noslow_clip1", "v2_slow250_clip2", "v2_clip3"):
        row["systems"][k] = measure(to_target(x, LIMITERS[k], target, ceil, true_peak_safe=True), x, target, ceiling)
    return [row]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--musdb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--songs", type=int, default=50)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--targets", nargs="+", type=float, default=[-14.0, -9.0])
    p.add_argument("--ceiling", type=float, default=-1.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--true-peak-safe", action="store_true")
    p.add_argument("--name", default="loudness.json")
    p.add_argument("--xl", default=None, help="musdb-XL gain-ratio folder: compare against the Ozone-limited versions at their loudness")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    jobs = [(d, a.targets, a.ceiling, a.seconds, a.true_peak_safe) for d in sorted(glob.glob(os.path.join(a.musdb, "*", "")))[: a.songs]]
    if a.xl:
        jobs = [(d, a.xl, a.seconds) for d in sorted(glob.glob(os.path.join(a.musdb, "*", "")))[: a.songs]]
    with Pool(a.workers) as pool:
        for rs in pool.imap_unordered(one_xl if a.xl else one, jobs):
            rows += rs
            for row in rs:
                print(row["song"][:22], row["target"], {k: (round(v["distortion_db"], 1), round(v["gain_spread_db"], 1), round(v["overshoot_db"], 2), round(v["lufs"], 1)) for k, v in row["systems"].items()}, flush=True)
    summ = {}
    if a.xl:
        summ["ozone"] = {k: {m: float(np.mean([r["systems"][k][m] for r in rows])) for m in rows[0]["systems"][k]} for k in rows[0]["systems"]}
        summ["ozone"]["input"] = dict(lufs=float(np.mean([r["input_lufs"] for r in rows])), plr=float(np.mean([r["input_plr"] for r in rows])),
                                      target=float(np.mean([r["target"] for r in rows])), ceiling_db=float(np.mean([r["ceiling_db"] for r in rows])))
        a.targets = []
    for target in a.targets:
        rs = [r for r in rows if r["target"] == target]
        summ[str(target)] = {k: {m: float(np.mean([r["systems"][k][m] for r in rs])) for m in rs[0]["systems"][k]} for k in rs[0]["systems"]}
        summ[str(target)]["input"] = dict(lufs=float(np.mean([r["input_lufs"] for r in rs])), plr=float(np.mean([r["input_plr"] for r in rs])))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, a.name), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
