"""Make the trial folders for the listening test (see remaster.listening).

  # restoration trials: needs held-out music, unseen-room impulse responses and the restoration checkpoint
  python -m remaster.build_listening_trials restoration --data data/raw/fma_small --rir data/raw/air/wav --ckpt remaster/checkpoints/best.pt --out listening_src

  # mastering trials: needs MUSDB18-HQ test mixes; --xl adds the Ozone-limited versions from musdb-XL
  python -m remaster.build_listening_trials mastering --musdb <musdb18hq/test> --xl <musdb_XL_ratio> --out listening_src

Restoration trials have a reference (the clean clip). Mastering trials have none: every version is a way of
finishing the same unmastered mix, and the listening page loudness-matches them.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import soundfile as sf

from .analysis import SR, loudest_window
from .data import list_tracks, load_audio, split_of
from .degrade import RIRBank, apply_echo, apply_reverb, match_level


def write(d, name, x):
    os.makedirs(d, exist_ok=True)
    sf.write(os.path.join(d, name + ".wav"), np.clip(x, -1, 1).T, SR, subtype="PCM_24")


def restoration(a):
    from .infer import enhance_auto, load_model
    models = dict(mix=load_model(a.ckpt))
    bank = RIRBank(a.rir)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(71).permutation(len(files))]
    n, plan, k = int(a.seconds * SR), ["reverb"] * a.reverb + ["echo"] * a.echo + ["clip"] * a.clip, 0
    for fi, f in enumerate(files):
        if k >= len(plan):
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < 30 * SR:
            continue
        ctx = x[:, 10 * SR: 12 * SR + n + 10 * SR][:, : n + 2 * SR]
        rng, kind = np.random.default_rng([71, fi]), plan[k]
        y = ctx
        if kind == "reverb":
            y, _ = apply_reverb(y, rng, bank, drr_db=float(rng.uniform(0, 6)), p_real=1.0)
        elif kind == "echo":
            y, _ = apply_echo(y, rng)
        y = y[:, -n:]
        if kind == "clip":
            thr = np.percentile(np.abs(y), float(rng.uniform(93, 98)))
            y = np.clip(y, -thr, thr)
        clean, deg = match_level(ctx[:, -n:], -20.0), match_level(y.astype(np.float32), -20.0)
        out, _ = enhance_auto(models, deg, use_stems=False, do_master=False, ride=0)
        d = os.path.join(a.out, f"{kind}_{k:02d}")
        write(d, "reference", clean); write(d, "damaged", deg); write(d, "lacquer", match_level(out, -20.0))
        k += 1
        print(d, flush=True)


def mastering(a):
    from .evaluate_loudness import sys_loudnorm
    from .mastering import master_track
    for k, d in enumerate(sorted(glob.glob(os.path.join(a.musdb, "*", "")))[: a.songs]):
        name = os.path.basename(d.rstrip("/"))
        x = sf.read(os.path.join(d, "mixture.wav"), dtype="float32", always_2d=True)[0].T
        s0, s1 = loudest_window(x, seconds=a.seconds)
        mix = x[:, s0:s1]
        out = os.path.join(a.out, f"master_{k:02d}")
        write(out, "unmastered", mix)
        write(out, "lacquer", master_track(mix, profile="loud")[0])
        write(out, "loudnorm", sys_loudnorm(mix, -9.0, -1.0))
        if a.xl and os.path.exists(os.path.join(a.xl, name + ".npy")):
            r = np.load(os.path.join(a.xl, name + ".npy"))
            r = r.T if r.ndim == 2 and r.shape[0] != x.shape[0] else r
            m = min(x.shape[1], r.shape[-1])
            write(out, "ozone", (x[:, :m] * r[..., :m])[:, s0:s1])
        print(out, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["restoration", "mastering"])
    p.add_argument("--out", required=True)
    p.add_argument("--data", nargs="+")
    p.add_argument("--rir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--musdb", default=None)
    p.add_argument("--xl", default=None)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--reverb", type=int, default=8)
    p.add_argument("--echo", type=int, default=4)
    p.add_argument("--clip", type=int, default=4)
    p.add_argument("--songs", type=int, default=8)
    a = p.parse_args()
    restoration(a) if a.cmd == "restoration" else mastering(a)


if __name__ == "__main__":
    main()
