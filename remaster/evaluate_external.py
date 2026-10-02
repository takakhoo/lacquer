"""Score dereverberation tools that run outside this package (UVR / MDX community models) on the clips of
evaluate_baselines.

  python -m remaster.evaluate_external dump  --data <music> --rir <IRs> --dir <work>      # writes in/NN.wav, clean/NN.wav
  ... run each tool on <work>/in, writing <work>/<name>/NN*.wav ...
  python -m remaster.evaluate_external score --dir <work> --out docs/evidence/baselines --names a b c
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch

from .data import list_tracks, load_audio, split_of
from .degrade import SR, RIRBank, apply_reverb, match_level
from .losses import log_spec_dist, si_sdr


def clips(data, rir, tracks, seconds, p_real=0.5):
    """Same files, seeds and reverbs as evaluate_baselines."""
    bank = RIRBank(rir)
    files = [f for f in list_tracks(data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(13).permutation(len(files))]
    n, k = int(seconds * SR), 0
    for fi, f in enumerate(files):
        if k >= tracks:
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < n + 2 * SR:
            continue
        ctx = x[:, : n + 2 * SR]
        rng = np.random.default_rng([13, fi])
        rev, log = apply_reverb(ctx, rng, bank, drr_db=float(rng.uniform(-3, 9)), p_real=p_real, p_algo=0.34)
        yield k, match_level(ctx[:, -n:], -20.0), match_level(rev[:, -n:], -20.0), log
        k += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["dump", "score"])
    p.add_argument("--data", nargs="+")
    p.add_argument("--rir", default=None)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--names", nargs="*", default=[])
    p.add_argument("--tracks", type=int, default=20)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--p-real", type=float, default=0.5)
    p.add_argument("--json", default="external.json")
    a = p.parse_args()
    if a.cmd == "dump":
        for d in ("in", "clean"):
            os.makedirs(os.path.join(a.dir, d), exist_ok=True)
        for k, clean, deg, log in clips(a.data, a.rir, a.tracks, a.seconds, a.p_real):
            sf.write(os.path.join(a.dir, "in", f"{k:02d}.wav"), deg.T, SR, subtype="FLOAT")
            sf.write(os.path.join(a.dir, "clean", f"{k:02d}.wav"), clean.T, SR, subtype="FLOAT")
            print(k, log["kind"], flush=True)
        return
    summ = {}
    for name in a.names:
        rows = []
        for cf in sorted(glob.glob(os.path.join(a.dir, "clean", "*.wav"))):
            k = os.path.basename(cf)[:2]
            hit = sorted(glob.glob(os.path.join(a.dir, name, f"{k}*")))
            if not hit:
                continue
            clean, y = sf.read(cf, dtype="float32")[0].T, sf.read(hit[0], dtype="float32")[0].T
            n = min(clean.shape[1], y.shape[1])
            y = match_level(y[:, :n], -20.0)
            c, o = torch.from_numpy(clean[:, :n])[None], torch.from_numpy(y)[None]
            rows.append(dict(clip=k, sisdr=si_sdr(o, c).item(), lsd=log_spec_dist(o, c).item()))
        summ[name] = dict(n=len(rows), sisdr=float(np.mean([r["sisdr"] for r in rows])), lsd=float(np.mean([r["lsd"] for r in rows])), rows=rows)
        print(name, summ[name]["n"], round(summ[name]["sisdr"], 2), round(summ[name]["lsd"], 2))
    if a.out:
        json.dump(summ, open(os.path.join(a.out, a.json), "w"), indent=1)


if __name__ == "__main__":
    main()
