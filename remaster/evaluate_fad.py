"""Frechet audio distance of damaged and restored clips against a reference set of clean music.

  python -m remaster.evaluate_fad make  --data <music> --rir <unseen rooms> --ckpt <ckpt> --dir runs/fad      # writes wav sets
  <embedding python> scripts/embed_fad.py runs/fad                                                           # writes *.npy per set
  python -m remaster.evaluate_fad score --dir runs/fad --out docs/evidence/fad

Sets: `reference` (clean clips from the training split), and for the same held-out clips `clean`, `damaged`
(reverb from rooms outside the training set) and `restored`. The distance of `clean` to the reference is the
floor: what two different samples of clean music measure.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf

from .data import list_tracks, load_audio, split_of
from .degrade import SR, RIRBank, apply_reverb, match_level


def frechet(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.asarray(linalg.sqrtm(s1.dot(s2))).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def make(a):
    from .infer import load_model, restore
    model, bank = load_model(a.ckpt), RIRBank(a.rir)
    files = list_tracks(a.data)
    n = int(a.seconds * SR)
    for name, split, count in (("reference", "train", a.reference), ("test", "test", a.clips)):
        fs = [f for f in files if split_of(f) == split]
        fs = [fs[i] for i in np.random.default_rng(97).permutation(len(fs))]
        k = 0
        for fi, f in enumerate(fs):
            if k >= count:
                break
            try:
                x = load_audio(f)
            except Exception:
                continue
            if x.shape[1] < n + 2 * SR:
                continue
            ctx = x[:, : n + 2 * SR]
            clean = match_level(ctx[:, -n:], -20.0)
            if name == "reference":
                sets = dict(reference=clean)
            else:
                rng = np.random.default_rng([97, fi])
                rev, _ = apply_reverb(ctx, rng, bank, drr_db=float(rng.uniform(-3, 9)), p_real=1.0)
                deg = match_level(rev[:, -n:], -20.0)
                sets = dict(clean=clean, damaged=deg, restored=match_level(restore(model, deg), -20.0))
            for s, y in sets.items():
                os.makedirs(os.path.join(a.dir, s), exist_ok=True)
                sf.write(os.path.join(a.dir, s, f"{k:04d}.wav"), np.clip(y, -1, 1).T, SR, subtype="PCM_16")
            k += 1
            if k % 50 == 0:
                print(name, k, flush=True)


def score(a):
    out = {}
    for emb in ("clap", "vggish"):
        ref = os.path.join(a.dir, f"reference.{emb}.npy")
        if not os.path.exists(ref):
            continue
        r = np.load(ref)
        out[emb] = dict(dim=int(r.shape[1]), n_reference=int(len(r)))
        for s in ("clean", "damaged", "restored"):
            e = np.load(os.path.join(a.dir, f"{s}.{emb}.npy"))
            out[emb][s] = frechet(r, e)
            out[emb]["n_" + s] = int(len(e))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "fad.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["make", "score"])
    p.add_argument("--data", nargs="+")
    p.add_argument("--rir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--clips", type=int, default=250)
    p.add_argument("--reference", type=int, default=1500)
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
