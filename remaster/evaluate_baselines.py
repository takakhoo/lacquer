"""Head-to-head on reverberant full mixes: classical WPE, the released vocal dereverb model, and Lacquer.

  python -m remaster.evaluate_baselines --ckpt remaster/checkpoints/best.pt --vocal-ckpt remaster/checkpoints/vocal_dereverb.pt \\
      --data <music> --rir <IRs> --out docs/evidence/baselines

WPE (weighted prediction error, Nakatani et al. 2010) is the standard blind dereverberation method from speech:
it predicts each STFT frame's late reverberation from earlier frames and subtracts it. It needs no training.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import list_tracks, load_audio, split_of
from .degrade import SR, RIRBank, apply_reverb, match_level
from .infer import load_model, restore
from .losses import log_spec_dist, si_sdr


def wpe_dereverb(x, taps=30, delay=2, iterations=3, size=2048, shift=512):
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe
    Y = stft(x, size=size, shift=shift).transpose(2, 0, 1)        # [F, channels, T]
    Z = wpe(Y, taps=taps, delay=delay, iterations=iterations, statistics_mode="full")
    y = istft(Z.transpose(1, 2, 0), size=size, shift=shift)
    out = np.zeros_like(x)
    n = min(x.shape[1], y.shape[1])
    out[:, :n] = y[:, :n]
    return out.astype(np.float32)


def score(y, clean):
    a, b = torch.from_numpy(y)[None], torch.from_numpy(clean)[None]
    return dict(sisdr=si_sdr(a, b).item(), lsd=log_spec_dist(a, b).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocal-ckpt", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--rir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=20)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--p-real", type=float, default=0.5, help="1.0 = every clip uses a measured room from --rir")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ours, vocal = load_model(a.ckpt), load_model(a.vocal_ckpt)
    bank = RIRBank(a.rir)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(13).permutation(len(files))]
    rows, n = [], int(a.seconds * SR)
    for fi, f in enumerate(files):
        if len(rows) >= a.tracks:
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < n + 2 * SR:
            continue
        ctx = x[:, : n + 2 * SR]
        rng = np.random.default_rng([13, fi])
        rev, log = apply_reverb(ctx, rng, bank, drr_db=float(rng.uniform(-3, 9)), p_real=a.p_real, p_algo=0.34)
        clean, deg = match_level(ctx[:, -n:], -20.0), match_level(rev[:, -n:], -20.0)
        row = dict(file=os.path.basename(f), reverb=log, input=score(deg, clean), wpe=score(match_level(wpe_dereverb(deg), -20.0), clean),
                   vocal_model=score(restore(vocal, deg), clean), lacquer=score(restore(ours, deg), clean))
        rows.append(row)
        print(len(rows), log["kind"], {k: round(row[k]["sisdr"], 1) for k in ("input", "wpe", "vocal_model", "lacquer")}, flush=True)
    summ = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in ("sisdr", "lsd")} for k in ("input", "wpe", "vocal_model", "lacquer")}
    summ["lacquer_beats_wpe"] = float(np.mean([r["lacquer"]["sisdr"] > r["wpe"]["sisdr"] for r in rows]))
    json.dump(dict(n=len(rows), summary=summ, rows=rows), open(os.path.join(a.out, "baselines.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
