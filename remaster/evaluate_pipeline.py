"""End-to-end check on held-out clips: which stage fixes what.

  python -m remaster.evaluate_pipeline --ckpt remaster/checkpoints/best.pt --data data/raw/fma_small --rir data/raw/mit_ir --out docs/evidence/pipeline

Compares the degraded input, DSP de-echo alone, the network alone, and both, against the clean clip.
Mastering is left out because it changes level and tone on purpose.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import list_tracks, load_audio, split_of
from .deecho import deecho
from .degrade import SR, RIRBank, degrade, match_level
from .infer import load_model, restore
from .losses import log_spec_dist, si_sdr

CONDS = ("reverb", "echo", "reverb+echo", "clip", "noise", "identity")


def score(y, clean):
    n = min(y.shape[1], clean.shape[1])
    a, b = torch.from_numpy(y[:, :n])[None], torch.from_numpy(clean[:, :n])[None]
    return dict(sisdr=si_sdr(a, b).item(), lsd=log_spec_dist(a, b).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--rir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=16)
    p.add_argument("--seconds", type=float, default=15.0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = load_model(a.ckpt)
    bank = RIRBank(a.rir)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(7).permutation(len(files))[: a.tracks]]
    rows = {c: {k: [] for k in ("input", "deecho", "network", "both")} for c in CONDS}
    for fi, f in enumerate(files):
        try:
            x = load_audio(f)
        except Exception:
            continue
        n = int(a.seconds * SR)
        if x.shape[1] < n:
            continue
        clean = match_level(x[:, :n], -20.0)
        for ci, cond in enumerate(CONDS):
            rng = np.random.default_rng([7, fi, ci])
            deg = clean if cond == "identity" else degrade(clean, rng, bank, effects=cond.split("+"))[0]
            de, _ = deecho(deg)
            rows[cond]["input"].append(score(deg, clean))
            rows[cond]["deecho"].append(score(de, clean))
            rows[cond]["network"].append(score(restore(model, deg), clean))
            rows[cond]["both"].append(score(restore(model, de), clean))
        print(f"{fi + 1}/{len(files)}", flush=True)
    summ = {c: {k: {m: float(np.mean([r[m] for r in v])) for m in ("sisdr", "lsd")} for k, v in d.items() if v} for c, d in rows.items()}
    json.dump(dict(ckpt=a.ckpt, n=len(rows["reverb"]["input"]), summary=summ), open(os.path.join(a.out, "pipeline.json"), "w"), indent=1)
    lines = ["| condition | input | de-echo only | network only | de-echo + network |", "|---|---:|---:|---:|---:|"]
    for c, d in summ.items():
        lines.append(f"| {c} | " + " | ".join(f"{d[k]['sisdr']:.1f} dB / {d[k]['lsd']:.1f}" for k in ("input", "deecho", "network", "both")) + " |")
    open(os.path.join(a.out, "pipeline.md"), "w").write("SI-SDR (dB, higher is better) / log-spectral distance (dB, lower is better)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
