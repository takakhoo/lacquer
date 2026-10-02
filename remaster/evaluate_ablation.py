"""Leave one stage out: what each repair stage contributes on clips with one fault or all three.

  python -m remaster.evaluate_ablation --ckpt remaster/checkpoints/best.pt --data data/raw/fma_small --rir data/raw/air/wav --out docs/evidence/ablation

Faults: room reverb from rooms outside the training set, a discrete echo, hard clipping, and all three stacked
in that order. The repair half of the pipeline runs complete and with each stage removed. No stems, level riding
or mastering, so the output stays comparable with the clean clip.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import list_tracks, load_audio, split_of
from .degrade import SR, RIRBank, apply_echo, apply_reverb, match_level
from .infer import enhance_auto, load_model
from .losses import log_spec_dist, si_sdr

CONDS = ("reverb", "echo", "clip", "all three", "clean")
VARIANTS = dict(full={}, no_declip=dict(do_declip=False), no_deecho=dict(do_deecho=False), no_network=dict(restore_strength=0.0))


def score(y, clean):
    a, b = torch.from_numpy(np.ascontiguousarray(y))[None], torch.from_numpy(np.ascontiguousarray(clean))[None]
    return dict(sisdr=si_sdr(a, b).item(), lsd=log_spec_dist(a, b).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--rir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=24)
    p.add_argument("--seconds", type=float, default=12.0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = dict(mix=load_model(a.ckpt))
    bank = RIRBank(a.rir)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(53).permutation(len(files))]
    rows, n = [], int(a.seconds * SR)
    for fi, f in enumerate(files):
        if len({r["file"] for r in rows}) >= a.tracks:
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < n + 2 * SR:
            continue
        ctx = x[:, : n + 2 * SR]
        clean = match_level(ctx[:, -n:], -20.0)
        for ci, cond in enumerate(CONDS):
            rng = np.random.default_rng([53, fi, ci])
            y = ctx
            if cond in ("reverb", "all three"):
                y, _ = apply_reverb(y, rng, bank, drr_db=float(rng.uniform(0, 9)), p_real=1.0)
            if cond in ("echo", "all three"):
                y, _ = apply_echo(y, rng)
            y = y[:, -n:]
            if cond in ("clip", "all three"):
                thr = np.percentile(np.abs(y), float(rng.uniform(93, 99.5)))
                y = np.clip(y, -thr, thr)
            deg = match_level(y.astype(np.float32), -20.0)
            row = dict(file=os.path.basename(f), cond=cond, input=score(deg, clean), variants={})
            for name, kw in VARIANTS.items():
                out, rep = enhance_auto(models, deg, use_stems=False, do_master=False, ride=0, **kw)
                row["variants"][name] = score(match_level(out, -20.0), clean)
                if name == "full":
                    row["decisions"] = rep["decisions"]
            rows.append(row)
            print(len(rows), cond, round(row["input"]["sisdr"], 1), {k: round(v["sisdr"], 1) for k, v in row["variants"].items()}, flush=True)
    summ = {}
    for cond in CONDS:
        rs = [r for r in rows if r["cond"] == cond]
        summ[cond] = dict(n=len(rs), input={m: float(np.mean([r["input"][m] for r in rs])) for m in ("sisdr", "lsd")},
                          **{v: {m: float(np.mean([r["variants"][v][m] for r in rs])) for m in ("sisdr", "lsd")} for v in VARIANTS})
    json.dump(dict(ckpt=a.ckpt, summary=summ, rows=rows), open(os.path.join(a.out, "ablation.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
