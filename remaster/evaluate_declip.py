"""Clipping: the DSP declipper against the masking network, on held-out music.

  python -m remaster.evaluate_declip --ckpt remaster/checkpoints/best.pt --data <music> --out docs/evidence/declip

Conditions: hard clipping at the 90th to 99.7th percentile of sample magnitude (the training range), tanh
saturation at the same thresholds, and clean input (false triggers).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import list_tracks, load_audio, split_of
from .declip import declip, find_clipping
from .degrade import SR, match_level
from .infer import load_model, restore
from .losses import log_spec_dist, si_sdr


def score(y, clean):
    a, b = torch.from_numpy(np.ascontiguousarray(y))[None], torch.from_numpy(np.ascontiguousarray(clean))[None]
    return dict(sisdr=si_sdr(a, b).item(), lsd=log_spec_dist(a, b).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=32)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    net = load_model(a.ckpt, a.device) if a.ckpt else None
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(21).permutation(len(files))]
    rows, n = [], int(a.seconds * SR)
    for fi, f in enumerate(files):
        if len(rows) >= a.tracks:
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < 30 * SR:
            continue
        clean = match_level(x[:, 20 * SR:20 * SR + n], -20.0)
        rng = np.random.default_rng([21, fi])
        pct = float(rng.uniform(90, 99.7))
        thr = np.percentile(np.abs(clean), pct)
        hard = np.clip(clean, -thr, thr).astype(np.float32)
        soft = (thr * np.tanh(clean / thr)).astype(np.float32)
        row = dict(file=os.path.basename(f), pct=pct, clean_trigger=any(c is not None for c in find_clipping(clean)))
        d_hard, rep = declip(hard)
        d_soft, rep_soft = declip(soft)
        row.update(detected=any(rep["clipped"]), soft_detected=any(rep_soft["clipped"]),
                   hard=dict(input=score(hard, clean), declip=score(d_hard, clean)),
                   soft=dict(input=score(soft, clean), declip=score(d_soft, clean)))
        if net is not None:
            row["hard"]["network"] = score(restore(net, hard), clean)
            row["hard"]["declip+network"] = score(restore(net, d_hard), clean)
            row["soft"]["network"] = score(restore(net, soft), clean)
        rows.append(row)
        print(len(rows), f"pct {pct:.1f}", {k: round(v["sisdr"], 1) for k, v in row["hard"].items()},
              "soft", {k: round(v["sisdr"], 1) for k, v in row["soft"].items()}, "clean trigger", row["clean_trigger"], flush=True)
    summ = {c: {k: {m: float(np.mean([r[c][k][m] for r in rows])) for m in ("sisdr", "lsd")} for k in rows[0][c]} for c in ("hard", "soft")}
    summ.update(n=len(rows), detected=float(np.mean([r["detected"] for r in rows])),
                soft_detected=float(np.mean([r["soft_detected"] for r in rows])),
                clean_triggers=int(sum(r["clean_trigger"] for r in rows)),
                declip_better=float(np.mean([r["hard"]["declip"]["sisdr"] > r["hard"]["input"]["sisdr"] for r in rows])),
                median_gain=float(np.median([r["hard"]["declip"]["sisdr"] - r["hard"]["input"]["sisdr"] for r in rows])))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, "declip.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
