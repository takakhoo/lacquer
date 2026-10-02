"""Does time attention lock onto the echo delay? A statistic over many clips and delays.

  python -m remaster.attention_probe --ckpt remaster/checkpoints/best.pt --data data/raw/fma_small --out docs/evidence/attention_lag.json

For each held-out clip an echo with a random delay is added (no reverb, so the echo is the only repeated
structure). The lag profile of time attention (mean weight against frames back, averaged over bands and
heads) is read from several layers, and its peak beyond the first few frames is compared with the true delay.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import list_tracks, load_audio, split_of
from .degrade import SR, match_level
from .infer import load_model

HOP = 512


@torch.no_grad()
def lag_profiles(model, x, layers, max_lag=60):
    grabs = {}
    hooks = []
    for li in layers:
        for blk in model.net.layers[li][0].layers:
            hooks.append(blk[0].attend.register_forward_pre_hook(lambda m, args, li=li: grabs.__setitem__(li, (args[0], args[1]))))
    model(torch.from_numpy(x)[None])
    for h in hooks:
        h.remove()
    out = {}
    for li in layers:
        q, k = grabs[li]                                           # [bands, heads, T, d]
        prof = torch.zeros(max_lag)
        for b in range(0, q.shape[0], 4):                          # every fourth band is plenty
            a = torch.softmax(q[b] @ k[b].transpose(-1, -2) * q.shape[-1] ** -0.5, dim=-1).mean(0)
            prof += torch.stack([torch.diagonal(a, -j).mean() for j in range(max_lag)])
        out[li] = (prof / len(range(0, q.shape[0], 4))).numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clips", type=int, default=30)
    a = p.parse_args()
    model = load_model(a.ckpt, device="cpu")
    layers = (0, 5, 11)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    rng = np.random.default_rng(17)
    files = [files[i] for i in rng.permutation(len(files))]
    rows = []
    for f in files:
        if len(rows) >= a.clips:
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < 12 * SR:
            continue
        x = x[:, 4 * SR: 10 * SR]
        delay = float(rng.uniform(0.09, 0.5)); gain = float(rng.uniform(0.3, 0.6))
        d = int(delay * SR)
        y = x.copy(); y[:, d:] += gain * x[:, :-d]
        clean, echo = match_level(x[:, -4 * SR:], -20.0), match_level(y[:, -4 * SR:], -20.0)
        true = delay * SR / HOP
        pe, pc = lag_profiles(model, echo, layers), lag_profiles(model, clean, layers)
        row = dict(file=os.path.basename(f), delay_ms=1000 * delay, gain=gain, true_lag_frames=true)
        for li in layers:
            ratio = pe[li][5:] / (pc[li][5:] + 1e-9)               # echo clip against the same clip without echo
            peak = int(np.argmax(ratio)) + 5
            row[f"L{li + 1}"] = dict(peak_lag=peak, hit=bool(abs(peak - true) <= 1.0), ratio_at_true=float(pe[li][int(round(true))] / (pc[li][int(round(true))] + 1e-9)))
        rows.append(row)
        print(len(rows), {k: (v["peak_lag"], v["hit"], round(v["ratio_at_true"], 2)) for k, v in row.items() if k.startswith("L")}, round(true, 1), flush=True)
    summ = {f"L{li + 1}": dict(hit_rate=float(np.mean([r[f"L{li + 1}"]["hit"] for r in rows])), median_ratio_at_true=float(np.median([r[f"L{li + 1}"]["ratio_at_true"] for r in rows]))) for li in layers}
    json.dump(dict(n=len(rows), summary=summ, rows=rows), open(a.out, "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
