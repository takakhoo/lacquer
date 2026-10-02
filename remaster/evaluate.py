"""Held-out evaluation: per-condition metrics, spectrogram panels, audio examples.

  python -m remaster.evaluate --ckpt runs/base/best.pt --data data/raw/fma_medium --rir data/raw/mit_ir --out eval/base
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from .data import build_fixed_set, list_tracks, split_of
from .infer import load_model
from .losses import log_spec_dist, sdr, si_sdr
from .model import SR


def panel(items, path):
    fig, axes = plt.subplots(len(items), 3, figsize=(15, 2.6 * len(items)), squeeze=False)
    for r, it in enumerate(items):
        for c, (k, title) in enumerate([("deg", "degraded input"), ("out", "model output"), ("clean", "clean target")]):
            x = it[k].mean(0).numpy()
            axes[r, c].specgram(x, NFFT=2048, Fs=SR, noverlap=1536, cmap="magma", vmin=-120, vmax=-30)
            axes[r, c].set_yscale("symlog", linthresh=500)
            axes[r, c].set_ylim(40, 20000)
            axes[r, c].set_title(f"{it['cond']}: {title}" + (f"  (SI-SDR {it['m_in']:.1f} -> {it['m_out']:.1f} dB)" if c == 1 else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--rir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=100)
    p.add_argument("--split", default="test")
    p.add_argument("--audio-examples", type=int, default=3)
    p.add_argument("--task", default="joint", choices=["joint", "artifact"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = load_model(a.ckpt)
    dev = next(model.parameters()).device
    files = [f for f in list_tracks(a.data) if split_of(f) == a.split]
    fixed = build_fixed_set(files, a.rir, n_tracks=a.tracks, seed=999, task=a.task)
    rows = {c: [] for c in dict.fromkeys(f["cond"] for f in fixed)}
    shown = {}
    for i in range(0, len(fixed), 8):
        chunk = fixed[i:i + 8]
        deg = torch.stack([c["deg"] for c in chunk]).to(dev)
        clean = torch.stack([c["clean"] for c in chunk]).to(dev)
        out = model(deg)
        m = dict(sisdr_in=si_sdr(deg, clean), sisdr_out=si_sdr(out, clean), sdr_in=sdr(deg, clean), sdr_out=sdr(out, clean),
                 lsd_in=log_spec_dist(deg, clean), lsd_out=log_spec_dist(out, clean))
        for j, c in enumerate(chunk):
            rows[c["cond"]].append({k: v[j].item() for k, v in m.items()})
            n = shown.get(c["cond"], 0)
            if n < a.audio_examples:
                shown[c["cond"]] = n + 1
                tag = f"{c['cond'].replace('+', '_')}_{n}"
                for k, t in (("deg", deg[j]), ("out", out[j]), ("clean", clean[j])):
                    sf.write(os.path.join(a.out, f"{tag}_{k}.wav"), t.cpu().numpy().T.clip(-1, 1), SR)
                if n == 0:
                    c.update(out=out[j].cpu(), m_in=m["sisdr_in"][j].item(), m_out=m["sisdr_out"][j].item())
    panel([c for c in fixed if "out" in c], os.path.join(a.out, "spectrograms.png"))
    summary = {c: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} | {"n": len(rs)} for c, rs in rows.items() if rs}
    json.dump(dict(ckpt=a.ckpt, summary=summary), open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    lines = ["| condition | n | SI-SDR in | SI-SDR out | gain | LSD in | LSD out |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c, s in summary.items():
        lines.append(f"| {c} | {s['n']} | {s['sisdr_in']:.2f} | {s['sisdr_out']:.2f} | {s['sisdr_out'] - s['sisdr_in']:+.2f} | {s['lsd_in']:.2f} | {s['lsd_out']:.2f} |")
    open(os.path.join(a.out, "metrics.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
