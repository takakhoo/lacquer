"""Do-no-harm audit: run the whole repair half on clean released tracks and count what fires.

  python -m remaster.evaluate_do_no_harm --data data/raw/fma_small --out docs/evidence/do_no_harm --tracks 100

A stage that acts on a clean track is a false alarm. Reported per stage, with how far the output moved from
the input (SI-SDR of output against input; identical audio is reported as 120 dB).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .data import list_tracks, load_audio, split_of
from .degrade import SR
from .infer import enhance_auto, load_model
from .losses import si_sdr

HERE = os.path.dirname(__file__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=100)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "best.pt"))
    p.add_argument("--vocal-ckpt", default=os.path.join(HERE, "checkpoints", "vocal_dereverb.pt"))
    p.add_argument("--controller-ckpt", default=os.path.join(HERE, "checkpoints", "controller.pt"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = dict(mix=load_model(a.ckpt), vocal=load_model(a.vocal_ckpt), controller=load_model(a.controller_ckpt))
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(61).permutation(len(files))]
    rows, n = [], int(a.seconds * SR)
    for f in files:
        if len(rows) >= a.tracks:
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < n:
            continue
        x = x[:, :n]
        y, rep = enhance_auto(models, x, do_master=False)
        d = " ".join(rep["decisions"])
        fired = dict(clipping="peaks rebuilt" in d, echo="Echo at" in d, room=rep.get("room", {}).get("action") == "reduce",
                     vocal=rep.get("vocal", {}).get("action") in ("reduce", "add"), balance="moved by" in d,
                     ride=rep.get("vu", {}).get("max_ride_db", 0.0) > 1.0)
        same = float(np.abs(y - x).max()) < 1e-6
        s = 120.0 if same else min(120.0, si_sdr(torch.from_numpy(y)[None], torch.from_numpy(x)[None]).item())
        rows.append(dict(file=os.path.basename(f), fired=fired, sisdr_vs_input=s, decisions=rep["decisions"], max_ride_db=rep.get("vu", {}).get("max_ride_db", 0.0)))
        print(len(rows), [k for k, v in fired.items() if v], round(s, 1), flush=True)
    summ = dict(n=len(rows), fired={k: int(sum(r["fired"][k] for r in rows)) for k in rows[0]["fired"]},
                untouched_by_repair=int(sum(not any(r["fired"][k] for k in ("clipping", "echo", "room", "vocal", "balance")) for r in rows)),
                sisdr_vs_input=dict(median=float(np.median([r["sisdr_vs_input"] for r in rows])), p10=float(np.percentile([r["sisdr_vs_input"] for r in rows], 10))))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, "do_no_harm.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
