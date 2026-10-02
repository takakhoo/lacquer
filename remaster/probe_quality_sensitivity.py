"""Does the reference-free quality predictor notice mastering faults?

  python -m remaster.probe_quality_sensitivity --data data/raw/fma_small --out docs/evidence/mastering

For held-out tracks, score the original and versions with a tone, width, dynamics or loudness fault. If the
original scores higher than its faulted versions, the predictor can steer a mastering search.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import list_tracks, load_audio, split_of
from .degrade import SR, match_level
from .evaluate_mastering import unmaster
from .quality import Aesthetics

KINDS = ("tilt", "bumps", "width", "peaky")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=60)
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = Aesthetics()
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(41).permutation(len(files))]
    n, rows = int(a.seconds * SR), []
    for fi, f in enumerate(files):
        if len(rows) >= a.tracks:
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < 25 * SR:
            continue
        x = x[:, 10 * SR:10 * SR + n]
        batch, infos = [match_level(x, -20.0)], [{}]
        for ki, k in enumerate(KINDS):
            for rep in range(2):
                y, info = unmaster(x, k, np.random.default_rng([41, fi, ki, rep]))
                batch.append(match_level(y, -20.0)); infos.append(dict(kind=k, **info))
        res = model(np.stack(batch), SR)
        rows.append(dict(file=os.path.basename(f), original=res[0], faulted=[dict(info=i, **r) for i, r in zip(infos[1:], res[1:])]))
        print(len(rows), round(res[0]["pq"], 2), [round(r["pq"] - res[0]["pq"], 2) for r in res[1:]], flush=True)
    summ = {}
    for k in KINDS:
        d = np.array([fr["pq"] - r["original"]["pq"] for r in rows for fr in r["faulted"] if fr["info"]["kind"] == k])
        dc = np.array([fr["ce"] - r["original"]["ce"] for r in rows for fr in r["faulted"] if fr["info"]["kind"] == k])
        summ[k] = dict(n=len(d), pq_change=float(d.mean()), original_higher=float((d < 0).mean()), ce_change=float(dc.mean()), ce_original_higher=float((dc < 0).mean()))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, "quality_sensitivity.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
