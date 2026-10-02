"""Echo stage alone, with and without the tempo-grid rule.

  python -m remaster.evaluate_deecho --data data/raw/fma_medium --out docs/evidence/deecho --tracks 100 --workers 16

Each held-out clip is tested clean and with an added echo (60 to 500 ms, gain 0.15 to 0.6, one to five taps).
The tempo-grid rule leaves an echo in place when its delay is a whole number of sixteenth notes or triplet
eighths: that protects delay effects and loops in clean music and costs the added echoes that happen to land
on the grid.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import torch

from .data import list_tracks, load_audio, split_of
from .deecho import deecho
from .degrade import SR, apply_echo, match_level
from .losses import si_sdr


def score(y, clean):
    return si_sdr(torch.from_numpy(np.ascontiguousarray(y))[None], torch.from_numpy(np.ascontiguousarray(clean))[None]).item()


def one(job):
    fi, f, seconds, z = job
    try:
        x = load_audio(f)
    except Exception:
        return None
    n = int(seconds * SR)
    if x.shape[1] < n:
        return None
    clean = match_level(x[:, :n], -20.0)
    echo, info = apply_echo(clean, np.random.default_rng([83, fi]))
    echo = match_level(echo, -20.0)
    row = dict(file=os.path.basename(f), info=info, input=score(echo, clean))
    for name, keep in (("always_remove", False), ("tempo_rule", True)):
        y, found = deecho(echo, keep_musical=keep, z_thresh=z)
        yc, found_c = deecho(clean, keep_musical=keep, z_thresh=z)
        row[name] = dict(sisdr=score(match_level(y, -20.0), clean), removed=sum(not e.get("kept") for e in found), kept=sum(bool(e.get("kept")) for e in found),
                         clean_removed=sum(not e.get("kept") for e in found_c), clean_kept=sum(bool(e.get("kept")) for e in found_c),
                         clean_sisdr=min(120.0, score(match_level(yc, -20.0), clean)))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=100)
    p.add_argument("--seconds", type=float, default=25.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--z", type=float, default=45.0, help="cepstral peak threshold")
    p.add_argument("--name", default="deecho.json")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(83).permutation(len(files))][: int(a.tracks * 1.3)]
    with Pool(a.workers) as pool:
        rows = [r for r in pool.imap(one, [(i, f, a.seconds, a.z) for i, f in enumerate(files)]) if r][: a.tracks]
    summ = dict(n=len(rows), input=float(np.mean([r["input"] for r in rows])))
    for name in ("always_remove", "tempo_rule"):
        summ[name] = dict(sisdr=float(np.mean([r[name]["sisdr"] for r in rows])), median_sisdr=float(np.median([r[name]["sisdr"] for r in rows])),
                          echoes_removed=int(sum(r[name]["removed"] > 0 for r in rows)), echoes_left_as_musical=int(sum(r[name]["removed"] == 0 and r[name]["kept"] > 0 for r in rows)),
                          clean_tracks_altered=int(sum(r[name]["clean_removed"] > 0 for r in rows)), clean_tracks_with_musical_echo=int(sum(r[name]["clean_kept"] > 0 for r in rows)))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, a.name), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
