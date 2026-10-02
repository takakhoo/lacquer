"""Corpus norms for dynamics: peak-to-loudness ratio (true peak minus integrated LUFS) and crest factor.

  python -m remaster.build_dynamics_norms --data data/raw/fma_small --n 600
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .data import list_tracks, load_audio
from .master import loudness, true_peak_db

OUT = os.path.join(os.path.dirname(__file__), "dynamics_norms.json")
PCTS = (5, 10, 25, 50, 75, 90, 95)


def one(f):
    try:
        x = load_audio(f)
        if x.shape[1] < 44100 * 20 or np.sqrt(np.mean(x ** 2)) < 1e-3:
            return None
        l = loudness(x)
        if not np.isfinite(l):
            return None
        return float(true_peak_db(x)) - l, float(20 * np.log10(np.abs(x).max() / np.sqrt(np.mean(x ** 2)))), l
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--n", type=int, default=600)
    a = p.parse_args()
    fs = list_tracks(a.data)
    fs = [fs[i] for i in np.random.default_rng(0).permutation(len(fs))[: a.n]]
    with ProcessPoolExecutor(8) as ex:
        r = np.array([v for v in ex.map(one, fs, chunksize=8) if v])
    pct = lambda col: {f"p{q}": round(float(v), 2) for q, v in zip(PCTS, np.percentile(r[:, col], PCTS))}
    out = dict(n=len(r), plr_db=pct(0), crest_db=pct(1), lufs=pct(2), source=f"{len(r)} clips from {a.data}")
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
