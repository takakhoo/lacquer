"""Estimate the mastering tonal-balance target as the median long-term spectrum of a music corpus.

  python -m remaster.build_target_curve --data data/raw/fma_medium --n 1500
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .data import list_tracks, load_audio
from .master import BAND_CENTERS, _CURVE_PATH, ltas_db


def one(f):
    try:
        x = load_audio(f)
        if x.shape[1] < 44100 * 10 or np.sqrt(np.mean(x ** 2)) < 1e-3:
            return None
        return ltas_db(x)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--out", default=_CURVE_PATH)
    a = p.parse_args()
    files = list_tracks(a.data)
    files = [files[i] for i in np.random.default_rng(0).permutation(len(files))[: a.n]]
    with ProcessPoolExecutor(16) as ex:
        rows = np.array([r for r in ex.map(one, files, chunksize=8) if r is not None])
    med = np.median(rows, axis=0)
    json.dump(dict(bands_hz=BAND_CENTERS.tolist(), ltas_db=med.tolist(), p10=np.percentile(rows, 10, axis=0).tolist(), p25=np.percentile(rows, 25, axis=0).tolist(),
                   p75=np.percentile(rows, 75, axis=0).tolist(), p90=np.percentile(rows, 90, axis=0).tolist(), n_tracks=len(rows), source=a.data), open(a.out, "w"), indent=1)
    print(len(rows), np.round(med, 1))


if __name__ == "__main__":
    main()
