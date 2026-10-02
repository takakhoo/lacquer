"""Measure the mastering features of a whole corpus, per genre, and of unmastered mixes and their stems.

  python -m remaster.build_mastering_norms --fma data/raw/fma_medium --metadata data/raw/fma_metadata/tracks.csv \\
      --musdb data/raw/musdb18hq --out data/norms --workers 48

Writes corpus.npz (one feature vector per track, with genre) and stems.npz (MUSDB mixtures and stems).
`remaster.mastering_norms` turns these into the percentile tables the mastering chain reads.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import soundfile as sf

from .analysis import SR, mastering_features, to_vector, vector_names
from .data import load_audio

STEMS = ("vocals", "drums", "bass", "other")


def fma_genres(path):
    """track id -> top-level genre, from the FMA tracks.csv (two header rows)."""
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        top, sub = next(r), next(r)
        next(r)
        col = next(i for i, (a, b) in enumerate(zip(top, sub)) if a == "track" and b == "genre_top")
        return {int(row[0]): row[col] for row in r if row and row[0].isdigit() and row[col]}


def one_fma(f):
    try:
        x = load_audio(f)
        if x.shape[1] < 10 * SR:
            return None
        return int(os.path.basename(f)[:6]), to_vector(mastering_features(x))
    except Exception:
        return None


def one_musdb(d):
    try:
        out = {}
        mix, sr = sf.read(os.path.join(d, "mixture.wav"), dtype="float32", always_2d=True)
        out["mix"] = to_vector(mastering_features(mix.T, sr))
        for s in STEMS:
            y, _ = sf.read(os.path.join(d, s + ".wav"), dtype="float32", always_2d=True)
            out[s] = to_vector(mastering_features(y.T, sr))
        return os.path.basename(d), "test" if "/test/" in d else "train", out
    except Exception as e:
        print("skip", d, e, flush=True)
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fma", default=None)
    p.add_argument("--metadata", default=None)
    p.add_argument("--musdb", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        if a.musdb:
            dirs = sorted(glob.glob(os.path.join(a.musdb, "*", "*", "")))
            res = [r for r in pool.imap_unordered(one_musdb, [d.rstrip("/") for d in dirs]) if r]
            np.savez(os.path.join(a.out, "stems.npz"), names=np.array([r[0] for r in res]), split=np.array([r[1] for r in res]),
                     feature_names=np.array(vector_names()), **{k: np.stack([r[2][k] for r in res]) for k in ("mix",) + STEMS})
            print("musdb", len(res), flush=True)
        if a.fma:
            genres = fma_genres(a.metadata) if a.metadata else {}
            files = sorted(glob.glob(os.path.join(a.fma, "*", "*.mp3")))
            files = files[: a.limit] if a.limit else files
            res = []
            for i, r in enumerate(pool.imap_unordered(one_fma, files, chunksize=16)):
                if r:
                    res.append(r)
                if i % 2000 == 0:
                    print(i, len(files), flush=True)
            np.savez(os.path.join(a.out, "corpus.npz"), ids=np.array([r[0] for r in res]), X=np.stack([r[1] for r in res]),
                     genre=np.array([genres.get(r[0], "") for r in res]), feature_names=np.array(vector_names()))
            print("fma", len(res), flush=True)


if __name__ == "__main__":
    main()
