"""Percentile tables of mastering features, per genre, read by the mastering chain.

  python -m remaster.mastering_norms --corpus data/norms/corpus.npz --stems data/norms/stems.npz [--quality pq.npz]

Writes remaster/mastering_norms.json. A feature's "normal range" for a style is its 10th to 90th percentile
over that style's tracks; with --quality only the better-produced half of each genre (by a reference-free
quality predictor) defines the range.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analysis import SCALARS, VECTORS, vector_names

PCTS = (5, 10, 25, 50, 75, 90, 95)
_PATH = os.path.join(os.path.dirname(__file__), "mastering_norms.json")
_CACHE = {}


def table(X):
    return {f"p{p}": np.nanpercentile(X, p, axis=0).round(3).tolist() for p in PCTS} | {"n": int(len(X))}


def build(corpus, stems=None, quality=None, min_tracks=150, keep=0.5):
    from .data import split_of
    c = np.load(corpus, allow_pickle=False)
    keep_rows = np.array([split_of(f"{int(i):06d}.mp3") == "train" for i in c["ids"]])   # held-out tracks never define the norms
    X, genre, ids = c["X"][keep_rows], c["genre"][keep_rows], c["ids"][keep_rows]
    out = dict(features=vector_names(), percentiles=list(PCTS), genres={}, quality_filtered=quality is not None)
    if quality is not None:
        q = np.load(quality)
        pq = dict(zip(q["ids"].tolist(), q["pq"].tolist()))
        score = np.array([pq.get(int(i), np.nan) for i in ids])
    out["genres"]["all"] = table(X)
    for g in sorted(set(genre.tolist()) - {""}):
        m = genre == g
        if m.sum() < min_tracks:
            continue
        if quality is not None:
            s = score[m]
            m2 = m.copy()
            m2[m] = s >= np.nanquantile(s, 1 - keep)
            m = m2
        out["genres"][g] = table(X[m])
    if stems:
        s = np.load(stems, allow_pickle=False)
        tr = s["split"] == "train"
        s = {k: (s[k][tr] if k in ("mix", "vocals", "drums", "bass", "other") else s[k]) for k in s.files}
        names = s["feature_names"].tolist()
        li = names.index("lufs")
        out["unmastered_mix"] = table(s["mix"])
        out["stems"] = {}
        for k in ("vocals", "drums", "bass", "other"):
            t = table(s[k])
            rel = s[k][:, li] - s["mix"][:, li]
            rel = rel[np.isfinite(rel)]
            t["level_re_mix"] = {f"p{p}": float(np.percentile(rel, p)) for p in PCTS}
            out["stems"][k] = t
    return out


def load():
    if "n" not in _CACHE:
        _CACHE["n"] = json.load(open(_PATH)) if os.path.exists(_PATH) else None
    return _CACHE["n"]


def genres():
    n = load()
    return [] if n is None else [g for g in n["genres"] if g != "all"]


def _ranges(t, names, lo, hi):
    low, med, high = (np.array(t[f"p{p}"]) for p in (lo, 50, hi))
    out = {}
    for k in SCALARS:
        j = names.index(k)
        out[k] = (float(low[j]), float(med[j]), float(high[j]))
    for k, size in VECTORS.items():
        j = names.index(f"{k}_0")
        out[k] = (low[j:j + size], med[j:j + size], high[j:j + size])
    return out


def ranges(genre="all", lo=10, hi=90, norms=None):
    """dict feature -> (low, median, high), scalars as floats and vector features as arrays."""
    n = norms or load()
    return _ranges(n["genres"].get(genre) or n["genres"]["all"], n["features"], lo, hi)


def stem_ranges(name, lo=10, hi=90, norms=None):
    """Same, for one instrument stem, plus "level" = stem loudness relative to the mix (p5, p10, p50, p90, p95)."""
    n = norms or load()
    t = n["stems"][name]
    out = _ranges(t, n["features"], lo, hi)
    # prefer levels measured on separated stems: that is what the pipeline can measure on a finished mix
    lv = (n.get("stems_separated") or {}).get(name) or t["level_re_mix"]
    out["level"] = tuple(lv[f"p{p}"] for p in (5, 10, 50, 90, 95))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--stems", default=None)
    p.add_argument("--quality", default=None)
    p.add_argument("--out", default=_PATH)
    a = p.parse_args()
    n = build(a.corpus, a.stems, a.quality)
    json.dump(n, open(a.out, "w"))
    print({g: t["n"] for g, t in n["genres"].items()})


if __name__ == "__main__":
    main()
