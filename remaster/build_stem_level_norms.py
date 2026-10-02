"""Instrument balance norms as the pipeline measures them: separate each mix, read stem loudness relative to the mix.

  python -m remaster.build_stem_level_norms --musdb data/raw/musdb18hq/train --fma data/raw/fma_medium --fma-tracks 1500 --out data/norms/stem_levels.npz
  python -m remaster.build_stem_level_norms --merge data/norms/stem_levels.npz          # writes into remaster/mastering_norms.json

Separated stems read about 1 LU lower than the true stems and a 30 s window differs from a whole song, so the
normal range has to come from the same measurement that is used on a new track.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf

from .analysis import SR, loudest_window
from .data import load_audio, split_of
from .mastering_norms import _PATH, PCTS
from .stem_master import stem_levels
from .stems import STEMS, separate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--musdb", default=None)
    p.add_argument("--fma", default=None)
    p.add_argument("--fma-tracks", type=int, default=1500)
    p.add_argument("--out", default=None)
    p.add_argument("--merge", default=None)
    p.add_argument("--source", default="musdb", choices=["musdb", "fma", "both"], help="which population defines the range when merging")
    a = p.parse_args()
    if a.merge:
        d = np.load(a.merge)
        keep = np.ones(len(d["source"]), bool) if a.source == "both" else d["source"] == a.source
        n = json.load(open(_PATH))
        n["stems_separated"] = {}
        for i, s in enumerate(STEMS):
            v = d["levels"][keep, i]
            v = v[np.isfinite(v) & (v > -22)]            # stems that are present
            n["stems_separated"][s] = {f"p{q}": float(np.percentile(v, q)) for q in PCTS} | {"n": int(len(v))}
        n["stems_separated"]["source"] = a.source
        json.dump(n, open(_PATH, "w"))
        print({s: {k: round(x, 2) for k, x in t.items() if k != "n"} for s, t in n["stems_separated"].items() if s != "source"})
        return
    files = []
    if a.musdb:
        files += [("musdb", f) for f in sorted(glob.glob(os.path.join(a.musdb, "*", "mixture.wav")))]
    if a.fma:
        fs = [f for f in sorted(glob.glob(os.path.join(a.fma, "*", "*.mp3"))) if split_of(f) == "train"]
        files += [("fma", fs[i]) for i in np.random.default_rng(5).permutation(len(fs))[: a.fma_tracks]]
    src, names, rows = [], [], []
    for k, (s, f) in enumerate(files):
        try:
            x = sf.read(f, dtype="float32", always_2d=True)[0].T if f.endswith(".wav") else load_audio(f)
            a0, b0 = loudest_window(x)
            x = x[:, a0:b0]
            if x.shape[1] < 15 * SR:
                continue
            lv = stem_levels(separate(x), x)
        except Exception as e:
            print("skip", f, e, flush=True)
            continue
        src.append(s); names.append(os.path.basename(os.path.dirname(f)) if s == "musdb" else os.path.basename(f)); rows.append([lv[t] for t in STEMS])
        if k % 50 == 0:
            print(k, len(files), flush=True)
            np.savez(a.out, source=np.array(src), names=np.array(names), levels=np.array(rows, dtype=np.float64))
    np.savez(a.out, source=np.array(src), names=np.array(names), levels=np.array(rows, dtype=np.float64))
    print("done", len(rows))


if __name__ == "__main__":
    main()
