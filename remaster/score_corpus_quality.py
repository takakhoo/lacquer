"""Score every track of a corpus with the reference-free quality predictor (Audiobox Aesthetics).

  python -m remaster.score_corpus_quality --fma data/raw/fma_medium --out data/norms/pq.npz

The scores select the better-produced half of each genre when the mastering norms are built.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import load_audio
from .degrade import SR
from .quality import Aesthetics


class Clips(Dataset):
    def __init__(self, files, seconds):
        self.files, self.n = files, int(seconds * SR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        try:
            x = load_audio(f)
            if x.shape[1] < self.n:
                raise ValueError
            a = (x.shape[1] - self.n) // 2
            return int(os.path.basename(f)[:6]), torch.from_numpy(x[:, a:a + self.n])
        except Exception:
            return -1, torch.zeros(2, self.n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fma", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.fma, "*", "*.mp3")))
    files = files[: a.limit] if a.limit else files
    model = Aesthetics()
    ids, rows = [], []
    for k, (i, x) in enumerate(DataLoader(Clips(files, a.seconds), batch_size=32, num_workers=a.workers)):
        ok = i >= 0
        if ok.any():
            res = model(x[ok], SR, batch=32)
            ids += i[ok].tolist()
            rows += [[r["pq"], r["ce"], r["cu"], r["pc"]] for r in res]
        if k % 50 == 0:
            print(k * 32, len(files), flush=True)
    r = np.array(rows)
    np.savez(a.out, ids=np.array(ids), pq=r[:, 0], ce=r[:, 1], cu=r[:, 2], pc=r[:, 3])
    print("scored", len(ids), "pq mean", r[:, 0].mean())


if __name__ == "__main__":
    main()
