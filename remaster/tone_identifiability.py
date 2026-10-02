"""How much of a tonal fault can be identified without a reference?

  python -m remaster.tone_identifiability --corpus data/norms/corpus.npz --stems data/norms/stems.npz --out docs/evidence/mastering

Works on the measured third-octave spectra of the corpus, where an EQ fault is an added curve. Part 1: how well
a held-out track's own tone can be predicted from the population, its genre, or its nearest neighbours in
descriptors that an EQ does not change. Part 2: how much of a simulated fault three blind estimators remove.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analysis import THIRD_OCT
from .data import split_of

TONE = (THIRD_OCT >= 60) & (THIRD_OCT <= 10000)
F = THIRD_OCT[TONE]


def faults(n, rng):
    """Same generator as evaluate_mastering: half tilts of 0.6 to 1.6 dB/octave, half one or two broad bumps of 3 to 8 dB."""
    out = np.zeros((n, len(F)))
    for i in range(n):
        if rng.random() < 0.5:
            out[i] = rng.choice([-1, 1]) * rng.uniform(0.6, 1.6) * np.log2(F / 1000)
        else:
            for _ in range(rng.integers(1, 3)):
                fc, g, w = np.exp(rng.uniform(np.log(80), np.log(8000))), rng.choice([-1, 1]) * rng.uniform(3, 8), rng.uniform(0.6, 1.2)
                out[i] += g * np.exp(-0.5 * (np.log2(F / fc) / w) ** 2)
    return out - out.mean(1, keepdims=True)


def rms(a):
    return float(np.sqrt((a ** 2).mean(1)).mean())


def estimators(Ltr, Lte, Sc):
    mu, St = Ltr.mean(0), np.cov(Ltr.T) + 1e-3 * np.eye(Ltr.shape[1])
    lo, hi = np.percentile(Ltr, 10, 0), np.percentile(Ltr, 90, 0)
    edge = lambda o: 0.7 * np.where(o > hi, o - hi, np.where(o < lo, o - lo, 0.0))
    mapg = lambda o: (o - mu) @ np.linalg.solve(Sc + St, Sc).T          # posterior mean of the fault under Gaussian tone and fault priors
    c = faults(len(Lte), np.random.default_rng(1))
    o = Lte + c
    return dict(n_train=int(len(Ltr)), n_test=int(len(Lte)), fault=rms(c),
                after=dict(range_rule=rms(c - edge(o)), gaussian_map=rms(c - mapg(o)), mean_match=rms(c - (o - mu))),
                clean_disturbed=dict(range_rule=rms(edge(Lte)), gaussian_map=rms(mapg(Lte)), mean_match=rms(Lte - mu)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--stems", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    c = np.load(a.corpus)
    names = c["feature_names"].tolist()
    li = [names.index(f"ltas_{i}") for i in range(len(THIRD_OCT))]
    inv = [i for i, k in enumerate(names) if k.startswith(("band_crest", "band_spread", "width_")) or k in ("lra", "plr", "psr", "corr", "corr_low")]
    X = c["X"]
    L = X[:, li][:, TONE]
    ok = np.isfinite(L).all(1) & (L.min(1) > -80) & np.isfinite(X[:, inv]).all(1)
    X, L, ids, genre = X[ok], L[ok], c["ids"][ok], c["genre"][ok]
    L = L - L.mean(1, keepdims=True)
    tr = np.array([split_of(f"{i:06d}.mp3") == "train" for i in ids])
    te = ~tr
    out = dict(bands=int(TONE.sum()), band_range_hz=[float(F[0]), float(F[-1])])
    # part 1: predictability of a track's own tone
    Z = np.clip(X[:, inv], -60, 60)
    Z = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-9)
    gm = np.stack([L[tr & (genre == g)].mean(0) if (tr & (genre == g)).sum() > 100 else L[tr].mean(0) for g in genre[te]])
    D = ((Z[te][:, None, :] - Z[tr][None, :, :]) ** 2).sum(-1)
    pred = dict(population_mean=rms(L[te] - L[tr].mean(0)), genre_mean=rms(L[te] - gm))
    for k in (5, 20, 100):
        idx = np.argpartition(D, k, axis=1)[:, :k]
        pred[f"nearest_{k}"] = rms(L[te] - L[tr][idx].mean(1))
    out["predictability_db"] = pred
    # part 2: blind estimators against a simulated fault
    Sc = np.cov(faults(20000, np.random.default_rng(0)).T)
    out["released_corpus"] = estimators(L[tr], L[te], Sc)
    s = np.load(a.stems)
    M = s["mix"][:, li][:, TONE]
    M = M - M.mean(1, keepdims=True)
    mtr = s["split"] == "train"
    out["professional_mixes"] = estimators(M[mtr], M[~mtr], Sc)
    out["professional_mixes"]["band_std_db"] = float(M[mtr].std(0).mean())
    out["released_corpus"]["band_std_db"] = float(L[tr].std(0).mean())
    sl = names.index("slope")
    dev = {k: s[k][:, sl] for k in ("mix", "vocals", "drums", "bass", "other")}
    fin = np.all([np.isfinite(v) for v in dev.values()], axis=0)
    out["slope_std_db_oct"] = {k: float(np.std(v[fin])) for k, v in dev.items()}
    out["slope_correlation_between_stems"] = {f"{a_}-{b_}": float(np.corrcoef(dev[a_][fin], dev[b_][fin])[0, 1]) for i, a_ in enumerate(("vocals", "drums", "bass", "other")) for b_ in ("vocals", "drums", "bass", "other")[i + 1:]}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "tone_identifiability.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
