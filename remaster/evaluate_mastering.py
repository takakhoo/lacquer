"""Un-mastering recovery: damage the mastering of held-out released tracks, master them again, and measure how
much of the damage each target scheme takes back.

  python -m remaster.evaluate_mastering --fma data/raw/fma_medium --corpus data/norms/corpus.npz --out docs/evidence/mastering --workers 40

Faults: spectral tilt, broad EQ bumps, stereo width, over-dynamic peaks. Target schemes: no correction (loudness
only), global norms, norms of the track's genre, and an oracle whose range is the original track itself.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy import signal

from .analysis import SCALARS, SR, THIRD_OCT, VECTORS, mastering_features
from .data import load_audio, split_of
from .mastering import master_track
from .mastering_norms import ranges

FAULTS = ("tilt", "bumps", "width", "peaky", "none")
TONE = (THIRD_OCT >= 30) & (THIRD_OCT <= 12500)


def eq_curve(x, freqs_db, sr=SR):
    f = np.concatenate([[0], THIRD_OCT, [sr / 2]])
    g = 10 ** (np.concatenate([[freqs_db[0]], freqs_db, [freqs_db[-1]]]) / 20)
    fir = signal.firwin2(4097, f, g, fs=sr)
    return np.stack([signal.fftconvolve(ch, fir, mode="same") for ch in x]).astype(np.float32)


def unmaster(x, kind, rng):
    if kind == "tilt":
        t = rng.choice([-1, 1]) * rng.uniform(0.6, 1.6)
        return eq_curve(x, t * np.log2(THIRD_OCT / 1000.0)), dict(tilt_db_oct=float(t))
    if kind == "bumps":
        c = np.zeros(len(THIRD_OCT))
        for _ in range(int(rng.integers(1, 3))):
            fc, g, w = np.exp(rng.uniform(np.log(80), np.log(8000))), rng.choice([-1, 1]) * rng.uniform(3, 8), rng.uniform(0.6, 1.2)
            c += g * np.exp(-0.5 * (np.log2(THIRD_OCT / fc) / w) ** 2)
        return eq_curve(x, c), dict(max_db=float(np.abs(c).max()))
    if kind == "width":
        g = rng.choice([-1, 1]) * rng.uniform(3, 9)
        mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2 * 10 ** (g / 20)
        return np.stack([mid + side, mid - side]).astype(np.float32), dict(side_db=float(g))
    if kind == "peaky":
        # upward expansion around the median level: loud moments get louder, as in a mix that was never controlled
        k = rng.uniform(0.3, 0.6)
        hop = 256
        env = np.abs(x).max(axis=0)
        n = len(env) // hop * hop
        lv = 20 * np.log10(env[:n].reshape(-1, hop).max(axis=1) + 1e-6)
        sm = signal.lfilter([0.3], [1, -0.7], lv)
        e = np.clip(k * (sm - np.median(sm)), -10, 10)
        g = np.interp(np.arange(x.shape[1]), np.arange(len(e)) * hop + hop / 2, e)
        return (x * 10 ** (g / 20)).astype(np.float32), dict(k=float(k))
    return x, {}


def errors(f, f0):
    w, w0 = np.maximum(f["width"], -40), np.maximum(f0["width"], -40)
    return dict(tone=float(np.sqrt(np.mean((f["ltas"] - f0["ltas"])[TONE] ** 2))), width=float(np.mean(np.abs(w - w0))),
                plr=float(abs(f["plr"] - f0["plr"])), crest=float(np.mean(np.abs(f["band_crest"] - f0["band_crest"]))))


def oracle_ranges(f0, tol=0.5):
    r = {}
    for k in SCALARS:
        r[k] = (f0[k] - tol, f0[k], f0[k] + tol)
    for k in VECTORS:
        v = np.asarray(f0[k], dtype=np.float64)
        r[k] = (v - tol, v, v + tol)
    r["corr_low"] = (f0["corr_low"] - 0.02, f0["corr_low"], 1.0)
    return r


def matchering_master(target, reference):
    """Matchering 2.0 (open-source reference mastering) with its defaults."""
    import tempfile

    import matchering as mg
    import soundfile as sf
    mg.log(lambda *_: None)
    with tempfile.TemporaryDirectory() as d:
        t, r, o = (os.path.join(d, n) for n in ("t.wav", "r.wav", "o.wav"))
        sf.write(t, target.T, SR, subtype="FLOAT"); sf.write(r, reference.T, SR, subtype="FLOAT")
        mg.process(target=t, reference=r, results=[mg.Result(o, subtype="FLOAT", use_limiter=True, normalize=False)])
        return sf.read(o, dtype="float32")[0].T


def one(job):
    f, genre, seed = job
    try:
        x = load_audio(f)
    except Exception:
        return []
    if x.shape[1] < 20 * SR:
        return []
    x = x[:, : 30 * SR]
    f0 = mastering_features(x)
    rows = []
    for fi, fault in enumerate(FAULTS):
        rng = np.random.default_rng([seed, fi])
        d, info = unmaster(x, fault, rng)
        fd = mastering_features(d)
        row = dict(file=os.path.basename(f), genre=genre, fault=fault, info=info, damaged=errors(fd, f0), schemes={})
        for scheme in ("loudness_only", "global", "genre", "oracle"):
            kw = dict(do_tonal=False, do_multiband=False, do_width=False, do_glue=False) if scheme == "loudness_only" else {}
            rg = oracle_ranges(f0) if scheme == "oracle" else ranges(genre if scheme == "genre" else "all")
            y, rep = master_track(d, rng=rg, **kw)
            row["schemes"][scheme] = errors(mastering_features(y), f0)
        y, _ = master_track(d, reference=x, match_reference_peak=True)
        row["schemes"]["reference"] = errors(mastering_features(y), f0)
        try:
            row["schemes"]["matchering"] = errors(mastering_features(matchering_master(d, x)), f0)
        except Exception as e:
            row["schemes"]["matchering"] = dict(tone=float("nan"), width=float("nan"), plr=float("nan"), crest=float("nan"))
        rows.append(row)
    return rows


def summarize(rows):
    out = {}
    for fault in FAULTS:
        rs = [r for r in rows if r["fault"] == fault]
        out[fault] = dict(n=len(rs), damaged={k: float(np.mean([r["damaged"][k] for r in rs])) for k in ("tone", "width", "plr", "crest")},
                          **{s: {k: float(np.nanmean([r["schemes"][s][k] for r in rs])) for k in ("tone", "width", "plr", "crest")} for s in rs[0]["schemes"]})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fma", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tracks", type=int, default=200)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = np.load(a.corpus)
    genre = dict(zip(c["ids"].tolist(), c["genre"].tolist()))
    known = set(ranges.__globals__["load"]()["genres"])
    files = [f for f in sorted(glob.glob(os.path.join(a.fma, "*", "*.mp3"))) if split_of(f) == "test" and genre.get(int(os.path.basename(f)[:6])) in known]
    files = [files[i] for i in np.random.default_rng(3).permutation(len(files))][: a.tracks]
    jobs = [(f, genre[int(os.path.basename(f)[:6])], i) for i, f in enumerate(files)]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    summ = summarize(rows)
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, "unmaster.json"), "w"), indent=1)
    for fault, s in summ.items():
        print(fault, s["n"], {k: {m: round(v, 2) for m, v in s[k].items()} for k in s if k != "n"})


if __name__ == "__main__":
    main()
