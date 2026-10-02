"""What a mastering engineer measures before touching anything: loudness, dynamics, tone, stereo image.

`mastering_features` returns one flat dict per track. The same function describes the reference corpus
(`build_mastering_norms`), the input of the mastering chain and its output, so every decision can be stated as
"this reading is outside the range of well-produced music in this style".
"""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from scipy import signal

SR = 44100
THIRD_OCT = 1000.0 * 2.0 ** (np.arange(-17, 13) / 3.0)          # 20 Hz .. 16 kHz
WIDTH_EDGES = (20, 120, 500, 2000, 8000, 20000)                 # bands for the stereo image
DYN_EDGES = (20, 120, 1000, 6000, 20000)                        # bands for multiband dynamics
_METERS = {}


def _meter(sr):
    if sr not in _METERS:
        _METERS[sr] = pyln.Meter(sr)
    return _METERS[sr]


def k_weight(x, sr=SR):
    y = x.astype(np.float64)
    for f in _meter(sr)._filters.values():
        y = signal.lfilter(f.b, f.a, y, axis=1)
    return y


def loudness_stats(x, sr=SR):
    """Integrated loudness, loudness range (EBU Tech 3342) and short-term series from one K-weighting pass."""
    k = k_weight(x, sr)
    hop = int(0.1 * sr)
    n = k.shape[1] // hop
    if n < 30:
        return dict(lufs=float("nan"), lra=float("nan"), short_term=np.zeros(0))
    p = (k[:, : n * hop] ** 2).reshape(k.shape[0], n, hop).mean(axis=2).sum(axis=0)      # 100 ms blocks
    c = np.concatenate([[0.0], np.cumsum(p)])
    mom = (c[4:] - c[:-4]) / 4                                                           # 400 ms, 75% overlap
    st = (c[30:] - c[:-30]) / 30                                                         # 3 s short-term
    to_l = lambda v: -0.691 + 10 * np.log10(np.maximum(v, 1e-12))
    # integrated: absolute gate at -70, relative gate 10 LU under the gated mean (BS.1770-4)
    g = mom[to_l(mom) > -70]
    if not len(g):
        return dict(lufs=float("nan"), lra=float("nan"), short_term=to_l(st))
    g = g[to_l(g) > to_l(g.mean()) - 10]
    lufs = float(to_l(g.mean())) if len(g) else float("nan")
    s = st[to_l(st) > -70]
    s = s[to_l(s) > to_l(s.mean()) - 20] if len(s) else s
    lra = float(np.percentile(to_l(s), 95) - np.percentile(to_l(s), 10)) if len(s) > 5 else float("nan")
    return dict(lufs=lufs, lra=lra, short_term=to_l(st))


def true_peak_db(x, sr=SR, os_factor=4):
    up = signal.resample_poly(x, os_factor, 1, axis=1)
    return float(20 * np.log10(np.abs(up).max() + 1e-12))


def band_powers(psd_f, psd, edges):
    return np.array([psd[(psd_f >= a) & (psd_f < b)].sum() + 1e-20 for a, b in zip(edges[:-1], edges[1:])])


def mastering_features(x, sr=SR):
    """x: float [2, N] (mono is duplicated). Returns scalars and small vectors, all in dB unless named otherwise."""
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    x = x.astype(np.float32)
    out = {}
    ls = loudness_stats(x, sr)
    tp = true_peak_db(x, sr)
    rms = float(10 * np.log10(np.mean(x ** 2) + 1e-12))
    out.update(lufs=ls["lufs"], lra=ls["lra"], true_peak=tp, plr=tp - ls["lufs"], crest=float(20 * np.log10(np.abs(x).max() + 1e-12)) - rms,
               dc=float(np.abs(x.mean(axis=1)).max()))
    # short-term dynamics: how far short-term loudness sits below the true peak, typical value
    if len(ls["short_term"]):
        st = ls["short_term"][ls["short_term"] > -70]
        out["psr"] = float(tp - np.percentile(st, 95)) if len(st) else float("nan")
    else:
        out["psr"] = float("nan")

    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
    f, pm = signal.welch(mid, sr, nperseg=8192, noverlap=4096)
    _, ps = signal.welch(side, sr, nperseg=8192, noverlap=4096)
    tot = pm + ps
    # tone: third-octave long-term spectrum relative to total power, plus one-number summaries
    lt = np.array([10 * np.log10(tot[(f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))].sum() + 1e-20) for fc in THIRD_OCT]) - 10 * np.log10(tot.sum() + 1e-20)
    out["ltas"] = lt
    sel = (THIRD_OCT >= 100) & (THIRD_OCT <= 10000)
    out["slope"] = float(np.polyfit(np.log2(THIRD_OCT[sel]), lt[sel], 1)[0]) - 3.0   # dB/octave of power density (a third-octave band widens 3 dB per octave)
    e = lambda a, b: 10 * np.log10(tot[(f >= a) & (f < b)].sum() / tot.sum() + 1e-20)
    out.update(sub=float(e(20, 60)), bass=float(e(60, 250)), mud=float(e(250, 500)), mids=float(e(500, 2000)), presence=float(e(2000, 5000)),
               sibilance=float(e(5000, 9000)), air=float(e(9000, 16000)))
    # stereo image: side/mid energy per band, and the plain channel correlation overall and in the low end
    wm, ws = band_powers(f, pm, WIDTH_EDGES), band_powers(f, ps, WIDTH_EDGES)
    out["width"] = 10 * np.log10(ws / wm)
    out["width_all"] = float(10 * np.log10(ps.sum() / (pm.sum() + 1e-20) + 1e-20))
    denom = float(np.sqrt(np.mean(x[0] ** 2) * np.mean(x[1] ** 2))) + 1e-12
    out["corr"] = float(np.mean(x[0] * x[1]) / denom)
    sos = signal.butter(4, 150, "low", fs=sr, output="sos")
    lo = signal.sosfilt(sos, x, axis=1)
    out["corr_low"] = float(np.mean(lo[0] * lo[1]) / (np.sqrt(np.mean(lo[0] ** 2) * np.mean(lo[1] ** 2)) + 1e-12))
    # multiband dynamics: per-band crest (99.9th percentile peak over RMS) and spread of the 400 ms level
    crest, spread = [], []
    m = x.mean(axis=0)
    for a, b in zip(DYN_EDGES[:-1], DYN_EDGES[1:]):
        sos = signal.butter(4, [a, min(b, sr / 2 - 100)], "band", fs=sr, output="sos")
        y = signal.sosfilt(sos, m)
        r = np.sqrt(np.mean(y ** 2) + 1e-14)
        crest.append(20 * np.log10(np.percentile(np.abs(y), 99.9) / r + 1e-9))
        hop = int(0.4 * sr)
        k = len(y) // hop
        lv = 10 * np.log10((y[: k * hop] ** 2).reshape(k, hop).mean(axis=1) + 1e-14)
        lv = lv[lv > lv.max() - 40]
        spread.append(float(np.percentile(lv, 90) - np.percentile(lv, 10)) if len(lv) > 4 else float("nan"))
    out["band_crest"], out["band_spread"] = np.array(crest), np.array(spread)
    return out


SCALARS = ("lufs", "lra", "true_peak", "plr", "crest", "psr", "slope", "sub", "bass", "mud", "mids", "presence", "sibilance", "air", "width_all", "corr", "corr_low")
VECTORS = dict(ltas=len(THIRD_OCT), width=len(WIDTH_EDGES) - 1, band_crest=len(DYN_EDGES) - 1, band_spread=len(DYN_EDGES) - 1)


def to_vector(feat):
    """Flatten to a fixed-order vector (see `vector_names`)."""
    return np.concatenate([[feat[k] for k in SCALARS]] + [np.asarray(feat[k], dtype=np.float64) for k in VECTORS])


def vector_names():
    return list(SCALARS) + [f"{k}_{i}" for k, n in VECTORS.items() for i in range(n)]
