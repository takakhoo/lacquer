"""Blind removal of discrete echoes by cepstral detection and exact inverse filtering.

A delayed copy y = x + a*g*x(t-d) multiplies the spectrum by (1 + a G e^{-jwd}). Its log is a ripple
in frequency, so the cepstrum shows a narrow peak at quefrency d whose shape is a*g. That gives the
delay and the echo path with no training, and the echo is then inverted exactly with a recursive
filter. Peaks are peeled one at a time, which also handles multi-tap and feedback delays.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

SR = 44100


def cepstrum(x, n=1 << 17):
    """Frame-averaged real cepstrum of the mid signal. n ~ 3 s so delays up to 600 ms sit well inside a frame."""
    m = x.mean(axis=0)
    if len(m) < n:
        m = np.pad(m, (0, n - len(m)))
    hop = n // 2
    w = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    k = 0
    for s in range(0, len(m) - n + 1, hop):
        acc += np.log(np.abs(np.fft.rfft(m[s:s + n] * w)) ** 2 + 1e-10)
        k += 1
    return np.fft.irfft(acc / max(k, 1))[: n // 2]


def find_echo(x, sr=SR, min_s=0.04, max_s=0.65, z_thresh=45.0, min_gain=0.04, width=48):
    """Return (delay_samples, path_fir, zscore) for the sharpest cepstral peak, or None."""
    c = cepstrum(x)
    lo, hi = int(min_s * sr), int(max_s * sr)
    seg = c[lo:hi]
    base = ndimage.median_filter(seg, 401)
    resid = seg - base
    mad = ndimage.median_filter(np.abs(resid), 2001) + 1e-9
    z = resid / (1.4826 * mad)
    i = int(np.argmax(z))
    # thresholds from held-out FMA: clean tracks sit at z ~ 12 (95th pct 39), echoed ones at z >= 45
    if z[i] < z_thresh or resid[i] < min_gain:
        return None
    d = lo + i
    # echo path: cepstrum just after the peak (a lowpassed repeat smears over a few dozen samples)
    path = (c[d:d + width] - base[min(i, len(base) - 1)]).copy()
    path[np.abs(path) < 3 * 1.4826 * mad[i]] = 0.0
    path[0] = c[d] - base[i]
    return d, path, float(z[i])


def remove_echo(x, d, path):
    """Invert y = x + path * x(t-d), i.e. apply 1 / (1 + path z^-d).

    The recursion only reaches back d samples or more, so it runs block by block (block = d samples)
    with plain vector operations.
    """
    path = np.asarray(path, dtype=np.float64)
    if np.abs(path).sum() >= 0.98:  # keep the recursion stable
        path = path * (0.98 / np.abs(path).sum())
    taps = [(k, p) for k, p in enumerate(path) if p != 0.0]
    out = x.astype(np.float64).copy()
    n = out.shape[1]
    for s in range(d, n, d):
        e = min(s + d, n)
        for k, p in taps:
            lo = s - d - k
            if lo >= 0:
                out[:, s:e] -= p * out[:, lo:lo + (e - s)]
            else:
                out[:, s - lo:e] -= p * out[:, 0:e - s + lo]
    return out.astype(np.float32)


def deecho(x, sr=SR, max_iter=4, refine=4, **kw):
    """Peel echoes until no sharp cepstral peak remains. Returns (audio, list of detections)."""
    found = []
    y = x
    for _ in range(max_iter):
        hit = find_echo(y, sr, **kw)
        if hit is None:
            break
        d, path, z = hit
        width = len(path)
        # The cepstrum reads the echo path only to first order and loses some amplitude at frame edges.
        # Fixed-point refinement: invert, look at what is left at the same quefrencies, add it back.
        best, best_res = remove_echo(y, d, path), None
        for _ in range(refine):
            c2 = cepstrum(best)
            left = c2[d:d + width] - np.median(c2[d + width:d + 8 * width])
            res = float(np.abs(left).sum())
            if best_res is not None and res >= best_res:
                break
            best_res, keep = res, best
            path = path + left
            best = remove_echo(y, d, path)
        y = best
        found.append(dict(delay_ms=1000 * d / sr, gain=float(path[0]), z=z))
    return y, found
