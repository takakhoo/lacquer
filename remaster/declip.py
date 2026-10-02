"""Declipping by consistent sparse reconstruction (A-SPADE, Kitic, Bertin and Gribonval 2015).

Hard clipping leaves exact knowledge behind: samples below the ceiling are untouched, and samples at the ceiling
were at least that large. The declipper looks for the sparsest spectrum whose waveform agrees with both facts.
Soft saturation leaves no flat ceiling, so it is not detected and passes through unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from .degrade import SR


def find_clipping(x, tol=1e-3, min_run=2, min_frac=2e-4):
    """Per channel (ceiling, high mask, low mask), or None when the channel shows no flat ceiling.

    A sample counts as clipped when it sits within tol of the channel peak inside a run of at least min_run
    such samples. min_frac is the share of clipped samples below which the channel is left alone.
    """
    out = []
    for ch in x:
        peak = float(np.abs(ch).max())
        if peak < 1e-4:
            out.append(None)
            continue
        hi, lo = ch >= peak * (1 - tol), ch <= -peak * (1 - tol)
        for m in (hi, lo):
            run = m & (np.roll(m, 1) | np.roll(m, -1))
            if min_run > 1:
                m &= run
        frac = (hi.sum() + lo.sum()) / ch.size
        # a real ceiling shows on both polarities
        out.append((peak * (1 - tol), hi, lo) if frac >= min_frac and hi.any() and lo.any() else None)
    return out


def _spade(yw, hi, lo, thr, red=2, s=2, r=1, eps=0.1, device="cpu"):
    """Batched A-SPADE over windowed frames. yw [F, win], hi/lo bool masks, thr [F, win] windowed ceiling."""
    win = yw.shape[1]
    P = red * win
    y = torch.from_numpy(yw).to(device)
    H, L = torch.from_numpy(hi).to(device), torch.from_numpy(lo).to(device)
    T = torch.from_numpy(thr).to(device)
    A = lambda v: torch.fft.rfft(v, n=P, norm="ortho")
    At = lambda c: torch.fft.irfft(c, n=P, norm="ortho")[:, :win]
    x, out = y.clone(), y.clone()
    u = torch.zeros(len(y), P // 2 + 1, dtype=torch.complex64, device=device)
    idx = torch.arange(len(y), device=device)
    bound = eps * y.norm(dim=1) + 1e-9
    k, it = s, 0
    while len(idx) and k <= P // 2 + 1:
        c = A(x) + u
        mag = c.abs()
        kth = mag.kthvalue(mag.shape[1] - k + 1, dim=1, keepdim=True).values
        z = torch.where(mag >= kth, c, torch.zeros_like(c))
        v = At(z - u)
        yi, Hi, Li, Ti = y[idx], H[idx], L[idx], T[idx]
        v = torch.where(~(Hi | Li), yi, v)
        v = torch.where(Hi, torch.maximum(v, Ti), v)
        x = torch.where(Li, torch.minimum(v, -Ti), v)
        res = A(x) - z
        done = res.norm(dim=1) <= bound[idx]
        out[idx[done]] = x[done]
        keep = ~done
        idx, x, u = idx[keep], x[keep], (u + res)[keep]
        it += 1
        if it % r == 0:
            k += s
    if len(idx):
        out[idx] = x
    return out.cpu().numpy()


def declip(x, sr=SR, win=2048, hop=512, eps=0.1, s=2, device="cpu", batch=4096):
    """Return (declipped audio, report). Channels without a flat ceiling are returned unchanged."""
    found = find_clipping(x)
    report = dict(clipped=[f is not None for f in found], share=[0.0] * len(x))
    if not any(report["clipped"]):
        return x, report
    w = np.sqrt(np.hanning(win + 1)[:-1]).astype(np.float32)
    y = x.astype(np.float32).copy()
    for ci, f in enumerate(found):
        if f is None:
            continue
        thr, hi, lo = f
        report["share"][ci] = float((hi.sum() + lo.sum()) / hi.size)
        n = x.shape[1]
        pad = win
        ch = np.pad(x[ci].astype(np.float32), (pad, pad + hop))
        hp, lp = np.pad(hi, (pad, pad + hop)), np.pad(lo, (pad, pad + hop))
        starts = np.arange(0, len(ch) - win + 1, hop)
        fr = np.lib.stride_tricks.sliding_window_view(ch, win)[starts]
        fh = np.lib.stride_tricks.sliding_window_view(hp, win)[starts]
        fl = np.lib.stride_tricks.sliding_window_view(lp, win)[starts]
        active = np.flatnonzero((fh | fl).any(1))
        rec = fr * w
        for a in range(0, len(active), batch):
            sel = active[a:a + batch]
            rec[sel] = _spade(np.ascontiguousarray(fr[sel] * w), np.ascontiguousarray(fh[sel]), np.ascontiguousarray(fl[sel]),
                              np.broadcast_to(thr * w, (len(sel), win)).copy(), eps=eps, s=s, device=device)
        acc, norm = np.zeros_like(ch), np.zeros_like(ch)
        for i, st in enumerate(starts):
            acc[st:st + win] += rec[i] * w
            norm[st:st + win] += w * w
        y[ci] = (acc / np.maximum(norm, 1e-6))[pad:pad + n]
    return y, report
