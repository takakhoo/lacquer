"""Deterministic mastering finish: tonal balance, loudness, true-peak limiting.

These steps have exact DSP answers, so no network is involved. Input/outputs are float [C, N].
"""
from __future__ import annotations

import json
import os

import numpy as np
import pyloudnorm as pyln
from scipy import ndimage, signal

SR = 44100
BAND_CENTERS = 1000.0 * 2.0 ** (np.arange(-17, 13) / 3.0)  # ~20 Hz .. 16 kHz, third octaves
_CURVE_PATH = os.path.join(os.path.dirname(__file__), "target_curve.json")


def ltas_db(x, sr=SR, centers=BAND_CENTERS):
    """Long-term average spectrum in third-octave bands, dB relative to total power."""
    f, p = signal.welch(x.mean(axis=0), sr, nperseg=8192, noverlap=4096)
    out = np.empty(len(centers))
    for i, fc in enumerate(centers):
        m = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        out[i] = 10 * np.log10(p[m].sum() + 1e-20) if m.any() else -200.0
    return out - 10 * np.log10(p.sum() + 1e-20)


def load_target_curve():
    """Return (median, low, high) third-octave curves. low/high bound the normal range of the corpus."""
    if os.path.exists(_CURVE_PATH):
        j = json.load(open(_CURVE_PATH))
        med = np.array(j["ltas_db"])
        return med, np.array(j.get("p10", med - 4.0)), np.array(j.get("p90", med + 4.0))
    # fallback: -4.5 dB/octave tilt above 100 Hz, typical of mastered popular music
    c = -4.5 * np.log2(np.maximum(BAND_CENTERS, 100.0) / 100.0)
    c[BAND_CENTERS < 40] -= 6 * np.log2(40 / BAND_CENTERS[BAND_CENTERS < 40])
    c = c - 10 * np.log10(np.sum(10 ** (c / 10)))
    return c, c - 4.0, c + 4.0


def tonal_balance(x, sr=SR, strength=0.6, max_db=6.0, mode="range", reference=None):
    """Correct the long-term spectrum with a smooth linear-phase EQ.

    mode="range": only bands outside the corpus 10th-90th percentile range are pulled back to its edge.
    Tracks differ from the corpus median by more than a typical EQ mistake (3.6 dB vs 2.6 dB RMS on
    held-out FMA), so pulling everything to the median damages normal tracks more than it repairs
    bad ones. mode="median" does that pull anyway, as a style choice.
    Passing `reference` audio matches its long-term spectrum instead (what mastering to a reference does).
    """
    med, low, high = load_target_curve()
    cur = ltas_db(x, sr)
    # bands far below the corpus are missing content (band-limited source); EQ cannot restore them
    valid = (BAND_CENTERS >= 30) & (BAND_CENTERS <= 16000) & (med - cur < 18)
    if reference is not None:
        ref = ltas_db(reference, sr)
        valid = (BAND_CENTERS >= 30) & (BAND_CENTERS <= 16000) & (np.abs(ref - cur) < 18)
        diff = ref - cur
        diff = diff - np.mean(diff[valid])
    elif mode == "median":
        diff = med - cur
        diff = diff - np.mean(diff[valid])
    else:
        diff = np.where(cur > high, high - cur, np.where(cur < low, low - cur, 0.0))
    diff = np.where(valid, diff, 0.0)
    diff = ndimage.gaussian_filter1d(diff, 1.0, mode="nearest")
    corr = np.clip(diff * strength, -max_db, max_db)
    freqs = np.concatenate([[0], BAND_CENTERS, [sr / 2]])
    gains = 10 ** (np.concatenate([[corr[0]], corr, [corr[-1]]]) / 20)
    fir = signal.firwin2(4097, freqs, gains, fs=sr)
    y = np.stack([signal.fftconvolve(ch, fir, mode="same") for ch in x])
    return y.astype(np.float32), dict(bands_hz=BAND_CENTERS.tolist(), correction_db=corr.tolist(), measured_db=cur.tolist(),
                                      target_db=med.tolist(), low_db=low.tolist(), high_db=high.tolist())


def true_peak_db(x, sr=SR, os_factor=4):
    up = signal.resample_poly(x, os_factor, 1, axis=1)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


def limiter(x, sr=SR, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=120.0, os_factor=4):
    """Stereo-linked lookahead limiter on 4x oversampled peaks."""
    ceil = 10 ** (ceiling_db / 20)
    up = signal.resample_poly(x, os_factor, 1, axis=1)
    n = x.shape[1]
    peak = np.abs(up[:, : n * os_factor]).reshape(x.shape[0], n, os_factor).max(axis=(0, 2))
    need = np.minimum(1.0, ceil / (peak + 1e-12))
    la = max(1, int(lookahead_ms * 1e-3 * sr))
    hop = 8
    m = len(need) // hop * hop
    need_d = np.concatenate([need[:m].reshape(-1, hop).min(axis=1), [need[m:].min()] if m < len(need) else []])
    rel = np.exp(-hop / (release_ms * 1e-3 * sr))
    g = np.empty_like(need_d)
    cur = 1.0
    for i, v in enumerate(need_d):
        cur = v if v < cur else rel * cur + (1 - rel) * 1.0
        cur = min(cur, v)
        g[i] = cur
    g = np.repeat(g, hop)[: len(need)]
    g = ndimage.minimum_filter1d(g, 2 * la + 1, mode="nearest")   # hold: react before the peak arrives
    g = ndimage.uniform_filter1d(g, la, mode="nearest")           # smooth the attack inside the hold window
    y = x * g
    over = np.abs(y).max() / ceil
    if over > 1:
        y = y / over
    return y.astype(np.float32), dict(max_gr_db=float(-20 * np.log10(g.min() + 1e-12)))


def loudness(x, sr=SR):
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(x.T.astype(np.float64)))


def master(x, sr=SR, target_lufs=-14.0, ceiling_db=-1.0, tonal_strength=0.6, max_limit_db=8.0, reference=None):
    """Tonal balance -> gain to target loudness -> true-peak limiter. Returns (audio, report).

    With `reference` audio, tone and loudness follow the reference instead of the corpus range and target_lufs.
    """
    report = dict(input_lufs=loudness(x, sr), input_true_peak_db=float(true_peak_db(x, sr)))
    b, a = signal.butter(2, 20 / (sr / 2), "high")
    y = signal.lfilter(b, a, x, axis=1).astype(np.float32)
    if tonal_strength > 0:
        y, report["tonal"] = tonal_balance(y, sr, strength=tonal_strength, max_db=9.0 if reference is not None else 6.0, reference=reference)
    if reference is not None and np.isfinite(loudness(reference, sr)):
        target_lufs = report["reference_lufs"] = loudness(reference, sr)
    lufs = loudness(y, sr)
    if np.isfinite(lufs):
        gain_db = target_lufs - lufs
        # do not ask the limiter for more than max_limit_db of reduction
        headroom = ceiling_db - true_peak_db(y, sr)
        gain_db = min(gain_db, headroom + max_limit_db)
        y = y * 10 ** (gain_db / 20)
        report["gain_db"] = float(gain_db)
    y, report["limiter"] = limiter(y, sr, ceiling_db=ceiling_db)
    # limiting lowers loudness slightly; one makeup pass keeps us near target
    lufs2 = loudness(y, sr)
    if np.isfinite(lufs2) and target_lufs - lufs2 > 0.3 and report["limiter"]["max_gr_db"] < max_limit_db:
        y = y * 10 ** (min(target_lufs - lufs2, 2.0) / 20)
        y, lim2 = limiter(y, sr, ceiling_db=ceiling_db)
        report["limiter"]["max_gr_db"] = max(report["limiter"]["max_gr_db"], lim2["max_gr_db"])
    report.update(output_lufs=loudness(y, sr), output_true_peak_db=float(true_peak_db(y, sr)))
    return y, report
