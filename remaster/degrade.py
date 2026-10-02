"""Degradations for training pairs. All functions take float32 [C, N] at SR and return [C, N].

Every effect returns audio time-aligned with its input so (degraded, clean) pairs stay
sample-aligned, which the waveform and complex-STFT losses depend on.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

SR = 44100
EFFECTS = ("reverb", "echo", "eq", "comp", "clip", "noise")


def _octave_bands(sr):
    edges = [0, 125, 250, 500, 1000, 2000, 4000, 8000, sr / 2]
    return list(zip(edges[:-1], edges[1:]))


def synth_rir(rng, sr=SR, rt60=None, channels=2):
    """Stereo RIR: unit direct path at t=0, sparse early reflections, band-dependent noise tail."""
    rt60 = rt60 if rt60 is not None else float(np.exp(rng.uniform(np.log(0.25), np.log(2.5))))
    n = int(sr * min(rt60 * 1.3, 3.0))
    t = np.arange(n) / sr
    hf_damp = rng.uniform(0.25, 0.9)
    lf_mult = rng.uniform(0.8, 1.4)
    out = np.zeros((channels, n), dtype=np.float64)
    bands = _octave_bands(sr)
    predelay = int(sr * rng.uniform(0.003, 0.04))
    for c in range(channels):
        noise = rng.standard_normal(n)
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        tail = np.zeros(n)
        for i, (lo, hi) in enumerate(bands):
            frac = i / (len(bands) - 1)
            band_rt = rt60 * (lf_mult * (1 - frac) + hf_damp * frac)
            m = (freqs >= lo) & (freqs < hi)
            band = np.fft.irfft(spec * m, n)
            tail += band * np.exp(-6.908 * t / max(band_rt, 0.05))
        onset = np.clip((np.arange(n) - predelay) / (0.01 * sr), 0, 1)
        tail *= onset
        for _ in range(rng.integers(3, 12)):
            d = int(sr * rng.uniform(0.004, 0.08))
            if d < n:
                tail[d] += rng.uniform(-1, 1) * 6 * np.sqrt(np.mean(tail ** 2) + 1e-12)
        out[c] = tail
    out /= np.sqrt(np.sum(out ** 2, axis=1, keepdims=True).mean()) + 1e-9
    return out.astype(np.float32)  # tail only, unit energy; direct path added by caller


def algo_rir(rng, sr=SR, rt60=None, channels=2):
    """Algorithmic (plugin-style) reverb tail: Schroeder/Moorer parallel damped combs into series allpasses.

    Room IRs and noise tails do not cover the comb-and-allpass sound of reverb plug-ins, which is what
    most produced music carries. The response is built in the frequency domain, so it is exact and fast.
    Returns a unit-energy tail [channels, n]; the caller adds the direct path.
    """
    rt60 = rt60 if rt60 is not None else float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    n = int(sr * min(rt60 * 1.6, 4.5))
    w = 2 * np.pi * np.fft.rfftfreq(n)
    z1 = np.exp(-1j * w)
    damp = rng.uniform(0.05, 0.55)
    lp = (1 - damp) / (1 - damp * z1)                       # one-pole lowpass inside each comb's feedback
    base = np.sort(rng.uniform(0.025, 0.08, size=rng.integers(6, 9)))
    pre = int(sr * rng.uniform(0.0, 0.04))
    out = np.zeros((channels, n))
    for c in range(channels):
        spread = 1.0 + (0.0 if c == 0 else rng.uniform(0.01, 0.05))  # slightly longer delays on the other side
        h = np.zeros(len(w), dtype=complex)
        for d_s in base:
            d = int(sr * d_s * spread) | 1
            g = 10 ** (-3 * d / (sr * rt60))
            zd = np.exp(-1j * w * d)
            h += zd / (1 - g * lp * zd)
        for d_s, g in ((0.005, 0.7), (0.0017, 0.7)):
            zd = np.exp(-1j * w * int(sr * d_s * spread))
            h *= (-g + zd) / (1 - g * zd)
        tail = np.fft.irfft(h * np.exp(-1j * w * pre), n)
        tail *= np.exp(-np.arange(n) / (0.9 * n) * 2.0)      # fade the circular wrap-around
        out[c] = tail
    out /= np.sqrt(np.sum(out ** 2, axis=1, keepdims=True).mean()) + 1e-9
    return out.astype(np.float32)


class RIRBank:
    """Real impulse responses, split at the direct-path peak into (direct, tail)."""

    def __init__(self, root=None, sr=SR):
        self.tails = []
        for f in sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)) if root else []:
            x, fs = sf.read(f, dtype="float32", always_2d=True)
            x = x.T
            if fs != sr:
                x = signal.resample_poly(x, sr, fs, axis=1).astype(np.float32)
            peak = int(np.argmax(np.abs(x).max(axis=0)))
            cut = peak + int(0.0025 * sr)
            tail = x[:, cut:cut + 3 * sr].copy()
            if tail.shape[1] < sr // 20:
                continue
            tail[:, : int(0.001 * sr)] *= np.linspace(0, 1, int(0.001 * sr), dtype=np.float32)
            tail /= np.sqrt(np.sum(tail ** 2, axis=1, keepdims=True).mean()) + 1e-9
            self.tails.append(tail)

    def __len__(self):
        return len(self.tails)

    def sample(self, rng):
        tail = self.tails[rng.integers(len(self.tails))]
        if tail.shape[0] == 1:
            # decorrelate a mono IR into stereo with a short random allpass-ish jitter
            shift = rng.integers(1, 40)
            tail = np.stack([tail[0], np.roll(tail[0], shift) * rng.choice([-1.0, 1.0])])
        return tail


def apply_reverb(x, rng, bank=None, drr_db=None, p_real=0.5, p_algo=0.0):
    """x + wet tail. Direct path is untouched, so the clean target stays aligned.

    p_algo is the share of non-real tails drawn from the algorithmic reverb. It defaults to 0 and draws no
    extra random number in that case, so fixed evaluation sets built before it existed stay identical.
    """
    if bank is not None and len(bank) and rng.random() < p_real:
        tail = bank.sample(rng)
        kind = "real"
    elif p_algo > 0 and rng.random() < p_algo:
        tail = algo_rir(rng, channels=x.shape[0])
        kind = "algo"
    else:
        tail = synth_rir(rng, channels=x.shape[0])
        kind = "synth"
    drr_db = drr_db if drr_db is not None else rng.uniform(-6, 12)
    wet = np.stack([signal.fftconvolve(x[c], tail[c % tail.shape[0]])[: x.shape[1]] for c in range(x.shape[0])])
    g = np.sqrt(np.mean(x ** 2) + 1e-12) / (np.sqrt(np.mean(wet ** 2)) + 1e-9) * 10 ** (-drr_db / 20)
    return (x + g * wet).astype(np.float32), dict(kind=kind, drr_db=float(drr_db))


def apply_echo(x, rng, sr=SR):
    """Discrete delay taps, optionally with feedback and a darker repeat."""
    delay = rng.uniform(0.06, 0.5)
    att = rng.uniform(0.15, 0.6)
    feedback = rng.random() < 0.5
    d = int(delay * sr)
    y = x.copy()
    rep = x
    if rng.random() < 0.5:
        b, a = signal.butter(1, rng.uniform(1500, 8000) / (sr / 2))
        rep = signal.lfilter(b, a, x, axis=1).astype(np.float32)
    n_taps = rng.integers(2, 6) if feedback else 1
    for k in range(1, n_taps + 1):
        if k * d >= x.shape[1]:
            break
        y[:, k * d:] += (att ** k) * rep[:, : -k * d]
    return y, dict(delay=float(delay), att=float(att), taps=int(n_taps))


def _peaking(fc, q, gain_db, sr):
    a_ = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a_, -2 * np.cos(w0), 1 - alpha * a_]
    a = [1 + alpha / a_, -2 * np.cos(w0), 1 - alpha / a_]
    return np.array(b) / a[0], np.array(a) / a[0]


def _shelf(fc, gain_db, sr, high):
    a_ = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    cs, sn = np.cos(w0), np.sin(w0)
    alpha = sn / 2 * np.sqrt(2)
    s = -1 if high else 1
    b = [a_ * ((a_ + 1) - s * (a_ - 1) * cs + 2 * np.sqrt(a_) * alpha),
         s * 2 * a_ * ((a_ - 1) - s * (a_ + 1) * cs),
         a_ * ((a_ + 1) - s * (a_ - 1) * cs - 2 * np.sqrt(a_) * alpha)]
    a = [(a_ + 1) + s * (a_ - 1) * cs + 2 * np.sqrt(a_) * alpha,
         -s * 2 * ((a_ - 1) + s * (a_ + 1) * cs),
         (a_ + 1) + s * (a_ - 1) * cs - 2 * np.sqrt(a_) * alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def apply_eq(x, rng, sr=SR, max_db=12.0):
    """1-4 random peaking/shelving biquads."""
    y = x.astype(np.float64)
    desc = []
    for _ in range(rng.integers(1, 5)):
        g = rng.uniform(-max_db, max_db)
        kind = rng.choice(["peak", "low", "high"], p=[0.6, 0.2, 0.2])
        if kind == "peak":
            fc = float(np.exp(rng.uniform(np.log(60), np.log(12000))))
            b, a = _peaking(fc, rng.uniform(0.4, 3.0), g, sr)
        elif kind == "low":
            fc = float(np.exp(rng.uniform(np.log(60), np.log(500))))
            b, a = _shelf(fc, g, sr, high=False)
        else:
            fc = float(np.exp(rng.uniform(np.log(2000), np.log(10000))))
            b, a = _shelf(fc, g, sr, high=True)
        y = signal.lfilter(b, a, y, axis=1)
        desc.append((str(kind), round(fc), round(float(g), 1)))
    return y.astype(np.float32), dict(bands=desc)


def apply_comp(x, rng, sr=SR):
    """Feed-forward compressor with attack/release, stereo-linked, makeup to match RMS."""
    # threshold relative to the clip's own level, so it bites on quiet and on already-loud material alike
    thr = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) + rng.uniform(-10, 6)
    ratio = rng.uniform(2.5, 12)
    att = rng.uniform(0.001, 0.03)
    rel = rng.uniform(0.04, 0.4)
    hop = 16
    level = np.abs(x).max(axis=0)
    n = len(level) // hop * hop
    env_in = level[:n].reshape(-1, hop).max(axis=1)
    db = 20 * np.log10(env_in + 1e-7)
    target = np.minimum(0.0, (thr - db) * (1 - 1 / ratio))
    ca = np.exp(-hop / (att * sr))
    cr = np.exp(-hop / (rel * sr))
    gr = np.empty_like(target)
    g = 0.0
    for i, tg in enumerate(target):
        c = ca if tg < g else cr
        g = c * g + (1 - c) * tg
        gr[i] = g
    gain = 10 ** (np.interp(np.arange(x.shape[1]), np.arange(len(gr)) * hop + hop / 2, gr) / 20)
    y = x * gain
    y *= np.sqrt(np.mean(x ** 2) + 1e-12) / (np.sqrt(np.mean(y ** 2)) + 1e-9)
    return y.astype(np.float32), dict(thr=float(thr), ratio=float(ratio), att=float(att), rel=float(rel))


def apply_clip(x, rng):
    """Hard or soft clipping at a percentile of the peak distribution."""
    pct = rng.uniform(90, 99.7)
    thr = np.percentile(np.abs(x), pct) + 1e-6
    if rng.random() < 0.5:
        y = np.clip(x, -thr, thr)
    else:
        y = thr * np.tanh(x / thr)
    return y.astype(np.float32), dict(pct=float(pct))


def apply_noise(x, rng, sr=SR):
    snr = rng.uniform(20, 45)
    n = rng.standard_normal(x.shape)
    if rng.random() < 0.5:
        b, a = signal.butter(1, 800 / (sr / 2))
        n = signal.lfilter(b, a, n, axis=1)
    n *= np.sqrt(np.mean(x ** 2) + 1e-12) / (np.sqrt(np.mean(n ** 2)) + 1e-9) * 10 ** (-snr / 20)
    return (x + n).astype(np.float32), dict(snr=float(snr))


def apply_ride(x, rng, sr=SR, max_db=8.0):
    """Level problems inside a track: sections of 1.5-5 s at different gains, with soft transitions."""
    n = x.shape[1]
    g = np.zeros(n)
    i = 0
    while i < n:
        ln = int(rng.uniform(1.5, 5.0) * sr)
        g[i:i + ln] = rng.uniform(-max_db, max_db)
        i += ln
    g = ndimage.uniform_filter1d(g, int(0.3 * sr), mode="nearest")
    g -= np.median(g)
    return (x * 10 ** (g / 20)).astype(np.float32), dict(range_db=float(g.max() - g.min()))


def apply_limit(x, rng, sr=SR):
    """Over-limiting ("loudness war"): drive the mix 6-16 dB into a brickwall limiter."""
    from .master import limiter
    drive = rng.uniform(6.0, 16.0)
    peak = np.abs(x).max() + 1e-9
    y, info = limiter(x / peak * 10 ** (drive / 20), sr, ceiling_db=-0.3, release_ms=float(rng.uniform(40, 200)))
    return (y * peak).astype(np.float32), dict(drive_db=float(drive), max_gr_db=info["max_gr_db"])


FN = dict(ride=apply_ride, limit=apply_limit, reverb=apply_reverb, echo=apply_echo, eq=apply_eq, comp=apply_comp, clip=apply_clip, noise=apply_noise)
P_DEFAULT = dict(reverb=0.55, echo=0.3, eq=0.45, comp=0.3, clip=0.15, noise=0.15)


def degrade(x, rng, bank=None, effects=None, p_identity=0.08, probs=P_DEFAULT, p_algo=0.0):
    """Apply a random chain. `effects` forces an explicit ordered list (used by eval)."""
    if effects is None:
        if rng.random() < p_identity:
            return x.copy(), {}
        effects = [e for e in EFFECTS if rng.random() < probs[e]]
        if not effects:
            effects = [str(rng.choice(EFFECTS[:4]))]
        # acoustic effects first, then signal-chain effects, mostly; shuffle sometimes
        if rng.random() < 0.3:
            rng.shuffle(effects)
    y, log = x, {}
    for e in effects:
        if e == "reverb":
            y, info = apply_reverb(y, rng, bank, p_algo=p_algo)
        else:
            y, info = FN[e](y, rng)
        log[e] = info
    return y, log


def rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def match_level(x, target_db=-20.0):
    return x * 10 ** ((target_db - rms_db(x)) / 20)
