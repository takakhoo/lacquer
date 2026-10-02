"""VU-style level metering and a slow gain rider for level problems inside a track.

The meter follows VU ballistics (about 300 ms to reach a steady reading). The rider works like an
engineer riding a fader while watching that meter: sections that sit well above or below the
track's usual level are eased back toward it. It is slow, bounded, has a dead zone, and holds its
gain through quiet passages so fades and breaks are not pumped up.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

SR = 44100
VU_REF_DBFS = -18.0  # 0 VU


def vu_trace(x, sr=SR, rate=20):
    """VU reading in dB (0 VU = -18 dBFS RMS sine) sampled `rate` times per second."""
    hop = sr // rate
    env = np.abs(x).mean(axis=0)
    n = len(env) // hop * hop
    env = env[:n].reshape(-1, hop).mean(axis=1)  # rectified average, as a VU movement does
    # two cascaded one-poles give the ~300 ms rise of the standard VU movement
    a = np.exp(-1.0 / (0.065 * rate))
    for _ in range(2):
        env = signal.lfilter([1 - a], [1, -a], env)
    # rectified average of a sine is 2/pi of its peak; reference a -18 dBFS RMS sine to 0 VU
    ref = 10 ** (VU_REF_DBFS / 20) * np.sqrt(2) * 2 / np.pi
    return 20 * np.log10(env / ref + 1e-9)


def ride_gain(x, sr=SR, rate=20, window_s=3.0, max_db=6.0, ratio=0.75, dead_db=1.5, gate_db=18.0, smooth_s=1.5):
    """Return (audio, report). report has the VU traces and the gain curve at `rate` Hz."""
    vu = vu_trace(x, sr, rate)
    if len(vu) < rate * 4:
        return x, dict(vu_before=vu.tolist(), vu_after=vu.tolist(), gain_db=[0.0] * len(vu), rate=rate, max_ride_db=0.0)
    loud = vu > np.percentile(vu, 95) - 40
    home = float(np.median(vu[loud]))                      # the track's usual level
    active = vu > home - gate_db                            # quiet passages do not steer the fader
    w = int(window_s * rate) | 1
    filled = np.where(active, vu, np.nan)
    # section level: median of active readings in a sliding window
    sec = np.array([np.nanmedian(filled[max(0, i - w // 2): i + w // 2 + 1]) if active[max(0, i - w // 2): i + w // 2 + 1].any() else np.nan
                    for i in range(len(vu))])
    want = home - sec
    want = np.sign(want) * np.maximum(np.abs(want) - dead_db, 0.0) * ratio
    want = np.clip(want, -max_db, max_db)
    # hold the last gain through gated passages, then smooth like a hand on a fader
    idx = np.where(~np.isnan(want), np.arange(len(want)), 0)
    np.maximum.accumulate(idx, out=idx)
    want = np.where(np.isnan(want[idx]), 0.0, want[idx])
    gain = ndimage.gaussian_filter1d(want, smooth_s * rate / 2, mode="nearest")
    hop = sr // rate
    g = np.interp(np.arange(x.shape[1]), (np.arange(len(gain)) + 0.5) * hop, gain)
    y = (x * 10 ** (g / 20)).astype(np.float32)
    return y, dict(vu_before=np.round(vu, 2).tolist(), vu_after=np.round(vu_trace(y, sr, rate), 2).tolist(),
                   gain_db=np.round(gain, 2).tolist(), rate=rate, home_vu=home, max_ride_db=float(np.abs(gain).max()))
