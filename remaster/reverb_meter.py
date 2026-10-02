"""Reverb measurement and the keep / reduce / add decision.

Wetness is measured with a dereverb model: wet_db = level of what the model removes relative to
what it keeps. Two meters with different meaning:
  * vocal model (trained dry-vs-reverberant vocals): absolute wetness of a vocal stem, so produced
    vocals have a measurable normal range and a stem can be too wet OR too dry.
  * full-mix model (trained to remove reverb added on top of finished productions): excess reverb only.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .degrade import rms_db, synth_rir
from .infer import restore

SR = 44100
_NORMS = os.path.join(os.path.dirname(__file__), "reverb_norms.json")


def active(x, floor_db=-45.0, frame=SR // 2):
    """Mask of half-second frames where the stem is actually playing."""
    n = x.shape[1] // frame
    lv = np.array([rms_db(x[:, i * frame:(i + 1) * frame]) for i in range(n)])
    return lv > max(floor_db, lv.max() - 35.0) if n else np.zeros(0, bool)


def wetness_db(x, dry, frame=SR // 2):
    """Removed-to-kept level ratio in dB over active frames. Returns None for a silent stem."""
    m = active(x, frame=frame)
    if m.sum() < 4:
        return None
    idx = np.flatnonzero(m)
    wet = np.concatenate([(x - dry)[:, i * frame:(i + 1) * frame] for i in idx], axis=1)
    keep = np.concatenate([dry[:, i * frame:(i + 1) * frame] for i in idx], axis=1)
    return float(rms_db(wet) - rms_db(keep))


def measure(model, x):
    """Return (dry estimate, wetness in dB or None)."""
    dry = restore(model, x)
    return dry, wetness_db(x, dry)


def measure_fast(model, x, windows=3, win_s=9.0):
    """Wetness from a few windows instead of the whole track, so clean tracks cost seconds.

    Windows are the most active ones in each third of the track. Returns the mean reading or None.
    """
    n, w = x.shape[1], int(win_s * SR)
    if n <= windows * w * 1.5:
        return measure(model, x)[1]
    reads = []
    for k in range(windows):
        lo, hi = k * n // windows, (k + 1) * n // windows - w
        starts = range(lo, max(lo + 1, hi), max(1, w // 2))
        best = max(starts, key=lambda s0: rms_db(x[:, s0:s0 + w]))
        seg = x[:, best:best + w]
        r = wetness_db(seg, restore(model, seg))
        if r is not None:
            reads.append(r)
    return float(np.mean(reads)) if reads else None


def load_norms():
    return json.load(open(_NORMS)) if os.path.exists(_NORMS) else {}


def decide(wet_db, low_db, high_db, low_target_db=None, high_target_db=None):
    """Pick an action and the target wetness. Inside [low, high] nothing is touched.

    Outside it, the stem is brought to a target inside the range rather than only to its edge, so an
    over-wet vocal lands near the 75th percentile of produced vocals and a bone-dry one near the 25th.
    """
    if wet_db is None:
        return "skip", None
    if wet_db > high_db:
        return "reduce", high_db if high_target_db is None else high_target_db
    if wet_db < low_db:
        return "add", low_db if low_target_db is None else low_target_db
    return "keep", wet_db


def apply_decision(x, dry, wet_db, action, target_db, rng=None):
    """Move wetness to target. Reduce: blend toward the dry estimate. Add: a short plate on the dry estimate."""
    if action == "reduce":
        # wet component scales linearly with (1 - s), so pick s to land on the target level
        s = float(np.clip(1.0 - 10 ** ((target_db - wet_db) / 20), 0.0, 1.0))
        return x + s * (dry - x), dict(strength=s)
    if action == "add":
        rng = rng or np.random.default_rng(0)
        tail = synth_rir(rng, rt60=0.9, channels=x.shape[0])
        from scipy import signal
        wet = np.stack([signal.fftconvolve(dry[c], tail[c])[: x.shape[1]] for c in range(x.shape[0])])
        cur = 10 ** (wet_db / 20) * 10 ** (rms_db(dry) / 20)
        want = 10 ** (target_db / 20) * 10 ** (rms_db(dry) / 20)
        g = np.sqrt(max(want ** 2 - cur ** 2, 0.0)) / (10 ** (rms_db(wet) / 20) + 1e-9)
        return (x + g * wet).astype(np.float32), dict(added_gain=float(g))
    return x, {}
