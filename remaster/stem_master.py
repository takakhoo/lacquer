"""Stem-level mastering: split the mix, compare each instrument with the normal range of its kind, correct what
is outside it, and put the pieces back.

The pieces go back as differences: output = mix + sum(processed stem - separated stem). A stem that needed
nothing contributes exactly zero, so separation artifacts never reach the output through an untouched stem.
"""
from __future__ import annotations

import numpy as np

from .analysis import SR
from .master import loudness
from .mastering import tonal
from .mastering_norms import stem_ranges
from .stems import STEMS, separate


def stem_levels(stems, x, sr=SR):
    """Loudness of each stem relative to the mix, in LU. A silent stem reads -inf."""
    ref = loudness(x, sr)
    return {k: (loudness(stems[k], sr) - ref) for k in STEMS}


def master_stems(x, sr=SR, stems=None, backend="demucs", do_balance=True, do_tone=True, max_gain_db=6.0, present_db=-22.0, tone_strength=0.7):
    """Returns (audio, report). `stems` may be passed in to skip separation."""
    stems = stems or separate(x, backend=backend)
    levels = stem_levels(stems, x, sr)
    rep = dict(levels=levels, decisions=[], stems={}, normal={k: [stem_ranges(k)["level"][i] for i in (0, 2, 4)] for k in STEMS})
    y = x.astype(np.float32).copy()
    for name in STEMS:
        s, lv = stems[name], levels[name]
        if not np.isfinite(lv) or lv < present_db:
            rep["stems"][name] = dict(action="absent")
            continue
        rng = stem_ranges(name)
        new, r = s, dict(level_lu=float(lv))
        if do_tone:
            new, r["tone"] = tonal(s, rng, sr, strength=tone_strength)
            if r["tone"]["action"] == "eq":
                rep["decisions"].append(f"{name.capitalize()} tone: {r['tone']['largest_db']:+.1f} dB at {r['tone']['largest_hz']:.0f} Hz to bring it inside the range of {name} stems.")
        if do_balance:
            p5, p10, _, p90, p95 = rng["level"]
            g = float(np.clip(p10 - lv, 0, max_gain_db)) if lv < p5 else float(np.clip(p90 - lv, -max_gain_db, 0)) if lv > p95 else 0.0
            r["gain_db"] = g
            if g:
                new = new * 10 ** (g / 20)
                rep["decisions"].append(f"{name.capitalize()} level: {lv:.1f} LU relative to the mix, outside the normal {p5:.1f} to {p95:.1f}: moved by {g:+.1f} dB.")
        y += new - s
        rep["stems"][name] = r
    if not rep["decisions"]:
        rep["decisions"].append("Stems: levels and tone of every instrument inside the normal range, left alone.")
    return y, rep
