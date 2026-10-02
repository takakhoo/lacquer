"""Stem-level mastering: split the mix, compare each instrument with the normal range of its kind, correct what
is outside it, and put the pieces back.

The pieces go back as differences: output = mix + sum(processed stem - separated stem). A stem that needed
nothing contributes exactly zero, so separation artifacts never reach the output through an untouched stem.
"""
from __future__ import annotations

import numpy as np

from .analysis import SR, loudest_window
from .master import loudness
from .mastering import tonal
from .mastering_norms import stem_ranges
from .stems import STEMS, separate


def stem_levels(stems, x, sr=SR):
    """Loudness of each stem relative to the mix, in LU, over the loudest 30 s of the mix. A silent stem reads -inf."""
    a, b = loudest_window(x, sr)
    ref = loudness(x[:, a:b], sr)
    return {k: (loudness(stems[k][:, a:b], sr) - ref) for k in STEMS}


def stem_notes(stems, x, sr=SR, present_db=-22.0):
    """Advisory readings per instrument against stems of professional mixes (5th to 95th percentile). Nothing is
    changed on the basis of these: instrument tone and dynamics vary too much between songs to correct blind (E30)."""
    from .analysis import mastering_features
    a, b = loudest_window(x, sr)
    levels, notes = stem_levels(stems, x, sr), []
    checks = (("slope", "tone", "darker than", "brighter than", "dB/octave"), ("plr", "dynamics", "denser than", "more dynamic than", "dB peak-to-loudness"),
              ("width_all", "stereo width", "narrower than", "wider than", "dB side/mid"))
    for name in STEMS:
        if not np.isfinite(levels[name]) or levels[name] < present_db:
            continue
        f, rng = mastering_features(stems[name][:, a:b], sr), stem_ranges(name, 5, 95)
        for key, label, low_word, high_word, unit in checks:
            lo, _, hi = rng[key]
            v = f[key]
            if not np.isfinite(v) or (key == "width_all" and v < -60):
                continue
            if v < lo or v > hi:
                kind = dict(vocals="vocal", drums="drum", bass="bass", other="accompaniment")[name]
                notes.append(f"{name.capitalize()} {label}: {v:.1f} {unit}, {low_word if v < lo else high_word} 95% of {kind} stems in professional mixes ({lo:.1f} to {hi:.1f}).")
    return notes


def master_stems(x, sr=SR, stems=None, backend="demucs", do_balance=True, do_tone=False, max_gain_db=6.0, present_db=-22.0, tone_strength=0.7,
                 only=("vocals",), margin_db=0.0):
    """Returns (audio, report). `stems` may be passed in to skip separation.

    Tone correction per stem is off by default: instrument tone varies more between songs than a typical fault
    (experiment E30), so pulling a stem toward the norm of its kind does more harm than good. Only the vocal level
    is corrected by default: with all four stems a third of unmodified songs has some stem outside its range,
    with the vocal alone 8% (E31). The other stems are reported. `margin_db` widens the normal range.
    """
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
        if do_balance and name in only:
            p5, p10, _, p90, p95 = rng["level"]
            g = float(np.clip(p10 - lv, 0, max_gain_db)) if lv < p5 - margin_db else float(np.clip(p90 - lv, -max_gain_db, 0)) if lv > p95 + margin_db else 0.0
            r["gain_db"] = g
            if g:
                new = new * 10 ** (g / 20)
                rep["decisions"].append(f"{name.capitalize()} level: {lv:.1f} LU relative to the mix, outside the normal {p5:.1f} to {p95:.1f}: moved by {g:+.1f} dB.")
        elif do_balance:
            p5, _, _, _, p95 = rng["level"]
            if lv < p5 or lv > p95:
                rep.setdefault("level_notes", []).append(f"{name.capitalize()} level: {lv:.1f} LU relative to the mix, outside the normal {p5:.1f} to {p95:.1f} of professional mixes.")
        y += new - s
        rep["stems"][name] = r
    rep["notes"] = rep.pop("level_notes", []) + stem_notes(stems, x, sr, present_db)
    if not rep["decisions"]:
        rep["decisions"].append("Instrument balance: vocal level inside the normal range of professional mixes, left alone.")
    return y, rep
