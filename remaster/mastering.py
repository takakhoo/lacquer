"""The mastering chain: measure, compare with the normal range of the style, act only outside it, measure again.

Order follows common mastering practice: clean-up, low-end mono, tonal balance, multiband dynamics, stereo
image, glue compression, loudness and true-peak limiting. Every step appends a plain-language decision.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .analysis import DYN_EDGES, SR, THIRD_OCT, WIDTH_EDGES, mastering_features
from .master import bus_compress, limiter_v2, loudness, true_peak_db
from .mastering_norms import ranges

PROFILES = dict(
    streaming=dict(lufs=-14.0, ceiling=-1.0, note="Spotify normalization level; AES TD1008 album level"),
    track=dict(lufs=-16.0, ceiling=-1.0, note="AES TD1008 track normalization level"),
    loud=dict(lufs=-9.0, ceiling=-2.0, note="louder than normalization; -2 dBTP because lossy encoders overshoot more on loud masters"),
    broadcast=dict(lufs=-23.0, ceiling=-1.0, note="EBU R128 programme loudness"),
    cinema=dict(lufs=-27.0, ceiling=-2.0, note="Netflix delivery level (dialogue-gated in the spec; programme loudness here)"),
)


def split_bands(x, edges, sr=SR, taps=4095):
    """Linear-phase complementary bands that sum back to x exactly. edges are the interior crossover frequencies."""
    lows = [np.stack([signal.fftconvolve(ch, signal.firwin(taps, e, fs=sr), mode="same") for ch in x]) for e in edges]
    return [lows[0]] + [lows[i] - lows[i - 1] for i in range(1, len(lows))] + [x - lows[-1]]


def reference_ranges(ref, sr=SR, tol=0.25):
    """Ranges that collapse onto a reference track: mastering to a reference is the same chain with a narrow target."""
    f = mastering_features(ref, sr)
    r = {k: (f[k] - tol, f[k], f[k] + tol) for k in ("plr", "width_all", "slope")}
    for k in ("ltas", "width", "band_crest", "band_spread"):
        v = np.asarray(f[k], dtype=np.float64)
        r[k] = (v - tol, v, v + tol)
    r["corr_low"] = (min(f["corr_low"], 0.9) - 0.02, f["corr_low"], 1.0)
    r["_lufs"], r["_true_peak"] = f["lufs"], f["true_peak"]
    return r


def _third_octave(sig, sr):
    f, p = signal.welch(sig, sr, nperseg=8192, noverlap=4096)
    return np.array([10 * np.log10(p[(f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))].sum() + 1e-20) for fc in THIRD_OCT])


def match_reference(x, ref, sr=SR, passes=2, max_db=12.0):
    """Reference mastering of tone and image in one step: the mid and the side channel each get a linear-phase
    EQ that moves their third-octave spectrum onto the reference's, relative to total level."""
    rm, rs = (ref[0] + ref[1]) / 2, (ref[0] - ref[1]) / 2
    tot_r = 10 * np.log10(np.mean(rm ** 2) + np.mean(rs ** 2) + 1e-20)
    tgt = [_third_octave(rm, sr) - tot_r, _third_octave(rs, sr) - tot_r]
    y, total = x.astype(np.float32), [np.zeros(len(THIRD_OCT)), np.zeros(len(THIRD_OCT))]
    for _ in range(passes):
        mid, side = (y[0] + y[1]) / 2, (y[0] - y[1]) / 2
        tot = 10 * np.log10(np.mean(mid ** 2) + np.mean(side ** 2) + 1e-20)
        out = []
        for ci, ch in enumerate((mid, side)):
            cur = _third_octave(ch, sr) - tot
            # bands with nothing in them on either side cannot be matched by an EQ (band-limited source, mono track)
            valid = (THIRD_OCT >= 30) & (THIRD_OCT <= 16000) & (cur > -90) & (tgt[ci] > -90) & (np.abs(tgt[ci] - cur) < 24)
            diff = np.where(valid, tgt[ci] - cur, 0.0)
            diff = diff - (np.mean(diff[valid]) if ci == 0 and valid.any() else 0.0)       # overall level is set later, by loudness
            corr = np.clip(ndimage.gaussian_filter1d(diff, 0.6, mode="nearest"), -max_db, max_db)
            total[ci] += corr
            freqs = np.concatenate([[0], THIRD_OCT, [sr / 2]])
            fir = signal.firwin2(4097, freqs, 10 ** (np.concatenate([[corr[0]], corr, [corr[-1]]]) / 20), fs=sr)
            out.append(signal.fftconvolve(ch, fir, mode="same"))
        y = np.stack([out[0] + out[1], out[0] - out[1]]).astype(np.float32)
    i = int(np.argmax(np.abs(total[0])))
    return y, dict(action="match", bands_hz=THIRD_OCT.tolist(), correction_db=total[0].tolist(), side_correction_db=total[1].tolist(),
                   largest_db=float(total[0][i]), largest_hz=float(THIRD_OCT[i]))


def tonal(x, rng, sr=SR, strength=0.7, max_db=6.0):
    """Third-octave bands outside the style's range are pulled to its edge with one smooth linear-phase EQ."""
    cur = mastering_features(x, sr)["ltas"]
    low, med, high = rng["ltas"]
    # bands far below the corpus are missing content (band-limited source); EQ cannot restore them
    valid = (THIRD_OCT >= 30) & (THIRD_OCT <= 16000) & (med - cur < 18)
    diff = np.where(cur > high, high - cur, np.where(cur < low, low - cur, 0.0))
    diff = ndimage.gaussian_filter1d(np.where(valid, diff, 0.0), 1.0, mode="nearest")
    corr = np.clip(diff * strength, -max_db, max_db)
    wanted = float(np.abs(diff).max())
    if np.abs(corr).max() < 0.25:
        return x, dict(action="keep", correction_db=corr.tolist(), wanted_db=wanted, bands_hz=THIRD_OCT.tolist(), measured_db=cur.tolist(),
                       target_db=med.tolist(), low_db=np.asarray(low).tolist(), high_db=np.asarray(high).tolist())
    freqs = np.concatenate([[0], THIRD_OCT, [sr / 2]])
    gains = 10 ** (np.concatenate([[corr[0]], corr, [corr[-1]]]) / 20)
    fir = signal.firwin2(4097, freqs, gains, fs=sr)
    y = np.stack([signal.fftconvolve(ch, fir, mode="same") for ch in x]).astype(np.float32)
    i = int(np.argmax(np.abs(corr)))
    return y, dict(action="eq", correction_db=corr.tolist(), largest_db=float(corr[i]), largest_hz=float(THIRD_OCT[i]), wanted_db=wanted,
                   bands_hz=THIRD_OCT.tolist(), measured_db=cur.tolist(), target_db=med.tolist(), low_db=np.asarray(low).tolist(), high_db=np.asarray(high).tolist())


def stereo_image(x, rng, feat, sr=SR, max_cut_db=6.0, max_boost_db=3.0):
    """Side level per band is moved to the edge of the style's range; a decorrelated low end is narrowed."""
    low, _, high = rng["width"]
    w = np.asarray(feat["width"])
    g = np.where(w > high, np.maximum(high - w, -max_cut_db), np.where((w < low) & (w > -40), np.minimum(low - w, max_boost_db), 0.0))
    # low end: narrow only when the channels are clearly decorrelated below 150 Hz (mono-sum loss), never as routine
    if feat["corr_low"] < 0.5:
        g[0] = min(g[0], -min(12.0, 24 * (0.5 - feat["corr_low"])))
    if np.abs(g).max() < 0.5:
        return x, dict(action="keep", side_gain_db=g.tolist())
    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
    bands = split_bands(side[None], WIDTH_EDGES[1:-1], sr)
    side2 = sum(b[0] * 10 ** (gi / 20) for b, gi in zip(bands, g))
    y = np.stack([mid + side2, mid - side2]).astype(np.float32)
    return y, dict(action="width", side_gain_db=g.tolist())


def multiband(x, rng, feat, sr=SR, max_excess_db=4.0):
    """A band whose peaks stand further above its level than the style's range gets 2:1 compression on the excess."""
    _, _, high = rng["band_crest"]
    excess = np.clip(np.asarray(feat["band_crest"]) - high, 0, max_excess_db)
    if excess.max() < 0.5:
        return x, dict(action="keep", excess_db=excess.tolist())
    bands, out, gr = split_bands(x, DYN_EDGES[1:-1], sr), [], []
    for b, e in zip(bands, excess):
        if e < 0.5:
            out.append(b); gr.append(0.0)
            continue
        level = 20 * np.log10(np.abs(b).max(axis=0)[::64] + 1e-7)
        y, r = bus_compress(b, sr, threshold_db=float(np.percentile(level, 99.5)) - 2.0 * e, ratio=2.0, attack_ms=10.0, release_ms=120.0)
        out.append(y); gr.append(r)
    return sum(out).astype(np.float32), dict(action="compress", excess_db=excess.tolist(), max_gr_db=gr)


def glue(x, rng, sr=SR):
    """Peak-to-loudness ratio against the style's range: compress the excess, protect what is already dense."""
    lufs = loudness(x, sr)
    if not np.isfinite(lufs):
        return x, dict(action="skip")
    low, _, high = rng["plr"]
    plr = float(true_peak_db(x, sr)) - lufs
    rep = dict(plr_db=plr, normal_range_db=[low, high])
    if plr > high:
        excess = min(plr - high, 4.0)
        level = 20 * np.log10(np.abs(x).max(axis=0)[::64] + 1e-7)
        y, gr = bus_compress(x, sr, threshold_db=float(np.percentile(level, 99.5)) - 2.0 * excess, ratio=2.0, attack_ms=25.0, release_ms=250.0)
        rep.update(action="compress", max_gr_db=gr)
        return y, rep
    rep["action"] = "protect" if plr < low else "keep"
    return x, rep


def finish(x, sr=SR, target_lufs=-14.0, ceiling_db=None, max_limit_db=3.0, clip_db=1.5):
    """Loudness and peaks. The limiter is given a budget (3 dB by default, the range engineers call transparent);
    a target that would need more is not reached, and the report says by how much. The ceiling is -1 dBTP, or
    -2 dBTP for masters louder than -14 LUFS, where lossy encoders overshoot more.
    """
    rep = {}
    lufs = loudness(x, sr)
    if not np.isfinite(lufs):
        return x, dict(limiter=dict(max_gr_db=0.0), ceiling_db=-1.0, shortfall_db=0.0)
    if ceiling_db is None:
        ceiling_db = -2.0 if target_lufs > -14.0 else -1.0
    headroom = ceiling_db - true_peak_db(x, sr)
    gain = min(target_lufs - lufs, headroom + max_limit_db)
    y, lim = limiter_v2(x * 10 ** (gain / 20), sr, ceiling_db=ceiling_db, clip_db=clip_db)
    l2 = loudness(y, sr)
    if target_lufs - l2 > 0.3 and gain < target_lufs - lufs - 0.1:
        pass                                                   # budget reached: stay below the target
    elif target_lufs - l2 > 0.3:
        y, lim = limiter_v2(x * 10 ** ((gain + min(target_lufs - l2, 1.5)) / 20), sr, ceiling_db=ceiling_db, clip_db=clip_db)
        l2 = loudness(y, sr)
    rep.update(gain_db=float(gain), limiter=lim, ceiling_db=float(ceiling_db), shortfall_db=float(max(0.0, target_lufs - l2)))
    return y, rep


def distance(feat, rng, keys=("ltas", "width", "band_crest", "plr", "corr_low")):
    """How far a track sits outside the style's range: RMS over features of the excess beyond the range, in dB."""
    ex = []
    for k in keys:
        low, _, high = rng[k]
        v = np.atleast_1d(np.asarray(feat[k], dtype=np.float64))
        low, high = np.atleast_1d(low), np.atleast_1d(high)
        e = np.where(v > high, v - high, np.where(v < low, low - v, 0.0))
        if k == "ltas":
            e = e[(THIRD_OCT >= 30) & (THIRD_OCT <= 12500)]
        if k == "corr_low":
            e = e * 10
        ex.append(e)
    return float(np.sqrt(np.mean(np.concatenate(ex) ** 2)))


def master_track(x, sr=SR, genre="all", profile="streaming", target_lufs=None, ceiling_db=None, lo=10, hi=90,
                 do_tonal=True, do_multiband=True, do_width=True, do_glue=True, norms=None, rng=None, reference=None, match_reference_peak=False):
    """Returns (audio, report). report["decisions"] lists what was measured and done, in order."""
    # blind moves stay inside what mastering engineers call normal (about 1.5 dB per band); a reference lifts that
    strength, max_db = 0.7, 1.5
    if reference is not None:
        # a reference replaces the style's range, and its loudness the delivery target unless one was given
        rng, strength, max_db = reference_ranges(reference, sr), 1.0, 12.0
        target_lufs = rng["_lufs"] if target_lufs is None and np.isfinite(rng["_lufs"]) else target_lufs
        if match_reference_peak and ceiling_db is None:
            ceiling_db = min(0.0, rng["_true_peak"])
    rng = rng or ranges(genre, lo, hi, norms)
    prof = PROFILES[profile]
    target_lufs = prof["lufs"] if target_lufs is None else target_lufs
    ceiling_db = prof.get("ceiling") if ceiling_db is None else ceiling_db
    before = mastering_features(x, sr)
    rep = dict(genre=genre, profile=profile, decisions=[], before={k: before[k] for k in ("lufs", "lra", "plr", "true_peak", "slope", "width_all", "corr_low")},
               distance_before=distance(before, rng))
    say = rep["decisions"].append
    y = x.astype(np.float32)
    # no routine high-pass or DC removal: only when the reading shows a problem
    if before["dc"] > 1e-3 or before["sub"] > rng.get("sub", (0, 0, 0.0))[2] + 6.0:
        b, a = signal.butter(2, 25 / (sr / 2), "high")
        y = signal.lfilter(b, a, y, axis=1).astype(np.float32)
        say("Clean-up: DC offset or excess sub-bass found, high-passed at 25 Hz.")
    if do_tonal and reference is not None:
        y, rep["tonal"] = match_reference(y, reference, sr)
        t = rep["tonal"]
        say(f"Tone and image: mid and side spectra matched to the reference, largest move {t['largest_db']:+.1f} dB at {t['largest_hz']:.0f} Hz.")
        do_width = False
    elif do_tonal:
        y, rep["tonal"] = tonal(y, rng, sr, strength, max_db)
        t = rep["tonal"]
        if t["action"] == "eq":
            say(f"Tone: largest correction {t['largest_db']:+.1f} dB at {t['largest_hz']:.0f} Hz."
                + (f" The reading is {t['wanted_db']:.1f} dB outside the range, more than mastering should move: a mix or stem problem." if reference is None and t["wanted_db"] > 4.0 else ""))
        else:
            say("Tone: inside the range of the style, left alone.")
    if do_multiband:
        y, rep["multiband"] = multiband(y, rng, mastering_features(y, sr), sr)
        m = rep["multiband"]
        names = ("sub and bass", "low mids", "upper mids", "highs")
        say("Band dynamics: " + ", ".join(f"{n} peaks {e:.1f} dB over the range, compressed" for n, e in zip(names, m["excess_db"]) if e >= 0.5) + "."
            if m["action"] == "compress" else "Band dynamics: inside the range, left alone.")
    if do_width:
        y, rep["width"] = stereo_image(y, rng, mastering_features(y, sr), sr, **(dict(max_cut_db=12.0, max_boost_db=12.0) if reference is not None else {}))
        if reference is not None and rep["width"]["action"] == "width":
            y, _ = stereo_image(y, rng, mastering_features(y, sr), sr, max_cut_db=6.0, max_boost_db=6.0)   # second pass, as for tone
        w = rep["width"]
        say("Stereo image: side level changed by " + ", ".join(f"{g:+.1f} dB" for g in w["side_gain_db"]) + " (low to high band)."
            if w["action"] == "width" else "Stereo image: inside the range, left alone.")
    max_limit = 6.0 if profile == "loud" or reference is not None else 3.0
    if do_glue:
        y, rep["glue"] = glue(y, rng, sr)
        g = rep["glue"]
        if g["action"] == "compress":
            say(f"Dynamics: peak-to-loudness ratio {g['plr_db']:.1f} dB is above the range ({g['normal_range_db'][1]:.1f}): 2:1 glue compression, {g['max_gr_db']:.1f} dB at most.")
        elif g["action"] == "protect":
            max_limit = 1.0
            say(f"Dynamics: already dense (peak-to-loudness {g['plr_db']:.1f} dB): no compression, limiter held back.")
        elif g["action"] == "keep":
            say(f"Dynamics: peak-to-loudness ratio {g['plr_db']:.1f} dB is normal for the style: left alone.")
    y, rep["finish"] = finish(y, sr, target_lufs, ceiling_db, max_limit)
    after = mastering_features(y, sr)
    rep["after"] = {k: after[k] for k in rep["before"]}
    rep["distance_after"] = distance(after, rng)
    f = rep["finish"]
    say(f"Loudness: {before['lufs']:.1f} to {after['lufs']:.1f} LUFS (target {target_lufs:.0f}), true peak {after['true_peak']:.1f} dBTP, limiter {f['limiter']['max_gr_db']:.1f} dB at most."
        + (f" Stopped {f['shortfall_db']:.1f} LU short of the target to stay inside the {max_limit:.0f} dB limiting budget." if f["shortfall_db"] > 0.5 else ""))
    if np.isfinite(before["lra"]) and np.isfinite(after["lra"]) and before["lra"] - after["lra"] > 1.0:
        say(f"Loudness range fell from {before['lra']:.1f} to {after['lra']:.1f} LU: macro-dynamics were reduced.")
    return y, rep
