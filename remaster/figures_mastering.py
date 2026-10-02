"""Figures for the mastering analysis: what released music measures like, per genre, against unmastered mixes.

  python -m remaster.figures_mastering        # writes docs/figures/mastering_*.png
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from .analysis import THIRD_OCT, WIDTH_EDGES
from .figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, RED, ROOT, TINT, YELLOW, save

EV = os.path.join(ROOT, "docs", "evidence", "mastering")
NORMS = os.path.join(os.path.dirname(__file__), "mastering_norms.json")


def _ax(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)


def _col(t, names, key, p):
    return np.array(t[f"p{p}"])[[names.index(k) for k in key]] if isinstance(key, list) else t[f"p{p}"][names.index(key)]


def fig_corpus():
    n = json.load(open(NORMS))
    names, G = n["features"], n["genres"]
    lt = [f"ltas_{i}" for i in range(len(THIRD_OCT))]
    sel = (THIRD_OCT >= 30) & (THIRD_OCT <= 14000)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), dpi=160)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.08, wspace=0.34, hspace=0.42)
    fig.text(0.07, 0.955, "What released music measures like", fontsize=14, fontweight="bold")
    fig.text(0.07, 0.925, f"{G['all']['n']:,} released tracks (FMA-large, training split) against {n['unmastered_mix']['n']} unmastered professional mixes (MUSDB18-HQ). "
             "Bars and bands span the 10th to 90th percentile.", fontsize=9, color=INK2)

    ax = axs[0, 0]
    _ax(ax, "Tone: third-octave spectrum", "frequency (Hz)", "level relative to total (dB)")
    ax.fill_between(THIRD_OCT[sel], _col(G["all"], names, lt, 10)[sel], _col(G["all"], names, lt, 90)[sel], color=TINT[INK2], lw=0)
    ax.plot(THIRD_OCT[sel], _col(G["all"], names, lt, 50)[sel], color=INK, lw=2, label="released, median")
    ax.plot(THIRD_OCT[sel], _col(n["unmastered_mix"], names, lt, 50)[sel], color=BLUE, lw=2, label="unmastered mixes")
    for g, c in (("Classical", AQUA), ("Hip-Hop", ORANGE)):
        ax.plot(THIRD_OCT[sel], _col(G[g], names, lt, 50)[sel], color=c, lw=1.4, label=g)
    ax.set_xscale("log"); ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000]); ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k"])
    ax.set_ylim(-45, -5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    order = sorted([g for g in G if g != "all"], key=lambda g: _col(G[g], names, "lufs", 50))
    rows = order + ["all", "unmastered"]
    tab = lambda g: n["unmastered_mix"] if g == "unmastered" else G[g]
    for ax, key, ttl, xl in ((axs[0, 1], "lufs", "Loudness by genre", "integrated loudness (LUFS)"), (axs[0, 2], "plr", "Peak-to-loudness ratio by genre", "true peak minus loudness (dB)")):
        _ax(ax, ttl, xl)
        for i, g in enumerate(rows):
            t = tab(g)
            c = BLUE if g == "unmastered" else INK if g == "all" else INK2
            ax.plot([_col(t, names, key, 10), _col(t, names, key, 90)], [i, i], color=c, lw=5 if g in ("all", "unmastered") else 3.5, alpha=0.9 if g in ("all", "unmastered") else 0.45, solid_capstyle="round")
            ax.plot([_col(t, names, key, 50)], [i], "o", color=c, ms=5.5, mec="white", mew=1.2)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([("all released" if g == "all" else "unmastered mixes" if g == "unmastered" else g) for g in rows], fontsize=8)
        ax.grid(axis="y", visible=False)

    ax = axs[1, 0]
    _ax(ax, "Stereo image: side level by band", None, "side minus mid (dB)")
    wk = [f"width_{i}" for i in range(len(WIDTH_EDGES) - 1)]
    x = np.arange(len(wk))
    for off, t, c, lab in ((-0.16, G["all"], INK, "released"), (0.16, n["unmastered_mix"], BLUE, "unmastered mixes")):
        lo, md, hi = (np.maximum(_col(t, names, wk, p), -45) for p in (10, 50, 90))
        ax.vlines(x + off, lo, hi, color=c, lw=6, alpha=0.75, label=lab)
        ax.plot(x + off, md, "o", color=c, ms=5.5, mec="white", mew=1.2)
    ax.set_xticks(x); ax.set_xticklabels(["< 120 Hz", "120-500", "500-2k", "2k-8k", "> 8k"], fontsize=8)
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axs[1, 1]
    _ax(ax, "Dynamics inside four bands", None, "peak over RMS in the band (dB)")
    ck = [f"band_crest_{i}" for i in range(4)]
    x = np.arange(4)
    for off, t, c, lab in ((-0.16, G["all"], INK, "released"), (0.16, n["unmastered_mix"], BLUE, "unmastered mixes")):
        lo, md, hi = (_col(t, names, ck, p) for p in (10, 50, 90))
        ax.vlines(x + off, lo, hi, color=c, lw=6, alpha=0.75, label=lab)
        ax.plot(x + off, md, "o", color=c, ms=5.5, mec="white", mew=1.2)
    ax.set_xticks(x); ax.set_xticklabels(["< 120 Hz", "120 Hz-1 k", "1 k-6 k", "> 6 k"], fontsize=8)
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axs[1, 2]
    _ax(ax, "Instrument balance in professional mixes", "stem loudness relative to the mix (LU)")
    stems = ("vocals", "other", "drums", "bass")
    for i, s in enumerate(stems):
        lv = n["stems"][s]["level_re_mix"]
        ax.plot([lv["p5"], lv["p95"]], [i, i], color=INK2, lw=3, alpha=0.4, solid_capstyle="round")
        ax.plot([lv["p25"], lv["p75"]], [i, i], color=INK2, lw=7, alpha=0.8, solid_capstyle="round")
        ax.plot([lv["p50"]], [i], "o", color=INK, ms=6, mec="white", mew=1.2)
        ax.text(lv["p95"] + 0.25, i, f"{lv['p50']:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(stems))); ax.set_yticklabels(["vocals", "other (guitars, keys)", "drums", "bass"], fontsize=8)
    ax.grid(axis="y", visible=False); ax.set_ylim(-0.6, len(stems) - 0.4)
    ax.text(0.0, -0.2, "thin: 5th to 95th percentile, thick: 25th to 75th, dot: median", transform=ax.transAxes, fontsize=7.5, color=INK2)
    save(fig, "mastering_corpus.png")


def fig_limiter():
    d = json.load(open(os.path.join(ROOT, "docs", "evidence", "loudness", "loudness_tp.json")))["summary"]["-9.0"]
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=160)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.12)
    fig.text(0.1, 0.945, "Reaching -9 LUFS under a -1 dBTP ceiling", fontsize=13, fontweight="bold")
    fig.text(0.1, 0.875, "50 unmastered mixes (MUSDB18-HQ test). Each limiter is driven to the same loudness\nand the same true peak. Lower left is better.", fontsize=8.6, color=INK2, linespacing=1.5)
    _ax(ax, "", "gain movement: spread of the applied gain (dB)", "distortion not explained by a smooth gain (dB)")
    ours = [("v2_noslow_clip0", "no clip stage"), ("v2_noslow_clip1", "1 dB clip stage"), ("v2_slow250_clip2", "2 dB"), ("v2_clip3", "3 dB")]
    xs, ys = [d[k]["gain_spread_db"] for k, _ in ours], [d[k]["distortion_db"] for k, _ in ours]
    ax.plot(xs, ys, "-", color=ORANGE, lw=1.6, zorder=2)
    ax.plot(xs, ys, "o", color=ORANGE, ms=8, mec="white", mew=1.4, zorder=3)
    for (k, lab), x, y in zip(ours, xs, ys):
        ax.annotate(lab, (x, y), xytext=(-8, -5), textcoords="offset points", ha="right", va="top", fontsize=8, color=INK2)
    ax.annotate("ours, two-stage limiter", ((xs[1] + xs[2]) / 2, (ys[1] + ys[2]) / 2), xytext=(10, 12), textcoords="offset points", fontsize=9, color=INK, fontweight="bold")
    for k, lab, dx, dy, ha in (("matchering", "Matchering 2.0", 8, 0, "left"), ("ffmpeg_alimiter", "ffmpeg alimiter", 8, 0, "left"), ("clip", "hard clip", 8, 0, "left"),
                               ("ours_v1", "ours, single stage", -8, 0, "right")):
        ax.plot([d[k]["gain_spread_db"]], [d[k]["distortion_db"]], "s" if k != "ours_v1" else "o", color=INK2 if k != "ours_v1" else TINT[ORANGE], ms=7.5, mec=INK2 if k == "ours_v1" else "white", mew=1.2, zorder=3)
        ax.annotate(lab, (d[k]["gain_spread_db"], d[k]["distortion_db"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 7.6); ax.set_ylim(-48, -20)
    save(fig, "mastering_limiter.png")


def fig_recovery():
    t = json.load(open(os.path.join(EV, "tone_identifiability.json")))
    u = json.load(open(os.path.join(EV, "unmaster.json")))["summary"]
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=160, gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.2, right=0.975, top=0.78, bottom=0.14, wspace=0.62)
    fig.text(0.035, 0.93, "Why blind tone correction is bounded, and what a reference buys", fontsize=13, fontweight="bold")
    fig.text(0.035, 0.87, "Third-octave tone, RMS over bands in dB. Left: 506 held-out released tracks. Right: 200 held-out tracks with a spectral tilt of 0.6 to 1.6 dB per octave.", fontsize=8.6, color=INK2)

    ax = axs[0]
    _ax(ax, "How far is a track's tone from the best guess?", "distance (dB)")
    p = t["predictability_db"]
    items = [("the average of all music", p["population_mean"], INK2), ("the average of its genre", p["genre_mean"], INK2), ("its 20 nearest neighbours", p["nearest_20"], INK2),
             ("a typical tonal fault", t["released_corpus"]["fault"], ORANGE)]
    for i, (lab, v, c) in enumerate(items[::-1]):
        ax.barh(i, v, height=0.56, color=c, alpha=0.85 if c == ORANGE else 0.55)
        ax.text(v + 0.08, i, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items[::-1]], fontsize=8.5)
    ax.set_xlim(0, 6.2); ax.grid(axis="y", visible=False)

    ax = axs[1]
    _ax(ax, "Tone error left after mastering a tilted copy", "tone error against the original (dB)")
    d = u["tilt"]
    items = [("damaged", d["damaged"]["tone"], INK2), ("blind, norms of all music", d["global"]["tone"], INK2), ("blind, norms of its genre", d["genre"]["tone"], INK2),
             ("Matchering 2.0, original as reference", d["matchering"]["tone"], BLUE), ("ours, original as reference", d["reference"]["tone"], ORANGE)]
    for i, (lab, v, c) in enumerate(items[::-1]):
        ax.barh(i, v, height=0.56, color=c, alpha=0.85 if c != INK2 else 0.55)
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items[::-1]], fontsize=8.5)
    ax.set_xlim(0, 3.6); ax.grid(axis="y", visible=False)
    save(fig, "mastering_recovery.png")


def fig_limiter_trace():
    from scipy import signal
    d = np.load(os.path.join(ROOT, "docs", "evidence", "loudness", "limiter_trace.npz"))
    sr, x, c = int(d["sr"]), d["input"], int(d["peak_index"])
    cols = [("clip", "hard clip"), ("matchering", "Matchering 2.0"), ("ozone", "Ozone 9 Maximizer"), ("ours", "ours, two-stage")]
    fig, axs = plt.subplots(3, 4, figsize=(15, 8.2), dpi=160, gridspec_kw=dict(height_ratios=[1, 0.75, 1.15]))
    fig.subplots_adjust(left=0.065, right=0.93, top=0.86, bottom=0.07, wspace=0.16, hspace=0.42)
    fig.text(0.065, 0.955, "What four limiters do to the same two seconds", fontsize=14, fontweight="bold")
    fig.text(0.065, 0.92, f"One unmastered mix, every limiter driven to the loudness ({float(d['target_lufs']):.1f} LUFS) and true peak of the Ozone version. "
             "The distortion row is what a smooth gain cannot explain.", fontsize=9, color=INK2)
    t = (np.arange(x.shape[1]) - c) / sr * 1000
    w = slice(max(0, c - int(0.004 * sr)), c + int(0.008 * sr))
    ref = 10 * np.log10(np.mean(x ** 2) + 1e-12)
    im = None
    for j, (k, name) in enumerate(cols):
        y, g, r = d[k], d[k + "_gain_db"], d[k + "_residual"]
        ax = axs[0, j]
        _ax(ax, name, "time around the peak (ms)", "sample value" if j == 0 else None)
        ax.plot(t[w], (x[0] * 10 ** (g / 20))[w], color=INK2, lw=1.0, alpha=0.55, label="input x fitted gain")
        ax.plot(t[w], y[0][w], color=ORANGE if k == "ours" else INK, lw=1.5, label="output")
        if j == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=1)
        ax = axs[1, j]
        _ax(ax, "", "time (s)", "applied gain (dB)" if j == 0 else None)
        tt = np.arange(len(g)) / sr
        ax.plot(tt, g - np.median(g), color=ORANGE if k == "ours" else INK, lw=1.1)
        ax.set_ylim(-9, 3)
        ax.text(0.02, 0.06, f"spread {np.percentile(g, 95) - np.percentile(g, 5):.1f} dB", transform=ax.transAxes, fontsize=8, color=INK2)
        ax = axs[2, j]
        f, tm, S = signal.spectrogram(r.mean(0), sr, nperseg=2048, noverlap=1536)
        db = 10 * np.log10(S * (f[1] - f[0]) * len(f) + 1e-14) - ref
        im = ax.pcolormesh(tm, f / 1000, db, vmin=-80, vmax=-20, cmap="magma", shading="auto", rasterized=True)
        ax.set_title("", loc="left"); ax.set_xlabel("time (s)", fontsize=8.5); ax.tick_params(length=0, labelsize=8)
        if j == 0:
            ax.set_ylabel("distortion: frequency (kHz)", fontsize=8.5)
        rel = 10 * np.log10(np.mean(r ** 2) / np.mean(y ** 2) + 1e-14)
        ax.text(0.02, 0.92, f"{rel:.0f} dB re signal", transform=ax.transAxes, fontsize=8, color="white", va="top")
        if j > 0:
            for a_ in axs[:, j]:
                a_.set_yticklabels([])
    for j in range(4):
        axs[0, j].set_ylim(axs[0, 0].get_ylim())
    cax = fig.add_axes([0.94, 0.07, 0.008, 0.3])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7, length=0); cb.set_label("dB re input level", fontsize=7.5); cb.outline.set_visible(False)
    save(fig, "mastering_limiter_trace.png")


if __name__ == "__main__":
    fig_corpus(); fig_limiter(); fig_recovery(); fig_limiter_trace()
