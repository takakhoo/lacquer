"""Paper-scale figures: drawn at the printed size (7.1 in across two columns, 3.4 in for one) with fonts in true
points, so they survive the DAFx two-column layout.

  python -m remaster.figures_paper      # writes docs/figures/paper_*.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import signal

from .analysis import THIRD_OCT
from .figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, OUT, RED, ROOT, SURFACE, TINT, YELLOW
from .figures_deep import HOP, NFFT, SR, db, logfreq, inset
from .figures_render import prepare, spec
from .mastering_norms import stem_ranges
import matplotlib.pyplot as plt

DPI = 400
F = dict(title=6.6, note=5.4, kind=4.9, tick=4.6, lab=4.9, ann=5.0)
ANN = dict(fontsize=F["ann"], color="white", zorder=7, bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.2))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    fig.patch.set_facecolor("white")
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, w); bg.set_ylim(0, h); bg.axis("off")
    return fig, bg


def tidy(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color(GRID); s.set_linewidth(0.5)
    ax.tick_params(labelsize=F["tick"], length=1.5, width=0.4, pad=1, color=INK2, labelcolor=INK2)
    return ax


def kind_label(ax, text):
    ax.text(0.97, 0.94, text, transform=ax.transAxes, fontsize=F["kind"], color=INK2, ha="right", va="top", style="italic", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))


def fig_pipeline(D):
    """Nine cards, two rows, full text width."""
    W, H = 7.1, 3.3
    fig, bg = canvas(W, H)
    cw, ch, gap = 1.34, 1.43, 0.1
    x0 = 0.0
    ys = {"REPAIR": H - 0.16 - ch, "MASTER": H - 0.16 - ch - 0.22 - ch}
    cells = {}
    for lab, n in (("REPAIR", 5), ("MASTER", 4)):
        y = ys[lab]
        bg.text(x0 + 0.02, y + ch + 0.035, lab, fontsize=5.2, fontweight="bold", color=INK2, va="bottom")
        for i in range(n):
            x = x0 + i * (cw + gap)
            cells[(lab, i)] = (x, y)
            if i < n - 1:
                bg.add_patch(FancyArrowPatch((x + cw, y + ch / 2), (x + cw + gap, y + ch / 2), arrowstyle="-|>", mutation_scale=5, lw=0.7, color=INK2, zorder=4))
    xr = x0 + 4 * (cw + gap) + cw / 2
    ym = ys["MASTER"] + ch / 2
    bg.plot([xr, xr, x0 + 3 * (cw + gap) + cw + gap * 0.6], [ys["REPAIR"], ym, ym], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=0)
    bg.text(xr - 0.04, (ys["REPAIR"] + ym) / 2, "repaired mix", fontsize=F["note"], color=INK2, va="center", ha="right", rotation=90)

    def cell(key, color, n, name, decision, kind):
        x, y = cells[key]
        bg.add_patch(FancyBboxPatch((x, y), cw, ch, boxstyle="round,pad=0.01,rounding_size=0.05", fc=TINT[color], ec=color, lw=0.8, zorder=1))
        bg.text(x + 0.06, y + ch - 0.05, f"{n}  {name}", fontsize=F["title"], fontweight="bold", color=INK, ha="left", va="top", zorder=5)
        ax = tidy(inset(fig, bg, x + 0.1, y + 0.55, cw - 0.17, ch - 0.55 - 0.24))
        ax.tick_params(labelsize=F["tick"], length=1.5, width=0.4, pad=1)
        bg.text(x + 0.06, y + 0.36, decision, fontsize=F["note"], color=INK2, ha="left", va="top", zorder=5, linespacing=1.2)
        kind_label(ax, kind)
        return ax

    LBL = dict(fontsize=F["lab"], labelpad=0.5)
    t, clean, delay_ms = D["t"], D["clean"], D["delay_ms"]
    # 1 clipping
    ax = cell(("REPAIR", 0), AQUA, 1, "Clipping", "flat tops on 3.5% of samples:\npeaks rebuilt by sparse\nreconstruction (A-SPADE)", "12 ms of waveform")
    c = int(np.argmax(np.abs(clean[0]))); sl = slice(max(0, c - 260), c + 270)
    tt = (np.arange(sl.start, sl.stop) - c) / SR * 1000
    ax.plot(tt, D["clipped"][0][sl], color=INK2, lw=0.6, alpha=0.8, label="clipped")
    ax.plot(tt, D["declipped"][0][sl], color=ORANGE, lw=0.8, label="rebuilt")
    ax.axhline(D["thr"], color=RED, lw=0.4, ls="--"); ax.axhline(-D["thr"], color=RED, lw=0.4, ls="--")
    ax.set_yticks([]); ax.set_xticks([])
    ax.legend(frameon=False, fontsize=F["kind"], loc="lower left", handlelength=1.0, borderaxespad=0.2)
    # 2 echo
    ax = cell(("REPAIR", 1), AQUA, 2, "Echo", f"cepstral peak at {delay_ms:.0f} ms ({D['z']:.0f}\nsigma), off the tempo grid:\nexact inverse filter applied", "real cepstrum")
    cep = np.asarray(t["cep_deg"]); q = np.arange(len(cep)) / SR * 1000
    m = (q > 20) & (q < 650)
    ax.plot(q[m], cep[m], color=INK, lw=0.45)
    pk = int(np.argmax(np.where(m, cep, -1)))
    ax.plot(q[pk], cep[pk], "o", color=ORANGE, ms=2.5, mec="white", mew=0.5)
    ax.annotate(f"{delay_ms:.0f} ms", (q[pk], cep[pk]), xytext=(6, -2), textcoords="offset points", fontsize=F["ann"], color=INK, fontweight="bold")
    ax.set_xlabel("quefrency (ms)", **LBL); ax.set_yticks([]); ax.set_xticks([100, 300, 500])
    ax.set_ylim(cep[m].min() * 1.1, cep[pk] * 1.35)
    # 3 room reverb
    ax = cell(("REPAIR", 2), ORANGE, 3, "Room reverb", "excess reverb reads -0.5 dB,\nfar above the -18 dB gate:\nmask applied at full strength", "predicted mask (dB)")
    mk = np.abs(np.asarray(t["mask"])).mean(0)
    img = logfreq(np.clip(20 * np.log10(mk + 1e-6), -18, 6), rows=200)
    ax.imshow(img, origin="lower", aspect="auto", cmap="RdYlBu_r", vmin=-18, vmax=6, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.04, "blue: attenuated   red: kept", transform=ax.transAxes, fontsize=F["kind"], color=INK, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
    # 4 vocal reverb
    ax = cell(("REPAIR", 3), ORANGE, 4, "Vocal reverb", "vocal stem wetness -12.4 dB,\ninside the normal -32 to\n+4 dB: left alone", "wetness meter")
    ax.set_xlim(-40, 12); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((-32, 0.3), 36, 0.3, fc=TINT[BLUE], ec=BLUE, lw=0.5))
    ax.add_patch(Rectangle((4, 0.3), 4, 0.3, fc=TINT[RED], ec=RED, lw=0.5))
    ax.plot([-12.4], [0.45], "o", color=INK, ms=3.5, mec="white", mew=0.6)
    ax.text(-14, 0.7, "normal", fontsize=F["kind"], color=INK2, ha="center")
    ax.text(6, 0.64, "too\nwet", fontsize=F["kind"], color=RED, ha="center", va="bottom", linespacing=1.0)
    ax.set_yticks([]); ax.set_xticks([-30, -20, -10, 0]); ax.set_xlabel("vocal reverb excess (dB)", **LBL)
    # 5 instrument balance
    ax = cell(("REPAIR", 4), ORANGE, 5, "Instrument balance", "separated-stem loudness vs.\nprofessional mixes: reported;\nvocal moved only on request", "stem level (LU)")
    names = ("vocals", "drums", "bass", "other")
    for i, nme in enumerate(names):
        r = stem_ranges(nme)["level"]
        ax.plot([r[0], r[4]], [i, i], color=BLUE, lw=3, alpha=0.35, solid_capstyle="round")
        if np.isfinite(D["lv"][nme]) and D["lv"][nme] > -22:
            ax.plot([D["lv"][nme]], [i], "o", color=INK, ms=2.5, mec="white", mew=0.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(names); ax.set_xlim(-22, 1); ax.set_xticks([-20, -10, 0]); ax.set_ylim(3.6, -0.9)
    # 6 level
    ax = cell(("MASTER", 0), ORANGE, 6, "Level", "controller trajectory within\n0.1 dB on this track: steady,\nleft alone (acts above 3 dB)", "controller gain (dB)")
    g = np.asarray(D["ses"]["gain_db"]); tg = np.linspace(0, D["ses"]["seconds"], len(g))
    ax.fill_between(tg, 0, g, color=ORANGE, alpha=0.3); ax.plot(tg, g, color=ORANGE, lw=0.7)
    ax.axhline(3, color=RED, lw=0.4, ls="--"); ax.axhline(-3, color=RED, lw=0.4, ls="--")
    ax.set_ylim(-4, 4); ax.set_yticks([-3, 0, 3]); ax.set_xticks([0, 10, 20]); ax.set_xlabel("time (s)", **LBL)
    # 7 tone
    ax = cell(("MASTER", 1), AQUA, 7, "Tone and image", "third-octave spectrum inside\nthe range of released music:\nleft alone (blind cap 1.5 dB)", "tone vs. p10 to p90")
    lo, _, hi = D["rng"]["ltas"]
    sel = (THIRD_OCT >= 40) & (THIRD_OCT <= 14000)
    ax.fill_between(THIRD_OCT[sel], lo[sel], hi[sel], color=TINT[BLUE], ec=BLUE, lw=0.3)
    ax.plot(THIRD_OCT[sel], D["feat"]["ltas"][sel], color=INK, lw=0.8)
    ax.set_xscale("log"); ax.set_xticks([100, 1000, 10000]); ax.set_xticklabels(["100", "1k", "10k"]); ax.set_yticks([])
    ax.set_ylim(-45, -3); ax.set_xlabel("Hz", **LBL); ax.minorticks_off()
    # 8 dynamics
    ax = cell(("MASTER", 2), AQUA, 8, "Dynamics", "peak-to-loudness 12.3 dB,\ninside 8 to 16; band crests\ninside their ranges: left alone", "crest by band (dB)")
    lo_c, _, hi_c = D["rng"]["band_crest"]; xb = np.arange(4)
    ax.vlines(xb, lo_c, hi_c, color=BLUE, lw=5, alpha=0.3)
    ax.plot(xb, D["feat"]["band_crest"], "o", color=INK, ms=2.5, mec="white", mew=0.5)
    ax.set_xticks(xb); ax.set_xticklabels(["<120", "120-1k", "1k-6k", ">6k"]); ax.set_yticks([10, 15, 20]); ax.set_ylim(8, 24)
    # 9 loudness
    ax = cell(("MASTER", 3), AQUA, 9, "Loudness and peaks", f"gain to the delivery target,\nthen a two-stage limiter:\n{D['lim']['max_gr_db']:.1f} dB of reduction, -1 dBTP", "12 dB hotter, then limited")
    tw = np.arange(clean.shape[1]) / SR
    ax.plot(tw[::6], D["hot"][0][::6], color=INK2, lw=0.25, alpha=0.45)
    ax.plot(tw[::6], D["loud"][0][::6], color=AQUA, lw=0.25, alpha=0.95)
    ax.axhline(10 ** (-1 / 20), color=RED, lw=0.4, ls="--"); ax.axhline(-10 ** (-1 / 20), color=RED, lw=0.4, ls="--")
    ax.set_ylim(-2.4, 2.4); ax.set_yticks([]); ax.set_xticks([0, 2, 4]); ax.set_xlabel("time (s)", **LBL)
    fig.savefig(os.path.join(OUT, "paper_pipeline.png"), facecolor="white")


def fig_before_after(D):
    """Two before/after spectrogram pairs on a zoomed window, one column wide."""
    W, H = 3.4, 2.72
    fig, bg = canvas(W, H)
    deg, deechoed, out, delay_ms, echo_gain = D["deg"], D["deechoed"], D["out"], D["delay_ms"], D["echo_gain"]
    t_on = D["t_on"]; t_cp = t_on + delay_ms / 1000
    t0, t1 = round(t_on - 0.22, 2), round(t_on + 1.0, 2)
    wb, hb = 1.52, 1.0
    panels = [("input", deg, "Echo: the copy of each hit", 0, 1), ("after the echo stage", deechoed, "", 1, 1),
              ("after the echo stage", deechoed, "Room reverb: the tail after each hit", 0, 0), ("after the network", out, "", 1, 0)]
    for lab, sig, head, col, row in panels:
        x = 0.1 + col * (wb + 0.14); y = 0.22 + row * (hb + 0.3)
        ax = tidy(inset(fig, bg, x, y, wb, hb))
        rows = spec(ax, sig, t0, t1)
        ax.tick_params(labelsize=F["tick"], length=1.5, width=0.4, pad=1)
        ax.text(0.03, 0.96, lab, transform=ax.transAxes, fontsize=F["ann"], color="white", va="top", fontweight="bold")
        if head:
            bg.text(x, y + hb + 0.03, head, fontsize=F["title"], fontweight="bold", color=INK, va="bottom")
        if (col, row) == (0, 1):
            ax.axvline(t_on, color="white", lw=0.6, ls="--", alpha=0.9)
            ax.axvline(t_cp, color=YELLOW, lw=0.7, ls="--", alpha=0.95)
            ax.annotate("", (t_cp, rows * 0.78), (t_on, rows * 0.78), arrowprops=dict(arrowstyle="<->", color=YELLOW, lw=0.7, shrinkA=0, shrinkB=0))
            ax.text((t_on + t_cp) / 2, rows * 0.80, f"{delay_ms:.0f} ms", ha="center", va="bottom", fontweight="bold", **ANN)
            ax.text(t_on - 0.015, rows * 0.55, "a hit", ha="right", va="center", **ANN)
            ax.text(t_cp + 0.015, rows * 0.55, f"its echo, {echo_gain:.2f}\nas loud", ha="left", va="center", **ANN)
        if (col, row) == (1, 1):
            ax.axvline(t_cp, color=YELLOW, lw=0.7, ls="--", alpha=0.95)
            ax.text(t_cp + 0.015, rows * 0.55, "copy gone,\nhit kept", ha="left", va="center", **ANN)
        if row == 0:
            ax.axvspan(t_on + 0.03, t_on + 0.19, color=YELLOW, alpha=0.16)
            ax.text(t_on + 0.11, rows * 0.55, "the room's tail" if col == 0 else "tail gone,\nhit kept", ha="center", va="center", **ANN)
        if row == 0:
            ax.set_xlabel("time (s)", fontsize=F["lab"], labelpad=0.5)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("40 Hz to 20 kHz, log", fontsize=F["lab"], labelpad=1)
        for xa in (0.1 + wb,):
            bg.add_patch(FancyArrowPatch((xa + 0.02, y + hb / 2), (xa + 0.12, y + hb / 2), arrowstyle="-|>", mutation_scale=5, lw=0.7, color=INK2, zorder=4))
    fig.savefig(os.path.join(OUT, "paper_before_after.png"), facecolor="white")


def fig_attention(D):
    """Layer-12 time attention and the lag profile for three layers, one column wide."""
    t, delay_ms = D["t"], D["delay_ms"]
    W, H = 3.4, 1.45
    fig, bg = canvas(W, H)
    A = np.asarray(t["att_time11"]); n = A.shape[0]
    ax = tidy(inset(fig, bg, 0.28, 0.26, 1.12, 1.12))
    ax.imshow(A, cmap="magma", vmin=0, vmax=np.percentile(A, 99.5), aspect="auto", extent=[0, n * HOP / SR, n * HOP / SR, 0], interpolation="nearest")
    ax.set_xlabel("attended frame (s)", fontsize=F["lab"], labelpad=0.5); ax.set_ylabel("query frame (s)", fontsize=F["lab"], labelpad=1)
    ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 2, 4])
    ax.text(0.97, 0.03, "layer 12, one band,\n8 heads averaged", transform=ax.transAxes, fontsize=F["kind"], color="white", va="bottom", ha="right")
    ax.annotate("", xy=(1.25, 1.25 + 0.21 + 0.04), xytext=(0.45, 2.45), arrowprops=dict(arrowstyle="-|>", color="white", lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(0.2, 2.55, f"second stripe,\n{delay_ms:.0f} ms back", fontsize=F["ann"], color="white", va="top")
    ax = tidy(inset(fig, bg, 1.86, 0.26, 1.46, 1.12))
    lags = np.arange(0, 60)
    for key, name, color in (("att_time0", "layer 1", INK2), ("att_time5", "layer 6", BLUE), ("att_time11", "layer 12", ORANGE)):
        M = np.asarray(t[key])
        prof = [np.mean([M[i, i - L] for i in range(L, n)]) for L in lags]
        ax.semilogy(lags * HOP / SR * 1000, prof, color=color, lw=0.8, label=name)
    ax.axvline(delay_ms, color=RED, lw=0.5, ls="--")
    ax.text(delay_ms + 8, ax.get_ylim()[1] * 0.6, f"echo delay\n{delay_ms:.0f} ms", fontsize=F["kind"], color=RED, va="top")
    ax.set_xlabel("how far back it looks (ms)", fontsize=F["lab"], labelpad=0.5); ax.set_ylabel("mean attention", fontsize=F["lab"], labelpad=1)
    ax.legend(frameon=False, fontsize=F["kind"], loc="upper right", handlelength=1.2, borderaxespad=0.2)
    ax.tick_params(which="minor", length=0)
    fig.savefig(os.path.join(OUT, "paper_attention.png"), facecolor="white")


def fig_limiter():
    """Four limiters on the same two seconds: the waveform at the largest peak and the applied gain, one column wide."""
    d = np.load(os.path.join(ROOT, "docs", "evidence", "loudness", "limiter_trace.npz"))
    sr, x, c = int(d["sr"]), d["input"], int(d["peak_index"])
    cols = [("clip", "hard clip"), ("matchering", "Matchering"), ("ozone", "Ozone 9"), ("ours", "ours")]
    W, H = 3.4, 1.6
    fig, bg = canvas(W, H)
    tms = (np.arange(x.shape[1]) - c) / sr * 1000
    w = slice(max(0, c - int(0.004 * sr)), c + int(0.008 * sr))
    pw = 0.72
    for j, (k, name) in enumerate(cols):
        y, g, r = d[k], d[k + "_gain_db"], d[k + "_residual"]
        xx = 0.3 + j * (pw + 0.07)
        ax = tidy(inset(fig, bg, xx, 0.92, pw, 0.52))
        ax.plot(tms[w], (x[0] * 10 ** (g / 20))[w], color=INK2, lw=0.5, alpha=0.55)
        ax.plot(tms[w], y[0][w], color=ORANGE if k == "ours" else INK, lw=0.7)
        ax.set_title(name, fontsize=F["title"], loc="left", pad=2, fontweight="bold")
        ax.set_ylim(-1.15, 1.15); ax.set_xticks([0, 5]); ax.set_yticks([-1, 0, 1])
        if j: ax.set_yticklabels([])
        else: ax.set_ylabel("at the peak", fontsize=F["lab"], labelpad=1)
        ax.set_xlabel("ms", fontsize=F["lab"], labelpad=0.5)
        ax = tidy(inset(fig, bg, xx, 0.2, pw, 0.45))
        tt = np.arange(len(g)) / sr
        ax.plot(tt, g - np.median(g), color=ORANGE if k == "ours" else INK, lw=0.5)
        ax.set_ylim(-9, 3); ax.set_yticks([-8, -4, 0]); ax.set_xticks([0, 1, 2])
        if j: ax.set_yticklabels([])
        else: ax.set_ylabel("gain (dB)", fontsize=F["lab"], labelpad=1)
        ax.set_xlabel("s", fontsize=F["lab"], labelpad=0.5)
        rel = 10 * np.log10(np.mean(r ** 2) / np.mean(y ** 2) + 1e-14)
        ax.text(0.03, 0.08, f"spread {np.percentile(g, 95) - np.percentile(g, 5):.1f} dB\ndistortion {rel:.0f} dB", transform=ax.transAxes, fontsize=F["kind"], color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "paper_limiter.png"), facecolor="white")


if __name__ == "__main__":
    D = prepare()
    fig_pipeline(D); fig_before_after(D); fig_attention(D); fig_limiter()
    print("wrote paper_pipeline/before_after/attention/limiter")
