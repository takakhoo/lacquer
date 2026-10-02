"""Stage-by-stage figures drawn from a real forward pass (docs/evidence/trace.npz, see remaster/trace.py).

  python -m remaster.trace --input <clip>     # capture
  python -m remaster.figures_deep             # draw docs/figures/stage_*.png

Every image inside these figures is data from that one run: the spectrograms, the token grid on the faces of
the tensor blocks, the attention maps, the mask. Nothing is illustrative filler.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D

from .figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, OUT, RED, ROOT, SURFACE, TINT, YELLOW

SR, NFFT, HOP = 44100, 2048, 512
MONO = "DejaVu Sans Mono"


def load():
    return np.load(os.path.join(ROOT, "docs", "evidence", "trace.npz"))


def db(x, floor=-90.0):
    return np.maximum(20 * np.log10(np.abs(x) + 1e-9), floor)


def logfreq(img_bins, rows=260, fmin=40.0, fmax=20000.0):
    """[1025, T] on linear bins -> [rows, T] on a log-frequency axis, low frequencies at the bottom row 0."""
    f = np.arange(img_bins.shape[0]) * SR / NFFT
    grid = np.geomspace(fmin, fmax, rows)
    return np.stack([np.interp(grid, f, img_bins[:, i]) for i in range(img_bins.shape[1])], axis=1)


def fig_ax(w, h):
    fig = plt.figure(figsize=(w, h), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, w); bg.set_ylim(0, h); bg.axis("off")
    return fig, bg


def inset(fig, bg, x, y, w, h):
    """Axes placed in the same inch coordinates as the background canvas."""
    (x0, x1), (y0, y1) = bg.get_xlim(), bg.get_ylim()
    ax = fig.add_axes([(x - x0) / (x1 - x0), (y - y0) / (y1 - y0), w / (x1 - x0), h / (y1 - y0)])
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(labelsize=7, length=2, color=INK2, labelcolor=INK2)
    return ax


def heading(bg, n, text, sub):
    H = bg.get_ylim()[1]
    bg.text(0.3, H - 0.28, f"STAGE {n}", fontsize=8.5, color=ORANGE, fontweight="bold", va="top")
    bg.text(0.3, H - 0.52, text, fontsize=14, fontweight="bold", va="top", color=INK)
    bg.text(0.3, H - 0.93, sub, fontsize=9, color=INK2, va="top")


def label(bg, x, y, text, size=8, color=INK2, ha="left", va="center", weight="normal", mono=False, stroke=None, z=20):
    tx = bg.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, family=MONO if mono else "DejaVu Sans", zorder=z)
    if stroke:
        import matplotlib.patheffects as pe
        tx.set_path_effects([pe.withStroke(linewidth=2.2, foreground=stroke)])


def arrow(bg, p, q, color=INK2, lw=1.3, rad=0.0, z=12):
    bg.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z))


def block(bg, x, y, w, h, depth, front, top=None, side=None, cmap="magma", vlim=None, edge=INK, z=2):
    """A tensor drawn as a box whose three visible faces carry real data.

    front: [rows, cols] or [rows, cols, 3] image, row 0 at the bottom. top/side: images for the receding faces.
    """
    dx, dy = depth * 0.62, depth * 0.40
    kw = dict(cmap=cmap, origin="lower", aspect="auto", interpolation="nearest")
    if vlim:
        kw.update(vmin=vlim[0], vmax=vlim[1])
    def face(img, tr, poly):
        clip = Polygon(poly, closed=True, transform=bg.transData)
        im = bg.imshow(img, extent=(0, 1, 0, 1), **({} if img.ndim == 3 else kw), **({"origin": "lower", "aspect": "auto", "interpolation": "nearest"} if img.ndim == 3 else {}))
        im.set_transform(tr + bg.transData); im.set_clip_path(clip); im.set_zorder(z)
        bg.add_patch(Polygon(poly, closed=True, fill=False, ec=edge, lw=0.9, zorder=z + 0.5))
    front_poly = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top_poly = [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)]
    side_poly = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
    shade = np.full((2, 2), 0.5)
    if top is None:
        bg.add_patch(Polygon(top_poly, closed=True, fc="#d9d6cf", ec=edge, lw=0.9, zorder=z))
    else:
        face(top, Affine2D.from_values(w, 0, dx, dy, x, y + h), top_poly)
    if side is None:
        bg.add_patch(Polygon(side_poly, closed=True, fc="#c4c0b7", ec=edge, lw=0.9, zorder=z))
    else:
        face(side, Affine2D.from_values(dx, dy, 0, h, x + w, y), side_poly)
    face(front, Affine2D.from_values(w, 0, 0, h, x, y), front_poly)
    bg.set_xlim(bg.get_xlim()); bg.set_ylim(bg.get_ylim())
    return dx, dy


def dim(bg, p, q, text, off=(0, -0.16), size=7.6, color=INK):
    """A dimension line with ticks and a label, like an engineering drawing."""
    (x0, y0), (x1, y1) = p, q
    bg.plot([x0, x1], [y0, y1], color=INK2, lw=0.8)
    for (xa, ya) in (p, q):
        nx, ny = -(y1 - y0), (x1 - x0)
        ln = np.hypot(nx, ny) + 1e-9
        bg.plot([xa - nx / ln * 0.05, xa + nx / ln * 0.05], [ya - ny / ln * 0.05, ya + ny / ln * 0.05], color=INK2, lw=0.8)
    ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    bg.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, fontsize=size, color=color, ha="center", va="center", rotation=ang if abs(ang) < 89 else 90, family=MONO)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ stage 1: sound -> spectrogram
def stage1(t):
    W, H = 14.0, 7.1
    fig, bg = fig_ax(W, H)
    bg.set_ylim(0.5, 7.6)
    heading(bg, 1, "From sound to a grid: the short-time Fourier transform",
            "The damaged clip (room reverb plus one 210 ms echo), 4 s of stereo at 44.1 kHz. Each 46 ms window becomes one column of 1025 complex numbers.")
    deg = t["deg"]; X = t["X"]
    n = deg.shape[1]; tt = np.arange(n) / SR
    # waveform, both channels
    axw = inset(fig, bg, 0.75, 4.55, 5.6, 1.55)
    step = 40
    for c, (off, col, name) in enumerate(((0.5, BLUE, "L"), (-0.5, ORANGE, "R"))):
        seg = deg[c, : n // step * step].reshape(-1, step)
        axw.fill_between(tt[::step][: len(seg)], off + seg.min(1) * 2.2, off + seg.max(1) * 2.2, color=col, lw=0)
        axw.text(-0.07, off, name, fontsize=8, color=INK2, ha="right", va="center")
    f0 = 150; s0 = f0 * HOP - NFFT // 2
    axw.axvspan(s0 / SR, (s0 + NFFT) / SR, color=YELLOW, alpha=0.55, lw=0)
    axw.set_xlim(0, 4); axw.set_ylim(-1.15, 1.15); axw.set_yticks([]); axw.set_xlabel("seconds", fontsize=7.5, color=INK2, labelpad=1)
    label(bg, 0.75, 6.28, "waveform", 9, INK, weight="bold"); label(bg, 1.75, 6.28, "[2, 176400]", 8, INK, mono=True)
    label(bg, 3.0, 6.28, "one analysis window, 2048 samples = 46 ms", 7.6, "#9a6b00")
    # one window times the Hann taper
    axz = inset(fig, bg, 0.75, 2.45, 2.55, 1.35)
    w = np.hanning(NFFT); seg = deg[0, s0:s0 + NFFT]
    axz.plot(np.arange(NFFT) / SR * 1000, seg, color=GRID, lw=0.7)
    axz.plot(np.arange(NFFT) / SR * 1000, seg * w, color=BLUE, lw=0.8)
    axz.plot(np.arange(NFFT) / SR * 1000, w * np.abs(seg).max(), color="#9a6b00", lw=1.0, ls=(0, (3, 2)))
    axz.set_xlim(0, NFFT / SR * 1000); axz.set_yticks([]); axz.set_xlabel("ms", fontsize=7.5, color=INK2, labelpad=1)
    label(bg, 0.75, 3.95, "window x Hann taper", 8.5, INK, weight="bold")
    # its spectrum
    axs = inset(fig, bg, 3.8, 2.45, 2.55, 1.35)
    col = X[0, :, f0]; fr = np.arange(1025) * SR / NFFT
    axs.fill_between(fr[1:], db(col[1:]), -90, color=TINT[BLUE], lw=0); axs.plot(fr[1:], db(col[1:]), color=BLUE, lw=0.8)
    axs.set_xscale("log"); axs.set_xlim(30, 22050); axs.set_ylim(-80, 45); axs.set_xticks([100, 1000, 10000]); axs.set_xticklabels(["100", "1k", "10k"])
    axs.set_xlabel("Hz", fontsize=7.5, color=INK2, labelpad=1)
    label(bg, 3.8, 3.95, "FFT of that window (dB)", 8.5, INK, weight="bold"); label(bg, 5.55, 3.95, "[1025]", 8, INK, mono=True)
    arrow(bg, (3.35, 3.1), (3.75, 3.1))
    label(bg, 0.75, 1.95, "The window then slides 512 samples (11.6 ms) and repeats: 345 columns for 4 s.", 8)
    label(bg, 0.75, 1.68, "Bins are 21.5 Hz apart, from 0 Hz to 22.05 kHz. Each value is complex: magnitude and phase.", 8)
    # the spectrogram tensor as two stacked blocks (L in front, R behind) with phase as the companion
    mag = db(X); bx, by, bw, bh = 7.25, 2.05, 4.6, 3.55
    img = logfreq(mag[1]); block(bg, bx + 0.42, by + 0.27, bw, bh, 0.55, img, vlim=(-55, 48), top=np.ones((2, 2, 3)) * 0.78, side=np.ones((2, 2, 3)) * 0.70, z=2)
    img = logfreq(mag[0]); dx, dy = block(bg, bx, by, bw, bh, 0.55, img, vlim=(-55, 48), top=np.ones((2, 2, 3)) * 0.86, side=np.ones((2, 2, 3)) * 0.78, z=4)
    fx = bx + bw * f0 / X.shape[2]
    bg.plot([fx, fx], [by, by + bh], color=YELLOW, lw=1.6, zorder=6)
    arrow(bg, (6.4, 3.1), (fx - 0.05, 3.1), color="#9a6b00", rad=-0.18)
    label(bg, bx, 6.28, "spectrogram", 9, INK, weight="bold"); label(bg, bx + 1.2, 6.28, "[2, 1025, 345] complex", 8, INK, mono=True)
    dim(bg, (bx, by - 0.14), (bx + bw, by - 0.14), "345 frames, 11.6 ms apart", (0, -0.17))
    dim(bg, (bx - 0.14, by), (bx - 0.14, by + bh), "1025 bins, 0 to 22 kHz", (-0.17, 0))
    dim(bg, (bx + bw + 0.1, by - 0.06), (bx + bw + 0.1 + 0.76, by - 0.06 + 0.5), "2 channels", (0.34, -0.12))
    for hz, lab in ((100, "100 Hz"), (1000, "1 kHz"), (10000, "10 kHz")):
        yy = by + bh * np.log(hz / 40) / np.log(20000 / 40)
        label(bg, bx + 0.06, yy, lab, 6.6, "#ffffff", stroke="#000000")
    # phase companion
    axp = inset(fig, bg, 12.55, 3.85, 1.2, 1.5)
    axp.imshow(np.angle(X[0])[40:140, 120:200], cmap="twilight", origin="lower", aspect="auto", interpolation="nearest")
    axp.set_xticks([]); axp.set_yticks([])
    label(bg, 12.55, 5.5, "phase (zoom)", 8, INK, weight="bold")
    label(bg, 12.55, 3.62, "kept, never", 7.4); label(bg, 12.55, 3.42, "thrown away", 7.4)
    label(bg, 7.25, 1.40, "Shown on a log frequency axis in decibels. Reverb is the smear after each note; the echo is the faint", 8)
    label(bg, 7.25, 1.13, "second copy of every onset 18 frames later. The network receives real and imaginary parts, both channels.", 8)
    save(fig, "stage_1_stft.png")


# ------------------------------------------------------------------ stage 2: band split -> tokens
def stage2(t):
    W, H = 14.0, 7.2
    fig, bg = fig_ax(W, H)
    bg.set_ylim(0.5, 7.7)
    heading(bg, 2, "Band split: 1025 bins become 62 tokens per frame",
            "Each band's bins, for both channels, real and imaginary parts, pass through that band's own linear layer to a 256-number token.")
    X = t["X"]; edges = t["band_edges_bins"]; K = len(edges) - 1
    mag = db(X[0])
    # spectrogram on a LINEAR-in-band axis: each band gets equal height, which is how the model sees it
    rows = np.stack([mag[edges[i]:edges[i + 1]].mean(0) for i in range(K)])
    ax1 = inset(fig, bg, 0.9, 1.75, 3.6, 4.35)
    ax1.imshow(logfreq(mag), cmap="magma", origin="lower", aspect="auto", vmin=-70, vmax=30, extent=(0, 4, 0, 1))
    hz = edges * SR / NFFT
    for e in hz[1:-1]:
        yy = np.log(max(e, 40) / 40) / np.log(20000 / 40)
        if 0 < yy < 1:
            ax1.axhline(yy, color="#ffffff", lw=0.35, alpha=0.75)
    ax1.set_yticks([np.log(f / 40) / np.log(500) for f in (100, 1000, 10000)]); ax1.set_yticklabels(["100 Hz", "1 kHz", "10 kHz"])
    ax1.set_xlabel("seconds", fontsize=7.5, color=INK2, labelpad=1)
    label(bg, 0.9, 6.28, "spectrogram with the 62 band edges", 9, INK, weight="bold")
    # band width profile
    ax2 = inset(fig, bg, 5.25, 4.35, 2.5, 1.75)
    widths = np.diff(edges)
    ax2.bar(np.arange(K), widths, color=ORANGE, width=0.8)
    ax2.set_yscale("log"); ax2.set_yticks([2, 10, 100]); ax2.set_yticklabels(["2", "10", "100"])
    ax2.set_xlabel("band index (low to high)", fontsize=7.5, color=INK2, labelpad=1); ax2.set_ylabel("bins in band", fontsize=7.5, color=INK2, labelpad=1)
    for s_ in ("top", "right"):
        ax2.spines[s_].set_visible(False)
    label(bg, 5.25, 6.28, "band widths", 9, INK, weight="bold")
    label(bg, 5.25, 3.78, f"2 bins (43 Hz) at the bottom,", 7.8); label(bg, 5.25, 3.55, f"{widths[-1]} bins (2.8 kHz) at the top.", 7.8)
    label(bg, 5.25, 3.18, "per band and frame:", 7.8, INK, weight="bold")
    label(bg, 5.25, 2.93, "bins x 2 ch x (re, im)", 7.6, INK, mono=True)
    label(bg, 5.25, 2.70, "-> normalize -> Linear", 7.6, INK, mono=True)
    label(bg, 5.25, 2.47, "-> 256 numbers", 7.6, INK, mono=True)
    label(bg, 5.25, 2.08, "62 separate linear layers,", 7.8); label(bg, 5.25, 1.85, "one per band.", 7.8)
    arrow(bg, (4.6, 3.9), (5.15, 3.9)); arrow(bg, (7.85, 3.9), (8.45, 3.9))
    # the token tensor, real data on all three faces
    rgb = t["tok_rgb_split"]                                  # [K, T, 3]
    sl = t["tok_slice"].astype(np.float32)                    # [T, K, 48] from the last layer, used for the depth faces
    top = sl[:, -1, :].T                                      # [48, T]: top band across time, depth going back
    side = sl[-1, :, :]                                       # [K, 48]: last frame, depth going back
    norm = lambda a: np.clip((a - np.percentile(a, 2)) / (np.percentile(a, 98) - np.percentile(a, 2) + 1e-9), 0, 1)
    bx, by, bw, bh, dp = 8.6, 1.95, 3.7, 3.3, 1.5
    dx, dy = block(bg, bx, by, bw, bh, dp, rgb, top=plt.cm.cividis(norm(top))[..., :3], side=plt.cm.cividis(norm(side))[..., :3])
    label(bg, bx, 6.28, "token grid", 9, INK, weight="bold"); label(bg, bx + 1.05, 6.28, "[345, 62, 256]", 8, INK, mono=True)
    dim(bg, (bx, by - 0.14), (bx + bw, by - 0.14), "345 frames", (0, -0.17))
    dim(bg, (bx - 0.14, by), (bx - 0.14, by + bh), "62 bands", (-0.17, 0))
    dim(bg, (bx + bw + 0.08, by - 0.08), (bx + bw + 0.08 + dx, by - 0.08 + dy), "256 numbers per token", (0.5, -0.2))
    label(bg, 0.9, 1.22, "Front face: every token reduced to three numbers (its top principal components) and shown as a color, so similar tokens look alike.", 8)
    label(bg, 0.9, 0.95, "Top and side faces: the first 48 of each token's 256 values. 345 x 62 = 21,390 tokens describe the 4 s clip.", 8)
    save(fig, "stage_2_band_split.png")


# ------------------------------------------------------------------ stage 3: what the layers do
def stage3(t):
    W, H = 14.0, 6.9
    fig, bg = fig_ax(W, H)
    heading(bg, 3, "Twelve layers: the token grid organizes itself",
            "The same [345, 62, 256] grid after the band split and after layers 1, 3, 6, 9 and 12. Color = each token's top three principal components.")
    X = t["X"]; edges = t["band_edges_bins"]; K = len(edges) - 1
    banded = np.stack([db(X[0])[edges[i]:edges[i + 1]].mean(0) for i in range(K)])
    panels = [("input, band-averaged", banded, "magma", "[62, 345] dB"), ("after band split", t["tok_rgb_split"], None, "tokens"),
              ("layer 1", t["tok_rgb0"], None, ""), ("layer 3", t["tok_rgb2"], None, ""), ("layer 6", t["tok_rgb5"], None, ""),
              ("layer 9", t["tok_rgb8"], None, ""), ("layer 12", t["tok_rgb11"], None, "")]
    bw, bh, gap = 1.62, 2.75, 0.3
    x = 0.45; y = 2.3
    for i, (name, img, cmap, sub) in enumerate(panels):
        kw = dict(vlim=(-55, 48), cmap=cmap) if cmap else {}
        block(bg, x, y, bw, bh, 0.34, img, **kw)
        label(bg, x, y + bh + 0.42, name, 8.6, INK, weight="bold")
        if i == 0:
            dim(bg, (x, y - 0.13), (x + bw, y - 0.13), "345 frames", (0, -0.16)); dim(bg, (x - 0.12, y), (x - 0.12, y + bh), "62 bands", (-0.16, 0))
        if i < len(panels) - 1:
            arrow(bg, (x + bw + 0.22, y + bh / 2), (x + bw + gap + 0.05, y + bh / 2), lw=1.1)
        x += bw + gap
    label(bg, 0.45, 1.45, "Right after the split the tokens look like noise: each band has its own projection and no band has seen any other.", 8.2)
    label(bg, 0.45, 1.17, "Within a few layers the grid takes on the shape of the music: horizontal bands in the lows and mids, vertical streaks at note onsets in the highs.", 8.2)
    label(bg, 0.45, 0.89, "Each panel has its own color mapping, so compare the structure between panels and not the colors.", 8.2)
    save(fig, "stage_3_layers.png")


# ------------------------------------------------------------------ stage 4: attention
def stage4(t):
    W, H = 14.0, 7.5
    fig, bg = fig_ax(W, H)
    heading(bg, 4, "Attention: the network looks back exactly one echo",
            "Time attention for one band (low mids), averaged over 8 heads. The clip has a 210 ms echo, which is 18 frames. Nobody told the model that.")
    T = t["att_time0"].shape[0]; lag_echo = 0.21 * SR / HOP
    xs = [0.75, 3.6, 6.45]
    for x, li, name in zip(xs, (0, 5, 11), ("layer 1", "layer 6", "layer 12")):
        ax = inset(fig, bg, x, 2.95, 2.45, 2.45)
        A = t[f"att_time{li}"]
        ax.imshow(np.log10(A + 1e-4), cmap="magma", origin="upper", aspect="equal", vmin=-4, vmax=-0.8, extent=(0, 4, 4, 0))
        ax.set_xlabel("attended frame (s)", fontsize=7.3, color=INK2, labelpad=1)
        if x == xs[0]:
            ax.set_ylabel("query frame (s)", fontsize=7.3, color=INK2, labelpad=1)
        ax.set_xticks([0, 2, 4]); ax.set_yticks([0, 2, 4])
        label(bg, x, 5.62, name, 9, INK, weight="bold"); label(bg, x + 0.72, 5.62, "[345, 345]", 7.6, INK, mono=True)
        if li == 11:
            ax.annotate("second stripe:\n18 frames back", xy=(1.55, 1.78), xytext=(0.25, 3.45), fontsize=7.2, color="#ffffff",
                        arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=0.9))
    # attention versus lag
    ax = inset(fig, bg, 9.55, 3.4, 4.0, 2.0)
    for li, col, name in ((0, INK2, "layer 1"), (5, BLUE, "layer 6"), (11, ORANGE, "layer 12")):
        A = t[f"att_time{li}"]
        prof = np.array([np.mean(np.diagonal(A, -k)) for k in range(0, 60)])
        ax.plot(np.arange(60) * HOP / SR * 1000, prof, color=col, lw=1.8, label=name)
    ax.axvline(210, color=RED, lw=1.0, ls=(0, (4, 3)))
    ax.set_yscale("log"); ax.set_xlim(0, 690); ax.set_xlabel("how far back it looks (ms)", fontsize=7.5, color=INK2, labelpad=1); ax.set_ylabel("mean attention", fontsize=7.5, color=INK2, labelpad=1)
    ax.text(222, ax.get_ylim()[1] * 0.55, "echo delay\n210 ms", fontsize=7.4, color=RED, va="top")
    ax.legend(frameon=False, fontsize=7.4, loc="upper right", handlelength=1.4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    label(bg, 9.55, 5.62, "attention against lag", 9, INK, weight="bold")
    # heads
    ax = inset(fig, bg, 9.55, 1.15, 4.0, 1.35)
    Hh = t["att_time_heads11"]; tf = int(t["att_frame_index"])
    lags = Hh[:, tf - 59: tf + 1][:, ::-1]
    ax.imshow(np.log10(lags + 1e-4), cmap="magma", aspect="auto", origin="upper", vmin=-4, vmax=-0.5, extent=(0, 60 * HOP / SR * 1000, 8.5, 0.5))
    ax.axvline(210, color="#ffffff", lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks([1, 4, 8]); ax.set_ylabel("head", fontsize=7.5, color=INK2, labelpad=1); ax.set_xlabel("lag (ms), one query frame, layer 12", fontsize=7.5, color=INK2, labelpad=1)
    label(bg, 9.55, 2.68, "the 8 heads, separately", 9, INK, weight="bold")
    # band attention
    ax = inset(fig, bg, 0.75, 0.75, 1.55, 1.55)
    ax.imshow(t["att_band11"], cmap="magma", origin="lower", aspect="equal", vmin=0, vmax=np.percentile(t["att_band11"], 99.5))
    ax.set_xticks([0, 30, 61]); ax.set_yticks([0, 30, 61]); ax.set_xlabel("attended band", fontsize=7.3, color=INK2, labelpad=1); ax.set_ylabel("query band", fontsize=7.3, color=INK2, labelpad=1)
    label(bg, 2.55, 2.15, "band attention, layer 12", 9, INK, weight="bold"); label(bg, 4.25, 2.15, "[62, 62]", 7.6, INK, mono=True)
    label(bg, 2.55, 1.85, "One frame, every band against every band. Mostly each band", 8)
    label(bg, 2.55, 1.60, "consults its neighbours (the diagonal) plus the lowest bands,", 8)
    label(bg, 2.55, 1.35, "where the fundamental of the music sits.", 8)
    label(bg, 2.55, 0.95, "Time attention is the half that finds reverb and echo; band attention", 8)
    label(bg, 2.55, 0.70, "is the half that knows what a spectrum should look like.", 8)
    save(fig, "stage_4_attention.png")


# ------------------------------------------------------------------ stage 5: mask and result
def stage5(t):
    W, H = 14.0, 8.0
    fig, bg = fig_ax(W, H)
    X, Y, C, M = t["X"][0], t["Y"][0], t["C"][0], t["mask"][0]
    def sisdr(a, b):
        a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); al = (a @ b) / (b @ b); return 10 * np.log10(((al * b) ** 2).sum() / ((a - al * b) ** 2).sum())
    s_in, s_out = sisdr(t["deg"], t["clean"]), sisdr(t["out"], t["clean"])
    rem, add = t["deg"] - t["out"], t["deg"] - t["clean"]
    frac = 100 * (1 - ((t["out"] - t["clean"]) ** 2).sum() / (add ** 2).sum())
    heading(bg, 5, "The mask: one complex gain for every cell, multiplied onto the input",
            f"Network alone on this clip (no DSP echo stage): {s_in:.1f} dB SI-SDR against the clean original before, {s_out:.1f} dB after. {frac:.0f}% of the added energy is gone.")
    bw, bh = 2.95, 2.25; y = 4.2
    def spec(x, img, name, shape, **kw):
        ax = inset(fig, bg, x, y, bw, bh)
        im = ax.imshow(img, origin="lower", aspect="auto", extent=(0, 4, 0, 1), **kw)
        ax.set_yticks([np.log(f / 40) / np.log(500) for f in (100, 1000, 10000)]); ax.set_yticklabels(["100", "1k", "10k"]); ax.set_xticks([0, 1, 2, 3, 4])
        label(bg, x, y + bh + 0.2, name, 9, INK, weight="bold"); label(bg, x + bw, y + bh + 0.2, shape, 7.4, INK, ha="right", mono=True)
        return ax, im
    sk = dict(cmap="magma", vmin=-55, vmax=48)
    spec(0.7, logfreq(db(X)), "damaged input", "[1025, 345]", **sk)
    axm, im = spec(4.35, logfreq(db(M, -40)), "mask (dB)", "[1025, 345] complex", cmap="RdBu_r", vmin=-24, vmax=24)
    spec(8.0, logfreq(db(Y)), "output", "[1025, 345]", **sk)
    spec(11.0 - 0.0, logfreq(db(C)), "clean original (never seen by the model)", "", **sk) if False else None
    label(bg, 3.82, y + bh / 2, "x", 17, INK, ha="center", weight="bold"); label(bg, 7.48, y + bh / 2, "=", 17, INK, ha="center", weight="bold")
    cax = inset(fig, bg, 4.35, y - 0.42, bw, 0.1); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_ticks([-24, -12, 0, 12, 24]); cb.ax.tick_params(labelsize=6.5, length=2)
    label(bg, 4.35 + bw / 2, y - 0.72, "attenuate   <   0 dB = untouched   <   boost", 7.2, ha="center")
    # clean target for reference
    ax = inset(fig, bg, 11.2, y, 2.45, bh)
    ax.imshow(logfreq(db(C)), origin="lower", aspect="auto", extent=(0, 4, 0, 1), **sk); ax.set_yticks([]); ax.set_xticks([0, 2, 4])
    label(bg, 11.2, y + bh + 0.2, "clean original", 9, INK, weight="bold"); label(bg, 11.2, y - 0.3, "for comparison; the model", 7.4); label(bg, 11.2, y - 0.52, "never sees it", 7.4)
    # removed vs added
    y2 = 0.95; bh2 = 1.75
    for x, img, name in ((0.7, X - Y, "what the model removed  (input - output)"), (4.35, X - C, "what was actually added  (input - clean)")):
        ax = inset(fig, bg, x, y2, bw, bh2)
        ax.imshow(logfreq(db(img)), origin="lower", aspect="auto", extent=(0, 4, 0, 1), **sk)
        ax.set_yticks([np.log(f / 40) / np.log(500) for f in (100, 1000, 10000)]); ax.set_yticklabels(["100", "1k", "10k"]); ax.set_xlabel("seconds", fontsize=7.3, color=INK2, labelpad=1)
        label(bg, x, y2 + bh2 + 0.2, name, 8.6, INK, weight="bold")
    # energy decay in one band
    ax = inset(fig, bg, 8.3, y2, 5.35, bh2)
    fr = np.arange(1025) * SR / NFFT; band = (fr > 500) & (fr < 2000)
    tt = np.arange(X.shape[1]) * HOP / SR
    for S_, col, name, lw in ((X, RED, "damaged", 1.3), (Y, ORANGE, "output", 1.9), (C, INK, "clean", 1.1)):
        ax.plot(tt, 10 * np.log10((np.abs(S_[band]) ** 2).sum(0) + 1e-9), color=col, lw=lw, label=name)
    ax.set_xlim(1.0, 3.0); ax.set_xlabel("seconds", fontsize=7.3, color=INK2, labelpad=1); ax.set_ylabel("dB", fontsize=7.3, color=INK2, labelpad=1)
    ax.legend(frameon=False, fontsize=7.4, ncol=3, loc="lower left", handlelength=1.4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    label(bg, 8.3, y2 + bh2 + 0.2, "energy between notes, 500 Hz to 2 kHz", 8.6, INK, weight="bold")
    label(bg, 8.3, y2 - 0.55, "Reverb and echo raise the level between notes. The output sits a little below the damaged input there and", 7.8)
    label(bg, 8.3, y2 - 0.80, "still above the clean original: most of this clip's gain is in timing and phase, and the repair is partial.", 7.8)
    save(fig, "stage_5_mask.png")


# ------------------------------------------------------------------ stage 6: echo
def stage6(t):
    W, H = 14.0, 5.6
    fig, bg = fig_ax(W, H)
    found_ms, found_g = float(t["echo_found_ms"][0]), float(t["echo_found_gain"][0])
    heading(bg, 6, "Echo is found without a network: a spike in the cepstrum",
            "A delayed copy puts a ripple on the log spectrum. The inverse FFT of the log spectrum (the cepstrum) turns that ripple into one spike at the delay.")
    ax = inset(fig, bg, 0.8, 1.15, 7.4, 3.0)
    q = np.arange(len(t["cep_deg"])) / SR * 1000
    m = (q > 40) & (q < 620)
    ax.plot(q[m], t["cep_ref"][m], color=INK2, lw=0.7, label="reverb only")
    ax.plot(q[m], t["cep_deg"][m], color=ORANGE, lw=1.0, label="reverb + echo")
    ax.plot(q[m], t["cep_fixed"][m] - 0.12, color=AQUA, lw=0.8, label="after the inverse filter (shifted down)")
    ax.annotate(f"spike at {found_ms:.1f} ms, height {found_g:.2f}\n(true echo: 210.0 ms, gain 0.45)", xy=(found_ms, t["cep_deg"][int(found_ms / 1000 * SR)]), xytext=(300, 0.33),
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    ax.set_xlabel("quefrency (ms): the delay a ripple corresponds to", fontsize=7.8, color=INK2, labelpad=2); ax.set_ylabel("cepstrum", fontsize=7.8, color=INK2, labelpad=1)
    ax.annotate("smaller dip at twice the delay (-a\u00b2/2):\nthe signature of a single echo", xy=(2 * found_ms, -0.088), xytext=(455, -0.075), fontsize=7.6, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.legend(frameon=False, fontsize=7.6, loc="center right", ncol=1, handlelength=1.4, bbox_to_anchor=(1.0, 0.62))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    x = 8.85
    label(bg, x, 4.05, "Why it works", 9.5, INK, weight="bold")
    label(bg, x, 3.70, "y[n] = x[n] + a x[n - d]", 9, INK, mono=True)
    label(bg, x, 3.38, "spectrum is multiplied by (1 + a e^{-jwd})", 8, INK2)
    label(bg, x, 3.12, "log of that is a cosine ripple in frequency, period 1/d", 8, INK2)
    label(bg, x, 2.86, "cepstrum of a ripple is a spike at d with height a", 8, INK2)
    label(bg, x, 2.40, "Undo it exactly", 9.5, INK, weight="bold")
    label(bg, x, 2.05, "x[n] = y[n] - a x[n - d]", 9, INK, mono=True)
    label(bg, x, 1.73, "a recursive filter, run block by block, refined until", 8, INK2)
    label(bg, x, 1.47, "the spike is gone. No training, 0.6 s per 25 s clip.", 8, INK2)
    label(bg, x, 1.05, "Held-out echo clips: 8.8 -> 25.2 dB SI-SDR (n = 79).", 8.4, "#0d7a55", weight="bold")
    save(fig, "stage_6_echo.png")


# ------------------------------------------------------------------ stage 7: level controller
def stage7(t):
    W, H = 14.0, 6.9
    fig, bg = fig_ax(W, H)
    heading(bg, 7, "The level controller turns one knob over time",
            "An 8 s clip with injected level faults. The network sees a mel spectrogram and outputs a gain trajectory; DSP applies it. Nothing else can change.")
    mel = t["ctrl_mel"]; g, go, act = t["ctrl_gain"], t["ctrl_gain_oracle"], t["ctrl_active"].astype(bool)
    tt = np.arange(len(g)) * HOP / SR
    ax = inset(fig, bg, 0.8, 3.05, 5.2, 2.05)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma", extent=(0, 8, 0, 128), vmin=np.percentile(mel, 5), vmax=np.percentile(mel, 99.5))
    ax.set_ylabel("mel band", fontsize=7.4, color=INK2, labelpad=1); ax.set_xticks(range(0, 9, 2))
    label(bg, 0.8, 5.3, "input: log-mel of the mid channel", 9, INK, weight="bold"); label(bg, 6.0, 5.3, "[128, 690]", 7.6, INK, ha="right", mono=True)
    def env(x):
        e = 10 * np.log10(np.convolve((x.mean(0) ** 2), np.ones(4410) / 4410, mode="same")[::2205] + 1e-9); return e
    ax = inset(fig, bg, 0.8, 0.95, 5.2, 1.6)
    te = np.arange(len(env(t["ctrl_in"]))) * 2205 / SR
    ax.plot(te, env(t["ctrl_clean"]), color=INK, lw=1.0, label="clean")
    ax.plot(te, env(t["ctrl_in"]), color=RED, lw=1.2, label="with level faults")
    ax.set_xlim(0, 8); ax.set_xlabel("seconds", fontsize=7.4, color=INK2, labelpad=1); ax.set_ylabel("level (dB)", fontsize=7.4, color=INK2, labelpad=1)
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower center", handlelength=1.4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    label(bg, 0.8, 2.72, "level over time", 9, INK, weight="bold")
    arrow(bg, (6.2, 3.2), (6.85, 3.2))
    ax = inset(fig, bg, 7.6, 3.05, 5.9, 2.05)
    go_c = go - go[act].mean(); g_c = g - g[act].mean()
    ax.plot(tt, go_c, color=INK, lw=1.1, label="ideal gain (needs the clean clip)")
    ax.plot(tt, g_c, color=ORANGE, lw=2.0, label="predicted gain (input only)")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlim(0, 8); ax.set_ylabel("gain (dB)", fontsize=7.4, color=INK2, labelpad=1)
    ax.legend(frameon=False, fontsize=7.4, ncol=1, loc="lower right", handlelength=1.4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    label(bg, 7.6, 5.3, "output: gain trajectory g(t)", 9, INK, weight="bold"); label(bg, 13.5, 5.3, "[690]", 7.6, INK, ha="right", mono=True)
    ax = inset(fig, bg, 7.6, 0.95, 5.9, 1.6)
    fixed = t["ctrl_in"] * 10 ** (np.interp(np.arange(t["ctrl_in"].shape[1]), (np.arange(len(g)) + 0.5) * HOP, g_c) / 20)
    ax.plot(te, env(t["ctrl_clean"]), color=INK, lw=1.0, label="clean")
    ef = env(fixed); ef = ef - (ef.mean() - env(t["ctrl_clean"]).mean())
    ax.plot(te, ef, color=ORANGE, lw=1.6, label="after the predicted gain (overall level matched)")
    ax.set_xlim(0, 8); ax.set_xlabel("seconds", fontsize=7.4, color=INK2, labelpad=1); ax.set_ylabel("level (dB)", fontsize=7.4, color=INK2, labelpad=1)
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower center", handlelength=1.4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    label(bg, 7.6, 2.72, "level over time, corrected", 9, INK, weight="bold")
    save(fig, "stage_7_controller.png")


# ------------------------------------------------------------------ the codec-token view of the same clip
def stage_encodec(t):
    if "enc_tokens" not in t.files:
        return
    W, H = 14.0, 7.2
    fig, bg = fig_ax(W, H)
    tok, lat, tok2 = t["enc_tokens"].astype(int), t["enc_latents"].astype(np.float32), t["enc_tokens_plus1db"].astype(int)
    same = float((tok == tok2).mean())
    def sisdr(a, b):
        a, b = a.ravel() - a.mean(), b.ravel() - b.mean(); al = (a @ b) / (b @ b); return 10 * np.log10(((al * b) ** 2).sum() / ((a - al * b) ** 2).sum())
    rt = sisdr(t["enc_roundtrip"], t["clean"])
    fig.text(0.3 / W, 1 - 0.28 / H, "THE CODEC-TOKEN VIEW", fontsize=8.5, color=ORANGE, fontweight="bold", va="top")
    bg.text(0.3, H - 0.52, "The same clean clip as EnCodec sees it", fontsize=14, fontweight="bold", va="top", color=INK)
    bg.text(0.3, H - 0.93, "48 kHz stereo EnCodec at 24 kbps. A token-prediction model reads these tokens and has to write a full set of them back.", fontsize=9, color=INK2, va="top")
    # latents block and token block
    norm = lambda a: np.clip((a - np.percentile(a, 2)) / (np.percentile(a, 98) - np.percentile(a, 2) + 1e-9), 0, 1)
    block(bg, 0.8, 3.55, 4.6, 2.1, 0.3, norm(lat), cmap="cividis", vlim=(0, 1))
    label(bg, 0.8, 5.95, "encoder latents (continuous)", 9, INK, weight="bold"); label(bg, 5.4, 5.95, "[128, 598]", 7.6, INK, ha="right", mono=True)
    dim(bg, (0.8, 3.41), (5.4, 3.41), "598 frames, 150 per second", (0, -0.16)); dim(bg, (0.66, 3.55), (0.66, 5.65), "128 dims", (-0.16, 0))
    arrow(bg, (5.72, 4.6), (6.28, 4.6)); label(bg, 6.0, 4.88, "quantize", 7.6, ha="center")
    block(bg, 6.7, 3.55, 4.6, 2.1, 0.3, tok[::-1], cmap="nipy_spectral", vlim=(0, 1023))
    label(bg, 6.7, 5.95, "tokens: 16 codebooks of 1024 entries", 9, INK, weight="bold"); label(bg, 11.3, 5.95, "[16, 598] integers", 7.6, INK, ha="right", mono=True)
    dim(bg, (6.7, 3.41), (11.3, 3.41), "598 frames", (0, -0.16)); dim(bg, (6.56, 3.55), (6.56, 5.65), "16 codebooks", (-0.16, 0))
    label(bg, 11.75, 5.45, "top row: codebook 1,", 7.8); label(bg, 11.75, 5.2, "the coarse one.", 7.8)
    label(bg, 11.75, 4.75, "rows below: residual", 7.8); label(bg, 11.75, 4.5, "refinements. Tokens", 7.8); label(bg, 11.75, 4.25, "rarely repeat, in", 7.8); label(bg, 11.75, 4.0, "any row.", 7.8)
    # how repetitive each codebook is
    ax = inset(fig, bg, 0.8, 0.95, 3.4, 1.7)
    rep_ = [(tok[i, 1:] == tok[i, :-1]).mean() for i in range(16)]
    ent = []
    for i in range(16):
        c = np.bincount(tok[i], minlength=1024) / tok.shape[1]; c = c[c > 0]; ent.append(-(c * np.log2(c)).sum())
    ax.bar(np.arange(1, 17), ent, color=BLUE, width=0.75)
    ax.axhline(np.log2(598), color=INK2, lw=0.9, ls=(0, (4, 3))); ax.text(16.4, np.log2(598) + 0.12, "max possible for 598 frames", fontsize=6.8, color=INK2, ha="right")
    ax.set_xticks([1, 4, 8, 12, 16]); ax.set_xlabel("codebook", fontsize=7.4, color=INK2, labelpad=1); ax.set_ylabel("entropy (bits)", fontsize=7.4, color=INK2, labelpad=1); ax.set_ylim(0, 10.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    label(bg, 0.8, 2.85, "every codebook is near maximum entropy", 9, INK, weight="bold")
    # facts
    x = 5.0
    label(bg, x, 2.75, "Three measurements on this clip", 9.5, INK, weight="bold")
    label(bg, x, 2.38, f"1.  Raise the level by 1 dB and re-encode: {100 * same:.1f}% of tokens are identical.", 8.4, INK)
    label(bg, x + 0.27, 2.12, "Level is stored outside the tokens, so a token model cannot see or fix gain.", 7.9)
    label(bg, x, 1.74, f"2.  Decode the clean tokens again: {rt:.1f} dB SI-SDR against the audio that went in.", 8.4, INK)
    label(bg, x + 0.27, 1.48, "That is the ceiling on this clip: a perfect token predictor still returns audio this far from the original.", 7.9)
    label(bg, x, 1.10, "3.  A probe on tokens detects damage with AUROC 0.68; on these latents, 0.78; one codebook does as well as 16.", 8.4, INK)
    label(bg, x + 0.27, 0.84, "The encoder is a usable analysis front end. The quantized tokens are the weakest view of the signal.", 7.9)
    save(fig, "stage_encodec.png")


if __name__ == "__main__":
    t = load()
    for f in (stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage_encodec):
        f(t)
