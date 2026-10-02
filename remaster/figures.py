"""Draw the README figures from measured shapes and logged curves.

  python -m remaster.figures            # writes docs/figures/*.png

Shapes in the diagrams are read from forward hooks (docs/evidence/shapes_bsroformer.json) or computed by
running the modules here; curves come from the val.jsonl files in docs/evidence/curves.
One visual language throughout: blue = signal, orange = learned, aqua = deterministic DSP, gray = data.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "figures")
CUR = os.path.join(ROOT, "docs", "evidence", "curves")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
TINT = {BLUE: "#e6f0fb", ORANGE: "#fdebe3", AQUA: "#e2f5ee", INK2: "#f1f0ec", RED: "#fbe6e6", YELLOW: "#fdf3d9"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), color=INK2, shape=None, title_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=TINT[color], ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.47 - i * 0.22, ln, ha="center", va="top", fontsize=7.8, color=INK2)
    if shape:
        ax.text(x + w / 2, y + 0.1, shape, ha="center", va="bottom", fontsize=7.8, family="DejaVu Sans Mono", color=INK)


def arrow(ax, p, q, color=INK2, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2))


def note(ax, x, y, text, color=INK2, size=8, ha="left", weight="normal"):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center", fontweight=weight)


def title(ax, w, h, text, sub=None):
    ax.text(0.3, h - 0.3, text, fontsize=13, fontweight="bold", va="top")
    if sub:
        ax.text(0.3, h - 0.68, sub, fontsize=8.8, color=INK2, va="top")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. the restoration network
def fig_architecture():
    sh = json.load(open(os.path.join(ROOT, "docs", "evidence", "shapes_bsroformer.json")))
    cfg = sh["cfg"]
    T = sh["shapes"]["band_split"][1][1]; K = sh["shapes"]["band_split"][1][2]; D = cfg["dim"]; L = cfg["depth"]
    Fb = cfg["stft_n_fft"] // 2 + 1
    W, H = 14.0, 8.75
    fig, ax = canvas(W, H)
    title(ax, W, H, "The restoration network: a band-split transformer that predicts a mask",
          f"Shapes measured on a 4 s stereo clip at 44.1 kHz.  {sh['params'] / 1e6:.1f}M parameters, {L} layers, width {D}.  Batch dimension omitted.")
    y, h = 5.35, 1.45
    xs = [0.3, 2.25, 4.2, 6.5, 9.75, 11.95]
    box(ax, xs[0], y, 1.65, h, "Stereo audio", ["4 s at 44.1 kHz"], BLUE, "[2, 176400]")
    box(ax, xs[1], y, 1.65, h, "STFT", ["window 2048, hop 512", "23 ms / 11.6 ms"], AQUA, f"[2, {Fb}, {T}] complex")
    box(ax, xs[2], y, 2.0, h, "Band split", [f"{K} bands, one Linear each", "re/im of L and R per bin"], ORANGE, f"[{T}, {K}, {D}]")
    box(ax, xs[3], y, 2.95, h, f"{L} x  time / band attention", ["time: each band attends over frames", "band: each frame attends over bands", "rotary positions, 8 heads"], ORANGE, f"[{T}, {K}, {D}]")
    box(ax, xs[4], y, 1.9, h, "Mask estimator", ["per-band MLP", "complex gain per bin"], ORANGE, f"[2, {Fb}, {T}] complex")
    box(ax, xs[5], y, 1.75, h, "Mask x input", ["then inverse STFT"], AQUA, "[2, 176400]")
    for a, b, wa in zip(xs[:-1], xs[1:], (1.65, 1.65, 2.0, 2.95, 1.9)):
        arrow(ax, (a + wa, y + h / 2), (b, y + h / 2))
    # the bypass: input spectrogram goes around the network to the multiply
    arrow(ax, (xs[1] + 0.83, y + h), (xs[5] + 0.6, y + h), color=BLUE, rad=-0.13, lw=1.6)
    note(ax, 8.0, y + h + 0.36, "the input spectrogram goes around the network: output = input x mask, and the mask starts at exactly 1,", BLUE, 8.3, "center")
    note(ax, 8.0, y + h + 0.13, "so an untrained network is a perfect pass-through and anything left at 1 comes out bit-exact", BLUE, 8.3, "center")

    # band layout strip
    widths = cfg["freqs_per_bands"]; edges = np.concatenate([[0], np.cumsum(widths)])
    hz = edges / (Fb - 1) * 22050
    x0, x1, yb = 0.6, 6.4, 3.55
    note(ax, x0, yb + 0.72, f"How {Fb} frequency bins become {K} tokens per frame", INK, 9.5, weight="bold")
    lx = lambda f: x0 + (x1 - x0) * np.log(np.maximum(f, 20) / 20) / np.log(22050 / 20)
    for i in range(K):
        ax.add_patch(Rectangle((lx(hz[i]), yb), lx(hz[i + 1]) - lx(hz[i]), 0.42, fc=TINT[ORANGE] if i % 2 else "#f9cdb9", ec=ORANGE, lw=0.4))
    for f, lab in ((50, "50 Hz"), (200, "200"), (1000, "1 kHz"), (4000, "4 k"), (16000, "16 k")):
        ax.plot([lx(f), lx(f)], [yb - 0.06, yb], color=INK2, lw=0.8); ax.text(lx(f), yb - 0.1, lab, ha="center", va="top", fontsize=7.5, color=INK2)
    note(ax, x0, yb - 0.5, f"narrow bands where pitch lives ({widths[0]} bins = {widths[0] * 21.5:.0f} Hz wide), wide bands at the top ({widths[-1]} bins = {widths[-1] * 21.5 / 1000:.1f} kHz)", INK2, 7.8)
    note(ax, x0, yb - 0.74, f"each band's bins x 2 channels x (re, im) go through their own Linear layer to a {D}-d token: [{T}, {Fb * 4}] -> [{T}, {K}, {D}]", INK2, 7.8)

    # attention grid
    gx, gy, gw, gh = 7.6, 2.05, 3.3, 2.05
    note(ax, gx, yb + 0.72, "One layer: two views of the same grid", INK, 9.5, weight="bold")
    nt, nk = 22, 10
    for i in range(nt):
        for j in range(nk):
            ax.add_patch(Rectangle((gx + i * gw / nt, gy + j * gh / nk), gw / nt * 0.9, gh / nk * 0.86, fc="#eceae4", ec="none"))
    for i in range(nt):
        ax.add_patch(Rectangle((gx + i * gw / nt, gy + 6 * gh / nk), gw / nt * 0.9, gh / nk * 0.86, fc=BLUE, ec="none", alpha=0.85))
    for j in range(nk):
        ax.add_patch(Rectangle((gx + 13 * gw / nt, gy + j * gh / nk), gw / nt * 0.9, gh / nk * 0.86, fc=ORANGE, ec="none", alpha=0.9))
    ax.text(gx + gw / 2, gy - 0.12, f"time: {T} frames (86 per second)", ha="center", va="top", fontsize=7.8, color=INK2)
    ax.text(gx - 0.1, gy + gh / 2, f"{K} bands", ha="right", va="center", fontsize=7.8, color=INK2, rotation=90)
    note(ax, gx + gw + 0.25, gy + 1.72, "time attention", BLUE, 8.8, weight="bold")
    note(ax, gx + gw + 0.25, gy + 1.48, f"one band across all frames:", INK2, 7.8)
    note(ax, gx + gw + 0.25, gy + 1.27, f"{K} sequences of length {T}", INK, 7.8)
    note(ax, gx + gw + 0.25, gy + 1.06, "sees reverb tails and echoes", INK2, 7.8)
    note(ax, gx + gw + 0.25, gy + 0.66, "band attention", ORANGE, 8.8, weight="bold")
    note(ax, gx + gw + 0.25, gy + 0.42, "one frame across all bands:", INK2, 7.8)
    note(ax, gx + gw + 0.25, gy + 0.21, f"{T} sequences of length {K}", INK, 7.8)
    note(ax, gx + gw + 0.25, gy + 0.0, "sees harmonics and timbre", INK2, 7.8)

    # training strip
    yb2 = 0.35
    ax.add_patch(FancyBboxPatch((0.3, yb2), W - 0.6, 1.05, boxstyle="round,pad=0.02,rounding_size=0.08", fc=TINT[INK2], ec=GRID, lw=1))
    note(ax, 0.55, yb2 + 0.82, "Training", INK, 9.5, weight="bold")
    note(ax, 0.55, yb2 + 0.55, "pairs made on the fly: clean music + (room or plug-in style reverb, echo, clipping, noise) -> damaged input, sample-aligned with its clean target", INK2, 8)
    note(ax, 0.55, yb2 + 0.30, "loss = waveform L1 + multi-resolution STFT (complex L1 + log-magnitude L1, windows 4096 to 256).  Started from a public vocal dereverb checkpoint, fine-tuned on full mixes.", INK2, 8)
    save(fig, "architecture.png")


# ---------------------------------------------------------------- 2. tokens vs mask
def fig_tokens_vs_mask():
    W, H = 14.0, 7.6
    fig, ax = canvas(W, H)
    title(ax, W, H, "Two routes to restoration", "Same 4 s stereo clip through both. One generates audio from codec tokens; the other corrects the input spectrogram.")
    # token-prediction row
    y, h = 4.65, 1.4
    note(ax, 0.3, y + h + 0.28, "Token prediction: EnCodec tokens through a 1-D U-Net (1.08 B parameters)", INK, 10.5, weight="bold")
    xs = [0.3, 2.15, 4.25, 6.3, 9.2, 11.25]
    box(ax, xs[0], y, 1.55, h, "Audio", ["4 s, resampled", "to 48 kHz"], BLUE, "[2, 192000]")
    box(ax, xs[1], y, 1.8, h, "EnCodec encoder", ["24 kbps, 150 frames/s", "16 residual codebooks"], INK2, "tokens [16, 598]")
    box(ax, xs[2], y, 1.75, h, "Token embedding", ["1024 entries per", "codebook"], ORANGE, "[384, 598]")
    box(ax, xs[3], y, 2.6, h, "1-D U-Net, CBAM + FiLM", ["768x299 -> 1536x150", "-> 3072x75 -> 6144x38 -> back"], ORANGE, "[384, 598]")
    box(ax, xs[4], y, 1.75, h, "16 heads + argmax", ["one 1024-way choice", "per codebook, per frame"], ORANGE, "logits [1024, 16, 598]")
    box(ax, xs[5], y, 2.4, h, "EnCodec decoder", ["audio is generated", "from tokens"], INK2, "[2, 192000]")
    for a, b, wa in zip(xs[:-1], xs[1:], (1.55, 1.8, 1.75, 2.6, 1.75)):
        arrow(ax, (a + wa, y + h / 2), (b, y + h / 2))
    probs = [(xs[1] + 0.9, "level is stored outside the tokens:\n100% of tokens unchanged after +1 dB"),
             (xs[4] + 0.87, "argmax has no gradient:\nonly cross-entropy trained the model"),
             (xs[5] + 1.2, "best possible output = the codec's own\nreconstruction: 9.6 dB SI-SDR vs clean")]
    for x, t in probs:
        ax.text(x, y - 0.18, t, ha="center", va="top", fontsize=7.8, color=RED)
        ax.plot([x, x], [y - 0.12, y], color=RED, lw=1.2)
    # lacquer row
    y2 = 1.0
    note(ax, 0.3, y2 + h + 1.0, "Lacquer: band-split transformer (51 M parameters) + DSP", INK, 10.5, weight="bold")
    xs2 = [0.3, 2.15, 4.6, 8.3, 10.6, 12.25]
    box(ax, xs2[0], y2, 1.55, h, "Audio", ["4 s at 44.1 kHz", "untouched"], BLUE, "[2, 176400]")
    box(ax, xs2[1], y2, 2.15, h, "STFT", ["no codec, no vocoder"], AQUA, "[2, 1025, 345] complex")
    box(ax, xs2[2], y2, 3.4, h, "Band-split transformer", ["12 x (time attention, band attention)", "tokens [345, 62, 256]"], ORANGE, "mask [2, 1025, 345] complex")
    box(ax, xs2[3], y2, 2.0, h, "Mask x input", ["mask starts at 1"], AQUA, "[2, 1025, 345]")
    box(ax, xs2[4], y2, 1.35, h, "iSTFT", [], AQUA, "[2, 176400]")
    box(ax, xs2[5], y2, 1.45, h, "Decisions", ["DSP stages,", "meters, mastering"], AQUA)
    for a, b, wa in zip(xs2[:-1], xs2[1:], (1.55, 2.15, 3.4, 2.0, 1.35)):
        arrow(ax, (a + wa, y2 + h / 2), (b, y2 + h / 2))
    arrow(ax, (xs2[1] + 1.07, y2 + h), (xs2[3] + 1.0, y2 + h), color=BLUE, rad=-0.2, lw=1.6)
    note(ax, (xs2[1] + 1.07 + xs2[3] + 1.0) / 2, y2 + h + 0.22, "input spectrogram, unchanged", BLUE, 7.8, "center")
    for x, t in [(xs2[1] + 1.07, "level, tone and phase all stay\nin the representation"), (xs2[2] + 1.7, "losses act on the waveform and\nspectrogram, with gradients"), (xs2[3] + 1.0, "no ceiling: what is not masked\npasses through unchanged")]:
        ax.text(x, y2 - 0.18, t, ha="center", va="top", fontsize=7.8, color="#0d7a55")
        ax.plot([x, x], [y2 - 0.12, y2], color=AQUA, lw=1.2)
    save(fig, "tokens_vs_mask.png")


# ---------------------------------------------------------------- 3. decision pipeline
def fig_pipeline():
    W, H = 14.0, 5.6
    fig, ax = canvas(W, H)
    title(ax, W, H, "The decision pipeline", "Each stage measures first and acts only outside a normal range. Numbers are the measured readings behind each rule.")
    y, h, w = 2.35, 2.2, 2.05
    stages = [
        ("1  Echo", AQUA, ["cepstrum of the mix", "", "sharp peak, z >= 45", "-> exact inverse filter", "", "clean music: z ~ 12"], "DSP"),
        ("2  Room reverb", ORANGE, ["network as a meter", "", "clean mixes: -41 dB", "light reverb: -20 dB", "heavy reverb: -4 dB", "gate at -24 dB"], "transformer"),
        ("3  Vocal reverb", ORANGE, ["Demucs vocal stem", "+ vocal dereverb model", "", "normal: -32 to 0 dB", "too wet: reduce", "bone dry: add plate"], "stems"),
        ("4  Level", ORANGE, ["controller network", "gain trajectory g(t)", "", "clamped to +-6 dB", "holds through", "quiet passages"], "controller"),
        ("5  Dynamics", AQUA, ["peak-to-loudness ratio", "", "normal: 8 to 16 dB", "peaky: 2:1 compression", "squashed: protect", "from more limiting"], "DSP"),
        ("6  Finish", AQUA, ["tonal range trim", "BS.1770 loudness", "true-peak limiter", "", "-14 LUFS, -1 dBTP", "or match a reference"], "DSP"),
    ]
    x = 0.3
    for i, (name, col, lines, tag) in enumerate(stages):
        box(ax, x, y, w, h, name, lines, col, title_size=10)
        ax.text(x + w / 2, y + 0.12, tag, ha="center", va="bottom", fontsize=7.5, color=INK2, style="italic")
        if i < len(stages) - 1:
            arrow(ax, (x + w, y + h / 2), (x + w + 0.21, y + h / 2))
        x += w + 0.21
    note(ax, 0.3, 1.75, "in: any stereo mix", BLUE, 9, weight="bold")
    note(ax, W - 0.3, 1.75, "out: restored and mastered mix + the list of decisions", BLUE, 9, ha="right", weight="bold")
    ax.add_patch(Rectangle((0.3, 0.45), 0.28, 0.28, fc=TINT[AQUA], ec=AQUA, lw=1.2)); note(ax, 0.68, 0.59, "deterministic DSP: the problem has an exact answer", INK2, 8.3)
    ax.add_patch(Rectangle((5.2, 0.45), 0.28, 0.28, fc=TINT[ORANGE], ec=ORANGE, lw=1.2)); note(ax, 5.58, 0.59, "learned: the problem needs a prior about what music sounds like", INK2, 8.3)
    note(ax, 0.3, 1.2, "A stage that finds nothing in excess does nothing, so a track that is already fine comes out unchanged apart from the loudness target.", INK2, 8.3)
    save(fig, "pipeline.png")


# ---------------------------------------------------------------- 4. controller + curriculum
def fig_controller():
    W, H = 14.0, 6.4
    fig, ax = canvas(W, H)
    title(ax, W, H, "The level controller: a network that turns knobs", "It outputs parameters, and DSP applies them, so it can only make EQ moves and fader moves. Shapes for an 8 s clip. 5.7M parameters.")
    y, h = 3.55, 1.45
    xs = [0.3, 2.2, 4.4, 6.75, 9.3, 11.6]
    box(ax, xs[0], y, 1.6, h, "Stereo audio", ["8 s at 44.1 kHz"], BLUE, "[2, 352800]")
    box(ax, xs[1], y, 1.9, h, "Mid/side log-mel", ["128 mel bands each", "86 frames per second"], AQUA, "[256, 690]")
    box(ax, xs[2], y, 2.05, h, "Conv, stride 4", ["21.5 frames per second"], ORANGE, "[172, 256]")
    box(ax, xs[3], y, 2.25, h, "6-layer transformer", ["width 256, 8 heads"], ORANGE, "[172, 256]")
    box(ax, xs[4], y + 0.85, 2.0, 0.95, "EQ head", ["mean over time"], ORANGE, "eq(f): [32] dB", 9)
    box(ax, xs[4], y - 0.35, 2.0, 0.95, "Gain head", ["one value per frame"], ORANGE, "g(t): [690] dB", 9)
    box(ax, xs[5], y, 2.1, h, "Apply as a mask", ["G(t, f) = eq(f) + g(t)", "separable in dB"], AQUA, "[1025, 690]")
    for a, b, wa in zip(xs[:3], xs[1:4], (1.6, 1.9, 2.05)):
        arrow(ax, (a + wa, y + h / 2), (b, y + h / 2))
    arrow(ax, (xs[3] + 2.25, y + h / 2 + 0.2), (xs[4], y + 1.3)); arrow(ax, (xs[3] + 2.25, y + h / 2 - 0.2), (xs[4], y + 0.15))
    arrow(ax, (xs[4] + 2.0, y + 1.3), (xs[5], y + h / 2 + 0.2)); arrow(ax, (xs[4] + 2.0, y + 0.15), (xs[5], y + h / 2 - 0.2))
    # curriculum
    yc = 1.25
    note(ax, 0.3, yc + 1.2, "Trained with a curriculum", INK, 10.5, weight="bold")
    stages = [("Stage 1: one effect", "EQ or compression or level riding, or clean", "LR x1.0, steps 0 to 10k"),
              ("Stage 2: one or two", "pairs of effects mixed in", "LR x0.6, 10k to 15k"),
              ("Stage 3: full mix", "each effect present independently", "LR x0.4, 15k onward")]
    x = 0.3
    for i, (a, b, c) in enumerate(stages):
        box(ax, x, yc - 0.1, 3.0, 1.05, a, [b, c], YELLOW, title_size=9)
        if i < 2:
            arrow(ax, (x + 3.0, yc + 0.42), (x + 3.25, yc + 0.42))
        x += 3.25
    note(ax, 10.3, yc + 0.78, "advance when validation stops improving", INK2, 8.2)
    note(ax, 10.3, yc + 0.52, "after a minimum stay in the stage", INK2, 8.2)
    note(ax, 10.3, yc + 0.20, "mixed from the start: learns later, ends at 1.76 dB", INK2, 8.2)
    note(ax, 10.3, yc - 0.06, "curriculum: 1.52 dB, a third of the damage to clean audio", "#0d7a55", 8.2)
    note(ax, 0.3, 0.45, "What it learned: level riding. Its EQ output stays near zero and it does not undo compression: both were measured to be ambiguous without a reference (experiments E5, E15).", INK2, 8.2)
    save(fig, "controller.png")


# ---------------------------------------------------------------- 5. curves
def _val(name):
    p = os.path.join(CUR, name + "_val.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def _style(ax, xlabel, ylabel, title_):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=8.5); ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title_, fontsize=10, fontweight="bold", loc="left", color=INK, pad=8)
    ax.tick_params(labelsize=8, length=0)


def fig_curves():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=160)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.14, wspace=0.26)
    fig.text(0.02, 0.955, "Training evidence", fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.895, "Validation on a fixed set of 32 held-out tracks. Points are logged checkpoints, lines connect them.", fontsize=8.8, color=INK2, va="top")
    ft = _val("ft_bs") + [v for v in _val("study_base") if v["step"] > 12000]
    ft = sorted({v["step"]: v for v in ft}.values(), key=lambda v: v["step"])
    sc = _val("art_small")
    ax = axes[0]
    for data, col, lab in ((ft, ORANGE, "fine-tuned from a vocal dereverb model"), (sc, BLUE, "trained from scratch")):
        if data:
            xs_, ys_ = [v["step"] for v in data], [v["conds"]["reverb"]["sisdr_out"] for v in data]
            ax.plot(xs_, ys_, color=col, lw=2, marker="o", ms=4, mfc=col, mec=SURFACE, mew=1)
            ax.text(0.97, 0.60 if col == ORANGE else 0.27, f"{lab}\n{ys_[-1]:.1f} dB at step {xs_[-1]:,}", transform=ax.transAxes, fontsize=7.8, color=INK2, ha="right", va="top")
    base = ft[0]["conds"]["reverb"]["sisdr_in"]
    ax.axhline(base, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(ax.get_xlim()[1], base + 0.12, f"damaged input {base:.1f} dB", fontsize=7.8, color=INK2, ha="right", va="bottom")
    _style(ax, "training step", "SI-SDR vs clean (dB)", "Reverb removal: the warm start is what works")
    ax = axes[1]
    if ft:
        xs_, ys_ = [v["step"] for v in ft], [v["conds"]["identity"]["sisdr_out"] for v in ft]
        ax.plot(xs_, ys_, color=ORANGE, lw=2, marker="o", ms=4, mec=SURFACE, mew=1)
        ax.axvline(2000, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(2250, min(ys_) + 0.3, "lossless MUSDB18-HQ and more\nclean pass-through examples added", fontsize=7.8, color=INK2, va="bottom")
        ax.annotate(f"{ys_[-1]:.0f} dB", (xs_[-1], ys_[-1]), xytext=(-4, 8), textcoords="offset points", fontsize=7.8, color=INK2, ha="right")
    _style(ax, "training step", "output vs input on clean audio (dB)", "Leaving clean audio alone (higher = less touched)")
    ax = axes[2]
    for name, col, lab, dy in (("ctrl_cur", ORANGE, "curriculum", -16), ("ctrl_mix", BLUE, "mixed from the start", 10), ("_old_ctrl_cur_l1", INK2, "L1 loss: never leaves 'do nothing'", 10)):
        d = _val(name)
        if d:
            xs_, ys_ = [v["step"] for v in d], [v["conds"]["ride"]["env_out"] for v in d]
            ax.plot(xs_, ys_, color=col, lw=2, marker="o", ms=3.2, mec=SURFACE, mew=0.8)
            pos = {"curriculum": (0.98, 0.14), "mixed from the start": (0.62, 0.30), "L1 loss: never leaves 'do nothing'": (0.25, 0.90)}[lab]
            ax.text(*pos, lab, transform=ax.transAxes, fontsize=7.8, color=INK2, ha="right" if lab == "curriculum" else "left", va="center")
    for s_, lab in ((10000, "stage 2"), (15000, "stage 3")):
        ax.axvline(s_, color=GRID, lw=1.2); ax.text(s_ + 300, 3.05, lab, fontsize=7.3, color=INK2, va="top")
    _style(ax, "training step", "level-envelope error (dB)", "Level controller: the curriculum helps")
    save(fig, "training_curves.png")


def fig_probe():
    p = os.path.join(ROOT, "docs", "evidence", "probe_results.json")
    r = json.load(open(p))
    effects = ["reverb", "echo", "eq", "comp", "clip", "noise"]
    reps = [("tok16", "EnCodec tokens (16 codebooks)", BLUE), ("latent", "EnCodec continuous latents", ORANGE), ("mel", "mel spectrogram", AQUA)]
    fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=160)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.13)
    fig.text(0.02, 0.955, "What EnCodec knows about damage", fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.885, "The same small probe, trained to detect each degradation from three representations of a 4 s clip. 0.5 is chance.", fontsize=8.8, color=INK2, va="top")
    bw = 0.25
    for i, (key, lab, col) in enumerate(reps):
        vals = [r[key]["auroc"][e] for e in effects]
        xs_ = np.arange(len(effects)) + (i - 1) * (bw + 0.02)
        ax.bar(xs_, [v - 0.5 for v in vals], bw, bottom=0.5, color=col, label=lab)
        for x, v in zip(xs_, vals):
            ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2, color=INK2)
    ax.set_xticks(np.arange(len(effects))); ax.set_xticklabels(["reverb", "echo", "EQ", "compression", "clipping", "noise"], fontsize=9, color=INK)
    ax.set_ylim(0.5, 1.03); ax.set_ylabel("detection AUROC", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.legend(frameon=False, fontsize=8.3, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.13), handlelength=1.2, columnspacing=1.6)
    save(fig, "encodec_probe.png")


if __name__ == "__main__":
    fig_architecture(); fig_tokens_vs_mask(); fig_pipeline(); fig_controller(); fig_curves(); fig_probe()
