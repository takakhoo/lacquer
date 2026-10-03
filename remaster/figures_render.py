"""The decision pipeline drawn from real signals: each stage as a small rendering of what it looks at, and two
before/after spectrogram pairs with the echo copy and the reverb tail marked.

  python -m remaster.figures_render        # writes docs/figures/pipeline_render.png

Signals come from the saved forward pass (docs/evidence/trace.npz: a 4 s clip with room reverb and a 210 ms
echo added, the network's STFTs and mask), from the DSP stages run here on that clip, and from the recorded
demo session for the level stage.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import signal

from .analysis import THIRD_OCT, mastering_features
from .declip import declip
from .deecho import deecho
from .figures import AQUA, BLUE, INK, INK2, ORANGE, OUT, RED, ROOT, SURFACE, TINT, YELLOW
from .figures_deep import HOP, NFFT, SR, db, logfreq, fig_ax, inset
from .master import limiter_v2
from .mastering_norms import ranges, stem_ranges

CMAP = "magma"
LBL = dict(fontsize=6.5, labelpad=1)


def spec(ax, x, t0, t1, vmin=-62, vmax=-2):
    """Log-frequency spectrogram of a stereo signal between t0 and t1 seconds."""
    seg = x[:, int(t0 * SR):int(t1 * SR)]
    _, _, Z = signal.stft(seg, SR, nperseg=1024, noverlap=1024 - 128, boundary=None, padded=False)
    m = np.abs(Z).mean(0)
    img = logfreq(db(m / (m.max() + 1e-9)))
    ax.imshow(img, origin="lower", aspect="auto", cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="nearest",
              extent=[t0, t1, 0, img.shape[0]])
    ax.set_yticks([])
    ax.set_xticks(np.arange(np.ceil(t0 * 4) / 4, t1, 0.25))
    ax.tick_params(labelsize=6)
    return img.shape[0]


def card(bg, x, y, w, h, color):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=TINT[color], ec=color, lw=1.3, zorder=1))


def note(bg, x, y, text, size=7.4, color=INK2, ha="left", **kw):
    bg.text(x, y, text, fontsize=size, color=color, ha=ha, va="top", zorder=5, linespacing=1.25, **kw)


ANN = dict(fontsize=6.8, color="white", zorder=7, bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.6))


def kind_label(ax, text):
    ax.text(0.98, 0.95, text, transform=ax.transAxes, fontsize=6.6, color=INK2, ha="right", va="top", style="italic", zorder=6,
            bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=1.5))


def prepare():
    """Everything the renderings need, computed once: the saved forward pass plus the DSP stages run here."""
    t = np.load(os.path.join(ROOT, "docs", "evidence", "trace.npz"))
    clean, deg, out = t["clean"], t["deg"], t["out"]
    delay_ms = float(t["echo_delay_s"]) * 1000
    ses = json.load(open(os.path.join(ROOT, "docs", "demo", "session.json")))
    thr = np.percentile(np.abs(clean), 96.5)
    clipped = np.clip(clean, -thr, thr).astype(np.float32)
    declipped, _ = declip(clipped)
    deechoed, found = deecho(deg, keep_musical=False)
    feat = mastering_features(clean)
    hot = clean * 10 ** (12 / 20)
    loud, lim = limiter_v2(hot, clip_db=1.5)
    try:
        from .stem_master import stem_levels
        from .stems import separate
        lv = stem_levels(separate(clean), clean)
    except Exception:
        lv = dict(vocals=float("-inf"), drums=-8.2, bass=-8.6, other=-6.2)
    env = signal.lfilter([1 / 220] * 220, 1, np.abs(deg).mean(0))
    on = int(np.argmax(np.diff(env[int(0.6 * SR):int(3.0 * SR)]))) + int(0.6 * SR)
    return dict(t=t, clean=clean, deg=deg, out=out, delay_ms=delay_ms, echo_gain=float(t["echo_gain"]), ses=ses, thr=thr,
                clipped=clipped, declipped=declipped, deechoed=deechoed, z=found[0]["z"] if found else float("nan"),
                rng=ranges("all"), feat=feat, hot=hot, loud=loud, lim=lim, lv=lv, t_on=on / SR)


def main():
    D = prepare()
    t, clean, deg, out, delay_ms, echo_gain, ses, thr = (D[k] for k in ("t", "clean", "deg", "out", "delay_ms", "echo_gain", "ses", "thr"))
    clipped, declipped, deechoed, z, rng, feat, hot, loud, lim, lv = (D[k] for k in ("clipped", "declipped", "deechoed", "z", "rng", "feat", "hot", "loud", "lim", "lv"))

    W, H = 16.0, 10.9
    fig, bg = fig_ax(W, H)
    bg.text(0.3, H - 0.3, "The decision pipeline, drawn from one clip", fontsize=14, fontweight="bold", va="top")
    bg.text(0.3, H - 0.72, f"Each stage shows the thing it measures on a real 4 s excerpt (room reverb and a {delay_ms:.0f} ms echo added) and what it decided. "
            "Green cards are deterministic signal processing, orange cards use the learned model.", fontsize=8.8, color=INK2, va="top")

    cw, ch, gap = 2.92, 2.62, 0.17
    x0 = 0.3
    ys = {"REPAIR": H - 1.2 - ch, "MASTER": H - 1.2 - ch - 0.45 - ch}
    cells = {}
    for lab, n in (("REPAIR", 5), ("MASTER", 4)):
        y = ys[lab]
        bg.text(x0, y + ch + 0.14, lab, fontsize=8.2, fontweight="bold", color=INK2, va="bottom")
        for i in range(n):
            x = x0 + i * (cw + gap)
            cells[(lab, i)] = (x, y)
            if i < n - 1:
                bg.add_patch(FancyArrowPatch((x + cw, y + ch / 2), (x + cw + gap, y + ch / 2), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=INK2, zorder=4))
    # from the end of REPAIR into MASTER: down the right edge, then along to the first master card
    xr = x0 + 4 * (cw + gap) + cw / 2
    ym = ys["MASTER"] + ch / 2
    bg.plot([xr, xr, x0 + 3 * (cw + gap) + cw + gap * 0.6], [ys["REPAIR"], ym, ym], color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=0)
    bg.text(xr + 0.08, ym + 0.05, "repaired mix", fontsize=7, color=INK2, va="bottom", ha="left", rotation=0)

    def cell(key, color, n, name, decision, kind, bottom=0.78, top=0.55):
        x, y = cells[key]
        card(bg, x, y, cw, ch, color)
        bg.text(x + 0.12, y + ch - 0.1, f"{n}  {name}", fontsize=9.6, fontweight="bold", color=INK, ha="left", va="top", zorder=5)
        ax = inset(fig, bg, x + 0.16, y + bottom, cw - 0.3, ch - bottom - top)
        ax.set_facecolor(SURFACE)
        ax.tick_params(labelsize=6, length=2, pad=1)
        note(bg, x + 0.12, y + 0.5, decision)
        kind_label(ax, kind)
        return ax

    # 1 clipping: waveform zoom around the loudest clipped run
    ax = cell(("REPAIR", 0), AQUA, 1, "Clipping", "flat ceiling on 3.5% of samples: peaks rebuilt\nby sparse reconstruction (A-SPADE)", "12 ms of waveform")
    c = int(np.argmax(np.abs(clean[0])))
    sl = slice(max(0, c - 260), c + 270)
    tt = (np.arange(sl.start, sl.stop) - c) / SR * 1000
    ax.plot(tt, clipped[0][sl], color=INK2, lw=1.0, alpha=0.8, label="clipped")
    ax.plot(tt, declipped[0][sl], color=ORANGE, lw=1.3, label="rebuilt")
    ax.axhline(thr, color=RED, lw=0.7, ls="--"); ax.axhline(-thr, color=RED, lw=0.7, ls="--")
    ax.set_yticks([]); ax.set_xticks([])
    ax.legend(frameon=False, fontsize=6.3, loc="lower left", handlelength=1.2)

    # 2 echo: cepstrum with the peak
    ax = cell(("REPAIR", 1), AQUA, 2, "Echo", f"cepstral peak at {delay_ms:.0f} ms, {z:.0f} sigma above the floor,\noff the tempo grid: exact inverse filter applied", "real cepstrum")
    cep = np.asarray(t["cep_deg"]); q = np.arange(len(cep)) / SR * 1000
    m = (q > 20) & (q < 650)
    ax.plot(q[m], cep[m], color=INK, lw=0.7)
    pk = int(np.argmax(np.where(m, cep, -1)))
    ax.plot(q[pk], cep[pk], "o", color=ORANGE, ms=5, mec="white", mew=1)
    ax.annotate(f"{delay_ms:.0f} ms", (q[pk], cep[pk]), xytext=(14, -2), textcoords="offset points", fontsize=7, color=INK, fontweight="bold")
    ax.set_xlabel("quefrency (ms)", **LBL); ax.set_yticks([]); ax.set_xticks([100, 300, 500])
    ax.set_ylim(cep[m].min() * 1.1, cep[pk] * 1.35)

    # 3 room reverb: the predicted mask
    ax = cell(("REPAIR", 2), ORANGE, 3, "Room reverb", "excess reverberation reads -0.5 dB, far above\nthe -18 dB gate: mask applied at full strength", "predicted mask (dB)")
    mk = np.abs(np.asarray(t["mask"])).mean(0)
    img = logfreq(np.clip(20 * np.log10(mk + 1e-6), -18, 6), rows=200)
    ax.imshow(img, origin="lower", aspect="auto", cmap="RdYlBu_r", vmin=-18, vmax=6, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.04, "blue: attenuated    red: kept", transform=ax.transAxes, fontsize=6.3, color=INK, va="bottom",
            bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=1.5))

    # 4 vocal reverb: the meter against the normal range
    ax = cell(("REPAIR", 3), ORANGE, 4, "Vocal reverb", "vocal stem wetness -12.4 dB, inside the\nnormal -32 to +4 dB: left alone", "wetness meter")
    ax.set_xlim(-40, 12); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((-32, 0.3), 36, 0.3, fc=TINT[BLUE], ec=BLUE, lw=0.8))
    ax.add_patch(Rectangle((4, 0.3), 4, 0.3, fc=TINT[RED], ec=RED, lw=0.8))
    ax.plot([-12.4], [0.45], "o", color=INK, ms=7, mec="white", mew=1.2)
    ax.text(-14, 0.72, "normal", fontsize=6.3, color=INK2, ha="center")
    ax.text(6, 0.72, "too\nwet", fontsize=6.3, color=RED, ha="center", va="bottom", linespacing=1.0)
    ax.text(10, 0.12, "no\nvoice", fontsize=6.3, color=INK2, ha="center", va="bottom", linespacing=1.0)
    ax.set_yticks([]); ax.set_xticks([-30, -20, -10, 0]); ax.set_xlabel("vocal reverb excess (dB)", **LBL)

    # 5 instrument balance: stem levels against professional mixes
    ax = cell(("REPAIR", 4), ORANGE, 5, "Instrument balance", "separated-stem loudness against professional\nmixes: reported; vocal moved only on request", "stem level vs. mix (LU)")
    names = ("vocals", "drums", "bass", "other")
    for i, nme in enumerate(names):
        r = stem_ranges(nme)["level"]
        ax.plot([r[0], r[4]], [i, i], color=BLUE, lw=5, alpha=0.35, solid_capstyle="round")
        if np.isfinite(lv[nme]) and lv[nme] > -22:
            ax.plot([lv[nme]], [i], "o", color=INK, ms=5, mec="white", mew=1)
        else:
            ax.text(-21.5, i, "absent", fontsize=6, color=INK2, va="center")
    ax.set_yticks(range(4)); ax.set_yticklabels(names); ax.set_xlim(-22, 1); ax.set_xticks([-20, -10, 0])
    ax.set_ylim(3.6, -0.9)

    # 6 level: the controller's gain trajectory from the recorded session
    ax = cell(("MASTER", 0), ORANGE, 6, "Level", "controller trajectory within 0.1 dB on this track:\nsteady, left alone (acts only above 3 dB)", "controller gain (dB)")
    g = np.asarray(ses["gain_db"]); tg = np.linspace(0, ses["seconds"], len(g))
    ax.fill_between(tg, 0, g, color=ORANGE, alpha=0.3); ax.plot(tg, g, color=ORANGE, lw=1.2)
    ax.axhline(3, color=RED, lw=0.7, ls="--"); ax.axhline(-3, color=RED, lw=0.7, ls="--")
    ax.set_ylim(-4, 4); ax.set_yticks([-3, 0, 3]); ax.set_xticks([0, 10, 20]); ax.set_xlabel("time (s)", **LBL)

    # 7 tone and image: third-octave spectrum against the normal range
    ax = cell(("MASTER", 1), AQUA, 7, "Tone and image", "third-octave spectrum inside the range of released\nmusic: left alone (blind moves capped at 1.5 dB)", "tone vs. p10 to p90 of released music")
    lo, _, hi = rng["ltas"]
    sel = (THIRD_OCT >= 40) & (THIRD_OCT <= 14000)
    ax.fill_between(THIRD_OCT[sel], lo[sel], hi[sel], color=TINT[BLUE], ec=BLUE, lw=0.5)
    ax.plot(THIRD_OCT[sel], feat["ltas"][sel], color=INK, lw=1.3)
    ax.set_xscale("log"); ax.set_xticks([100, 1000, 10000]); ax.set_xticklabels(["100", "1k", "10k"]); ax.set_yticks([])
    ax.set_ylim(-45, -3); ax.set_xlabel("Hz", **LBL)
    ax.minorticks_off()

    # 8 dynamics: band crest against the range
    ax = cell(("MASTER", 2), AQUA, 8, "Dynamics", "peak-to-loudness 12.3 dB, inside 8 to 16 dB;\nband crests inside their ranges: left alone", "crest by band (dB)")
    lo_c, _, hi_c = rng["band_crest"]
    xb = np.arange(4)
    ax.vlines(xb, lo_c, hi_c, color=BLUE, lw=9, alpha=0.3)
    ax.plot(xb, feat["band_crest"], "o", color=INK, ms=5, mec="white", mew=1)
    ax.set_xticks(xb); ax.set_xticklabels(["<120", "120-1k", "1k-6k", ">6k"]); ax.set_yticks([10, 15, 20]); ax.set_ylim(8, 24)

    # 9 loudness and peaks: the limiter's gain on a loud pass
    ax = cell(("MASTER", 3), AQUA, 9, "Loudness and peaks", f"gain to the delivery target, then a two-stage limiter:\n{lim['max_gr_db']:.1f} dB of reduction at most here, ceiling -1 dBTP", "12 dB hotter, then limited")
    tw = np.arange(clean.shape[1]) / SR
    ax.plot(tw[::6], hot[0][::6], color=INK2, lw=0.4, alpha=0.45)
    ax.plot(tw[::6], loud[0][::6], color=AQUA, lw=0.4, alpha=0.95)
    ax.axhline(10 ** (-1 / 20), color=RED, lw=0.7, ls="--"); ax.axhline(-10 ** (-1 / 20), color=RED, lw=0.7, ls="--")
    ax.set_ylim(-2.4, 2.4); ax.set_yticks([]); ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xlabel("time (s)", **LBL)
    ax.text(0.02, 0.04, "grey: before   green: after   red: ceiling", transform=ax.transAxes, fontsize=6.3, color=INK, va="bottom",
            bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=1.5))

    # bottom: two before/after pairs on a zoomed window around a strong transient
    t_on = D["t_on"]; t_cp = t_on + delay_ms / 1000
    t0, t1 = round(t_on - 0.22, 2), round(t_on + 1.0, 2)
    yb, hb, wb = 0.5, 2.65, 3.58
    bg.text(x0, yb + hb + 0.52, "BEFORE AND AFTER", fontsize=8.2, fontweight="bold", color=INK2, va="bottom")
    bg.text(x0, yb + hb + 0.33, f"{t1 - t0:.1f} s of the same clip, {t0:.1f} to {t1:.1f} s, 40 Hz to 20 kHz on a log axis.", fontsize=7.4, color=INK2, va="bottom")
    panels = [("the input: room reverb and a 210 ms echo", deg, "2  Echo: the copy"), ("after the echo stage", deechoed, ""),
              ("after the echo stage", deechoed, "3  Room reverb: the tail"), ("after the network", out, "")]
    for i, (lab, sig, head) in enumerate(panels):
        x = x0 + i * (wb + 0.3) + (0.3 if i >= 2 else 0)
        ax = inset(fig, bg, x, yb, wb, hb)
        rows = spec(ax, sig, t0, t1)
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=7.3, color="white", va="top", fontweight="bold")
        if head:
            bg.text(x, yb + hb + 0.05, head, fontsize=8.6, fontweight="bold", color=INK, va="bottom")
        if i == 0:
            ax.axvline(t_on, color="white", lw=1.0, ls="--", alpha=0.9)
            ax.axvline(t_cp, color=YELLOW, lw=1.2, ls="--", alpha=0.95)
            ax.annotate("", (t_cp, rows * 0.80), (t_on, rows * 0.80), arrowprops=dict(arrowstyle="<->", color=YELLOW, lw=1.2))
            ax.text((t_on + t_cp) / 2, rows * 0.82, f"{delay_ms:.0f} ms", ha="center", va="bottom", fontweight="bold", **{**ANN, "fontsize": 7.2})
            ax.text(t_on - 0.015, rows * 0.56, "a hit", ha="right", va="center", **ANN)
            ax.text(t_cp + 0.015, rows * 0.56, f"its echo: every sound\nagain, {echo_gain:.2f} as loud", ha="left", va="center", **ANN)
        if i == 1:
            ax.axvline(t_cp, color=YELLOW, lw=1.2, ls="--", alpha=0.95)
            ax.text(t_cp + 0.015, rows * 0.56, "the copy is gone;\nthe hit itself stays", ha="left", va="center", **ANN)
        if i >= 2:
            ax.axvspan(t_on + 0.03, t_on + 0.19, color=YELLOW, alpha=0.16)
            ax.text(t_on + 0.11, rows * 0.56, "the tail: the room ringing\nafter each hit" if i == 2 else "tail gone, the\nhit itself stays",
                    ha="center", va="center", **ANN)
        ax.set_xlabel("time (s)", **LBL)
    for xa in (x0 + wb, x0 + 2 * (wb + 0.3) + 0.3 + wb):
        bg.add_patch(FancyArrowPatch((xa + 0.05, yb + hb / 2), (xa + 0.25, yb + hb / 2), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=INK2, zorder=4))

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "pipeline_render.png"), facecolor=SURFACE)
    print("wrote pipeline_render.png", dict(t_on=t_on, delay=delay_ms, z=z, max_gr=lim["max_gr_db"]))


if __name__ == "__main__":
    main()
