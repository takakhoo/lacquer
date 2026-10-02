"""Record one restoration session and replay it.

  python -m remaster.demo replay                      # terminal animation, needs only numpy
  python -m remaster.demo record --input track.wav    # needs the checkpoints; writes docs/demo/

`record` runs the full pipeline on a clip (optionally damaging it first), then stores small
spectrograms, VU traces and the decision list in docs/demo/session.json and renders docs/demo/restore.gif.
`replay` plays that session back in the terminal: the damaged spectrogram, a sweep that reveals the
restored one, and the decisions in the order they were made.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(__file__)
DEMO = os.path.join(HERE, "..", "docs", "demo")
SR = 44100


def small_spec(x, rows=48, cols=160):
    """Log-frequency spectrogram (40 Hz - 16 kHz) as uint8 [rows, cols], row 0 = highest band."""
    from scipy import signal
    f, _, s = signal.stft(x.mean(axis=0), SR, nperseg=4096, noverlap=3072)
    db = 20 * np.log10(np.abs(s) + 1e-9)
    grid = np.geomspace(40, 16000, rows)
    img = np.stack([np.interp(grid, f, db[:, i]) for i in range(db.shape[1])], axis=1)
    idx = np.linspace(0, img.shape[1] - 1, cols).astype(int)
    edges = np.append(idx, img.shape[1])
    img = np.stack([img[:, a:max(b, a + 1)].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    img = np.clip((img + 95) / 75, 0, 1)
    return (img[::-1] * 255).astype(np.uint8)


def record(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from .app import read_any
    from .degrade import RIRBank, degrade, match_level
    from .infer import enhance_auto, load_model
    ck = os.path.join(HERE, "checkpoints")
    models = dict(mix=load_model(os.path.join(ck, "best.pt")), vocal=load_model(os.path.join(ck, "vocal_dereverb.pt")),
                  controller=load_model(os.path.join(ck, "controller.pt")) if os.path.exists(os.path.join(ck, "controller.pt")) else None)
    x = read_any(a.input)[:, : int(a.seconds * SR)]
    if a.stress != "none":
        rng = np.random.default_rng(a.seed)
        x = match_level(degrade(x, rng, RIRBank(a.rir if os.path.isdir(a.rir) else None), effects=a.stress.split("+"))[0], -20.0)
    y, rep = enhance_auto(models, x)
    os.makedirs(DEMO, exist_ok=True)
    ya = match_level(y, 10 * np.log10(np.mean(x ** 2) + 1e-12))  # compare spectra at equal level
    sess = dict(title=a.title, seconds=x.shape[1] / SR, stress=a.stress, before=small_spec(x).tolist(), after=small_spec(ya).tolist(),
                decisions=rep["decisions"], vu_before=rep["vu"]["vu_before"][::4], vu_after=rep["vu"]["vu_after"][::4], gain_db=rep["vu"]["gain_db"][::4],
                loudness=dict(before=rep["master"]["input_lufs"], after=rep["master"]["output_lufs"]),
                true_peak=dict(before=rep["master"]["input_true_peak_db"], after=rep["master"]["output_true_peak_db"]))
    json.dump(sess, open(os.path.join(DEMO, "session.json"), "w"), default=float)
    # animated figure: a sweep reveals the restored spectrogram while the decisions appear
    from scipy import signal
    def big(sig):
        f, _, s = signal.stft(sig.mean(axis=0), SR, nperseg=4096, noverlap=3584)
        db = 20 * np.log10(np.abs(s) + 1e-9)
        grid = np.geomspace(40, 18000, 300)
        return np.stack([np.interp(grid, f, db[:, i]) for i in range(db.shape[1])], axis=1)[::-1]
    B, A = big(x), big(ya)
    n_frames, frames = 44, []
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#e8e6e1", "axes.labelcolor": "#8b8f99", "xtick.color": "#8b8f99", "ytick.color": "#8b8f99"})
    for k in range(n_frames):
        frac = min(1.0, max(0.0, (k - 6) / (n_frames - 16)))
        fig = plt.figure(figsize=(9.6, 5.4), dpi=90, facecolor="#0e0f12")
        ax = fig.add_axes([0.06, 0.40, 0.90, 0.50]); ax.set_facecolor("#000")
        cut = int(frac * B.shape[1])
        img = np.concatenate([A[:, :cut], B[:, cut:]], axis=1)
        ax.imshow(img, aspect="auto", cmap="magma", vmin=-95, vmax=-20, extent=[0, sess["seconds"], 0, 1])
        if 0 < frac < 1:
            ax.axvline(frac * sess["seconds"], color="#ffffff", lw=1.5)
        ax.set_yticks([np.log(f / 40) / np.log(18000 / 40) for f in (100, 1000, 10000)]); ax.set_yticklabels(["100", "1k", "10k"], fontsize=8)
        ax.set_xlabel("seconds", fontsize=8); ax.tick_params(labelsize=8)
        fig.text(0.06, 0.935, a.title, fontsize=13, fontweight="bold")
        fig.text(0.96, 0.935, "restored" if frac >= 1 else "damaged" if frac <= 0 else "restoring", fontsize=11, ha="right", color="#ffb35c" if frac > 0 else "#7aa2ff")
        shown = int(np.ceil(frac * len(sess["decisions"]))) if frac > 0 else 0
        for i, d in enumerate(sess["decisions"][:shown]):
            fig.text(0.06, 0.28 - i * 0.052, "✓  " + d, fontsize=9.5, color="#e8e6e1")
        if frac >= 1:
            fig.text(0.06, 0.28 - len(sess["decisions"]) * 0.052 - 0.01,
                     f"Loudness {sess['loudness']['before']:.1f} → {sess['loudness']['after']:.1f} LUFS   true peak {sess['true_peak']['before']:.1f} → {sess['true_peak']['after']:.1f} dBTP",
                     fontsize=9.5, color="#6fd49a")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(96, method=Image.Quantize.MEDIANCUT))
        plt.close(fig)
    frames[0].save(os.path.join(DEMO, "restore.gif"), save_all=True, append_images=frames[1:], duration=[110] * (n_frames - 1) + [3500], loop=0, optimize=True)
    print("wrote", os.path.join(DEMO, "session.json"), "and restore.gif;", len(sess["decisions"]), "decisions")
    for d in sess["decisions"]:
        print("  ", d)


RAMP = " .:-=+*#%@"


def _row(vals, color=True):
    if not color:
        return "".join(RAMP[min(9, int(v) * 10 // 256)] for v in vals)
    out = []
    for v in vals:
        # magma-like ramp in 256-color terminals: black -> purple -> orange -> pale
        c = (16, 53, 54, 90, 126, 162, 198, 204, 209, 215, 221, 229)[min(11, int(v) * 12 // 256)]
        out.append(f"\033[48;5;{c}m ")
    return "".join(out) + "\033[0m"


def replay(a):
    path = a.session or os.path.join(DEMO, "session.json")
    s = json.load(open(path))
    before, after = np.array(s["before"], dtype=np.uint8), np.array(s["after"], dtype=np.uint8)
    rows, cols = before.shape
    color = sys.stdout.isatty() and not a.plain
    steps = [0] + list(range(0, cols + 1, max(1, cols // 40)))[1:] + [cols]
    print(f"\n  {s['title']}   ({s['seconds']:.0f} s, damage: {s['stress']})\n")
    shown = 0
    for n, cut in enumerate(steps):
        img = np.concatenate([after[:, :cut], before[:, cut:]], axis=1)
        if n and sys.stdout.isatty():
            sys.stdout.write(f"\033[{rows + 2 + shown}A")
        label = "damaged " if cut == 0 else "restored" if cut == cols else "restoring"
        sys.stdout.write(f"  16k │ {label:<60}\n")
        for r in range(rows):
            sys.stdout.write("      │" + _row(img[r], color) + "\n")
        sys.stdout.write("  40Hz└" + "─" * cut + ("▲" if 0 < cut < cols else "") + "\n")
        want = int(np.ceil(cut / cols * len(s["decisions"])))
        for d in s["decisions"][:shown]:
            sys.stdout.write("   ✓ " + d + "\n")
        for d in s["decisions"][shown:want]:
            sys.stdout.write("   ✓ " + d + "\n")
        shown = want
        sys.stdout.flush()
        if sys.stdout.isatty():
            time.sleep(a.delay if 0 < cut < cols else 0.8)
        elif cut not in (0, cols):
            continue
    print(f"\n  Loudness {s['loudness']['before']:.1f} -> {s['loudness']['after']:.1f} LUFS, true peak {s['true_peak']['before']:.1f} -> {s['true_peak']['after']:.1f} dBTP\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--input", required=True)
    r.add_argument("--title", default="Lacquer: restoring a damaged mix")
    r.add_argument("--stress", default="reverb+echo")
    r.add_argument("--seconds", type=float, default=20.0)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--rir", default="data/raw/mit_ir")
    q = sub.add_parser("replay")
    q.add_argument("--session", default=None)
    q.add_argument("--delay", type=float, default=0.06)
    q.add_argument("--plain", action="store_true", help="ASCII shading instead of terminal colors")
    a = p.parse_args()
    record(a) if a.cmd == "record" else replay(a)
