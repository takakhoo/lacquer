"""Reference-free quality check: does the restored audio score better than the damaged audio?

SI-SDR only says how close the output is to one clean reference. This asks a listener-style question
with Meta's Audiobox Aesthetics predictor (production quality and content enjoyment, 1-10), which needs
no reference, and checks with chroma/rhythm similarity that the music itself was not changed.

  python -m remaster.evaluate_quality --ckpt runs/x/best.pt --data <music> --rir <IRs> --out evidence/quality
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import list_tracks, load_audio, split_of
from .degrade import SR, RIRBank, degrade, match_level
from .descriptors import content_similarity
from .infer import enhance_auto, load_model
from .quality import Aesthetics

CONDS = ("reverb", "echo", "reverb+echo", "clip", "noise", "clean")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--rir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=24)
    p.add_argument("--seconds", type=float, default=12.0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = dict(mix=load_model(a.ckpt), vocal=None)
    scorer, bank = Aesthetics(), RIRBank(a.rir)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(5).permutation(len(files))[: a.tracks * 2]]
    rows, n = [], int(a.seconds * SR)
    for fi, f in enumerate(files):
        if len({r["file"] for r in rows}) >= a.tracks:
            break
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < n + SR:
            continue
        clean = match_level(x[:, SR // 2: SR // 2 + n], -20.0)
        for ci, cond in enumerate(CONDS):
            rng = np.random.default_rng([5, fi, ci])
            deg = clean if cond == "clean" else match_level(degrade(clean, rng, bank, effects=cond.split("+"))[0], -20.0)
            out, rep = enhance_auto(models, deg, use_stems=False, do_master=False, ride=0)
            out = match_level(out, -20.0)  # the predictor should judge quality, not loudness
            s = scorer(np.stack([clean, deg, out]), SR)
            sim = content_similarity(clean, out, SR)
            rows.append(dict(file=os.path.basename(f), cond=cond, clean=s[0], degraded=s[1], restored=s[2], similarity=sim,
                             room=rep.get("room", {}).get("action"), echoes=len(rep.get("echoes", []))))
        print(len({r["file"] for r in rows}), "tracks", flush=True)
    summ = {}
    for c in CONDS:
        rs = [r for r in rows if r["cond"] == c]
        summ[c] = dict(n=len(rs), **{f"{k}_{m}": float(np.mean([r[k][m] for r in rs])) for k in ("clean", "degraded", "restored") for m in ("pq", "ce")},
                       restored_better_pq=float(np.mean([r["restored"]["pq"] > r["degraded"]["pq"] for r in rs])),
                       **{f"sim_{k}": float(np.mean([r["similarity"][k] for r in rs])) for k in rows[0]["similarity"]})
    json.dump(dict(ckpt=a.ckpt, summary=summ, rows=rows), open(os.path.join(a.out, "quality.json"), "w"), indent=1)
    lines = ["| condition | n | PQ clean | PQ damaged | PQ restored | restored > damaged | CE clean | CE damaged | CE restored |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, s in summ.items():
        lines.append(f"| {c} | {s['n']} | {s['clean_pq']:.2f} | {s['degraded_pq']:.2f} | {s['restored_pq']:.2f} | {100 * s['restored_better_pq']:.0f}% | {s['clean_ce']:.2f} | {s['degraded_ce']:.2f} | {s['restored_ce']:.2f} |")
    lines += ["", "Content similarity, restored vs clean (1.0 = identical): " + "; ".join(f"{c}: " + ", ".join(f"{k[4:]} {v:.3f}" for k, v in s.items() if k.startswith("sim_")) for c, s in summ.items())]
    open(os.path.join(a.out, "quality.md"), "w").write("Audiobox Aesthetics, 1-10. PQ = production quality, CE = content enjoyment. All clips level-matched.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
