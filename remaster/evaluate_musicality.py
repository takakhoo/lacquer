"""Musical descriptors before and after restoration: does the repair bring the music's own properties back?

SI-SDR compares waveforms. These compare what a listener would call the character of the track: how long
notes ring (decay), how clear the pulse and the key are, how bright and how dynamic it is, and whether the
notes and rhythm are the same ones. For each descriptor the question is how much of the gap that the damage
opened between the damaged and the clean clip is closed by the restored one.

  python -m remaster.evaluate_musicality --ckpt remaster/checkpoints/best.pt --data <music> --rir <IRs> --out docs/evidence/musicality
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import list_tracks, load_audio, split_of
from .degrade import SR, RIRBank, degrade, match_level
from .descriptors import content_similarity, describe
from .infer import enhance_auto, load_model

CONDS = ("reverb", "echo", "reverb+echo", "clip", "noise", "clean")
KEYS = ("decay_s", "pulse_clarity", "key_clarity", "flux", "flatness", "crest_db", "dynamics_db", "centroid_hz")
NAMES = dict(decay_s="note decay (s)", pulse_clarity="pulse clarity", key_clarity="key clarity", flux="spectral flux", flatness="spectral flatness",
             crest_db="crest factor (dB)", dynamics_db="dynamics (dB)", centroid_hz="brightness (Hz)")


def summarize(rows, out, ckpt):
    summ = {}
    for c in CONDS:
        rs = [r for r in rows if r["cond"] == c]
        s = dict(n=len(rs))
        for k in KEYS:
            # a descriptor can be undefined on a clip (no clear note decay): use the clips where all three exist
            ok = [r for r in rs if all(np.isfinite(r[w][k]) for w in ("clean", "damaged", "restored"))]
            if not ok:
                s[k] = dict(n=0)
                continue
            ed = float(np.mean([abs(r["damaged"][k] - r["clean"][k]) for r in ok])); er = float(np.mean([abs(r["restored"][k] - r["clean"][k]) for r in ok]))
            s[k] = dict(n=len(ok), clean=float(np.mean([r["clean"][k] for r in ok])), damaged=float(np.mean([r["damaged"][k] for r in ok])), restored=float(np.mean([r["restored"][k] for r in ok])),
                        err_damaged=ed, err_restored=er, recovered=float(1 - er / ed) if ed > 1e-9 else None)
        for k in ("chroma_sim", "rhythm_sim"):
            s[k] = dict(damaged=float(np.mean([r["sim_damaged"][k] for r in rs])), restored=float(np.mean([r["sim_restored"][k] for r in rs])))
        summ[c] = s
    json.dump(dict(ckpt=ckpt, summary=summ, rows=rows), open(os.path.join(out, "musicality.json"), "w"), indent=1, default=float)
    lines = []
    for c in CONDS[:-1]:
        s = summ[c]
        lines += [f"### {c} (n = {s['n']})", "", "| descriptor | clean | damaged | restored | gap closed |", "|---|---:|---:|---:|---:|"]
        for k in KEYS:
            v = s[k]
            if not v["n"]:
                continue
            name = NAMES[k] + ("" if v["n"] == s["n"] else f" ({v['n']} clips)")
            lines.append(f"| {name} | {v['clean']:.3g} | {v['damaged']:.3g} | {v['restored']:.3g} | " + ("n/a" if v["recovered"] is None else f"{100 * v['recovered']:.0f}%") + " |")
        lines += [f"| same notes (chroma similarity to clean) | 1 | {s['chroma_sim']['damaged']:.3f} | {s['chroma_sim']['restored']:.3f} | |",
                  f"| same rhythm (onset-envelope correlation) | 1 | {s['rhythm_sim']['damaged']:.3f} | {s['rhythm_sim']['restored']:.3f} | |", ""]
    open(os.path.join(out, "musicality.md"), "w").write("Gap closed = 1 - |restored - clean| / |damaged - clean|, averaged per clip. Negative means the restored clip is further from clean than the damaged one.\n\n" + "\n".join(lines))
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--rir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=24)
    p.add_argument("--seconds", type=float, default=15.0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = dict(mix=load_model(a.ckpt), vocal=None)
    bank = RIRBank(a.rir)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    files = [files[i] for i in np.random.default_rng(9).permutation(len(files))[: a.tracks * 2]]
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
        dc = describe(clean, SR)
        for ci, cond in enumerate(CONDS):
            rng = np.random.default_rng([9, fi, ci])
            deg = clean if cond == "clean" else match_level(degrade(clean, rng, bank, effects=cond.split("+"))[0], -20.0)
            out, _ = enhance_auto(models, deg, use_stems=False, do_master=False, ride=0)
            out = match_level(out, -20.0)
            rows.append(dict(file=os.path.basename(f), cond=cond, clean=dc, damaged=describe(deg, SR), restored=describe(out, SR),
                             sim_damaged=content_similarity(clean, deg, SR), sim_restored=content_similarity(clean, out, SR)))
        print(len({r["file"] for r in rows}), "tracks", flush=True)
    summarize(rows, a.out, a.ckpt)

if __name__ == "__main__":
    main()
