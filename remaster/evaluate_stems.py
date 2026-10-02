"""Full-mix model vs stem-aware pipeline on MUSDB18-HQ test songs (ground-truth stems available).

  python -m remaster.evaluate_stems --musdb data/raw/musdb18hq/test --rir data/raw/mit_ir --mix-ckpt runs/ft_bs/best.pt \\
      --vocal-ckpt pretrained/vocal_dereverb.pt --out evidence/stems

Scenarios: room (reverb on the whole mix), wet_vocal (reverb on the vocal stem only), clean.
Target is always the clean mix. Mastering and de-echo are off so only reverb handling is compared.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch

from .degrade import SR, RIRBank, apply_reverb
from .infer import enhance, enhance_auto, enhance_stems, load_model
from .losses import log_spec_dist, si_sdr
from .reverb_meter import active


def score(y, ref):
    a, b = torch.from_numpy(y)[None], torch.from_numpy(ref)[None]
    return dict(sisdr=si_sdr(a, b).item(), lsd=log_spec_dist(a, b).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--musdb", required=True)
    p.add_argument("--rir", required=True)
    p.add_argument("--mix-ckpt", required=True)
    p.add_argument("--vocal-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--songs", type=int, default=16)
    p.add_argument("--seconds", type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = dict(mix=load_model(a.mix_ckpt), vocal=load_model(a.vocal_ckpt))
    bank = RIRBank(a.rir)
    rows, decisions = [], []
    for si, d in enumerate(sorted(glob.glob(os.path.join(a.musdb, "*")))):
        if len({r["song"] for r in rows}) >= a.songs:
            break
        v, _ = sf.read(d + "/vocals.wav", dtype="float32")
        v = v.T
        n = a.seconds * SR
        best = max(range(0, max(1, v.shape[1] - n), 10 * SR), key=lambda s: active(v[:, s:s + n]).mean())
        if active(v[:, best:best + n]).mean() < 0.6:
            continue
        st = {k: sf.read(f"{d}/{k}.wav", start=best, frames=n, dtype="float32")[0].T.copy() for k in ("vocals", "drums", "bass", "other")}
        clean = sum(st.values())
        rng = np.random.default_rng([11, si])
        scen = dict(
            room=apply_reverb(clean, rng, bank, drr_db=rng.uniform(0, 9))[0],
            wet_vocal=clean - st["vocals"] + apply_reverb(st["vocals"], rng, bank, drr_db=rng.uniform(-3, 6))[0],
            clean=clean,
        )
        for name, deg in scen.items():
            full, rep_f = enhance(models["mix"], deg, do_master=False, do_deecho=False)
            stem, rep_s = enhance_stems(models, deg, do_master=False, do_deecho=False, allow_add=False)
            auto, rep_a = enhance_auto(models, deg, do_master=False, do_deecho=False, allow_add=False, ride=0)
            rows.append(dict(song=os.path.basename(d), scenario=name, input=score(deg, clean), full_mix=score(full, clean), stems=score(stem, clean), auto=score(auto, clean),
                             auto_room=rep_a["room"]["action"], auto_vocal=rep_a["vocal"]["action"]))
            decisions.append(dict(song=os.path.basename(d), scenario=name, full_mix_bypassed=rep_f.get("restore_bypassed"),
                                  stems={k: (v_["action"], None if v_["wetness_db"] is None else round(v_["wetness_db"], 1)) for k, v_ in rep_s["stems"].items()}))
            print(rows[-1]["song"][:28], name, {k: round(rows[-1][k]["sisdr"], 1) for k in ("input", "full_mix", "stems", "auto")}, rows[-1]["auto_room"], rows[-1]["auto_vocal"], flush=True)
    summ = {}
    for sc in ("room", "wet_vocal", "clean"):
        rs = [r for r in rows if r["scenario"] == sc]
        summ[sc] = {m: {k: float(np.mean([r[m][k] for r in rs])) for k in ("sisdr", "lsd")} for m in ("input", "full_mix", "stems", "auto")}
        summ[sc]["auto_actions"] = dict(room_reduce=sum(r["auto_room"] == "reduce" for r in rs), vocal_reduce=sum(r["auto_vocal"] == "reduce" for r in rs))
        acts = [d_["stems"] for d_ in decisions if d_["scenario"] == sc]
        summ[sc]["actions"] = {k: {a_: sum(x[k][0] == a_ for x in acts) for a_ in ("reduce", "keep", "add", "skip")} for k in ("vocals", "drums", "bass", "other")}
        summ[sc]["n"] = len(rs)
    json.dump(dict(summary=summ, rows=rows, decisions=decisions), open(os.path.join(a.out, "stems_eval.json"), "w"), indent=1)
    lines = ["| scenario | n | input | full-mix model | stem pipeline | auto (two-level) |", "|---|---:|---:|---:|---:|---:|"]
    for sc, s in summ.items():
        lines.append(f"| {sc} | {s['n']} | " + " | ".join(f"{s[m]['sisdr']:.1f} dB / {s[m]['lsd']:.1f}" for m in ("input", "full_mix", "stems", "auto")) + " |")
    lines += ["", "Auto decisions: " + "; ".join(f"{sc}: room reduce {s['auto_actions']['room_reduce']}/{s['n']}, vocal reduce {s['auto_actions']['vocal_reduce']}/{s['n']}" for sc, s in summ.items())]
    lines += ["", "Decisions (count of songs per action):"]
    for sc, s in summ.items():
        lines.append(f"- {sc}: " + "; ".join(f"{k} " + ", ".join(f"{a_} {n}" for a_, n in v.items() if n) for k, v in s["actions"].items()))
    open(os.path.join(a.out, "stems_eval.md"), "w").write("SI-SDR dB (higher is better) / log-spectral distance dB (lower is better), against the clean mix\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
