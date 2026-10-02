"""Reference mastering head-to-head on the un-mastering recovery clips: ours, Matchering 2.0, ITO-Master.

  python -m remaster.evaluate_reference_baselines --fma data/raw/fma_medium --corpus data/norms/corpus.npz --out runs/reference_eval \\
      --ito-repo /scratch/$USER/ITO-Master --ito-python /scratch/$USER/venvs/ito310/bin/python

Same tracks, faults and seeds as evaluate_mastering. Each system gets the damaged clip and the original as its
reference. ITO-Master (Koo et al., ISMIR 2025) runs through its released inference script, white-box model, with
and without its inference-time optimization.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import numpy as np
import soundfile as sf

from .analysis import SR, mastering_features
from .data import load_audio, split_of
from .evaluate_mastering import errors, matchering_master, unmaster
from .mastering import master_track
from .mastering_norms import load

FAULTS = ("tilt", "bumps", "width")


def ito(py, repo, inp, ref, out_dir, optimize, device):
    cmd = [py, "inference.py", "--input_path", inp, "--reference_path", ref, "--model_type", "white_box", "--inference_device", device, "--output_dir_path", out_dir]
    if optimize:
        cmd += ["--perform_ito", "--ito_reference_path", ref, "--ito_objective", "AudioFeatureLoss", "--num_steps", "100"]
    subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    files = sorted(glob.glob(os.path.join(out_dir, "*.wav")), key=os.path.getmtime)
    pick = [f for f in files if "init" not in os.path.basename(f)] if optimize else [f for f in files if "init" in os.path.basename(f)]
    y, sr = sf.read((pick or files)[-1], dtype="float32", always_2d=True)
    assert sr == SR
    return y.T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fma", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tracks", type=int, default=40)
    p.add_argument("--ito-repo", default=None)
    p.add_argument("--ito-python", default=None)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.out = os.path.abspath(a.out)                     # the ITO-Master script runs from its own directory
    os.makedirs(a.out, exist_ok=True)
    c = np.load(a.corpus)
    genre = dict(zip(c["ids"].tolist(), c["genre"].tolist()))
    known = set(load()["genres"])
    files = [f for f in sorted(glob.glob(os.path.join(a.fma, "*", "*.mp3"))) if split_of(f) == "test" and genre.get(int(os.path.basename(f)[:6])) in known]
    files = [files[i] for i in np.random.default_rng(3).permutation(len(files))][: a.tracks]
    rows = []
    for seed, f in enumerate(files):
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < 20 * SR:
            continue
        x = x[:, : 30 * SR]
        f0 = mastering_features(x)
        for fi, fault in enumerate(FAULTS):
            d, info = unmaster(x, fault, np.random.default_rng([seed, fi]))   # fault index matches evaluate_mastering.FAULTS order
            work = os.path.join(a.out, "work", f"{seed:03d}_{fault}")
            os.makedirs(work, exist_ok=True)
            ip, rp = os.path.join(work, "input.wav"), os.path.join(work, "reference.wav")
            sf.write(ip, d.T, SR, subtype="PCM_24"); sf.write(rp, x.T, SR, subtype="PCM_24")
            row = dict(file=os.path.basename(f), fault=fault, info=info, damaged=errors(mastering_features(d), f0), systems={})
            y, _ = master_track(d, reference=x, match_reference_peak=True)
            row["systems"]["ours"] = errors(mastering_features(y), f0)
            row["systems"]["matchering"] = errors(mastering_features(matchering_master(d, x)), f0)
            if a.ito_repo:
                for name, opt in (("ito_master", False), ("ito_master_optimized", True)):
                    try:
                        y = ito(a.ito_python, a.ito_repo, ip, rp, os.path.join(work, name), opt, a.device)
                        row["systems"][name] = errors(mastering_features(y), f0)
                    except Exception as e:
                        print("ito failed", name, (getattr(e, "stderr", b"") or b"").decode()[-400:] or str(e)[-300:], flush=True)
            rows.append(row)
            print(seed, fault, "damaged", {k: round(v, 2) for k, v in row["damaged"].items() if k in ("tone", "width")},
                  {s: (round(v["tone"], 2), round(v["width"], 2)) for s, v in row["systems"].items()}, flush=True)
            json.dump(rows, open(os.path.join(a.out, "reference_rows.json"), "w"))
    summ = {}
    for fault in FAULTS:
        rs = [r for r in rows if r["fault"] == fault]
        names = sorted(set.intersection(*[set(r["systems"]) for r in rs]))
        summ[fault] = dict(n=len(rs), damaged={k: float(np.mean([r["damaged"][k] for r in rs])) for k in ("tone", "width", "plr", "crest")},
                           **{s: {k: float(np.mean([r["systems"][s][k] for r in rs])) for k in ("tone", "width", "plr", "crest")} for s in names})
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, "reference.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
