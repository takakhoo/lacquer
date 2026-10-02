"""Write a labeled gallery of training pairs: audio, parameters, and spectrograms per degradation.

  python -m remaster.dump_samples --data data/raw/fma_small --rir data/raw/mit_ir --out docs/evidence/dataset_samples
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .data import list_tracks, load_audio, make_pair, split_of
from .degrade import EFFECTS, SR, RIRBank


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--rir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--per-effect", type=int, default=3)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bank = RIRBank(a.rir)
    files = [f for f in list_tracks(a.data) if split_of(f) == "test"]
    rng0 = np.random.default_rng(3)
    conds = [[e] for e in EFFECTS] + [["reverb", "echo"], ["reverb", "echo", "clip", "noise"]]
    index = []
    fig, axes = plt.subplots(len(conds), 2 * a.per_effect, figsize=(5.2 * a.per_effect * 2 / 2, 2.3 * len(conds)), squeeze=False)
    for ci, effects in enumerate(conds):
        for k in range(a.per_effect):
            while True:
                f = files[rng0.integers(len(files))]
                try:
                    x = load_audio(f)
                except Exception:
                    continue
                if x.shape[1] >= 12 * SR:
                    break
            x = x[:, 4 * SR: 12 * SR]
            rng = np.random.default_rng([3, ci, k])
            deg, clean, log = make_pair(x, rng, bank, 6 * SR, effects=effects)
            tag = f"{'_'.join(effects)}_{k}"
            sf.write(os.path.join(a.out, tag + "_degraded.wav"), deg.T.clip(-1, 1), SR)
            sf.write(os.path.join(a.out, tag + "_clean.wav"), clean.T.clip(-1, 1), SR)
            index.append(dict(id=tag, source=os.path.basename(f), effects=effects, params=log))
            for j, (sig, name) in enumerate(((clean, "clean"), (deg, "+".join(effects)))):
                ax = axes[ci, 2 * k + j]
                ax.specgram(sig.mean(0), NFFT=2048, Fs=SR, noverlap=1536, cmap="magma", vmin=-120, vmax=-30)
                ax.set_yscale("symlog", linthresh=500); ax.set_ylim(40, 20000)
                ax.set_title(name, fontsize=8); ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "gallery.png"), dpi=100)
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1, default=float)
    print(len(index), "pairs written to", a.out)


if __name__ == "__main__":
    main()
