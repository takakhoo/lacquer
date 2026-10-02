"""Can a fault inside one instrument be fixed from the finished mix?

  python -m remaster.evaluate_stem_master --musdb data/raw/musdb18hq/test --out docs/evidence/stem_master

For each MUSDB18-HQ test song a 30 s excerpt is rebuilt from its true stems with one fault: a level error or a
tone error on one stem. Three systems try to undo it from the mix alone: the mix-level tone correction, the stem
engine on separated stems, and the stem engine on the true (faulty) stems as the bound set by perfect separation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch

from .analysis import SR
from .degrade import match_level
from .evaluate_mastering import unmaster
from .losses import log_spec_dist, si_sdr
from .mastering import tonal
from .mastering_norms import ranges
from .stem_master import master_stems
from .stems import STEMS, separate

FAULTS = [("level", s) for s in ("vocals", "drums", "bass")] + [("tone", s) for s in ("vocals", "drums", "bass")] + [("none", "")]


def score(y, ref):
    a, b = torch.from_numpy(match_level(y, -20.0))[None], torch.from_numpy(match_level(ref, -20.0))[None]
    return dict(sisdr=si_sdr(a, b).item(), lsd=log_spec_dist(a, b).item())


def stem_gains(y, true):
    """Least-squares gain of each true stem inside y, in dB."""
    A = np.stack([true[s].reshape(-1) for s in STEMS], 1).astype(np.float64)
    g, *_ = np.linalg.lstsq(A, y.reshape(-1).astype(np.float64), rcond=None)
    return {s: float(20 * np.log10(max(abs(v), 1e-4))) for s, v in zip(STEMS, g)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--musdb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--songs", type=int, default=50)
    p.add_argument("--seconds", type=float, default=30.0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, n = [], int(a.seconds * SR)
    for si, d in enumerate(sorted(glob.glob(os.path.join(a.musdb, "*", "")))[: a.songs]):
        true = {s: sf.read(os.path.join(d, s + ".wav"), dtype="float32", always_2d=True)[0].T for s in STEMS}
        # the 30 s where the vocal is most active
        v = (true["vocals"] ** 2).mean(axis=0)
        c = np.concatenate([[0], np.cumsum(v)])
        start = int(np.argmax(c[n::SR] - c[:-n:SR])) * SR if len(v) > n + SR else 0
        true = {s: x[:, start:start + n] for s, x in true.items()}
        clean = sum(true.values())
        for fi, (kind, stem) in enumerate(FAULTS):
            rng = np.random.default_rng([31, si, fi])
            faulty = dict(true)
            info = {}
            if kind == "level":
                info["gain_db"] = float(rng.choice([-1, 1]) * rng.uniform(4, 9))
                faulty[stem] = true[stem] * 10 ** (info["gain_db"] / 20)
            elif kind == "tone":
                faulty[stem], info = unmaster(true[stem], "bumps", rng)
            mix = sum(faulty.values()).astype(np.float32)
            sep = separate(mix)
            out_mix, _ = tonal(mix, ranges("all"))
            out_sep, rep = master_stems(mix, stems=sep)
            oracle = dict(faulty, residual=np.zeros_like(mix))
            out_orc, _ = master_stems(mix, stems=oracle)
            row = dict(song=os.path.basename(d.rstrip("/")), kind=kind, stem=stem, info=info, decisions=rep["decisions"],
                       input=score(mix, clean), mix_level=score(out_mix, clean), stems=score(out_sep, clean), stems_true=score(out_orc, clean))
            if kind == "level":
                others = [s for s in ("vocals", "drums", "bass", "other") if s != stem]
                for k, y in (("input", mix), ("stems", out_sep), ("stems_true", out_orc)):
                    g = stem_gains(y, true)
                    row[k]["balance_err_db"] = abs(g[stem] - float(np.mean([g[s] for s in others])))
            rows.append(row)
            print(row["song"][:24], kind, stem, {k: round(row[k]["sisdr"], 1) for k in ("input", "mix_level", "stems", "stems_true")},
                  {k: round(row[k].get("balance_err_db", 0), 1) for k in ("input", "stems", "stems_true")}, flush=True)
    summ = {}
    for kind, stem in FAULTS:
        rs = [r for r in rows if r["kind"] == kind and r["stem"] == stem]
        summ[f"{kind}:{stem}"] = dict(n=len(rs), **{k: {m: float(np.mean([r[k][m] for r in rs])) for m in rs[0][k]} for k in ("input", "mix_level", "stems", "stems_true")})
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, "stem_master.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
