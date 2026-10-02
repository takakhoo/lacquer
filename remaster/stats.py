"""Paired statistics for the per-clip evaluation files: mean gain, bootstrap 95% interval, Wilcoxon signed-rank test.

  python -m remaster.stats
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

EV = os.path.join(os.path.dirname(__file__), "..", "docs", "evidence")


def paired(a, b, n_boot=10000, seed=0):
    """b minus a, per clip."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return dict(n=int(len(d)), mean=float(d.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], improved=float((d > 0).mean()), wilcoxon_p=p)


def main():
    out = {}
    for name, path, a, b in (("reverb, training families (E18)", "baselines/baselines.json", "input", "lacquer"),
                             ("reverb, unseen rooms (E22)", "heldout_rooms/baselines.json", "input", "lacquer"),
                             ("reverb, unseen rooms, WPE (E22)", "heldout_rooms/baselines.json", "input", "wpe")):
        rows = json.load(open(os.path.join(EV, path)))["rows"]
        out[name] = paired([r[a]["sisdr"] for r in rows], [r[b]["sisdr"] for r in rows])
    rows = json.load(open(os.path.join(EV, "declip/declip.json")))["rows"]
    out["hard clipping, declipper (E21)"] = paired([r["hard"]["input"]["sisdr"] for r in rows], [r["hard"]["declip"]["sisdr"] for r in rows])
    out["hard clipping, network (E21)"] = paired([r["hard"]["input"]["sisdr"] for r in rows], [r["hard"]["network"]["sisdr"] for r in rows])
    for k, v in out.items():
        print(f"{k:40s} n={v['n']:3d}  {v['mean']:+.2f} dB  [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  improved {100 * v['improved']:.0f}%  p={v['wilcoxon_p']:.1e}")
    json.dump(out, open(os.path.join(EV, "paired_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
