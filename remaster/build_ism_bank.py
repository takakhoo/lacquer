"""Simulated stereo room impulse responses for training variety: shoebox rooms, image sources for the early
part and ray tracing for the tail, frequency-dependent absorption, a spaced microphone pair.

  python -m remaster.build_ism_bank --out data/raw/ism_rir --n 3000 --workers 48
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

SR = 44100
BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]


def one(job):
    seed, out = job
    import pyroomacoustics as pra
    path = os.path.join(out, f"ism_{seed:05d}.wav")
    if os.path.exists(path):
        return 1
    rng = np.random.default_rng([77, seed])
    for _ in range(20):
        dims = np.array([rng.uniform(3, 28), rng.uniform(3, 20), rng.uniform(2.4, 10)])
        rt60 = float(np.exp(rng.uniform(np.log(0.25), np.log(2.8))))
        try:
            absorb, _ = pra.inverse_sabine(rt60, dims)
        except ValueError:
            continue
        # rooms absorb more at high frequencies; the tilt and its irregularity vary per room
        tilt = np.linspace(-1, 1, len(BANDS)) * rng.uniform(0.0, 0.6) + rng.normal(0, 0.12, len(BANDS))
        coeffs = np.clip(absorb * np.exp(tilt), 0.01, 0.95)
        mat = pra.Material(energy_absorption=dict(coeffs=coeffs.tolist(), center_freqs=BANDS), scattering=float(rng.uniform(0.05, 0.45)))
        room = pra.ShoeBox(dims, fs=SR, materials=mat, max_order=int(rng.integers(3, 9)), ray_tracing=True, air_absorption=True)
        room.set_ray_tracing(n_rays=6000, receiver_radius=0.5, energy_thres=1e-7, time_thres=3.2)
        src = np.array([rng.uniform(0.6, dims[0] - 0.6), rng.uniform(0.6, dims[1] - 0.6), rng.uniform(0.8, min(2.2, dims[2] - 0.3))])
        ctr = np.array([rng.uniform(0.8, dims[0] - 0.8), rng.uniform(0.8, dims[1] - 0.8), rng.uniform(1.0, min(2.0, dims[2] - 0.4))])
        if np.linalg.norm(src - ctr) < 0.8:
            continue
        ang, half = rng.uniform(0, 2 * np.pi), rng.uniform(0.08, 0.35)
        off = half * np.array([np.cos(ang), np.sin(ang), 0.0])
        mics = np.stack([ctr - off, ctr + off], 1)
        if (mics[:2] < 0.2).any() or (mics[0] > dims[0] - 0.2).any() or (mics[1] > dims[1] - 0.2).any():
            continue
        room.add_source(src.tolist())
        room.add_microphone_array(mics)
        room.compute_rir()
        a, b = room.rir[0][0], room.rir[1][0]
        n = min(max(len(a), len(b)), int(3.3 * SR))
        x = np.zeros((n, 2), dtype=np.float32)
        x[: min(n, len(a)), 0], x[: min(n, len(b)), 1] = a[:n], b[:n]
        x /= np.abs(x).max() + 1e-9
        sf.write(path, x, SR, subtype="FLOAT")
        return 1
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        done = sum(pool.imap_unordered(one, [(i, a.out) for i in range(a.n)], chunksize=4))
    print("wrote", done)


if __name__ == "__main__":
    main()
