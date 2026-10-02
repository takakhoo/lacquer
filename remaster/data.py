"""On-the-fly (degraded, clean) pairs from a folder of music files."""
from __future__ import annotations

import glob
import hashlib
import os

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch.utils.data import Dataset

from .degrade import SR, EFFECTS, RIRBank, apply_comp, apply_eq, degrade, match_level, rms_db

REF_DB = -20.0


def list_tracks(roots, exts=(".mp3", ".wav", ".flac")):
    files = []
    for root in roots:
        for f in glob.iglob(os.path.join(root, "**", "*"), recursive=True):
            if f.lower().endswith(exts) and "/._" not in f:
                files.append(f)
    return sorted(files)


def split_of(path):
    """Stable split by filename hash: ~1% val, ~1% test, rest train. MUSDB keeps its own test dir."""
    if "/musdb18hq/" in path:
        return "test" if "/musdb18hq/test/" in path else "train"
    h = int(hashlib.md5(os.path.basename(os.path.dirname(path) + path).encode()).hexdigest(), 16) % 100
    return "val" if h == 0 else "test" if h == 1 else "train"


def load_audio(path, start_s=None, dur_s=None):
    """Return float32 [2, N] at SR."""
    info = sf.info(path)
    if start_s is None:
        x, fs = sf.read(path, dtype="float32", always_2d=True)
    else:
        x, fs = sf.read(path, start=int(start_s * info.samplerate), frames=int(dur_s * info.samplerate), dtype="float32", always_2d=True)
    x = x.T
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    x = x[:2]
    if fs != SR:
        x = signal.resample_poly(x, SR, fs, axis=1).astype(np.float32)
    return x


ARTIFACTS = ("reverb", "echo", "clip", "noise")
P_ARTIFACT = dict(reverb=0.6, echo=0.35, clip=0.2, noise=0.2, eq=0.0, comp=0.0)


def make_pair(clean_ctx, rng, bank, seg, effects=None, task="joint", p_identity=0.08, p_algo=0.0):
    """Degrade a segment that has leading context, then keep the last `seg` samples of both.

    The context gives reverb tails and echoes of preceding audio a chance to land inside the crop,
    as they would in a real recording.

    task="joint": every effect is a degradation to undo.
    task="artifact": only reverb/echo/clip/noise are undone. EQ and compression are applied to the
    clean side as augmentation, so the target keeps whatever tone and dynamics the input has.
    """
    if task.startswith("tone:"):  # single-effect variant for diagnostics, e.g. "tone:ride"
        if effects is None:
            name = task.split(":")[1]
            pool = ["eq", "comp", "ride"]
            if rng.random() < p_identity:
                effects = []
            elif name == "s1":    # curriculum stage 1: exactly one effect
                effects = [str(rng.choice(pool))]
            elif name == "s2":    # stage 2: one or two effects
                effects = [str(e) for e in rng.choice(pool, size=rng.integers(1, 3), replace=False)]
            else:
                effects = name.split("+")
        task = "tone"
    if task == "tone":
        # tone and dynamics only: EQ, compression, level riding. These are corrected by predicted DSP parameters.
        if effects is None:
            effects = [] if rng.random() < p_identity else ([e for e, p in (("eq", 0.6), ("comp", 0.5), ("ride", 0.3)) if rng.random() < p] or ["eq"])
    if task == "artifact":
        if rng.random() < 0.5:
            clean_ctx, _ = apply_eq(clean_ctx, rng, max_db=6.0)
        if rng.random() < 0.3:
            clean_ctx, _ = apply_comp(clean_ctx, rng)
        if effects is None and rng.random() >= p_identity:
            effects = [e for e in ARTIFACTS if rng.random() < P_ARTIFACT[e]] or [str(rng.choice(ARTIFACTS[:2]))]
        elif effects is None:
            effects = []
    deg, log = degrade(clean_ctx, rng, bank, effects=effects, p_algo=p_algo)
    clean, deg = clean_ctx[:, -seg:], deg[:, -seg:]
    clean, deg = match_level(clean, REF_DB), match_level(deg, REF_DB)
    return deg, clean, log


class PairDataset(Dataset):
    def __init__(self, files, rir_root=None, seg_s=4.0, ctx_s=2.0, epoch_len=100000, seed=0, task="joint",
                 groups=None, weights=None, p_identity=0.08, p_algo=0.0):
        """`groups` is a list of file lists sampled with probabilities `weights` (e.g. MP3 corpus vs lossless)."""
        self.task, self.p_identity, self.p_algo = task, p_identity, p_algo
        self.groups = groups or [files]
        self.weights = np.array(weights or [1.0] * len(self.groups), dtype=np.float64)
        self.weights /= self.weights.sum()
        self.files, self.seg, self.ctx = files, int(seg_s * SR), int(ctx_s * SR)
        self.rir_root, self.bank, self.epoch_len, self.seed = rir_root, None, epoch_len, seed

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        if self.bank is None:
            self.bank = RIRBank(self.rir_root)
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, i, info.id if info else 0, int.from_bytes(os.urandom(4), "little")])
        need = self.seg + self.ctx
        for _ in range(20):
            group = self.groups[rng.choice(len(self.groups), p=self.weights)]
            f = group[rng.integers(len(group))]
            try:
                if f.lower().endswith((".wav", ".flac")):
                    # long lossless files: seek to a random window instead of decoding the whole track
                    info = sf.info(f)
                    win = (need / SR) + 0.5
                    start = rng.uniform(0, max(0.0, info.duration - win))
                    x = load_audio(f, start, win)
                else:
                    x = load_audio(f)
            except Exception:
                continue
            if x.shape[1] < need or not np.isfinite(x).all():
                continue
            s = rng.integers(0, x.shape[1] - need + 1)
            x = x[:, s:s + need]
            if rms_db(x[:, -self.seg:]) < -45:
                continue
            if rng.random() < 0.5:
                x = x[::-1].copy()
            deg, clean, _ = make_pair(x, rng, self.bank, self.seg, task=self.task, p_identity=self.p_identity, p_algo=self.p_algo)
            g = 10 ** (rng.uniform(-12, 4) / 20)
            deg, clean = deg * g, clean * g
            if not (np.isfinite(deg).all() and np.abs(deg).max() < 50):
                continue
            return torch.from_numpy(np.ascontiguousarray(deg)), torch.from_numpy(np.ascontiguousarray(clean))
        raise RuntimeError("could not sample a valid training pair")


VAL_CONDITIONS = ("reverb", "echo", "eq", "comp", "clip", "noise", "reverb+echo", "reverb+eq+comp", "identity")
VAL_ARTIFACT = ("reverb", "echo", "clip", "noise", "reverb+echo", "reverb+echo+clip", "identity")
VAL_TONE = ("eq", "comp", "ride", "eq+comp", "eq+comp+ride", "identity", "limit")  # append only: earlier items keep their seeds


def build_fixed_set(files, rir_root, n_tracks=32, seg_s=6.0, ctx_s=2.0, seed=1234, task="joint"):
    """Deterministic eval set: every track under every condition. Returns list of dicts."""
    bank = RIRBank(rir_root)
    rng0 = np.random.default_rng(seed)
    pick = rng0.permutation(len(files))
    seg, ctx = int(seg_s * SR), int(ctx_s * SR)
    out = []
    for idx in pick:
        if len({o["file"] for o in out}) >= n_tracks:
            break
        f = files[idx]
        try:
            x = load_audio(f)
        except Exception:
            continue
        if x.shape[1] < seg + ctx + SR:
            continue
        s = (x.shape[1] - seg - ctx) // 2
        x = x[:, s:s + seg + ctx]
        if rms_db(x[:, -seg:]) < -40:
            continue
        for ci, cond in enumerate(VAL_ARTIFACT if task == "artifact" else VAL_TONE if task.startswith("tone") else VAL_CONDITIONS):
            rng = np.random.default_rng([seed, int(idx), ci])
            effects = [] if cond == "identity" else cond.split("+")
            deg, clean, log = make_pair(x, rng, bank, seg, effects=effects, task=task.split(":")[0])
            out.append(dict(file=f, cond=cond, deg=torch.from_numpy(np.ascontiguousarray(deg)), clean=torch.from_numpy(np.ascontiguousarray(clean)), log=log))
    return out
