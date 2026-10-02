"""What do EnCodec representations know about degradations?

The thesis tried to synthesize clean audio from EnCodec tokens and hit the codec's quality ceiling.
This asks the opposite question: are tokens a good *analysis* representation for deciding what is
wrong with a track? Identical small probes are trained on (a) discrete tokens, (b) continuous
encoder latents, (c) a log-mel spectrogram, to detect each degradation and regress its strength.

  python -m remaster.token_probe build --data data/raw/fma_medium --rir data/raw/mit_ir --out data/probe --n 32000
  python -m remaster.token_probe train --out data/probe
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .data import list_tracks, load_audio, split_of
from .degrade import EFFECTS, SR, RIRBank, degrade, match_level, rms_db

SEG = 4 * SR
REG = ("reverb_drr", "echo_delay_ms", "echo_att", "noise_snr", "clip_pct", "comp_ratio", "eq_max_db")


class Clips(Dataset):
    def __init__(self, files, rir, n, seed):
        self.files, self.rir, self.n, self.seed, self.bank = files, rir, n, seed, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.bank is None:
            self.bank = RIRBank(self.rir)
        rng = np.random.default_rng([self.seed, i])
        while True:
            f = self.files[rng.integers(len(self.files))]
            try:
                x = load_audio(f)
            except Exception:
                continue
            if x.shape[1] < SEG + 2 * SR:
                continue
            s = rng.integers(0, x.shape[1] - SEG - 2 * SR + 1)
            x = x[:, s:s + SEG + 2 * SR]
            if rms_db(x[:, -SEG:]) < -45:
                continue
            effects = [e for e in EFFECTS if rng.random() < 0.3]
            y, log = degrade(x, rng, self.bank, effects=effects)
            y = match_level(y[:, -SEG:], -20.0)
            if not np.isfinite(y).all():
                continue
            cls = np.array([e in log for e in EFFECTS], dtype=np.float32)
            reg = np.full(len(REG), np.nan, dtype=np.float32)
            if "reverb" in log: reg[0] = log["reverb"]["drr_db"]
            if "echo" in log: reg[1], reg[2] = log["echo"]["delay"] * 1000, log["echo"]["att"]
            if "noise" in log: reg[3] = log["noise"]["snr"]
            if "clip" in log: reg[4] = log["clip"]["pct"]
            if "comp" in log: reg[5] = log["comp"]["ratio"]
            if "eq" in log: reg[6] = max(abs(b[2]) for b in log["eq"]["bands"])
            return torch.from_numpy(y), torch.from_numpy(cls), torch.from_numpy(reg)


@torch.no_grad()
def encodec_features(codec, wav48):
    """wav48 [B, 2, N] -> tokens [B, 16, T] (int), latents [B, 128, T] at 150 Hz, per-segment scales [B, S]."""
    seg, stride = codec.segment_length, codec.segment_stride
    toks, lats, scales = [], [], []
    for off in range(0, wav48.shape[-1], stride):
        x = wav48[..., off:off + seg]
        mono = x.mean(dim=1, keepdim=True)
        scale = mono.pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-8
        emb = codec.encoder(x / scale)
        codes = codec.quantizer.encode(emb, codec.frame_rate, codec.bandwidth).transpose(0, 1)
        keep = emb.shape[-1] if off + seg >= wav48.shape[-1] else int(round(stride / seg * emb.shape[-1]))
        toks.append(codes[..., :keep]); lats.append(emb[..., :keep]); scales.append(scale.view(-1))
    return torch.cat(toks, -1), torch.cat(lats, -1), torch.stack(scales, 1)


def build(a):
    from encodec import EncodecModel
    torch.multiprocessing.set_sharing_strategy("file_system")  # many workers x many batches exhausts file descriptors otherwise
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    codec = EncodecModel.encodec_model_48khz().to(dev).eval()
    codec.set_target_bandwidth(24.0)
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=2048, hop_length=512, n_mels=128).to(dev)
    files = list_tracks(a.data)
    for split, n, seed in (("train", a.n, 1), ("test", a.n // 10, 2)):
        fs = [f for f in files if split_of(f) == ("train" if split == "train" else "test")]
        dl = DataLoader(Clips(fs, a.rir, n, seed), batch_size=32, num_workers=16)
        T, L, M, C, R = [], [], [], [], []
        for i, (y, cls, reg) in enumerate(dl):
            y = y.to(dev)
            tok, lat, _ = encodec_features(codec, torchaudio.functional.resample(y, SR, 48000))
            T.append(tok.short().cpu()); L.append(lat.half().cpu())
            M.append(torch.log(mel(y.mean(1)) + 1e-5).half().cpu()); C.append(cls); R.append(reg)
            if i % 50 == 0:
                print(split, i * 32, "/", n, tok.shape, lat.shape, M[-1].shape, flush=True)
        torch.save(dict(tok=torch.cat(T), lat=torch.cat(L), mel=torch.cat(M), cls=torch.cat(C), reg=torch.cat(R)), os.path.join(a.out, f"{split}.pt"))


class Probe(nn.Module):
    """Same trunk for every input type; only the input projection differs."""

    def __init__(self, kind, n_q=16, dim=256, depth=4):
        super().__init__()
        self.kind, self.n_q = kind, n_q
        if kind == "tok":
            self.emb = nn.ModuleList([nn.Embedding(1024, dim) for _ in range(n_q)])
        else:
            self.inp = nn.Conv1d(128, dim, 3, padding=1)
        self.down = nn.Conv1d(dim, dim, 4, stride=4 if kind != "mel" else 2)  # ~37.5 Hz (EnCodec) / ~43 Hz (mel)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.pos = nn.Parameter(torch.zeros(1, 400, dim))
        self.cls = nn.Linear(dim, len(EFFECTS))
        self.reg = nn.Linear(dim, len(REG))

    def forward(self, x):
        if self.kind == "tok":
            h = sum(e(x[:, i].long()) for i, e in enumerate(self.emb)).transpose(1, 2)
        else:
            h = self.inp(x.float())
        h = self.down(h).transpose(1, 2)
        h = self.trunk(h + self.pos[:, : h.shape[1]])
        h = h.mean(1)
        return self.cls(h), self.reg(h)


def auroc(score, label):
    order = np.argsort(score)
    rank = np.empty(len(score)); rank[order] = np.arange(1, len(score) + 1)
    pos = label > 0.5
    return float((rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(pos.sum() * (~pos).sum(), 1))


def train(a):
    dev = "cuda"
    tr, te = torch.load(os.path.join(a.out, "train.pt")), torch.load(os.path.join(a.out, "test.pt"))
    mu = torch.nanmean(tr["reg"], 0)
    sd = torch.sqrt(torch.nanmean((tr["reg"] - mu) ** 2, 0))
    lat_sd = tr["lat"][:2000].float().std()
    mel_mu, mel_sd = tr["mel"][:2000].float().mean(), tr["mel"][:2000].float().std()
    results = {}
    variants = [("tok16", "tok", 16), ("tok8", "tok", 8), ("tok4", "tok", 4), ("tok1", "tok", 1), ("latent", "lat", 0), ("mel", "mel", 0)]
    for name, kind, nq in variants:
        torch.manual_seed(0)
        def feats(d, idx):
            if kind == "tok": return d["tok"][idx][:, :nq].to(dev)
            if kind == "lat": return (d["lat"][idx].float() / lat_sd).to(dev)
            return ((d["mel"][idx].float() - mel_mu) / mel_sd).to(dev)
        model = Probe(kind, n_q=max(nq, 1)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.01)
        n = len(tr["cls"]); steps = a.epochs * n // 64
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
        model.train(); step = 0
        for ep in range(a.epochs):
            perm = torch.randperm(n)
            for i in range(0, n - 63, 64):
                idx = perm[i:i + 64]
                lc, lr_ = model(feats(tr, idx))
                reg = ((tr["reg"][idx] - mu) / sd).to(dev)
                m = ~torch.isnan(reg)
                loss = nn.functional.binary_cross_entropy_with_logits(lc, tr["cls"][idx].to(dev)) + nn.functional.smooth_l1_loss(lr_[m], reg[m])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            print(name, "epoch", ep, "loss", round(loss.item(), 4), flush=True)
        model.eval(); P, Rg = [], []
        with torch.no_grad():
            for i in range(0, len(te["cls"]), 128):
                idx = torch.arange(i, min(i + 128, len(te["cls"])))
                lc, lr_ = model(feats(te, idx)); P.append(lc.sigmoid().cpu()); Rg.append((lr_.cpu() * sd + mu))
        P, Rg = torch.cat(P).numpy(), torch.cat(Rg).numpy()
        lab, reg = te["cls"].numpy(), te["reg"].numpy()
        r = dict(auroc={e: auroc(P[:, j], lab[:, j]) for j, e in enumerate(EFFECTS)},
                 mae={k: float(np.nanmean(np.abs(Rg[:, j] - reg[:, j]))) for j, k in enumerate(REG)},
                 mae_predict_mean={k: float(np.nanmean(np.abs(mu[j].item() - reg[:, j]))) for j, k in enumerate(REG)})
        r["mean_auroc"] = float(np.mean(list(r["auroc"].values())))
        results[name] = r
        print(name, json.dumps(r), flush=True)
        json.dump(results, open(os.path.join(a.out, "probe_results.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "train"])
    p.add_argument("--data", nargs="+")
    p.add_argument("--rir")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=32000)
    p.add_argument("--epochs", type=int, default=6)
    a = p.parse_args()
    build(a) if a.cmd == "build" else train(a)
