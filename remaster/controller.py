"""Tone and dynamics controller: a network that turns knobs instead of generating audio.

It listens to a segment and outputs (1) a static EQ curve and (2) a slow broadband gain trajectory,
both in dB. They are applied to the input spectrogram as G(t, f) = eq(f) + gain(t), a mask
constrained to be separable in the log domain. Everything it can do is an EQ move or a fader move,
so it cannot add artifacts, and its output can be read and overridden like a mastering chain.

  python -m remaster.controller --data data/raw/fma_medium data/raw/musdb18hq/train --rir data/raw/mit_ir --out runs/ctrl
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from .data import PairDataset, build_fixed_set, list_tracks, split_of
from .losses import log_spec_dist, si_sdr
from .master import BAND_CENTERS

SR = 44100
N_EQ = 32
EQ_FREQS = np.geomspace(25.0, 20000.0, N_EQ)


class Controller(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8, n_fft=2048, hop=512, max_eq_db=15.0, max_gain_db=15.0, bands=1, fast_gain=False):
        super().__init__()
        self.cfg = dict(arch="controller", dim=dim, depth=depth, heads=heads, n_fft=n_fft, hop=hop, max_eq_db=max_eq_db, max_gain_db=max_gain_db, bands=bands, fast_gain=fast_gain)
        self.fast_gain = fast_gain
        self.n_fft, self.hop, self.max_eq, self.max_gain, self.bands = n_fft, hop, max_eq_db, max_gain_db, bands
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.mel = torchaudio.transforms.MelScale(128, SR, n_stft=n_fft // 2 + 1)
        # mid and side log-mel, 86 fps -> 21.5 fps
        self.inp = nn.Sequential(nn.Conv1d(256, dim, 5, padding=2), nn.GELU(), nn.Conv1d(dim, dim, 4, stride=4))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.pos = nn.Parameter(torch.zeros(1, 1024, dim))
        self.eq_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, N_EQ))
        self.gain_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, bands))
        if fast_gain:
            # A compressor moves faster than the 21.5 fps trunk. The fast head sees the trunk's context
            # (what kind of processing is present) plus local full-rate features (86 fps) of the level itself.
            self.local = nn.Sequential(nn.Conv1d(256, dim // 2, 5, padding=2), nn.GELU(), nn.Conv1d(dim // 2, dim // 2, 9, padding=4), nn.GELU())
            self.gain_head = nn.Sequential(nn.Linear(dim + dim // 2, dim), nn.GELU(), nn.Linear(dim, bands))
        for h in (self.eq_head, self.gain_head):  # start as an exact pass-through
            nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)
        # linear interpolation matrix from the EQ control points to STFT bins (log-frequency axis)
        f = np.fft.rfftfreq(n_fft, 1 / SR)
        lf, lc = np.log(np.maximum(f, EQ_FREQS[0])), np.log(EQ_FREQS)
        idx = np.clip(np.searchsorted(lc, lf) - 1, 0, N_EQ - 2)
        w = np.clip((lf - lc[idx]) / (lc[idx + 1] - lc[idx]), 0, 1)
        m = np.zeros((len(f), N_EQ), dtype=np.float32)
        m[np.arange(len(f)), idx] = 1 - w
        m[np.arange(len(f)), idx + 1] += w
        self.register_buffer("interp", torch.from_numpy(m), persistent=False)

    def stft(self, x):
        b, c, n = x.shape
        s = torch.stft(x.reshape(b * c, n), self.n_fft, self.hop, window=self.window, return_complex=True)
        return s.view(b, c, *s.shape[-2:])

    def params(self, x):
        """x [B, 2, N] -> eq_db [B, N_EQ], gain_db [B, T] (T = STFT frames), spec."""
        spec = self.stft(x)
        mid, side = (spec[:, 0] + spec[:, 1]) / 2, (spec[:, 0] - spec[:, 1]) / 2
        feats = torch.cat([torch.log(self.mel(mid.abs().pow(2)) + 1e-7), torch.log(self.mel(side.abs().pow(2)) + 1e-7)], dim=1)
        feats = (feats + 5.0) / 5.0
        h = self.inp(feats).transpose(1, 2)
        h = self.trunk(h + self.pos[:, : h.shape[1]])
        eq = self.max_eq * torch.tanh(self.eq_head(h.mean(1)) / self.max_eq)
        eq = eq - eq.mean(dim=1, keepdim=True)  # broadband level belongs to the gain trajectory
        if self.fast_gain:
            up = F.interpolate(h.transpose(1, 2), size=spec.shape[-1], mode="linear", align_corners=False)
            hg = torch.cat([up, self.local(feats)], dim=1).transpose(1, 2)
            g = (self.max_gain * torch.tanh(self.gain_head(hg) / self.max_gain))[..., 0]
        else:
            g = self.max_gain * torch.tanh(self.gain_head(h) / self.max_gain)  # [B, T/4, bands]
            g = F.interpolate(g.transpose(1, 2), size=spec.shape[-1], mode="linear", align_corners=False)[:, 0]
        return eq, g, spec

    def apply(self, spec, eq, g, length):
        gain_db = (eq @ self.interp.T)[:, None, :, None] + g[:, None, None, :]
        return torch.istft((spec * 10 ** (gain_db / 20)).flatten(0, 1), self.n_fft, self.hop, window=self.window, length=length).view(spec.shape[0], spec.shape[1], length)

    def forward(self, x, return_params=False):
        eq, g, spec = self.params(x)
        y = self.apply(spec, eq, g, x.shape[-1])
        return (y, eq, g) if return_params else y

    @torch.no_grad()
    def oracle(self, deg, clean):
        """Best separable correction given the clean signal: per-band long-term level ratio (EQ) and
        per-frame broadband level ratio (gain). Used as a training target and as the upper bound of
        what an EQ curve plus a gain trajectory can repair.
        Returns eq [B, N_EQ], gain [B, T], and a mask of frames loud enough to define the gain."""
        D, C = self.stft(deg).abs().pow(2).mean(1), self.stft(clean).abs().pow(2).mean(1)  # [B, F, T]
        g = 10 * torch.log10((C.sum(1) + 1e-6) / (D.sum(1) + 1e-6))
        frame = 10 * torch.log10(C.sum(1) + 1e-6)
        act = frame > frame.amax(1, keepdim=True) - 40
        g = torch.where(act, g, torch.zeros_like(g)).clamp(-self.max_gain, self.max_gain)
        # EQ after the gain is accounted for, pooled onto the control points
        Dg = D * 10 ** (g[:, None, :] / 10)
        w = self.interp / self.interp.sum(0, keepdim=True).clamp_min(1e-6)   # [F, N_EQ], columns sum to 1
        num = torch.einsum("bft,fk->bk", C * act[:, None, :], w)
        den = torch.einsum("bft,fk->bk", Dg * act[:, None, :], w)
        eq = (10 * torch.log10((num + 1e-6) / (den + 1e-6))).clamp(-self.max_eq, self.max_eq)
        shift = eq.mean(1, keepdim=True)
        return eq - shift, (g + shift).clamp(-self.max_gain, self.max_gain), act


class MagLoss(nn.Module):
    """Magnitude-only multi-resolution loss. Phase is left out on purpose: the correction is zero-phase
    while the damage was minimum-phase, so a phase-sensitive loss would punish a perfect EQ fix."""

    def __init__(self, ffts=(4096, 2048, 1024, 512)):
        super().__init__()
        self.ffts = ffts
        for n in ffts:
            self.register_buffer(f"w{n}", torch.hann_window(n), persistent=False)

    def forward(self, p, t):
        loss = 0.0
        for n in self.ffts:
            w = getattr(self, f"w{n}")
            P = torch.view_as_real(torch.stft(p.flatten(0, 1), n, n // 4, window=w, return_complex=True)).pow(2).sum(-1).add(1e-8).sqrt()
            T = torch.view_as_real(torch.stft(t.flatten(0, 1), n, n // 4, window=w, return_complex=True)).pow(2).sum(-1).add(1e-8).sqrt()
            loss = loss + (torch.log(P + 1e-3) - torch.log(T + 1e-3)).abs().mean() + (P - T).abs().mean()
        return loss / len(self.ffts)


def third_octave_db(x):
    """[B, 2, N] -> [B, bands] long-term third-octave levels in dB (relative), for the tonal metric."""
    w = torch.hann_window(8192, device=x.device)
    p = torch.stft(x.mean(1), 8192, 4096, window=w, return_complex=True).abs().pow(2).mean(-1)
    f = torch.fft.rfftfreq(8192, 1 / SR).to(x.device)
    out = []
    for fc in BAND_CENTERS[(BAND_CENTERS >= 40) & (BAND_CENTERS <= 12000)]:
        m = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        out.append(10 * torch.log10(p[:, m].sum(1) + 1e-12))
    return torch.stack(out, 1)


def tonal_dist(a, b):
    d = third_octave_db(a) - third_octave_db(b)
    return (d - d.mean(1, keepdim=True)).pow(2).mean(1).sqrt()


def env_dist(a, b, hop=2205):
    """RMS difference of 50 ms level envelopes in dB (after removing the mean offset): dynamics error."""
    ea = 10 * torch.log10(F.avg_pool1d(a.mean(1, keepdim=True).pow(2), hop).squeeze(1) + 1e-7)
    eb = 10 * torch.log10(F.avg_pool1d(b.mean(1, keepdim=True).pow(2), hop).squeeze(1) + 1e-7)
    act = eb > eb.amax(1, keepdim=True) - 35
    d = (ea - eb)
    d = d - (d * act).sum(1, keepdim=True) / act.sum(1, keepdim=True).clamp_min(1)
    return ((d.pow(2) * act).sum(1) / act.sum(1).clamp_min(1)).sqrt()


@torch.no_grad()
def evaluate(model, fixed, dev, batch=8):
    model.eval()
    rows = {}
    for i in range(0, len(fixed), batch):
        ch = fixed[i:i + batch]
        deg = torch.stack([c["deg"] for c in ch]).to(dev); clean = torch.stack([c["clean"] for c in ch]).to(dev)
        out = model(deg)
        oeq, og, _ = model.oracle(deg, clean)
        orc = model.apply(model.stft(deg), oeq, og, deg.shape[-1])
        m = dict(tonal_in=tonal_dist(deg, clean), tonal_out=tonal_dist(out, clean), tonal_oracle=tonal_dist(orc, clean),
                 env_in=env_dist(deg, clean), env_out=env_dist(out, clean), env_oracle=env_dist(orc, clean),
                 lsd_in=log_spec_dist(deg, clean), lsd_out=log_spec_dist(out, clean), lsd_oracle=log_spec_dist(orc, clean),
                 sisdr_in=si_sdr(deg, clean), sisdr_out=si_sdr(out, clean))
        for j, c in enumerate(ch):
            r = rows.setdefault(c["cond"], {})
            for k, v in m.items():
                r.setdefault(k, []).append(v[j].item())
    model.train()
    return {c: {k: float(np.mean(v)) for k, v in r.items()} for c, r in rows.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--data-weights", nargs="+", type=float, default=None)
    p.add_argument("--rir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seg", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--resume", default=None)
    p.add_argument("--w-param", type=float, default=0.05)
    p.add_argument("--param-loss", default="huber", choices=["huber", "l1"])
    p.add_argument("--fast-gain", action="store_true", help="gain head at the full 86 fps frame rate")
    p.add_argument("--p-identity", type=float, default=0.15)
    p.add_argument("--curriculum", action="store_true",
                   help="single effect -> pairs -> full mix, advancing on a validation plateau (the thesis schedule)")
    p.add_argument("--min-stage-steps", type=int, default=4000)
    p.add_argument("--max-stage-steps", type=int, default=12000)
    p.add_argument("--task", default="tone", help='"tone" or a single-effect diagnostic like "tone:ride"')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    groups = [[f for f in list_tracks([d]) if split_of(f) == "train"] for d in a.data]
    val_files = [f for f in list_tracks(a.data[:1]) if split_of(f) == "val"]
    fixed = build_fixed_set(val_files, a.rir, n_tracks=32, seg_s=8.0, task="tone")
    model = Controller(dim=a.dim, depth=a.depth, fast_gain=a.fast_gain).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)
    print("params %.2fM, val items %d" % (sum(q.numel() for q in model.parameters()) / 1e6, len(fixed)), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    step, best = 0, 1e9
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev, weights_only=True)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step, best = ck["step"], ck["best"]
    crit = MagLoss().to(dev)
    # Curriculum, ported from the thesis trainer (Curriculum_Tokenize_Master/token_train.py): stages get
    # harder, a stage ends when validation stops improving (after a minimum stay), and the LR steps down.
    stages = [("tone:s1", 1.0), ("tone:s2", 0.6), ("tone", 0.4)] if a.curriculum else [(a.task, 1.0)]
    stage_i, stage_start, hist, lr_scale = 0, 0, [], [1.0]
    if a.resume and os.path.exists(a.resume):
        stage_i, stage_start, hist = ck.get("stage_i", 0), ck.get("stage_start", 0), ck.get("hist", [])
    lr_scale[0] = stages[stage_i][1]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: lr_scale[0] * min(1.0, (s_ + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s_ / a.steps)))), last_epoch=step - 1 if step else -1)

    def loader(task):
        ds = PairDataset([f for g in groups for f in g], a.rir, seg_s=a.seg, ctx_s=0.5, task=task, groups=groups, weights=a.data_weights, p_identity=a.p_identity)
        return DataLoader(ds, batch_size=a.batch, num_workers=a.workers, drop_last=True, persistent_workers=True, prefetch_factor=4)

    dl = loader(stages[stage_i][0])
    print(f"=== stage {stage_i}: {stages[stage_i][0]} (lr x{lr_scale[0]})", flush=True)
    log = open(os.path.join(a.out, "train.jsonl"), "a")
    t0, acc = time.time(), 0.0
    while step < a.steps:
        advance = False
        for deg, clean in dl:
            deg, clean = deg.to(dev), clean.to(dev)
            out, eq, g = model(deg, return_params=True)
            oeq, og, act = model.oracle(deg, clean)
            l_audio = crit(out, clean)
            # direct supervision of the knobs: the audio loss alone gave no usable gradient across tracks
            if a.param_loss == "l1":
                l_param = (eq - oeq).abs().mean() + ((g - og).abs() * act).sum() / act.sum().clamp_min(1)
            else:
                # Huber, quadratic within 6 dB. With L1 the many frames that need no correction pull the output
                # to zero with constant force and the network never leaves "do nothing" (E12).
                l_param = F.huber_loss(eq, oeq, delta=6.0) + (F.huber_loss(g, og, delta=6.0, reduction="none") * act).sum() / act.sum().clamp_min(1)
            loss = l_audio + a.w_param * l_param
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                continue
            opt.step(); sched.step(); step += 1; acc += loss.item()
            with torch.no_grad():
                d = min(0.999, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
            if step % 100 == 0:
                row = dict(step=step, loss=acc / 100, lr=sched.get_last_lr()[0], it_s=100 / (time.time() - t0))
                log.write(json.dumps(row) + "\n"); log.flush(); print(row, flush=True); t0, acc = time.time(), 0.0
            if step % 500 == 0:
                torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, best=best, cfg=model.cfg,
                                stage_i=stage_i, stage_start=stage_start, hist=hist), os.path.join(a.out, "last.pt"))
            if step % a.val_every == 0 or step == a.steps:
                res = evaluate(ema, fixed, dev)
                score = float(np.mean([r["lsd_out"] for r in res.values()]))
                open(os.path.join(a.out, "val.jsonl"), "a").write(json.dumps(dict(step=step, score=score, conds=res)) + "\n")
                print(f"[val {step}] mean LSD {score:.3f} | " + " ".join(f"{c}: tonal {r['tonal_in']:.2f}->{r['tonal_out']:.2f} (oracle {r['tonal_oracle']:.2f}) env {r['env_in']:.2f}->{r['env_out']:.2f} ({r['env_oracle']:.2f}) lsd {r['lsd_in']:.2f}->{r['lsd_out']:.2f} ({r['lsd_oracle']:.2f})" for c, r in res.items()), flush=True)
                if score < best:
                    best = score
                    torch.save(dict(ema=ema.state_dict(), cfg=model.cfg, step=step, val=res), os.path.join(a.out, "best.pt"))
                hist.append(score)
                if stage_i < len(stages) - 1:
                    stayed = step - stage_start
                    # plateau: the last three validations did not beat the earlier best in this stage by 1%
                    plateau = len(hist) >= 5 and min(hist[-3:]) > 0.99 * min(hist[:-3])
                    advance = stayed >= a.max_stage_steps or (stayed >= a.min_stage_steps and plateau)
                t0 = time.time()
            if step >= a.steps or advance:
                break
        if advance:
            stage_i, stage_start, hist = stage_i + 1, step, []
            lr_scale[0] = stages[stage_i][1]
            del dl
            dl = loader(stages[stage_i][0])
            print(f"=== stage {stage_i}: {stages[stage_i][0]} (lr x{lr_scale[0]}) at step {step}", flush=True)


if __name__ == "__main__":
    main()


@torch.no_grad()
def ride_with_controller(model, x, chunk_s=8.0, hop_s=4.0, rate=20):
    """Full-track level riding with the learned gain trajectory.

    The controller was trained on 8 s windows, so the track is covered with overlapping windows and
    the per-window trajectories are cross-faded. Only the gain output is used: it is a plain time-domain
    multiplication, so nothing passes through an STFT. Returns (audio, report) shaped like vu.ride_gain.
    """
    import numpy as np
    from .degrade import rms_db
    from .vu import vu_trace
    dev = next(model.parameters()).device
    n = x.shape[1]
    chunk, hop = int(chunk_s * SR), int(hop_s * SR)
    # the model expects roughly -20 dB RMS input; use the loud half of the track as the reference
    lv = sorted(rms_db(x[:, i:i + SR]) for i in range(0, max(1, n - SR + 1), SR))
    ref = float(np.mean(lv[len(lv) // 2:]))
    xt = torch.from_numpy(x * 10 ** ((-20.0 - ref) / 20))
    starts = list(range(0, max(1, n - chunk // 2), hop))
    acc, wsum = np.zeros(n), np.zeros(n)
    for s0 in starts:
        seg = xt[:, s0:s0 + chunk]
        m = seg.shape[1]
        if m < SR:
            continue
        if m < chunk:
            seg = torch.nn.functional.pad(seg, (0, chunk - m))
        _, g, _ = model.params(seg[None].to(dev))
        g = g[0].float().cpu().numpy()
        gs = np.interp(np.arange(m), (np.arange(len(g)) + 0.5) * model.hop, g)
        # each window's trajectory is only defined up to a constant (windows are level-normalized on their
        # own), so shift it to agree with what is already stitched where they overlap
        seen = wsum[s0:s0 + m] > 0
        if seen.any():
            gs += np.mean(acc[s0:s0 + m][seen] / wsum[s0:s0 + m][seen] - gs[seen])
        w = np.minimum(np.arange(1, m + 1), np.arange(m, 0, -1)).astype(np.float64)
        acc[s0:s0 + m] += gs * w
        wsum[s0:s0 + m] += w
    gain = acc / np.maximum(wsum, 1e-9)
    gain -= np.median(gain)
    y = (x * 10 ** (gain / 20)).astype(np.float32)
    step = SR // rate
    return y, dict(vu_before=np.round(vu_trace(x, SR, rate), 2).tolist(), vu_after=np.round(vu_trace(y, SR, rate), 2).tolist(),
                   gain_db=np.round(gain[step // 2::step], 2).tolist(), rate=rate, max_ride_db=float(np.abs(gain).max()), method="controller")
