"""Train the restorer. Example:
  python -m remaster.train --data data/raw/fma_medium --rir data/raw/mit_ir --out remaster/runs/base
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .data import PairDataset, build_fixed_set, list_tracks, split_of
from .losses import MultiResSTFT, log_spec_dist, si_sdr
from .model import Restorer, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--data-weights", nargs="+", type=float, default=None, help="sampling probability per --data root")
    p.add_argument("--p-identity", type=float, default=0.08)
    p.add_argument("--p-algo", type=float, default=0.0, help="share of non-real reverb tails drawn from the algorithmic (plug-in style) reverb")
    p.add_argument("--rir", default=None)
    p.add_argument("--train-rir", nargs="*", default=[], help="extra impulse-response roots used for training only; validation keeps --rir")
    p.add_argument("--out", required=True)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--no-level-feats", action="store_true")
    p.add_argument("--task", default="joint", choices=["joint", "artifact"])
    p.add_argument("--arch", default="restorer", choices=["restorer", "msst_bs", "msst_mel", "unet"])
    p.add_argument("--cbam", action="store_true")
    p.add_argument("--film", action="store_true")
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--unet-depth", type=int, default=4)
    p.add_argument("--msst-config", default=None)
    p.add_argument("--init", default=None, help="released MSST checkpoint to fine-tune from")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--stop-at", type=int, default=None, help="end the run at this step without changing the LR schedule (--steps)")
    p.add_argument("--lr-scale", type=float, default=1.0, help="multiply the scheduled LR, for branching a run with a different rate")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seg", type=float, default=4.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--val-every", type=int, default=2500)
    p.add_argument("--val-tracks", type=int, default=32)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--w-wave", type=float, default=10.0)
    p.add_argument("--w-complex", type=float, default=1.0)
    p.add_argument("--w-log", type=float, default=0.5)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, fixed, device, batch=8):
    model.eval()
    rows = {c: dict(n=0, sisdr_in=0.0, sisdr_out=0.0, lsd_in=0.0, lsd_out=0.0) for c in dict.fromkeys(f["cond"] for f in fixed)}
    for i in range(0, len(fixed), batch):
        chunk = fixed[i:i + batch]
        deg = torch.stack([c["deg"] for c in chunk]).to(device)
        clean = torch.stack([c["clean"] for c in chunk]).to(device)
        out = model(deg)
        m = dict(sisdr_in=si_sdr(deg, clean), sisdr_out=si_sdr(out, clean), lsd_in=log_spec_dist(deg, clean), lsd_out=log_spec_dist(out, clean))
        for j, c in enumerate(chunk):
            r = rows[c["cond"]]
            r["n"] += 1
            for k, v in m.items():
                r[k] += v[j].item()
    model.train()
    return {c: {k: (v / r["n"] if k != "n" else v) for k, v in r.items()} for c, r in rows.items() if r["n"]}


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    groups = [[f for f in list_tracks([d]) if split_of(f) == "train"] for d in a.data]
    files = list_tracks(a.data[:1])  # validation always comes from the first root so runs stay comparable
    train_files = [f for g in groups for f in g]
    val_files = [f for f in files if split_of(f) == "val"]
    print("train files per root:", [len(g) for g in groups], flush=True)
    print(f"tracks: {len(files)} total, {len(train_files)} train, {len(val_files)} val", flush=True)
    fixed = build_fixed_set(val_files, a.rir, n_tracks=a.val_tracks, task=a.task)
    print(f"fixed val set: {len(fixed)} items", flush=True)

    if a.arch == "restorer":
        model = Restorer(dim=a.dim, depth=a.depth, heads=a.heads, residual=a.residual, level_feats=not a.no_level_feats).to(dev)
    elif a.arch == "unet":
        from .unet import UNet
        model = UNet(base=a.base, depth=a.unet_depth, cbam=a.cbam, film=a.film).to(dev)
    else:
        from .pretrained import from_msst
        model = from_msst(a.arch, a.msst_config, a.init).to(dev)
    if a.eval_only:
        res = evaluate(model, fixed, dev, batch=4)
        print("[zero-shot] " + " ".join(f"{c}:{r['sisdr_in']:.1f}->{r['sisdr_out']:.1f} (lsd {r['lsd_in']:.1f}->{r['lsd_out']:.1f})" for c, r in res.items()), flush=True)
        json.dump(res, open(os.path.join(a.out, "zero_shot.json"), "w"), indent=1)
        return
    ema = copy.deepcopy(model).requires_grad_(False)
    print(f"params: {count_params(model) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: a.lr_scale * min(1.0, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    # the MSST models autocast to float16 (their code path needs it), which requires loss scaling
    scaler = torch.amp.GradScaler("cuda", enabled=a.arch.startswith("msst") and dev.type == "cuda")
    mrstft = MultiResSTFT(w_complex=a.w_complex, w_log=a.w_log).to(dev)
    step = 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev, weights_only=True)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best_resume = ck["step"], ck.get("best", -1e9)
        print(f"resumed from step {step}", flush=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    ds = PairDataset(train_files, [a.rir] + a.train_rir if a.train_rir else a.rir, seg_s=a.seg, seed=a.seed, task=a.task, groups=groups, weights=a.data_weights, p_identity=a.p_identity, p_algo=a.p_algo)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    log = open(os.path.join(a.out, "train.jsonl"), "a")
    t0, acc, best = time.time(), {}, (best_resume if a.resume and os.path.exists(a.resume) else -1e9)
    model.train()
    end = min(a.steps, a.stop_at or a.steps)
    while step < end:
        for deg, clean in dl:
            deg, clean = deg.to(dev, non_blocking=True), clean.to(dev, non_blocking=True)
            out = model(deg)
            l_wave = (out - clean).abs().mean()
            l_c, l_l = mrstft(out, clean)
            loss = a.w_wave * l_wave + l_c + l_l
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                print("non-finite gradient, skipping batch", flush=True)
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(opt); scaler.update(); sched.step()
            with torch.no_grad():
                d = min(a.ema, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
            step += 1
            for k, v in dict(loss=loss, wave=l_wave, cplx=l_c, logmag=l_l, gn=gn).items():
                acc[k] = acc.get(k, 0.0) + float(v)
            if step % 100 == 0:
                row = {k: v / 100 for k, v in acc.items()}
                row.update(step=step, lr=sched.get_last_lr()[0], it_s=100 / (time.time() - t0))
                log.write(json.dumps(row) + "\n"); log.flush()
                print(" ".join(f"{k}={v}" if k == "step" else f"{k}={v:.4g}" for k, v in row.items()), flush=True)
                t0, acc = time.time(), {}
            if step % a.ckpt_every == 0:
                torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, cfg=model.cfg, best=best), os.path.join(a.out, "last.pt.tmp"))
                os.replace(os.path.join(a.out, "last.pt.tmp"), os.path.join(a.out, "last.pt"))
            if step % a.val_every == 0 or step == end:
                res = evaluate(ema, fixed, dev)
                gain = sum(r["sisdr_out"] - r["sisdr_in"] for c, r in res.items() if c != "identity") / (len(res) - 1)
                with open(os.path.join(a.out, "val.jsonl"), "a") as f:
                    f.write(json.dumps(dict(step=step, mean_sisdr_gain=gain, conds=res)) + "\n")
                print(f"[val {step}] mean SI-SDR gain {gain:+.2f} dB | " + " ".join(f"{c}:{r['sisdr_in']:.1f}->{r['sisdr_out']:.1f} (lsd {r['lsd_in']:.1f}->{r['lsd_out']:.1f})" for c, r in res.items()), flush=True)
                if gain > best:
                    best = gain
                    torch.save(dict(ema=ema.state_dict(), cfg=model.cfg, step=step, val=res, task=a.task), os.path.join(a.out, "best.pt"))
                t0 = time.time()
            if step >= end:
                break


if __name__ == "__main__":
    main()
