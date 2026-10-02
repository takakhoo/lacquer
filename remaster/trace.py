"""Run one damaged clip through every stage and save what is inside, for the figures.

  python -m remaster.trace --input data/examples/023041.mp3

Writes docs/evidence/trace.npz: waveforms, spectrograms, the token grid after selected layers, attention
maps, the predicted mask, what was removed against what was added, the echo cepstrum, the controller's
outputs, mastering curves, and EnCodec tokens and latents for the same clip.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .app import read_any
from .deecho import cepstrum, deecho, find_echo
from .degrade import SR, RIRBank, apply_echo, apply_reverb, apply_ride, match_level
from .infer import load_model

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "docs", "evidence", "trace.npz")


def pca_rgb(tok, ref=None):
    """tok [T, K, D] -> [K, T, 3] in 0..1 from the top three principal components."""
    t, k, d = tok.shape
    x = tok.reshape(-1, d).astype(np.float64)
    x = x - x.mean(0)
    if ref is None:
        _, _, vt = np.linalg.svd(x[:: max(1, len(x) // 4000)], full_matrices=False)
        ref = vt[:3]
    p = x @ ref.T
    lo, hi = np.percentile(p, 1, axis=0), np.percentile(p, 99, axis=0)
    return np.clip((p - lo) / (hi - lo + 1e-9), 0, 1).reshape(t, k, 3).transpose(1, 0, 2).astype(np.float32), ref


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--start", type=float, default=6.0)
    p.add_argument("--rir", default="data/raw/mit_ir")
    a = p.parse_args()
    ck = os.path.join(HERE, "checkpoints")
    model = load_model(os.path.join(ck, "best.pt"), device="cpu")
    net = model.net
    rng = np.random.default_rng(11)
    full = read_any(a.input)
    n = 4 * SR
    s0 = int(a.start * SR)
    ctx = full[:, s0 - 2 * SR: s0 + n]                       # 2 s of lead-in so tails land inside the window
    rev, rlog = apply_reverb(ctx, rng, RIRBank(a.rir), drr_db=1.0, p_real=1.0)
    d = int(0.21 * SR)
    deg_ctx = rev.copy(); deg_ctx[:, d:] += 0.45 * rev[:, :-d]   # one clean echo, 210 ms, so it is easy to find in the maps
    g = 10 ** ((-20.0 - 10 * np.log10(np.mean(deg_ctx[:, -n:] ** 2))) / 20)
    clean, deg = (ctx[:, -n:] * g).astype(np.float32), (deg_ctx[:, -n:] * g).astype(np.float32)
    out = {"clean": clean, "deg": deg, "sr": SR, "echo_delay_s": 0.21, "echo_gain": 0.45, "drr_db": rlog["drr_db"]}

    # ---- hooks: tokens after the band split and after each layer, and q/k of chosen attention modules
    grabs = {}
    net.band_split.register_forward_hook(lambda m, i, o: grabs.__setitem__("split", o[0].numpy()))
    for li, layer in enumerate(net.layers):
        layer[1].register_forward_hook(lambda m, i, o, li=li: grabs.__setitem__(f"layer{li}", (o[0] if isinstance(o, tuple) else o).numpy()))
    att_layers = (0, 5, 11)
    def qk_hook(key):
        def f(mod, args):
            q, k = args[0], args[1]
            grabs[key] = (q.numpy(), k.numpy())
        return f
    for li in att_layers:
        for wi, which in enumerate(("time", "band")):
            for blk in net.layers[li][wi].layers:
                blk[0].attend.register_forward_pre_hook(qk_hook(f"{which}{li}"))
    x = torch.from_numpy(deg)[None]
    y = model(x)[0].numpy()
    out["out"] = y.astype(np.float32)
    win = torch.hann_window(2048)
    S = lambda w: torch.stft(torch.from_numpy(w), 2048, 512, window=win, return_complex=True).numpy()
    X, Y, C = S(deg), S(y), S(clean)
    out["X"], out["Y"], out["C"] = X.astype(np.complex64), Y.astype(np.complex64), C.astype(np.complex64)
    out["mask"] = (Y / (X + 1e-6 * np.abs(X).max())).astype(np.complex64)
    T, K = grabs["split"].shape[0], grabs["split"].shape[1]
    out["tok_split_norm"] = np.linalg.norm(grabs["split"], axis=-1).T.astype(np.float32)      # [K, T]
    rgb, ref = pca_rgb(grabs["layer11"].reshape(T, K, -1))
    for li in (0, 2, 5, 8, 11):
        tok = grabs[f"layer{li}"].reshape(T, K, -1)
        out[f"tok_rgb{li}"], _ = pca_rgb(tok)
        out[f"tok_norm{li}"] = np.linalg.norm(tok, axis=-1).T.astype(np.float32)
    out["tok_rgb_split"], _ = pca_rgb(grabs["split"])
    out["tok_slice"] = grabs["layer11"].reshape(T, K, -1)[:, :, :48].astype(np.float16)          # a corner of the real tensor
    # attention maps: softmax(q k^T / sqrt(d)), averaged over heads
    kb = 24                                                   # a band in the low mids
    tf = int(T * 0.62)
    for li in att_layers:
        q, k = grabs[f"time{li}"]                             # [K, heads, T, d]
        a_ = torch.softmax(torch.from_numpy(q[kb]) @ torch.from_numpy(k[kb]).transpose(-1, -2) * q.shape[-1] ** -0.5, dim=-1)
        out[f"att_time{li}"] = a_.mean(0).numpy().astype(np.float32)                              # [T, T]
        out[f"att_time_heads{li}"] = a_[:, tf].numpy().astype(np.float32)                        # [heads, T] for one query frame
        q, k = grabs[f"band{li}"]                             # [T, heads, K, d]
        b_ = torch.softmax(torch.from_numpy(q[tf]) @ torch.from_numpy(k[tf]).transpose(-1, -2) * q.shape[-1] ** -0.5, dim=-1)
        out[f"att_band{li}"] = b_.mean(0).numpy().astype(np.float32)                              # [K, K]
    out["att_band_index"], out["att_frame_index"] = kb, tf
    widths = np.array(model.cfg["model"]["freqs_per_bands"]); out["band_edges_bins"] = np.concatenate([[0], np.cumsum(widths)])

    # ---- echo stage on the longer clip
    long_clean = full[:, s0 - 2 * SR: s0 + 16 * SR]
    long_rev, _ = apply_reverb(long_clean, np.random.default_rng(11), RIRBank(a.rir), drr_db=1.0, p_real=1.0)
    long_deg = long_rev.copy(); long_deg[:, d:] += 0.45 * long_rev[:, :-d]
    cz = cepstrum(long_deg); cr = cepstrum(long_rev)
    fixed, found = deecho(long_deg)
    out["cep_deg"], out["cep_ref"], out["cep_fixed"] = cz[: int(0.7 * SR)].astype(np.float32), cr[: int(0.7 * SR)].astype(np.float32), cepstrum(fixed)[: int(0.7 * SR)].astype(np.float32)
    out["echo_found_ms"] = np.array([f["delay_ms"] for f in found]); out["echo_found_gain"] = np.array([f["gain"] for f in found])

    # ---- level controller on a clip with a level fault
    cpath = os.path.join(ck, "controller.pt")
    if os.path.exists(cpath):
        ctrl = load_model(cpath, device="cpu")
        c8 = match_level(full[:, s0: s0 + 8 * SR], -20.0)
        ride, _ = apply_ride(c8, np.random.default_rng(5))
        ride = match_level(ride, -20.0)
        eq, gt, spec = ctrl.params(torch.from_numpy(ride)[None])
        oeq, og, act = ctrl.oracle(torch.from_numpy(ride)[None], torch.from_numpy(c8)[None])
        out.update(ctrl_in=ride, ctrl_clean=c8, ctrl_gain=gt[0].numpy(), ctrl_gain_oracle=og[0].numpy(), ctrl_eq=eq[0].numpy(), ctrl_eq_oracle=oeq[0].numpy(), ctrl_active=act[0].numpy())
        mid = (spec[0, 0] + spec[0, 1]) / 2
        out["ctrl_mel"] = torch.log(ctrl.mel(mid.abs().pow(2)) + 1e-7).numpy().astype(np.float32)

    # ---- EnCodec view of the same clean clip
    try:
        import torchaudio
        from encodec import EncodecModel
        from .token_probe import encodec_features
        codec = EncodecModel.encodec_model_48khz().eval(); codec.set_target_bandwidth(24.0)
        w48 = torchaudio.functional.resample(torch.from_numpy(clean), SR, 48000)[None]
        tok, lat, _ = encodec_features(codec, w48)
        tok2, _, _ = encodec_features(codec, w48 * 10 ** (1 / 20))
        rt = codec.decode(codec.encode(w48))[0, :, : w48.shape[-1]]
        rt = torchaudio.functional.resample(rt, 48000, SR).numpy()[:, :n]
        out.update(enc_tokens=tok[0].numpy().astype(np.int16), enc_latents=lat[0].numpy().astype(np.float16), enc_tokens_plus1db=tok2[0].numpy().astype(np.int16), enc_roundtrip=rt.astype(np.float32))
    except Exception as e:  # the EnCodec figures are optional
        print("encodec part skipped:", e)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB;", "tokens", grabs["split"].shape, "echo found", found)


if __name__ == "__main__":
    main()
