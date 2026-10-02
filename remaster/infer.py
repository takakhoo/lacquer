"""Full-track restoration with chunked overlap-add, plus the end-to-end `enhance` entry point."""
from __future__ import annotations

import numpy as np
import torch

from .deecho import deecho
from .degrade import rms_db
from .master import master
from .vu import ride_gain
from .model import SR
from .pretrained import build

REF_DB = -20.0


def load_model(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build(ck["cfg"])
    model.load_state_dict(ck["ema"])
    return model.to(device).eval()


@torch.no_grad()
def restore(model, x, chunk_s=6.0, overlap_s=1.5, batch=4):
    """x: float32 [2, N] at 44.1 kHz. Returns restored audio at the input's level."""
    dev = next(model.parameters()).device
    n = x.shape[1]
    # level reference from the loud part of the track, so quiet intros do not inflate the gain
    frame = SR
    lv = [rms_db(x[:, i:i + frame]) for i in range(0, max(1, n - frame + 1), frame)]
    ref = np.mean(sorted(lv)[len(lv) // 2:])
    g = 10 ** ((REF_DB - ref) / 20)
    chunk, ov = int(chunk_s * SR), int(overlap_s * SR)
    hop = chunk - ov
    starts = list(range(0, max(1, n - ov), hop))
    xt = torch.from_numpy(x * g)
    pad = starts[-1] + chunk - n
    if pad > 0:
        xt = torch.nn.functional.pad(xt, (0, pad))
    out = torch.zeros_like(xt)
    wsum = torch.zeros(xt.shape[1])
    w = torch.ones(chunk)
    ramp = torch.linspace(0, 1, ov)
    w[:ov], w[-ov:] = ramp, ramp.flip(0)
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        seg = torch.stack([xt[:, s:s + chunk] for s in ss]).to(dev)
        y = model(seg).cpu()
        for s, yy in zip(ss, y):
            ww = w.clone()
            if s == 0:
                ww[:ov] = 1
            out[:, s:s + chunk] += yy * ww
            wsum[s:s + chunk] += ww
    out = (out / wsum.clamp_min(1e-6))[:, :n] / g
    return out.numpy()


def enhance(model, x, restore_strength=1.0, do_master=True, do_deecho=True, bypass_db=-24.0, **master_kw):
    """DSP de-echo -> neural restoration (reverb, clipping, noise) -> mastering finish."""
    report = {}
    y = x
    if do_deecho:
        y, report["echoes"] = deecho(y)
        x = y
    if model is not None and restore_strength > 0:
        r = restore(model, x)
        delta = float(rms_db(r - x) - rms_db(x))
        report["restore_delta_db"] = delta
        # Do no harm: the network perturbs clean audio at roughly -29 dB. A proposed change at or below
        # that level means it found nothing to fix, so the input passes through untouched.
        report["restore_bypassed"] = delta < bypass_db
        if not report["restore_bypassed"]:
            y = x + restore_strength * (r - x)
    if do_master:
        y, report["master"] = master(y, **master_kw)
    return y.astype(np.float32), report


def enhance_stems(models, x, backend="demucs", reverb_bias_db=0.0, allow_add=True, do_master=True, do_deecho=True,
                  excess_gate_db=-24.0, **master_kw):
    """Stem-aware pipeline: de-echo -> separate -> per-stem reverb decision -> remix -> master.

    models: dict with "mix" (full-mix restorer) and optionally "vocal" (vocal dereverb model).
    Vocals get an absolute wetness measurement and are steered into the normal range of produced
    vocals (reduced if too wet, a plate added if too dry). Instrument stems only lose reverb the
    full-mix model flags as excess. reverb_bias_db shifts every target: negative is drier.
    """
    from .reverb_meter import apply_decision, decide, load_norms, measure
    from .stems import STEMS, remix, separate

    report = {"mode": "stems", "backend": backend, "stems": {}}
    y = x
    if do_deecho:
        y, report["echoes"] = deecho(y)
    stems = separate(y, backend=backend)
    norms = load_norms()
    for name in STEMS:
        st = stems[name]
        if name == "vocals" and models.get("vocal") is not None and "vocals" in norms:
            dry, wet = measure(models["vocal"], st)
            nv = norms["vocals"]
            action, target = decide(wet, nv["low_db"] + reverb_bias_db, nv["high_db"] + reverb_bias_db,
                                    nv["low_target_db"] + reverb_bias_db, nv["high_target_db"] + reverb_bias_db)
            if action == "add" and not allow_add:
                action = "keep"
        else:
            dry, wet = measure(models["mix"], st)
            # the full-mix model only reports excess reverb, so the choice is remove it or leave it
            gate = excess_gate_db + reverb_bias_db
            action, target = ("skip", None) if wet is None else ("reduce", gate) if wet > gate else ("keep", wet)
        info = dict(wetness_db=wet, action=action, target_db=target)
        if action in ("reduce", "add"):
            stems[name], extra = apply_decision(st, dry, wet, action, target)
            info.update(extra)
        report["stems"][name] = info
    y = remix(stems)
    if do_master:
        y, report["master"] = master(y, **master_kw)
    return y.astype(np.float32), report


def enhance_auto(models, x, backend="demucs", use_stems=True, reverb_bias_db=0.0, allow_add=True, restore_strength=1.0,
                 ride=0.75, do_master=True, do_deecho=True, excess_gate_db=-24.0, **master_kw):
    """The full decision pipeline.

    1. Echo: cepstral detection, exact inverse if one is found.
    2. Room reverb: the full-mix model doubles as a meter for reverb in excess of a normal production.
       Above the gate it is removed from the whole mix; below it the network is bypassed.
       (On MUSDB test, whole-mix removal beats per-stem removal for room reverb: 8.8 vs 5.2 dB.)
    3. Vocal ambience: the vocal stem gets an absolute wetness reading and is steered into the normal
       range of produced vocals, down if too wet, up if too dry. Stems are only remixed when the
       vocal is actually changed, so separation artifacts never reach a track that needed nothing.
    4. Level: VU-style gain riding, then tonal balance, loudness and limiting.
    reverb_bias_db shifts the reverb targets (negative = drier, positive = wetter).
    """
    from .reverb_meter import apply_decision, decide, load_norms, measure_fast
    from .stems import remix, separate

    report = {"mode": "auto", "decisions": []}
    y = x
    if do_deecho:
        y, report["echoes"] = deecho(y)
        for e in report["echoes"]:
            report["decisions"].append(f"Echo at {e['delay_ms']:.0f} ms (gain {e['gain']:.2f}): removed.")
    mix_model, vocal_model = models.get("mix"), models.get("vocal")
    if mix_model is not None and restore_strength > 0:
        wet = measure_fast(mix_model, y)  # full-track pass only if something needs removing
        gate = excess_gate_db + reverb_bias_db
        report["room"] = dict(excess_db=wet, gate_db=gate)
        if wet is not None and wet > gate:
            # soft gate: borderline readings get a partial correction, clear ones the full one
            s = float(np.clip((wet - gate) / 8.0, 0.0, 1.0)) * restore_strength
            y = y + s * (restore(mix_model, y) - y)
            report["room"].update(action="reduce", strength=s)
            report["decisions"].append(f"Room reverb / noise / clipping: excess measured at {wet:.1f} dB (gate {gate:.0f}): removed at {100 * s:.0f}% strength.")
        else:
            report["room"]["action"] = "keep"
            report["decisions"].append("Room reverb: none in excess of a normal production" + ("" if wet is None else f" ({wet:.1f} dB)") + ": left alone.")
    norms = load_norms()
    if use_stems and vocal_model is not None and "vocals" in norms:
        stems = separate(y, backend=backend)
        v_wet = measure_fast(vocal_model, stems["vocals"])
        nv = norms["vocals"]
        action, target = decide(v_wet, nv["low_db"] + reverb_bias_db, nv["high_db"] + reverb_bias_db,
                                nv["low_target_db"] + reverb_bias_db, nv["high_target_db"] + reverb_bias_db)
        if action == "add" and not allow_add:
            action = "keep"
        if v_wet is not None and v_wet > nv.get("max_credible_db", 8.0):
            action = "unsure"  # the meter removed almost everything: this stem is not a voice it understands
        report["vocal"] = dict(wetness_db=v_wet, action=action, target_db=target, backend=backend,
                               normal_range_db=[nv["low_db"] + reverb_bias_db, nv["high_db"] + reverb_bias_db])
        if action in ("reduce", "add"):
            stems["vocals"], extra = apply_decision(stems["vocals"], restore(vocal_model, stems["vocals"]), v_wet, action, target)
            report["vocal"].update(extra)
            y = remix(stems)
            verb = "reduced" if action == "reduce" else "ambience added"
            report["decisions"].append(f"Vocal reverb: {v_wet:.1f} dB, outside the normal range: {verb} to {target:.1f} dB.")
        elif action == "keep":
            report["decisions"].append(f"Vocal reverb: {v_wet:.1f} dB, inside the normal range: left alone.")
        elif action == "unsure":
            report["decisions"].append(f"Vocal reverb: reading of {v_wet:.1f} dB is not credible for a voice: left alone.")
        else:
            report["decisions"].append("Vocal: no active vocal found.")
    if ride > 0:
        if models.get("controller") is not None:
            # learned gain trajectory: fixes more than the VU rider at the same disturbance to clean audio (E14)
            from .controller import ride_with_controller
            yr, report["vu"] = ride_with_controller(models["controller"], y)
            y = y + min(1.0, ride / 0.75) * (yr - y)
        else:
            y, report["vu"] = ride_gain(y, ratio=ride)
        how = "learned controller" if report["vu"].get("method") == "controller" else "VU rider"
        report["decisions"].append(f"Level ({how}): rode the gain by up to {report['vu']['max_ride_db']:.1f} dB.")
    if do_master:
        y, report["master"] = master(y, **master_kw)
    return y.astype(np.float32), report
