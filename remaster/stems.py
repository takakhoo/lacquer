"""Split a full mix into stems so each can be restored with its own policy.

Backends: "demucs" (local, HT-Demucs from Meta) and "elevenlabs" (remote API, opt-in).
Separators do not sum back to the mix exactly, so the leftover (mix minus the sum of stems) is
returned as "residual" and added back at remix time. That keeps an untouched remix bit-close to the input.
"""
from __future__ import annotations

import numpy as np
import torch

SR = 44100
STEMS = ("vocals", "drums", "bass", "other")
_CACHE = {}


def _demucs(device):
    if "demucs" not in _CACHE:
        from demucs.pretrained import get_model
        _CACHE["demucs"] = get_model("htdemucs").to(device).eval()
    return _CACHE["demucs"]


@torch.no_grad()
def separate(x, backend="demucs", device=None):
    """x: float32 [2, N] at 44.1 kHz -> dict of stems (same shape) plus "residual"."""
    if backend == "elevenlabs":
        from .elevenlabs_backend import separate_elevenlabs
        stems = separate_elevenlabs(x)
    else:
        from demucs.apply import apply_model
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model = _demucs(device)
        wav = torch.from_numpy(x)
        ref = wav.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        out = apply_model(model, ((wav - mu) / sd)[None], device=device, shifts=0, split=True, overlap=0.25, progress=False)[0]
        out = (out * sd + mu).cpu().numpy()
        stems = {name: out[i].astype(np.float32) for i, name in enumerate(model.sources)}
    stems["residual"] = (x - sum(stems[k] for k in stems)).astype(np.float32)
    return stems


def remix(stems):
    return sum(stems.values()).astype(np.float32)
