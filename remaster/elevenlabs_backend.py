"""Optional stem separation through the ElevenLabs API. UNTESTED against the live service.

Written from the official API reference (POST /v1/music/stem-separation, multipart `file`,
`stem_variation_id`, query `output_format`, response application/zip). It uploads the track to
ElevenLabs, so it only runs when the caller asks for backend="elevenlabs" and ELEVENLABS_API_KEY is set
in the environment. Music is not covered by their zero-retention mode.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
URL = "https://api.elevenlabs.io/v1/music/stem-separation"
# pcm_44100 needs the Pro plan and mp3_44100_192 needs Creator; fall back down the list on a 4xx
FORMATS = ("mp3_44100_192", "mp3_44100_128")


def available():
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def _align(stem, n, lag):
    out = np.zeros((2, n), dtype=np.float32)
    src = stem[:, max(lag, 0):]
    dst = max(-lag, 0)
    m = min(n - dst, src.shape[1])
    if m > 0:
        out[:, dst:dst + m] = src[:, :m]
    return out


def separate_elevenlabs(x):
    """x: float32 [2, N] at 44.1 kHz -> dict(vocals, drums, bass, other), time-aligned to x."""
    import requests

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    buf = io.BytesIO()
    sf.write(buf, x.T, SR, format="FLAC")
    last = None
    for fmt in FORMATS:
        buf.seek(0)
        r = requests.post(URL, headers={"xi-api-key": key}, params={"output_format": fmt},
                          data={"stem_variation_id": "six_stems_v1"}, files={"file": ("mix.flac", buf, "audio/flac")}, timeout=900)
        if r.ok:
            break
        last = f"{r.status_code}: {r.text[:300]}"
    else:
        raise RuntimeError(f"ElevenLabs stem separation failed ({last})")
    raw = {}
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            try:
                a, fs = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
            except Exception:
                continue
            a = a.T
            if a.shape[0] == 1:
                a = np.repeat(a, 2, axis=0)
            if fs != SR:
                a = signal.resample_poly(a, SR, fs, axis=1).astype(np.float32)
            raw[os.path.splitext(os.path.basename(name))[0].lower()] = a[:2]
    if not raw:
        raise RuntimeError("ElevenLabs returned no decodable stems")
    # MP3 adds encoder delay: find one lag for all stems by correlating their sum with the mix
    n = x.shape[1]
    total = sum(_align(a, n, 0) for a in raw.values()).mean(0)
    seg = slice(0, min(n, SR * 30))
    c = signal.correlate(total[seg], x.mean(0)[seg], mode="full", method="fft")
    lag = int(np.argmax(c) - (seg.stop - seg.start - 1))
    lag = lag if abs(lag) < SR // 4 else 0
    stems = {k: np.zeros((2, n), dtype=np.float32) for k in ("vocals", "drums", "bass", "other")}
    for name, a in raw.items():
        key_ = next((k for k in ("vocals", "drums", "bass") if k in name), "other")  # guitar, piano, other -> other
        stems[key_] += _align(a, n, lag)
    return stems
