"""Local web app: upload a track, hear and see the before/after.

  python -m remaster.app --ckpt remaster/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tempfile
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy import signal

from .degrade import RIRBank, degrade
from .infer import enhance_auto, load_model
from . import elevenlabs_backend
from .master import loudness, true_peak_db
from .mastering import PROFILES
from .mastering_norms import genres as mastering_genres
from .model import SR

HERE = os.path.dirname(__file__)
WORK = os.path.join(tempfile.gettempdir(), "remaster_sessions")
os.makedirs(WORK, exist_ok=True)
app = FastAPI()
STATE = dict(model=None, vocal=None, controller=None, bank=None, ckpt=None, examples={})


def read_any(path):
    """Decode anything ffmpeg understands to float32 [2, N] at 44.1 kHz."""
    try:
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        x = x.T
        if fs != SR:
            x = signal.resample_poly(x, SR, fs, axis=1).astype(np.float32)
    except Exception:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
        x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    return np.ascontiguousarray(x[:2])


def spectrogram_png(x, path, vmin=-90, vmax=-10):
    """Axis-free log-frequency spectrogram, 20 Hz .. 20 kHz, so the UI can overlay a playhead.

    Drawn at a fixed RMS so before/after differ by content and not by overall loudness.
    """
    x = x * 10 ** ((-20.0 - 10 * np.log10(np.mean(x ** 2) + 1e-12)) / 20)
    f, t, s = signal.stft(x.mean(axis=0), SR, nperseg=4096, noverlap=3072)
    db = 20 * np.log10(np.abs(s) + 1e-9)
    grid = np.geomspace(20, 20000, 360)
    img = np.stack([np.interp(grid, f, db[:, i]) for i in range(db.shape[1])], axis=1)
    cols = min(img.shape[1], 1600)
    idx = np.linspace(0, img.shape[1] - 1, cols).astype(int)
    plt.imsave(path, img[::-1, idx], cmap="magma", vmin=vmin, vmax=vmax)


def stats(x):
    return dict(lufs=round(loudness(x), 2), true_peak_db=round(float(true_peak_db(x)), 2),
                crest_db=round(float(20 * np.log10(np.abs(x).max() / (np.sqrt(np.mean(x ** 2)) + 1e-12) + 1e-12)), 2))


@app.post("/api/process")
async def process(file: UploadFile = File(None), reference: UploadFile = File(None), example: str = Form(""), restore_strength: float = Form(1.0), do_master: bool = Form(True),
                  target_lufs: float = Form(-14.0), tonal_strength: float = Form(0.6), stress: str = Form("none"),
                  do_deecho: bool = Form(True), use_stems: bool = Form(True), backend: str = Form("demucs"),
                  reverb_bias_db: float = Form(0.0), ride: float = Form(0.75), allow_add: bool = Form(False), fix_balance: bool = Form(False),
                  genre: str = Form("all"), profile: str = Form("streaming")):
    sid = uuid.uuid4().hex[:12]
    d = os.path.join(WORK, sid)
    os.makedirs(d)
    if example in STATE["examples"]:
        src = STATE["examples"][example]
    elif file is not None:
        src = os.path.join(d, "upload" + os.path.splitext(file.filename or "")[1])
        with open(src, "wb") as f:
            f.write(await file.read())
    else:
        return JSONResponse(dict(error="no audio provided"), status_code=400)
    try:
        x = read_any(src)
    except Exception as e:
        return JSONResponse(dict(error=f"could not decode audio: {e}"), status_code=400)
    x = x[:, : SR * 600]
    stress_log = {}
    if stress != "none":
        rng = np.random.default_rng(abs(hash(sid)) % 2 ** 32)
        peak = np.abs(x).max()
        x, stress_log = degrade(x, rng, STATE["bank"], effects=stress.split("+"))
        x = x * min(1.0, 0.98 * peak / (np.abs(x).max() + 1e-9)) if peak > 0 else x
    ref = None
    if reference is not None and reference.filename:
        rp = os.path.join(d, "reference" + os.path.splitext(reference.filename)[1])
        with open(rp, "wb") as f:
            f.write(await reference.read())
        try:
            ref = read_any(rp)[:, : SR * 600]
        except Exception as e:
            return JSONResponse(dict(error=f"could not decode reference: {e}"), status_code=400)
    if backend == "elevenlabs" and not elevenlabs_backend.available():
        return JSONResponse(dict(error="ELEVENLABS_API_KEY is not set in the environment of this server"), status_code=400)
    try:
        y, report = enhance_auto(dict(mix=STATE["model"], vocal=STATE["vocal"], controller=STATE["controller"]), x, reference=ref, restore_strength=restore_strength,
                                 do_master=do_master, do_deecho=do_deecho, use_stems=use_stems, backend=backend,
                                 reverb_bias_db=reverb_bias_db, ride=ride, allow_add=allow_add, stem_balance="fix" if fix_balance else "report",
                                 genre=genre, profile=profile, target_lufs=None if profile != "custom" else target_lufs, tonal_strength=tonal_strength)
    except Exception as e:
        return JSONResponse(dict(error=f"processing failed: {e}"), status_code=500)
    sf.write(os.path.join(d, "before.wav"), x.T, SR, subtype="PCM_24")
    sf.write(os.path.join(d, "after.wav"), y.T, SR, subtype="PCM_24")
    spectrogram_png(x, os.path.join(d, "before.png"))
    spectrogram_png(y, os.path.join(d, "after.png"))
    out = dict(id=sid, duration=x.shape[1] / SR, before=stats(x), after=stats(y), report=report, stress=stress_log,
               model=os.path.basename(STATE["ckpt"]) if STATE["ckpt"] else None)
    json.dump(out, open(os.path.join(d, "report.json"), "w"), default=float)
    return JSONResponse(json.loads(json.dumps(out, default=float)))


@app.get("/api/file/{sid}/{name}")
def get_file(sid: str, name: str):
    if not sid.isalnum() or name not in {"before.wav", "after.wav", "before.png", "after.png", "report.json"}:
        return JSONResponse(dict(error="not found"), status_code=404)
    return FileResponse(os.path.join(WORK, sid, name))


@app.get("/api/info")
def info():
    return dict(model=os.path.basename(STATE["ckpt"]) if STATE["ckpt"] else None, has_rirs=bool(STATE["bank"] and len(STATE["bank"])),
                examples=sorted(STATE["examples"]), vocal_model=STATE["vocal"] is not None, controller=STATE["controller"] is not None, elevenlabs=elevenlabs_backend.available(),
                genres=mastering_genres(), profiles={k: v["note"] for k, v in PROFILES.items()})


app.mount("/", StaticFiles(directory=os.path.join(HERE, "web"), html=True), name="web")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "best.pt"))
    p.add_argument("--vocal-ckpt", default=os.path.join(HERE, "checkpoints", "vocal_dereverb.pt"))
    p.add_argument("--controller-ckpt", default=os.path.join(HERE, "checkpoints", "controller.pt"))
    p.add_argument("--rir", default="data/raw/mit_ir")
    p.add_argument("--examples", default="data/examples", help="folder of demo tracks offered in the UI")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    if os.path.exists(a.ckpt):
        STATE["model"], STATE["ckpt"] = load_model(a.ckpt), a.ckpt
    else:
        print(f"no checkpoint at {a.ckpt}: running mastering-only")
    if os.path.exists(a.vocal_ckpt):
        STATE["vocal"] = load_model(a.vocal_ckpt)
    if os.path.exists(a.controller_ckpt):
        STATE["controller"] = load_model(a.controller_ckpt)
    if os.path.isdir(a.examples):
        STATE["examples"] = {f: os.path.join(a.examples, f) for f in sorted(os.listdir(a.examples)) if f.lower().endswith((".mp3", ".wav", ".flac"))}
    STATE["bank"] = RIRBank(a.rir if os.path.isdir(a.rir) else None)
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
