"""Embed the wav sets written by `remaster.evaluate_fad make` with CLAP (one vector per clip) and VGGish (one
vector per 0.96 s). Runs in an environment that has laion_clap and torch.hub access; writes <set>.<name>.npy."""
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

root = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
sets = [s for s in ("reference", "clean", "damaged", "restored") if os.path.isdir(os.path.join(root, s))]


def load(f, sr):
    x, fs = sf.read(f, dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    return torchaudio.functional.resample(x, fs, sr)


import laion_clap
clap = laion_clap.CLAP_Module(enable_fusion=False).to(dev)       # default HTSAT-tiny audio tower with the 630k-audioset checkpoint
clap.load_ckpt(os.environ.get("CLAP_CKPT") or None) if os.environ.get("CLAP_CKPT") else clap.load_ckpt()
for s in sets:
    files = sorted(glob.glob(os.path.join(root, s, "*.wav")))
    out = []
    for i in range(0, len(files), 16):
        batch = torch.stack([load(f, 48000)[: 48000 * 10] for f in files[i:i + 16]])
        with torch.no_grad():
            out.append(clap.get_audio_embedding_from_data(x=batch.to(dev), use_tensor=True).cpu().numpy())
    np.save(os.path.join(root, f"{s}.clap.npy"), np.concatenate(out))
    print(s, "clap", len(files), flush=True)
del clap

vgg = torch.hub.load("harritaylor/torchvggish", "vggish")
vgg.postprocess = False
vgg.eval().to(dev)
vgg.device = dev
for s in sets:
    files = sorted(glob.glob(os.path.join(root, s, "*.wav")))
    out = []
    for f in files:
        with torch.no_grad():
            out.append(vgg.forward(load(f, 16000).numpy(), 16000).cpu().numpy())
    np.save(os.path.join(root, f"{s}.vggish.npy"), np.concatenate(out))
    print(s, "vggish", len(files), flush=True)
