"""Wrap MSST RoFormer models (vendored in third_party/msst) so they train and infer like Restorer."""
from __future__ import annotations

import torch
import torch.nn as nn
import yaml


class MSSTRestorer(nn.Module):
    def __init__(self, kind, model_cfg):
        super().__init__()
        if kind == "msst_bs":
            from .third_party.msst.bs_roformer import BSRoformer as M
        elif kind == "msst_mel":
            from .third_party.msst.mel_band_roformer import MelBandRoformer as M
        else:
            raise ValueError(kind)
        model_cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in model_cfg.items()}
        self.cfg = dict(arch=kind, model=model_cfg)
        self.net = M(**model_cfg)
        self.amp = True

    def forward(self, x):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=self.amp and x.device.type == "cuda"):
            y = self.net(x)
        y = y.float()
        return y[:, 0] if y.ndim == 4 else y


def from_msst(kind, config_path, ckpt_path=None):
    """Build from an MSST yaml config and optionally load its released checkpoint."""
    cfg = yaml.load(open(config_path), Loader=yaml.FullLoader)["model"]
    m = MSSTRestorer(kind, dict(cfg))
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = sd.get("state", sd.get("state_dict", sd))
        print("loaded pretrained:", m.net.load_state_dict(sd, strict=True))
    return m


def build(cfg):
    """Rebuild any supported architecture from a checkpoint's cfg dict."""
    if cfg.get("arch", "restorer") == "restorer":
        from .model import Restorer
        return Restorer(**{k: v for k, v in cfg.items() if k != "arch"})
    if cfg["arch"] == "unet":
        from .unet import UNet
        return UNet(**{k: v for k, v in cfg.items() if k != "arch"})
    if cfg["arch"] == "controller":
        from .controller import Controller
        return Controller(**{k: v for k, v in cfg.items() if k != "arch"})
    return MSSTRestorer(cfg["arch"], cfg["model"])
