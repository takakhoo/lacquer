"""Spectrogram U-Net baseline with the thesis building blocks (CBAM, FiLM) as switches.

Exists to answer one question with a measurement: on the same task, data and budget as the band-split
transformer, does a conv U-Net compete, and do CBAM and (unconditional) FiLM change anything?
Like the main model it predicts a complex STFT mask and starts as an identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 44100


class CBAM(nn.Module):
    """Channel attention then spatial attention (Woo et al. 2018), 2-D version of the thesis block."""

    def __init__(self, ch, reduction=8, kernel=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(ch, max(4, ch // reduction), 1), nn.ReLU(), nn.Conv2d(max(4, ch // reduction), ch, 1))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def forward(self, x):
        w = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        x = x * w
        s = torch.sigmoid(self.spatial(torch.cat([x.amax(1, keepdim=True), x.mean(1, keepdim=True)], 1)))
        return x * s


class FiLM(nn.Module):
    """The thesis FiLM: learned per-channel scale and shift with no conditioning input."""

    def __init__(self, ch):
        super().__init__()
        self.gamma, self.beta = nn.Parameter(torch.ones(1, ch, 1, 1)), nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta


class Block(nn.Module):
    def __init__(self, cin, cout, cbam, film):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU(),
                                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU())
        self.att = CBAM(cout) if cbam else nn.Identity()
        self.film = FiLM(cout) if film else nn.Identity()

    def forward(self, x):
        return self.proj(x) + self.film(self.att(self.body(x)))


class UNet(nn.Module):
    def __init__(self, base=32, depth=5, cbam=False, film=False, n_fft=2048, hop=512, channels=2):
        super().__init__()
        self.cfg = dict(arch="unet", base=base, depth=depth, cbam=cbam, film=film, n_fft=n_fft, hop=hop, channels=channels)
        self.n_fft, self.hop, self.depth, self.amp = n_fft, hop, depth, True
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        chs = [min(base * 2 ** i, 512) for i in range(depth + 1)]
        self.inp = nn.Conv2d(channels * 2 + channels, chs[0], 3, padding=1)
        self.enc = nn.ModuleList([Block(chs[i], chs[i], cbam, film) for i in range(depth)])
        self.down = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1) for i in range(depth)])
        self.mid = Block(chs[-1], chs[-1], cbam, film)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 4, stride=2, padding=1) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([Block(chs[i] * 2, chs[i], cbam, film) for i in reversed(range(depth))])
        self.out = nn.Conv2d(chs[0], channels * 2, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, n = x.shape
        spec = torch.stft(x.reshape(b * c, n), self.n_fft, self.hop, window=self.window, return_complex=True).view(b, c, self.n_fft // 2 + 1, -1)
        f, t = spec.shape[-2:]
        # power-compressed real/imag plus log magnitude, so level is visible to the convs
        mag = spec.abs().clamp_min(1e-8)
        comp = torch.view_as_real(spec * mag.pow(-0.7)).permute(0, 1, 4, 2, 3).reshape(b, c * 2, f, t)
        h = torch.cat([comp, torch.log10(mag + 1e-4) / 2], 1)[:, :, : f - 1]
        m = 2 ** self.depth
        h = F.pad(h, (0, (-t) % m))
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=self.amp and x.device.type == "cuda"):
            h = self.inp(h)
            skips = []
            for blk, dn in zip(self.enc, self.down):
                h = blk(h); skips.append(h); h = dn(h)
            h = self.mid(h)
            for up, blk in zip(self.up, self.dec):
                h = blk(torch.cat([up(h), skips.pop()], 1))
            h = self.out(h)
        h = F.pad(h.float()[..., :t], (0, 0, 0, 1))  # restore the dropped Nyquist bin
        mask = torch.view_as_complex(h.view(b, c, 2, f, t).permute(0, 1, 3, 4, 2).contiguous()) + 1.0
        return torch.istft((spec * mask).reshape(b * c, f, t), self.n_fft, self.hop, window=self.window, length=n).view(b, c, n)
