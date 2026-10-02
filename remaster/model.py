"""Band-split transformer that predicts a complex STFT mask for stereo 44.1 kHz music.

The network starts as an exact identity (mask = 1 + 0j), so clean input passes through
untouched and training only has to learn the correction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 44100


def band_edges_hz():
    e = list(range(0, 1000, 100)) + list(range(1000, 4000, 250)) + list(range(4000, 8000, 500))
    e += list(range(8000, 16000, 1000)) + list(range(16000, 20000, 2000)) + [20000, SR // 2]
    return e


def band_bins(n_fft, sr=SR):
    n_bins = n_fft // 2 + 1
    edges = [min(n_bins, int(round(h / (sr / 2) * (n_bins - 1)))) for h in band_edges_hz()]
    edges[-1] = n_bins
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.g * math.sqrt(x.shape[-1])


def rope(x, base=10000.0):
    """Rotary position embedding over the sequence axis of [B, H, N, D]."""
    n, d = x.shape[-2], x.shape[-1]
    freqs = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d))
    ang = torch.arange(n, device=x.device, dtype=x.dtype)[:, None] * freqs[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, dim, heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ff_mult, dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.n2(x))


class BandSplit(nn.Module):
    """Per-band projection of [unit-norm spectral shape, log band level per channel].

    Stock BS-RoFormer keeps only the normalized shape. Level is what EQ, compression and reverb
    decay change, so it is fed explicitly here.
    """

    def __init__(self, bands, dim, feat_per_bin, level_feats=True):
        super().__init__()
        self.bands, self.level_feats, self.ch = bands, level_feats, feat_per_bin // 2
        extra = self.ch if level_feats else 0
        self.norm = nn.ModuleList([RMSNorm((b - a) * feat_per_bin) for a, b in bands])
        self.proj = nn.ModuleList([nn.Linear((b - a) * feat_per_bin + extra, dim) for a, b in bands])

    def forward(self, x):  # [B, T, F, feat] with feat = channels * (re, im)
        out = []
        for norm, proj, (a, b) in zip(self.norm, self.proj, self.bands):
            xb = x[:, :, a:b]
            h = norm(xb.flatten(2))
            if self.level_feats:
                e = xb.unflatten(-1, (self.ch, 2)).pow(2).sum(-1).mean(2)  # [B, T, channels] mean bin power
                h = torch.cat([h, 0.5 * torch.log10(e + 1e-8) + 1.0], dim=-1)
            out.append(proj(h))
        return torch.stack(out, dim=2)


class MaskHead(nn.Module):
    def __init__(self, bands, dim, feat_per_bin, mult=4):
        super().__init__()
        self.feat = feat_per_bin
        self.heads = nn.ModuleList([
            nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim * mult), nn.Tanh(), nn.Linear(dim * mult, (b - a) * feat_per_bin))
            for a, b in bands])
        for h in self.heads:
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def forward(self, x):  # [B, T, K, D] -> [B, T, F, feat]
        return torch.cat([h(x[:, :, i]).unflatten(-1, (-1, self.feat)) for i, h in enumerate(self.heads)], dim=2)


class Restorer(nn.Module):
    def __init__(self, dim=128, depth=6, heads=8, n_fft=2048, hop=512, channels=2, dropout=0.0, residual=False, level_feats=True):
        super().__init__()
        self.cfg = dict(dim=dim, depth=depth, heads=heads, n_fft=n_fft, hop=hop, channels=channels, dropout=dropout, residual=residual, level_feats=level_feats)
        self.n_fft, self.hop, self.channels, self.residual = n_fft, hop, channels, residual
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        bands = band_bins(n_fft)
        feat = channels * 2
        self.split = BandSplit(bands, dim, feat, level_feats)
        self.time_blocks = nn.ModuleList([Block(dim, heads, dropout=dropout) for _ in range(depth)])
        self.band_blocks = nn.ModuleList([Block(dim, heads, dropout=dropout) for _ in range(depth)])
        self.head = MaskHead(bands, dim, feat * (2 if residual else 1))
        self.amp = True

    def stft(self, x):  # [B, C, N] -> complex [B, C, F, T]
        b, c, n = x.shape
        s = torch.stft(x.reshape(b * c, n), self.n_fft, self.hop, window=self.window, return_complex=True)
        return s.view(b, c, *s.shape[-2:])

    def istft(self, s, length):
        b, c = s.shape[:2]
        x = torch.istft(s.reshape(b * c, *s.shape[-2:]), self.n_fft, self.hop, window=self.window, length=length)
        return x.view(b, c, length)

    def forward(self, x):
        n = x.shape[-1]
        spec = self.stft(x)
        b, c, f, t = spec.shape
        feats = torch.view_as_real(spec).permute(0, 3, 2, 1, 4).reshape(b, t, f, c * 2)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=self.amp and x.device.type == "cuda"):
            h = self.split(feats)  # [B, T, K, D]
            k = h.shape[2]
            for tb, bb in zip(self.time_blocks, self.band_blocks):
                h = tb(h.transpose(1, 2).reshape(b * k, t, -1)).view(b, k, t, -1).transpose(1, 2)
                h = bb(h.reshape(b * t, k, -1)).view(b, t, k, -1)
            m = self.head(h)
        h = m
        m = h.float().view(b, t, f, -1, c, 2).permute(3, 0, 4, 2, 1, 5).contiguous()
        out = spec * (torch.view_as_complex(m[0]) + 1.0)
        if self.residual:
            # additive branch can synthesize content the mask cannot reach (e.g. bins the input lacks)
            out = out + torch.view_as_complex(m[1]) * spec.abs().mean(dim=(1, 2, 3), keepdim=True)
        return self.istft(out, n)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
