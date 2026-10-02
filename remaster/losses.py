import torch
import torch.nn.functional as F


class MultiResSTFT(torch.nn.Module):
    """Complex L1 + log-magnitude L1 at several resolutions.

    The log term weights quiet content (reverb tails, echo repeats, noise floor) that a
    linear-domain loss barely sees.
    """

    def __init__(self, ffts=(4096, 2048, 1024, 512, 256), w_complex=1.0, w_log=1.0):
        super().__init__()
        self.ffts, self.w_complex, self.w_log = ffts, w_complex, w_log
        for n in ffts:
            self.register_buffer(f"win{n}", torch.hann_window(n), persistent=False)

    def spec(self, x, n):
        return torch.stft(x.reshape(-1, x.shape[-1]), n, n // 4, window=getattr(self, f"win{n}"), return_complex=True)

    def forward(self, pred, target):
        lc = ll = 0.0
        for n in self.ffts:
            p, t = torch.view_as_real(self.spec(pred, n)), torch.view_as_real(self.spec(target, n))
            lc = lc + (p - t).abs().mean()
            # explicit sqrt(re^2 + im^2 + eps): complex abs() has an undefined gradient at exactly 0 on some backends
            pm, tm = (p.pow(2).sum(-1) + 1e-8).sqrt(), (t.pow(2).sum(-1) + 1e-8).sqrt()
            ll = ll + (torch.log(pm + 1e-3) - torch.log(tm + 1e-3)).abs().mean()
        k = len(self.ffts)
        return self.w_complex * lc / k, self.w_log * ll / k


def si_sdr(pred, target, eps=1e-8):
    """Scale-invariant SDR in dB over the last axis (channels flattened)."""
    p, t = pred.flatten(1), target.flatten(1)
    p, t = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    a = (p * t).sum(1, keepdim=True) / (t.pow(2).sum(1, keepdim=True) + eps)
    return 10 * torch.log10((a * t).pow(2).sum(1) / ((p - a * t).pow(2).sum(1) + eps) + eps)


def sdr(pred, target, eps=1e-8):
    p, t = pred.flatten(1), target.flatten(1)
    return 10 * torch.log10(t.pow(2).sum(1) / ((p - t).pow(2).sum(1) + eps) + eps)


def log_spec_dist(pred, target, n_fft=2048):
    """Log-spectral distance in dB (lower is better)."""
    w = torch.hann_window(n_fft, device=pred.device)
    p = torch.stft(pred.reshape(-1, pred.shape[-1]), n_fft, n_fft // 4, window=w, return_complex=True).abs()
    t = torch.stft(target.reshape(-1, target.shape[-1]), n_fft, n_fft // 4, window=w, return_complex=True).abs()
    d = (20 * torch.log10(p + 1e-4) - 20 * torch.log10(t + 1e-4)).pow(2).mean(1).sqrt().mean(1)
    return d.view(pred.shape[0], -1).mean(1)
