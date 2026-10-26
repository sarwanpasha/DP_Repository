"""Equivariant active-set predictor for the circulant theta LP.

Two node sets: frequencies k in {1..n'} (constraints) and generators r in R (variables).  The only
coupling is through the LP's own linear maps J (generators -> frequencies) and J^T, applied
channel-wise by FFT.  Pointwise MLPs are shared across nodes.  Hence the network commutes with
the multiplier action (k, r) -> (u^{-1} k, u r), u in Z_n^*, and has no n-dependent parameters.

A plain 1D-CNN-over-frequency baseline is included for ablation (not equivariant).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class FFTOps:
    """Batched J / J^T for a fixed n and index set R (torch)."""

    def __init__(self, n, R, device):
        self.n = n; self.nprime = (n - 1) // 2
        self.R = torch.as_tensor(R, dtype=torch.long, device=device)
        self.nR = n - self.R
        self.m = len(R)
        self.scale = 1.0 / math.sqrt(2.0 * n)          # ||J||_2 = sqrt(2n)
        self.device = device

    def J(self, z):                                      # z: [B,C,m] -> [B,C,n'] (reps 1..n')
        B, C, _ = z.shape
        vt = torch.zeros(B, C, self.n, device=z.device, dtype=z.dtype)
        vt.index_add_(2, self.R, z); vt.index_add_(2, self.nR, z)
        f = torch.fft.fft(vt, dim=2).real
        return f[..., 1:self.nprime + 1] * self.scale

    def JT(self, h):                                     # h: [B,C,n'] -> [B,C,m]
        B, C, _ = h.shape
        u = torch.zeros(B, C, self.n, device=h.device, dtype=h.dtype)
        u[..., 1:self.nprime + 1] = h; u[..., self.nprime + 1:] = h.flip(-1)
        f = torch.fft.fft(u, dim=2).real
        return 2.0 * f[..., self.R] * self.scale


def mlp(i, o, hidden):
    return nn.Sequential(nn.Conv1d(i, hidden, 1), nn.GELU(), nn.Conv1d(hidden, o, 1))


class EquivariantPredictor(nn.Module):
    def __init__(self, f_in, g_in, C=64, layers=6, hidden=128):
        super().__init__()
        self.enc_f = nn.Conv1d(f_in, C, 1); self.enc_g = nn.Conv1d(g_in, C, 1)
        self.upd_f = nn.ModuleList([mlp(2 * C, C, hidden) for _ in range(layers)])
        self.upd_g = nn.ModuleList([mlp(2 * C, C, hidden) for _ in range(layers)])
        self.norm_f = nn.ModuleList([nn.GroupNorm(1, C) for _ in range(layers)])
        self.norm_g = nn.ModuleList([nn.GroupNorm(1, C) for _ in range(layers)])
        self.out = nn.Conv1d(C, 1, 1)
        self.layers = layers

    def forward(self, ff, gf, ops: FFTOps):
        # ff: [B, f_in, n']  frequency-side features;  gf: [B, g_in, m] generator-side features
        h = self.enc_f(ff); z = self.enc_g(gf)
        for l in range(self.layers):
            z = z + self.upd_g[l](torch.cat([z, ops.JT(self.norm_f[l](h))], 1))
            h = h + self.upd_f[l](torch.cat([h, ops.J(self.norm_g[l](z))], 1))
        return self.out(h).squeeze(1)                    # logits [B, n']


class CNNBaseline(nn.Module):
    """Non-equivariant baseline: dilated 1D CNN over the frequency axis (generator features are
    scattered onto the frequency axis as an extra channel, which is what a naive model would do)."""

    def __init__(self, f_in, g_in, C=64, layers=6, k=5):
        super().__init__()
        self.enc = nn.Conv1d(f_in + g_in, C, 1)
        self.convs = nn.ModuleList([nn.Conv1d(C, C, k, padding=(k // 2) * 2 ** (l % 4), dilation=2 ** (l % 4),
                                              padding_mode="circular") for l in range(layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, C) for _ in range(layers)])
        self.out = nn.Conv1d(C, 1, 1)

    def forward(self, ff, gf, ops: FFTOps):
        B, Cg, mm = gf.shape
        gfull = torch.zeros(B, Cg, ff.shape[2] + 1, device=ff.device, dtype=ff.dtype)
        R = ops.R if ops.R.dim() == 2 else ops.R[None, :].expand(B, -1)
        mask = ops.mask if hasattr(ops, "mask") else torch.ones_like(R, dtype=torch.bool)
        idx = torch.where(mask, R - 1, torch.full_like(R, ff.shape[2]))  # padded -> dummy slot
        gfull.scatter_(2, idx[:, None, :].expand(B, Cg, mm), gf * mask[:, None, :])
        gfull = gfull[..., :ff.shape[2]]
        h = self.enc(torch.cat([ff, gfull], 1))
        for conv, nm in zip(self.convs, self.norms):
            h = h + conv(F.gelu(nm(h)))
        return self.out(h).squeeze(1)


def topm(logits, m):
    """Boolean mask of the m largest logits per row."""
    idx = torch.topk(logits, m, dim=1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool); mask.scatter_(1, idx, True)
    return mask
