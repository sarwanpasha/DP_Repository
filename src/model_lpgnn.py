"""LP-GNN baseline (Chen et al. 2023 style): bipartite constraint-variable message passing with
learned edge functions of the entry J_{k,r}, applied as dense matmuls (O(n m) per layer).
Same forward signature as EquivariantPredictor: forward(ff, gf, ops)."""
import math, torch, torch.nn as nn
from model import mlp


class LPGNN(nn.Module):
    def __init__(self, f_in, g_in, C=64, layers=6, hidden=128, E=8):
        super().__init__()
        self.C, self.E, self.layers = C, E, layers
        self.enc_f = nn.Conv1d(f_in, C, 1); self.enc_g = nn.Conv1d(g_in, C, 1)
        self.edge = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, E))
        self.g2f = nn.ModuleList([nn.Conv1d(C, C * E, 1) for _ in range(layers)])
        self.f2g = nn.ModuleList([nn.Conv1d(C, C * E, 1) for _ in range(layers)])
        self.upd_f = nn.ModuleList([mlp(2 * C, C, hidden) for _ in range(layers)])
        self.upd_g = nn.ModuleList([mlp(2 * C, C, hidden) for _ in range(layers)])
        self.norm_f = nn.ModuleList([nn.GroupNorm(1, C) for _ in range(layers)])
        self.norm_g = nn.ModuleList([nn.GroupNorm(1, C) for _ in range(layers)])
        self.out = nn.Conv1d(C, 1, 1)

    def edges(self, ops, B, device, dtype):
        n = ops.n; nprime = (n - 1) // 2
        R = ops.R if ops.R.dim() == 2 else ops.R[None, :].expand(B, -1)
        mask = ops.mask if hasattr(ops, "mask") else torch.ones_like(R, dtype=torch.bool)
        k = torch.arange(1, nprime + 1, device=device, dtype=torch.float64)
        W = 2.0 * torch.cos(2 * math.pi * k[None, :, None] * R[:, None, :].double() / n)  # [B,n',m]
        Ew = self.edge(W.to(dtype)[..., None]) * mask[:, None, :, None].to(dtype)  # [B,n',m,E]
        return Ew / math.sqrt(2.0 * n)

    def forward(self, ff, gf, ops):
        B = ff.shape[0]; C, E = self.C, self.E
        Ew = self.edges(ops, B, ff.device, ff.dtype)
        h = self.enc_f(ff); z = self.enc_g(gf)
        for l in range(self.layers):
            hh = self.f2g[l](self.norm_f[l](h)).view(B, E, C, -1)          # [B,E,C,n']
            msg_z = torch.einsum("bkme,beck->bcm", Ew, hh)                # [B,C,m]
            z = z + self.upd_g[l](torch.cat([z, msg_z], 1))
            zz = self.g2f[l](self.norm_g[l](z)).view(B, E, C, -1)          # [B,E,C,m]
            msg_h = torch.einsum("bkme,becm->bck", Ew, zz)                # [B,C,n']
            h = h + self.upd_f[l](torch.cat([h, msg_h], 1))
        return self.out(h).squeeze(1)
