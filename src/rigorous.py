"""Rigorous enclosure [theta_lo, theta_up] of theta(G) from a basis A of the circulant theta LP.
Primal: for any x with g(x) = 1 + J x >= 0,  theta >= 1 + 2 sum(x).  If min g >= -delta then
x/(1+delta) is feasible, so theta >= 1 + 2 sum(x)/(1+delta).
Dual: for any y >= 0 with rho = J'y + 2*1, every feasible x satisfies |x_r| <= 1 (Lemma: x_r =
(1/2n) sum_k J_kr g_k and sum_k g_k = n), hence theta <= 1 + sum(y) + ||rho||_1.
All products are evaluated in float64 by direct cosine sums with exact integer argument reduction,
and the accumulated rounding error is bounded a priori and added to delta and rho.
Works for 1-D circulants (kr mod n) and, through `dots`, for Z_n x Z_n Cayley graphs."""
import math, torch

U = 2.0 ** -53          # unit roundoff, float64
C_COS = 32.0 * U        # absolute error bound of 2cos(2 pi d/n) evaluated in float64 with exact integer d (argument rounding plus cos rounding)


def gamma(k):
    return k * U / (1 - k * U)


def cos_block(dots, n):
    """2 cos(2 pi dots / n) with dots an int64 tensor already reduced mod n."""
    return 2.0 * torch.cos(2.0 * math.pi * dots.double() / n)


def enclosure(n, N, Rvec, Avec, dots, dev, chunk=2048):
    """n: modulus; N: number of constraints (= n for 1-D, n^2 for 2-D); Rvec: long tensor of generators;
    Avec: long tensor of the m basis constraints; dots(kidx, Rvec) -> int64 [len(kidx), m] dot products mod n.
    Returns dict with lo, up, and diagnostics."""
    m = len(Rvec)
    JA = cos_block(dots(Avec, Rvec), n)
    ones = torch.ones(m, device=dev)
    x = torch.linalg.solve(JA, -ones); lam = torch.linalg.solve(JA.T, -2.0 * ones)
    # one step of iterative refinement for x and lam
    x = x - torch.linalg.solve(JA, JA @ x + ones); lam = lam - torch.linalg.solve(JA.T, JA.T @ lam + 2.0 * ones)
    sx = x.abs().sum().item()
    e_g = 2.0 * sx * C_COS + gamma(m + 2) * (1.0 + 2.0 * sx)       # rigorous bound on |computed g_k - true g_k|
    # primal: g = 1 + J x over all N constraints, in chunks
    gmin = float("inf"); allk = torch.arange(N, device=dev)
    for i in range(0, N, chunk):
        kidx = allk[i:i + chunk]
        g = 1.0 + cos_block(dots(kidx, Rvec), n) @ x
        gmin = min(gmin, g.min().item())
    delta = max(0.0, -(gmin - e_g))
    sumx = x.sum().item(); e_sum = gamma(m) * sx
    lo = 1.0 + 2.0 * (sumx - e_sum) / (1.0 + delta)
    # dual: y_k = lam_i/2 on basis rows k and on -k (same row), clipped at 0
    y = torch.clamp(lam, min=0.0) / 2.0
    # rho_r = sum over all N constraints of J_kr y_k + 2 = 2 * sum_{i in A} J_{A_i r} y_i + 2 (rows k and -k each carry y_i)
    rho = 2.0 * (JA.T @ y) + 2.0
    sy = y.sum().item()
    e_rho = 2.0 * (2.0 * sy * C_COS + gamma(m + 2) * (2.0 * sy))   # per entry, factor 2 for the two rows per basis element
    rho1 = rho.abs().sum().item() + m * e_rho
    up = 1.0 + 2.0 * sy + gamma(m) * 2.0 * sy + rho1               # sum over all N rows of y = 2 * sum(y on reps)
    return dict(lo=lo, up=up, width=up - lo, gmin=gmin, e_g=e_g, delta=delta, rho1=rho1, ny=2.0 * sy, sumx=sumx)


def dots_1d(n):
    def f(kidx, Rvec):
        return (kidx[:, None] * Rvec[None, :]) % n
    return f
