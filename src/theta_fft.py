"""
FFT-accelerated interior-point solver for the Lovasz theta of circulant graphs.

Circulant graph G on Z_n with symmetric connection set S (edges), non-edge set
T = Z_n \ (S u {0}), T = -T.  Theta LP (Fourier form):

    theta(G) = max  1 + sum_{t in T} x_t
               s.t. g_k(x) = 1 + sum_{t in T} x_t cos(2 pi k t / n) >= 0,  k = 0..n-1.

Symmetric coordinates: R = {min(t, n-t) : t in T}, m = |R|.
    objective 1 + 2 sum_r x_r,   g(x) = 1 + J x,   J_{k,r} = 2 cos(2 pi k r / n).

J v      = Re FFT(vtilde)         (vtilde symmetric zero-padding of v)   O(n log n)
J^T u    = 2 Re FFT(u) restricted to R                                   O(n log n)
H(x)     = J^T D J,  D = diag(g^{-2})                                    matvec O(n log n)
H_{r,r'} = a_{r-r'} + a_{r+r'},  a_j = 2 Re FFT(D)_j   (Toeplitz + Hankel)

Everything here is written from scratch for the DP_iclr project.
"""
import numpy as np
import time


# ----------------------------------------------------------------------------
# instance generation
# ----------------------------------------------------------------------------
def random_circulant(n, p=0.5, rng=None):
    """Random circulant on Z_n (n odd).  Returns sorted array R of non-edge
    generator representatives in {1..(n-1)/2}."""
    rng = np.random.default_rng(rng)
    half = (n - 1) // 2
    reps = np.arange(1, half + 1)
    is_edge = rng.random(half) < p
    R = reps[~is_edge]
    return R


# ----------------------------------------------------------------------------
# FFT operators
# ----------------------------------------------------------------------------
class ThetaOps:
    """FFT implementations of J, J^T and H = J^T D J for a fixed n and R."""

    def __init__(self, n, R):
        self.n = int(n)
        self.R = np.asarray(R, dtype=np.int64)
        self.m = len(self.R)
        self.nR = self.n - self.R  # mirror indices

    # J v : R^m -> R^n
    def J(self, v):
        vt = np.zeros(self.n)
        vt[self.R] = v
        vt[self.nR] += v          # symmetric zero padding (R never contains n/2 for odd n)
        return np.fft.fft(vt).real

    # J^T u : R^n -> R^m
    def JT(self, u):
        f = np.fft.fft(u).real
        return 2.0 * f[self.R]

    # Hessian-vector product with weights D (length n)
    def H(self, D, v):
        return self.JT(D * self.J(v))

    # symbol a_j = 2 Re FFT(D)_j,   H_{rr'} = a_{r-r'} + a_{r+r'}
    def symbol(self, D):
        return 2.0 * np.fft.fft(D).real

    def H_diag(self, D):
        a = self.symbol(D)
        return a[0] + a[(2 * self.R) % self.n]

    def H_dense(self, D):
        a = self.symbol(D)
        r = self.R
        return a[(r[:, None] - r[None, :]) % self.n] + a[(r[:, None] + r[None, :]) % self.n]

    def J_dense(self):
        k = np.arange(self.n)
        return 2.0 * np.cos(2 * np.pi * np.outer(k, self.R) / self.n)


# ----------------------------------------------------------------------------
# preconditioned conjugate gradient with iteration accounting
# ----------------------------------------------------------------------------
def pcg(matvec, b, precond=None, x0=None, tol=1e-10, maxiter=5000):
    """Solve A x = b, A SPD via preconditioned CG.  precond(r) ~ A^{-1} r.
    Returns x, info dict (iters, relres, converged)."""
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - matvec(x) if x0 is not None else b.copy()
    bnorm = np.linalg.norm(b) + 1e-300
    z = precond(r) if precond is not None else r
    p = z.copy()
    rz = r @ z
    it = 0
    relres = np.linalg.norm(r) / bnorm
    while relres > tol and it < maxiter:
        Ap = matvec(p)
        pAp = p @ Ap
        if pAp <= 0:
            break
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        it += 1
        relres = np.linalg.norm(r) / bnorm
        if relres <= tol:
            break
        z = precond(r) if precond is not None else r
        rz_new = r @ z
        beta = rz_new / rz
        rz = rz_new
        p = z + beta * p
    return x, dict(iters=it, relres=relres, converged=relres <= tol)


# ----------------------------------------------------------------------------
# classical preconditioners (all O(n log n) or O(m) to apply)
# ----------------------------------------------------------------------------
def make_precond(name, ops, D, g=None):
    """Return a callable r -> P^{-1} r, or None."""
    if name in (None, "none"):
        return None
    if name == "jacobi":
        d = ops.H_diag(D)
        return lambda r: r / d
    if name == "circulant":
        # circulant embedding of the Toeplitz part: treat H as principal
        # submatrix of J_full^T D J_full whose inverse is ~ J_full^T D^{-1} J_full / n^2
        Dinv = 1.0 / D
        scale = 1.0 / (4.0 * ops.n ** 2)   # J^T D J for full R has J^T J = 2n I -> (2n)^2
        return lambda r: scale * ops.JT(Dinv * ops.J(r)) * 1.0
    raise ValueError(name)


# ----------------------------------------------------------------------------
# barrier interior-point method
# ----------------------------------------------------------------------------
def solve_theta(n, R, tol=1e-8, mu=10.0, t0=1.0, precond="none",
                cg_tol=1e-10, cg_maxiter=None, newton_tol=1e-9, max_newton=200,
                verbose=False, record=None, dense_newton=False, max_total_newton=None):
    """Long-step barrier method for the circulant theta LP.

    minimise  f_t(x) = -t c^T x - sum_k log g_k(x),   c = 2*ones(m),  g = 1 + J x
    Newton:   H dx = t c + J^T (1/g),   H = J^T diag(1/g^2) J.

    Returns dict with theta, x, and per-Newton-step CG statistics.
    """
    ops = ThetaOps(n, R)
    m = ops.m
    if m == 0:
        return dict(theta=1.0, x=np.zeros(0), newton_steps=0, cg_iters=[], t_final=None)
    c = 2.0 * np.ones(m)
    x = np.zeros(m)                 # analytic centre: g = 1
    g = 1.0 + ops.J(x)
    t = t0
    nu = float(n)                   # barrier parameter
    cg_maxiter = cg_maxiter or 4 * m + 50
    stats = dict(cg_iters=[], newton_per_outer=[], decrements=[], t_hist=[], Dmax=[])
    total_newton = 0
    outer = 0
    t_start = time.time()

    while True:
        # ---- centre for current t ----
        nsteps = 0
        for _ in range(max_newton):
            D = 1.0 / g ** 2
            grad = -t * c - ops.JT(1.0 / g)          # gradient of f_t
            rhs = -grad
            if dense_newton:
                Hd = ops.H_dense(D)
                dx = np.linalg.solve(Hd, rhs)
                cg_info = dict(iters=0, relres=0.0, converged=True)
            else:
                P = make_precond(precond, ops, D, g)
                dx, cg_info = pcg(lambda v: ops.H(D, v), rhs, precond=P,
                                  tol=cg_tol, maxiter=cg_maxiter)
            lam2 = rhs @ dx                          # Newton decrement squared
            stats["cg_iters"].append(cg_info["iters"])
            stats["decrements"].append(np.sqrt(max(lam2, 0)))
            stats["t_hist"].append(t)
            stats["Dmax"].append(D.max())
            nsteps += 1
            total_newton += 1
            if lam2 / 2 < newton_tol:
                break
            if max_total_newton is not None and total_newton >= max_total_newton:
                break
            # ---- damped step / backtracking line search (feasibility + Armijo) ----
            Jdx = ops.J(dx)
            # largest feasible step
            neg = Jdx < 0
            smax = np.min(-g[neg] / Jdx[neg]) if neg.any() else np.inf
            s = min(1.0, 0.99 * smax)
            f0 = -t * (c @ x) - np.sum(np.log(g))
            while True:
                gn = g + s * Jdx
                if np.all(gn > 0):
                    fn = -t * (c @ (x + s * dx)) - np.sum(np.log(gn))
                    if fn <= f0 + 1e-4 * s * (grad @ dx):
                        break
                s *= 0.5
                if s < 1e-12:
                    break
            x = x + s * dx
            g = gn
        stats["newton_per_outer"].append(nsteps)
        outer += 1
        gap = nu / t                                  # duality gap bound
        if verbose:
            print(f"outer {outer:3d}  t={t:.3e}  gap<={gap:.2e}  newton={nsteps}  "
                  f"cg(last)={stats['cg_iters'][-1]}  obj={1 + c @ x:.8f}")
        if gap < tol:
            break
        if max_total_newton is not None and total_newton >= max_total_newton:
            break
        t *= mu

    theta = 1.0 + c @ x
    stats.update(dict(theta=theta, x=x, g=g, n=n, m=m, outer=outer,
                      newton_steps=total_newton, total_cg=int(np.sum(stats["cg_iters"])),
                      time=time.time() - t_start, t_final=t))
    return stats


# ----------------------------------------------------------------------------
# reference solver (dense LP) for verification at small n
# ----------------------------------------------------------------------------
def theta_reference(n, R):
    from scipy.optimize import linprog
    ops = ThetaOps(n, R)
    Jd = ops.J_dense()
    m = ops.m
    # max 1 + 2 sum x  s.t. -J x <= 1
    res = linprog(c=-2.0 * np.ones(m), A_ub=-Jd, b_ub=np.ones(n), bounds=[(None, None)] * m,
                  method="highs")
    return 1.0 + (-res.fun), res.x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n in [23, 61, 101]:
        R = random_circulant(n, rng=rng)
        ops = ThetaOps(n, R)
        # operator checks
        v = rng.standard_normal(ops.m); u = rng.standard_normal(n); D = rng.random(n) + 0.1
        Jd = ops.J_dense()
        print(f"n={n} m={ops.m}  |Jv-Jd v|={np.abs(ops.J(v) - Jd @ v).max():.2e}"
              f"  |JTu-Jd^T u|={np.abs(ops.JT(u) - Jd.T @ u).max():.2e}"
              f"  |Hv-dense|={np.abs(ops.H(D, v) - ops.H_dense(D) @ v).max():.2e}"
              f"  |H-JtDJ|={np.abs(ops.H_dense(D) - Jd.T @ (D[:, None] * Jd)).max():.2e}"
              f"  |diag|={np.abs(ops.H_diag(D) - np.diag(ops.H_dense(D))).max():.2e}")
        th_ref, _ = theta_reference(n, R)
        st = solve_theta(n, R, tol=1e-8)
        print(f"   theta_ref={th_ref:.8f}  theta_ipm={st['theta']:.8f}  diff={abs(th_ref - st['theta']):.2e}"
              f"  newton={st['newton_steps']} total_cg={st['total_cg']} maxcg={max(st['cg_iters'])} time={st['time']:.2f}s")
