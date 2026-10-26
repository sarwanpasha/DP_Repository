"""Two-phase active-set (simplex) correction from a predicted basis for the theta LP.
Phase A: dual simplex on perturbed cost (basis dual feasible by construction) until primal feasible.
Phase B: primal simplex on the true cost until all multipliers >= 0.  Dense backend keeps an explicit
inverse of the cosine minor with Sherman-Morrison rank-1 updates (refactor every 50 pivots); the
FFT backend uses CGNR on J_A^T J_A.
"""
import numpy as np
from theta_fft import ThetaOps, pcg


class Basis:
    def __init__(self, ops: ThetaOps, A, solver="dense", counter=None, tol=1e-12):
        self.ops = ops; self.n = ops.n; self.m = ops.m; self.nprime = (ops.n - 1) // 2
        self.A = np.array(sorted(A), dtype=np.int64)
        assert len(self.A) == self.m
        self.solver = solver; self.tol = tol
        self.counter = counter
        self.inner_iters = []
        self.refresh()

    def _fft(self, k=1):
        if self.counter is not None: self.counter.n_fft += k

    def refresh(self):
        if self.solver == "dense":
            self.JA = 2.0 * np.cos(2 * np.pi * np.outer(self.A, self.ops.R) / self.n)
            self.Binv = np.linalg.inv(self.JA)
            self.n_updates = 0
        self.mask = np.zeros(self.n); self.mask[self.A] = 1; self.mask[self.n - self.A] = 1

    def JA_mv(self, z): self._fft(); return self.ops.J(z)[self.A]
    def JAT_mv(self, w):
        u = np.zeros(self.n); u[self.A] = w; u[self.n - self.A] = w; self._fft(); return 0.5 * self.ops.JT(u)
    def GA_mv(self, z): self._fft(2); return 0.5 * self.ops.JT(self.mask * self.ops.J(z))

    def solve(self, w, x0=None):
        if self.solver == "dense":
            return self.Binv @ w
        z, info = pcg(self.GA_mv, self.JAT_mv(w), x0=x0, tol=self.tol, maxiter=8 * self.m)
        self.inner_iters.append(info["iters"]); return z

    def solve_T(self, v, x0=None):
        if self.solver == "dense":
            return self.Binv.T @ v
        z, info = pcg(self.GA_mv, v, x0=x0, tol=self.tol, maxiter=8 * self.m)
        self.inner_iters.append(info["iters"]); return self.JA_mv(z)

    def row(self, k):
        return 2.0 * np.cos(2 * np.pi * k * self.ops.R / self.n)

    def g_all(self, x):
        self._fft(); return 1.0 + self.ops.J(x)

    def swap(self, k_out, k_in):
        if self.solver == "dense" and self.n_updates < 50:
            i = int(np.where(self.A == k_out)[0][0])
            u = self.row(k_in) - self.JA[i]
            Binv_ei = self.Binv[:, i].copy(); uT_Binv = u @ self.Binv
            denom = 1.0 + uT_Binv[i]
            if abs(denom) > 1e-10:
                self.Binv -= np.outer(Binv_ei, uT_Binv) / denom
                self.JA[i] = self.row(k_in)
                Anew = self.A.copy(); Anew[i] = k_in
                order = np.argsort(Anew)
                self.A = Anew[order]; self.JA = self.JA[order]; self.Binv = self.Binv[:, order]
                self.mask = np.zeros(self.n); self.mask[self.A] = 1; self.mask[self.n - self.A] = 1
                self.n_updates += 1
                return
        A = set(self.A.tolist()); A.remove(int(k_out)); A.add(int(k_in))
        self.A = np.array(sorted(A), dtype=np.int64); self.refresh()


def simplex_correct(ops, A0, solver="dense", counter=None, max_pivots=None, eps=1e-9, verbose=False,
                    feas_tol=1e-9):
    m = ops.m; nprime = (ops.n - 1) // 2
    max_pivots = max_pivots or 4 * m
    B = Basis(ops, A0, solver=solver, counter=counter)
    c = 2.0 * np.ones(m)
    x = B.solve(-np.ones(m)); lam = B.solve_T(-c)
    g = B.g_all(x)
    pivA = pivB = 0
    log = []
    lam_t = np.maximum(lam, eps)
    while True:
        gr = g[1:nprime + 1]
        viol = np.where(gr < -feas_tol)[0] + 1
        viol = viol[~np.isin(viol, B.A)]
        if len(viol) == 0:
            break
        if pivA + pivB >= max_pivots:
            return dict(ok=False, theta=1 + c @ x, x=x, lam=lam, A=B.A, pivots=pivA + pivB, phaseA=pivA, phaseB=pivB, log=log)
        k_in = viol[np.argmin(gr[viol - 1])]
        w = B.solve_T(B.row(k_in))
        pos = w > 1e-14
        if not pos.any():
            i_out = np.argmin(lam_t)
        else:
            ratios = np.full(m, np.inf); ratios[pos] = lam_t[pos] / w[pos]
            i_out = np.argmin(ratios)
        t = lam_t[i_out] / w[i_out] if w[i_out] > 1e-14 else 0.0
        lam_t = lam_t - t * w; lam_t[i_out] = t
        k_out = B.A[i_out]
        lam_map = {int(k): lam_t[i] for i, k in enumerate(B.A)}; lam_map[int(k_in)] = t; lam_map.pop(int(k_out), None)
        B.swap(k_out, k_in)
        lam_t = np.array([lam_map[int(k)] for k in B.A]); lam_t = np.maximum(lam_t, 0.0)
        x = B.solve(-np.ones(m), x0=x); g = B.g_all(x)
        pivA += 1
    lam = B.solve_T(-c)
    while True:
        i_neg = np.where(lam < -feas_tol)[0]
        if len(i_neg) == 0:
            break
        if pivA + pivB >= max_pivots:
            return dict(ok=False, theta=1 + c @ x, x=x, lam=lam, A=B.A, pivots=pivA + pivB, phaseA=pivA, phaseB=pivB, log=log)
        i = i_neg[np.argmin(lam[i_neg])]
        d = B.solve(np.eye(m)[i])
        Jd = ops.J(d); B._fft()
        blocking = Jd < -1e-14
        blocking[B.A] = False; blocking[ops.n - B.A] = False
        if not blocking.any():
            return dict(ok=False, unbounded=True, theta=np.inf, x=x, lam=lam, A=B.A, pivots=pivA + pivB, phaseA=pivA, phaseB=pivB, log=log)
        ratios = np.full(ops.n, np.inf); ratios[blocking] = -g[blocking] / Jd[blocking]
        k_in_full = int(np.argmin(ratios)); t = ratios[k_in_full]
        k_in = min(k_in_full, ops.n - k_in_full) if k_in_full != 0 else 0
        if k_in == 0:
            return dict(ok=False, theta=1 + c @ x, x=x, lam=lam, A=B.A, pivots=pivA + pivB, phaseA=pivA, phaseB=pivB, log=log)
        x = x + t * d; g = g + t * Jd
        k_out = B.A[i]
        B.swap(k_out, k_in)
        lam = B.solve_T(-c, x0=lam)
        pivB += 1
    theta = 1.0 + c @ x
    g = B.g_all(x)
    ok = g.min() >= -1e-7 and lam.min() >= -1e-7 and abs(theta - (1 + lam.sum())) < 1e-6 * max(1, theta)
    return dict(ok=bool(ok), theta=theta, x=x, lam=lam, A=B.A, pivots=pivA + pivB, phaseA=pivA, phaseB=pivB,
                gmin=g.min(), lmin=lam.min(), inner_iters=B.inner_iters, log=log)
