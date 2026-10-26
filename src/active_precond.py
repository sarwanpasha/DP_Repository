"""Active-set (cosine-minor) preconditioner for the FFT-IPM Newton system.

H = sum_{k=0}^{n-1} D_k j_k j_k^T = D_0 j_0 j_0^T + 2 sum_{k=1}^{n'} D_k j_k j_k^T,   n' = (n-1)/2

Pick A = the |A| representatives k in {1..n'} with the largest D_k.  Then
    H_A = 2 J_A^T D_A J_A,   P^{-1} = (1/2) J_A^+ D_A^{-1} (J_A^+)^T
and P^{-1} H = I + low-rank(<= |A| - m) + O(max D_I / min D_A).

The inner solves with the cosine minor J_A (rows k in A of J) are D-free.
Two implementations: dense (prototype) and CGNR on G_A = J^T 1_A J via FFT.
"""
import numpy as np
from theta_fft import ThetaOps, pcg


class ActiveSetPrecond:
    def __init__(self, ops: ThetaOps, D, size=None, inner="dense", inner_tol=1e-12,
                 inner_maxiter=None, inner_precond=None):
        self.ops = ops
        n, m = ops.n, ops.m
        nprime = (n - 1) // 2
        reps = np.arange(1, nprime + 1)
        Dr = D[reps]
        size = m if size is None else size
        order = np.argsort(-Dr)
        self.A = reps[order[:size]]                 # active representatives
        self.DA = Dr[order[:size]]
        self.gap = (Dr[order[size - 1]] / Dr[order[size]]) if size < nprime else np.inf
        self.inner = inner
        self.inner_iters = []
        self.inner_tol = inner_tol
        self.inner_maxiter = inner_maxiter or 4 * m
        self.inner_precond = inner_precond
        if inner == "dense":
            JA = 2.0 * np.cos(2 * np.pi * np.outer(self.A, ops.R) / n)
            self.JA = JA
            if size == m:
                self.lu = np.linalg.inv(JA)          # prototype: explicit inverse
            else:
                self.JA_pinv = np.linalg.pinv(JA)
        # mask for FFT-based G_A = J^T 1_A J
        self.maskA = np.zeros(n)
        self.maskA[self.A] = 1.0
        self.maskA[n - self.A] = 1.0

    # J_A z  (R^m -> R^{|A|})   and   J_A^T w (R^{|A|} -> R^m) via FFT
    def JA_mv(self, z):
        return self.ops.J(z)[self.A]

    def JAT_mv(self, w):
        u = np.zeros(self.ops.n)
        u[self.A] = w
        u[self.ops.n - self.A] = w
        return 0.5 * self.ops.JT(u)   # both k and n-k rows carry w; J^T sums them -> halve

    def GA_mv(self, z):
        # (J_A^T J_A) z = sum_{k in A} j_k j_k^T z  = (1/2) J^T (1_A^{sym} * J z)
        return 0.5 * self.ops.JT(self.maskA * self.ops.J(z))

    def solve_JA(self, w):
        """z = J_A^+ w"""
        if self.inner == "dense":
            return self.lu @ w if hasattr(self, "lu") else self.JA_pinv @ w
        # CGNR: (J_A^T J_A) z = J_A^T w
        rhs = self.JAT_mv(w)
        z, info = pcg(self.GA_mv, rhs, precond=self.inner_precond, tol=self.inner_tol,
                      maxiter=self.inner_maxiter)
        self.inner_iters.append(info["iters"])
        return z

    def solve_JAT(self, v):
        """y = (J_A^+)^T v = J_A (J_A^T J_A)^{-1} v"""
        if self.inner == "dense":
            return self.lu.T @ v if hasattr(self, "lu") else self.JA_pinv.T @ v
        z, info = pcg(self.GA_mv, v, precond=self.inner_precond, tol=self.inner_tol,
                      maxiter=self.inner_maxiter)
        self.inner_iters.append(info["iters"])
        return self.JA_mv(z)

    def __call__(self, r):
        y = self.solve_JAT(r)          # (J_A^+)^T r
        y = y / self.DA                # D_A^{-1}
        return 0.5 * self.solve_JA(y)  # (1/2) J_A^+ ...


if __name__ == "__main__":
    import sys
    from theta_fft import random_circulant, solve_theta
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 211
    R = random_circulant(n, rng=0)
    ops = ThetaOps(n, R); m = ops.m
    st = solve_theta(n, R, tol=1e-8, precond="none")
    print(f"n={n} m={m} theta={st['theta']:.8f} newton={st['newton_steps']} total_cg={st['total_cg']}")
    # replay: rebuild a few D's along the path is not stored; use final g and a synthetic
    # mid-path by scaling. Instead re-run IPM and hook: simplest -- test on final D and on
    # D from moderately deep steps by re-solving with fewer outer iterations.
    for tol in [1e-2, 1e-4, 1e-6, 1e-8]:
        s = solve_theta(n, R, tol=tol, precond="none")
        D = 1.0 / s["g"] ** 2
        rng = np.random.default_rng(1); b = rng.standard_normal(m)
        _, i0 = pcg(lambda v: ops.H(D, v), b, tol=1e-10, maxiter=8 * m)
        P = ActiveSetPrecond(ops, D, inner="dense")
        _, i1 = pcg(lambda v: ops.H(D, v), b, precond=P, tol=1e-10, maxiter=8 * m)
        # check FFT J_A ops against dense
        z = rng.standard_normal(m); w = rng.standard_normal(m)
        e1 = np.abs(P.JA_mv(z) - P.JA @ z).max(); e2 = np.abs(P.JAT_mv(w) - P.JA.T @ w).max()
        e3 = np.abs(P.GA_mv(z) - P.JA.T @ (P.JA @ z)).max()
        Pc = ActiveSetPrecond(ops, D, inner="cg", inner_tol=1e-13)
        _, i2 = pcg(lambda v: ops.H(D, v), b, precond=Pc, tol=1e-10, maxiter=8 * m)
        kJA = np.linalg.cond(P.JA)
        print(f"  tol={tol:.0e} Dmax={D.max():.1e} gap(D_A,min/D_I,max)={P.gap:.1e}  CG none={i0['iters']:4d}"
              f"  active(dense)={i1['iters']:3d}  active(cgnr)={i2['iters']:3d} inner mean={np.mean(Pc.inner_iters):.0f} max={np.max(Pc.inner_iters)}"
              f"  cond(J_A)={kJA:.2e}  opcheck={max(e1,e2,e3):.1e}")
