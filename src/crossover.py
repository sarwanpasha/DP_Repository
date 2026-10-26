"""Certified crossover for the circulant theta LP.

Given a candidate active set A (m frequency representatives in {1..n'}), the LP vertex is
    J_A x = -1_m,        primal;   g = 1 + J x  must be >= 0
    J_A^T lam = -2 1_m,  dual;     lam >= 0
and then theta = 1 + 2 sum x = 1 + sum lam exactly (LP duality).  Both solves use only FFT
matvecs with the cosine minor (CGNR on G_A = J_A^T J_A, or GMRES on J_A).

If the certificate fails, an active-set exchange loop swaps violated primal constraints in
and negative multipliers out, warm-starting the iterative solves.
"""
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from theta_fft import ThetaOps, pcg


class FFTCounter:
    def __init__(self): self.n_fft = 0


class Minor:
    """Cosine minor J_A (rows A of J) with FFT matvecs and iterative solves."""

    def __init__(self, ops: ThetaOps, A, counter=None):
        self.ops = ops; self.n = ops.n; self.m = ops.m
        self.A = np.asarray(sorted(A), dtype=np.int64)
        assert len(self.A) == self.m, (len(self.A), self.m)
        self.mask = np.zeros(self.n); self.mask[self.A] = 1.0; self.mask[self.n - self.A] = 1.0
        self.c = counter or FFTCounter()
        self.iters = []

    def J(self, z):
        self.c.n_fft += 1
        return self.ops.J(z)

    def JT(self, u):
        self.c.n_fft += 1
        return self.ops.JT(u)

    def JA(self, z):                       # R^m -> R^m
        return self.J(z)[self.A]

    def JAT(self, w):                      # R^m -> R^m
        u = np.zeros(self.n); u[self.A] = w; u[self.n - self.A] = w
        return 0.5 * self.JT(u)

    def GA(self, z):                       # J_A^T J_A z
        return 0.5 * self.JT(self.mask * self.J(z))

    def solve(self, w, x0=None, tol=1e-12, method="cgnr", maxiter=None):
        """z with J_A z = w."""
        maxiter = maxiter or 6 * self.m
        if method == "dense":
            JA = 2.0 * np.cos(2 * np.pi * np.outer(self.A, self.ops.R) / self.n)
            return np.linalg.solve(JA, w)
        if method == "gmres":
            cnt = [0]
            op = LinearOperator((self.m, self.m), matvec=self.JA, dtype=float)
            z, flag = gmres(op, w, x0=x0, rtol=tol, restart=min(self.m, 3000), maxiter=3,
                            callback=lambda _: cnt.__setitem__(0, cnt[0] + 1), callback_type="pr_norm")
            self.iters.append(cnt[0]); return z
        # CGNR: J_A^T J_A z = J_A^T w
        z, info = pcg(self.GA, self.JAT(w), x0=x0, tol=tol, maxiter=maxiter)
        self.iters.append(info["iters"]); return z

    def solve_T(self, v, x0=None, tol=1e-12, method="cgnr", maxiter=None):
        """y with J_A^T y = v, i.e. y = J_A (J_A^T J_A)^{-1} v."""
        maxiter = maxiter or 6 * self.m
        if method == "dense":
            JA = 2.0 * np.cos(2 * np.pi * np.outer(self.A, self.ops.R) / self.n)
            return np.linalg.solve(JA.T, v)
        if method == "gmres":
            cnt = [0]
            op = LinearOperator((self.m, self.m), matvec=self.JAT, dtype=float)
            y, flag = gmres(op, v, x0=x0, rtol=tol, restart=min(self.m, 3000), maxiter=3,
                            callback=lambda _: cnt.__setitem__(0, cnt[0] + 1), callback_type="pr_norm")
            self.iters.append(cnt[0]); return y
        z, info = pcg(self.GA, v, x0=x0, tol=tol, maxiter=maxiter)
        self.iters.append(info["iters"])
        return self.JA(z)


def certify(ops, A, method="cgnr", tol=1e-12, feas_tol=1e-7, counter=None, x0=None, lam0=None):
    """Attempt certified crossover at active set A.  Returns dict."""
    M = Minor(ops, A, counter)
    m = ops.m
    x = M.solve(-np.ones(m), x0=x0, tol=tol, method=method)
    g = 1.0 + M.J(x)
    lam = M.solve_T(-2.0 * np.ones(m), x0=lam0, tol=tol, method=method)
    theta_p = 1.0 + 2.0 * x.sum(); theta_d = 1.0 + lam.sum()
    gmin = g.min(); lmin = lam.min()
    ok = gmin >= -feas_tol * max(1.0, abs(theta_p)) and lmin >= -feas_tol * max(1.0, abs(lam).max()) \
        and abs(theta_p - theta_d) <= 1e-6 * max(1.0, abs(theta_p))
    return dict(ok=bool(ok), theta=theta_p, theta_d=theta_d, x=x, g=g, lam=lam, gmin=gmin, lmin=lmin,
                minor=M, iters=list(M.iters))


def exchange_loop(ops, A0, max_swaps=50, method="cgnr", tol=1e-12, feas_tol=1e-7, counter=None,
                  verbose=False, max_batch=8):
    """Active-set exchange: enter most violated primal constraints, leave most negative duals."""
    n = ops.n; nprime = (n - 1) // 2
    A = np.array(sorted(A0), dtype=np.int64)
    hist = []
    x0 = None; lam0 = None
    for it in range(max_swaps + 1):
        c = certify(ops, A, method=method, tol=tol, feas_tol=feas_tol, counter=counter, x0=x0, lam0=lam0)
        hist.append(dict(swap=it, gmin=c["gmin"], lmin=c["lmin"], theta=c["theta"], iters=c["iters"]))
        if verbose:
            print(f"   exchange {it}: gmin={c['gmin']:.2e} lmin={c['lmin']:.2e} theta={c['theta']:.8f} "
                  f"inner iters={c['iters']}")
        if c["ok"] or it == max_swaps:
            c["swaps"] = it; c["hist"] = hist; c["A"] = A
            return c
        g = c["g"][1:nprime + 1]          # reps 1..n'
        lam = c["lam"]                     # indexed by sorted A
        inA = np.zeros(nprime + 1, bool); inA[A] = True
        cand_in = np.array([k for k in np.argsort(g) + 1 if not inA[k] and g[k - 1] < -feas_tol])
        cand_out_idx = np.argsort(lam)
        cand_out = np.array([A[i] for i in cand_out_idx if lam[i] < -feas_tol])
        q = max(1, min(len(cand_in), len(cand_out), max_batch))
        if len(cand_in) == 0:               # dual infeasible only: replace most negative lam by least-positive-g outside
            cand_in = np.array([k for k in np.argsort(g) + 1 if not inA[k]])
        if len(cand_out) == 0:              # primal infeasible only: drop smallest multipliers
            cand_out = np.array([A[i] for i in cand_out_idx])
        q = max(1, min(q, len(cand_in), len(cand_out)))
        A = np.array(sorted(set(A.tolist()) - set(cand_out[:q].tolist()) | set(cand_in[:q].tolist())), dtype=np.int64)
        x0 = c["x"]; lam0 = c["lam"]


def predict_static(ops):
    """Heuristic predictor: top-m of the edge-set cosine transform S^(k)."""
    n = ops.n; nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    S = np.setdiff1d(reps, ops.R)
    s_ind = np.zeros(n); s_ind[S] = 1; s_ind[n - S] = 1
    Shat = np.fft.fft(s_ind).real[reps]
    return reps[np.argsort(-Shat)[:ops.m]], Shat


def predict_prefix(n, R, n_newton, counter=None):
    """Run the plain FFT-IPM for a fixed number of Newton steps; return top-m of D and the state."""
    from theta_fft import solve_theta
    st = solve_theta(n, R, tol=1e-30, precond="none", max_total_newton=n_newton)
    D = 1.0 / st["g"] ** 2
    nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    A = reps[np.argsort(-D[reps])[:len(R)]]
    if counter is not None:
        counter.n_fft += 2 * st["total_cg"] + 4 * st["newton_steps"]
    return A, st


if __name__ == "__main__":
    import sys, time
    from theta_fft import random_circulant, solve_theta
    ns = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [211, 401, 809]
    for n in ns:
        for seed in range(2):
            R = random_circulant(n, rng=seed); ops = ThetaOps(n, R); m = ops.m
            t0 = time.time()
            ref = solve_theta(n, R, tol=1e-9, precond="none", dense_newton=True) if n <= 1601 else None
            th_ref = ref["theta"] if ref else float("nan")
            print(f"n={n} m={m} seed={seed} theta_ref={th_ref:.8f} (dense IPM {time.time()-t0:.1f}s, "
                  f"newton={ref['newton_steps'] if ref else '-'})")
            # (a) static predictor
            cnt = FFTCounter(); A0, _ = predict_static(ops)
            t0 = time.time(); c = exchange_loop(ops, A0, counter=cnt, max_swaps=60)
            print(f"   static : ok={c['ok']} swaps={c['swaps']} theta={c['theta']:.8f} err={abs(c['theta']-th_ref):.1e} "
                  f"FFTs={cnt.n_fft} inner-iters(last)={c['iters']} time={time.time()-t0:.1f}s")
            # (b) IPM prefix predictors
            for N0 in [15, 25, 40]:
                cnt = FFTCounter(); t0 = time.time()
                A0, st = predict_prefix(n, R, N0, counter=cnt); fft_prefix = cnt.n_fft
                c = exchange_loop(ops, A0, counter=cnt, max_swaps=60)
                print(f"   prefix{N0:3d}: ok={c['ok']} swaps={c['swaps']} theta={c['theta']:.8f} err={abs(c['theta']-th_ref):.1e} "
                      f"FFTs={cnt.n_fft} (prefix {fft_prefix}) inner-iters(last)={c['iters']} time={time.time()-t0:.1f}s")
