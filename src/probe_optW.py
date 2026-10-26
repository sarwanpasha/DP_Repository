"""Probe: capacity of FFT-applicable preconditioner families on real deep-boundary
Newton systems.  Family 1: P^{-1} = J^T diag(W) J  (W >= 0, free).  Optimal W by
least squares min ||J^T W J H - I||_F.  Reports condition numbers.
"""
import numpy as np, sys
from scipy.optimize import nnls
from theta_fft import random_circulant, solve_theta, ThetaOps

def kappa(A):
    w = np.linalg.eigvalsh((A + A.T) / 2)
    return w[-1] / max(w[0], 1e-300), w

def cg_iters_est(A, tol=1e-10):
    # actual CG iteration count on A x = b for random b (unpreconditioned CG on the
    # symmetrised operator)
    from theta_fft import pcg
    rng = np.random.default_rng(0); b = rng.standard_normal(A.shape[0])
    _, info = pcg(lambda v: A @ v, b, tol=tol, maxiter=20 * A.shape[0])
    return info["iters"]

for n in [101, 211, 401]:
    R = random_circulant(n, rng=0)
    ops = ThetaOps(n, R); m = ops.m
    st = solve_theta(n, R, tol=1e-8, precond="none")
    g = st["g"]; D = 1.0 / g ** 2                     # final (deepest boundary) weights
    H = ops.H_dense(D)
    Jd = ops.J_dense()
    kH, wH = kappa(H)
    # Jacobi
    dj = np.diag(H); kJ, _ = kappa(H / np.sqrt(np.outer(dj, dj)))
    # least-squares optimal W:  sum_k W_k (j_k j_k^T H) = I
    # columns: vec(j_k j_k^T H)
    G = np.empty((m * m, n))
    for k in range(n):
        jk = Jd[k]
        G[:, k] = np.outer(jk, jk @ H).ravel()
    target = np.eye(m).ravel()
    W_ls, res_ls = np.linalg.lstsq(G, target, rcond=None)[:2]
    W_nn, rn = nnls(G, target, maxiter=50 * n)
    for name, W in [("lsq", W_ls), ("nnls", W_nn)]:
        Pinv = Jd.T @ (W[:, None] * Jd)
        # symmetric preconditioned operator Pinv^{1/2} H Pinv^{1/2} has same spectrum as Pinv H
        ev = np.linalg.eigvals(Pinv @ H).real
        ev = np.sort(ev)
        kap = ev[-1] / max(ev[0], 1e-300) if ev[0] > 0 else np.inf
        print(f"n={n} m={m}  {name}: kappa(P^-1 H)={kap:.3e}  min eig={ev[0]:.3e} max={ev[-1]:.3e}  "
              f"W range [{W.min():.2e},{W.max():.2e}]  ||P^-1H-I||_F={np.linalg.norm(Pinv@H-np.eye(m)):.3f}")
    # circulant baseline W = 1/D
    Pc = Jd.T @ ((1.0 / D)[:, None] * Jd); evc = np.sort(np.linalg.eigvals(Pc @ H).real)
    print(f"n={n} m={m}  kappa(H)={kH:.3e}  kappa(Jacobi)={kJ:.3e}  kappa(circ)={evc[-1]/evc[0]:.3e}  "
          f"CG iters(H)={cg_iters_est(H)}  Dmax={D.max():.2e} Dmin={D.min():.2e}  #D>1e3: {(D>1e3).sum()}")
    # eigenvalue spectrum shape of H
    print("   H eigs (log10 quantiles 0,.1,.25,.5,.75,.9,1):",
          np.round(np.log10(np.quantile(wH, [0, .1, .25, .5, .75, .9, 1])), 2))
    sys.stdout.flush()
