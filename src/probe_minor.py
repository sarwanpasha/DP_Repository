"""Scaling of the cosine minor J_A: condition number, singular value profile, and inner
solver iteration counts (CGNR on G_A, GMRES on J_A), vs n."""
import sys, json, time
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from theta_fft import random_circulant, solve_theta, ThetaOps, pcg
from active_precond import ActiveSetPrecond

ns = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [211, 401, 809]
tol_path = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4   # how deep along the path
for n in ns:
    for seed in range(2):
        R = random_circulant(n, rng=seed)
        ops = ThetaOps(n, R); m = ops.m
        t0 = time.time()
        # get a deep-path D cheaply: run IPM with dense Newton (exact) to moderate tol
        s = solve_theta(n, R, tol=tol_path, precond="none", dense_newton=True)
        D = 1.0 / s["g"] ** 2
        P = ActiveSetPrecond(ops, D, inner="dense")
        sv = np.linalg.svd(P.JA, compute_uv=False)
        kap = sv[0] / sv[-1]
        rng = np.random.default_rng(0); w = rng.standard_normal(m)
        # CGNR iterations
        Pc = ActiveSetPrecond(ops, D, inner="cg", inner_tol=1e-12)
        Pc.solve_JA(w); it_cgnr = Pc.inner_iters[-1]
        # GMRES on J_A directly (restart-free)
        cnt = [0]
        def cb(_): cnt[0] += 1
        A = LinearOperator((m, m), matvec=Pc.JA_mv, dtype=float)
        z, flag = gmres(A, w, rtol=1e-12, restart=m, maxiter=1, callback=cb, callback_type="pr_norm")
        it_gm = cnt[0]
        # outer PCG on H with exact P
        b = rng.standard_normal(m)
        _, io = pcg(lambda v: ops.H(D, v), b, precond=P, tol=1e-10, maxiter=8 * m)
        q = np.quantile(sv / sv[0], [0, .01, .05, .1, .25, .5])
        print(f"n={n:5d} m={m:4d} seed={seed} Dmax={D.max():.1e} gap={P.gap:.1e} cond(J_A)={kap:8.1f} "
              f"sv/svmax quantiles[0,1,5,10,25,50%]={np.round(q,4)} "
              f"CGNR={it_cgnr:4d} GMRES={it_gm:4d} outerPCG={io['iters']} ({time.time()-t0:.0f}s)", flush=True)
