"""Generate labelled instances: random circulant -> exact optimal active set (certified).

For each instance we store
  n, R (non-edge reps), A (optimal active reps, |A|=m), theta, x*, lam*,
  static feature Shat(k) (edge-set cosine transform on reps 1..n'),
  prefix features: g_k and log D_k after N0 in {5,10,15,25} Newton steps of the plain FFT-IPM.
Labels are obtained with a dense-Newton IPM (exact to 1e-9) and certified by the crossover
(J_A x=-1, J_A^T lam=-2, g>=0, lam>=0).  Degenerate/uncertified instances are flagged.

usage: python gen_data.py n_list seed_start seed_end out.npz
"""
import sys, time, numpy as np
from theta_fft import random_circulant, solve_theta, ThetaOps
from crossover import certify

def features_static(ops):
    n = ops.n; nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    S = np.setdiff1d(reps, ops.R)
    s_ind = np.zeros(n); s_ind[S] = 1; s_ind[n - S] = 1
    Shat = np.fft.fft(s_ind).real[reps]           # = sum_{s in S} 2cos(2 pi k s/n)
    edge_mask = np.zeros(nprime); edge_mask[S - 1] = 1.0
    return Shat, edge_mask

def prefix_features(n, R, N0s):
    out = {}
    for N0 in N0s:
        st = solve_theta(n, R, tol=1e-30, precond="none", max_total_newton=N0, cg_tol=1e-8)
        nprime = (n - 1) // 2
        g = st["g"][1:nprime + 1]
        out[N0] = dict(g=g.astype(np.float32), fft_cost=2 * st["total_cg"] + 4 * st["newton_steps"],
                       x=st["x"].astype(np.float32))
    return out

def gen_instance(n, seed, N0s=(5, 10, 15, 25)):
    R = random_circulant(n, rng=seed); ops = ThetaOps(n, R); m = ops.m
    nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    t0 = time.time()
    st = solve_theta(n, R, tol=1e-9, precond="none", dense_newton=True)
    D = (1.0 / st["g"] ** 2)[reps]
    order = np.argsort(-D)
    A = np.sort(reps[order[:m]]); gap = D[order[m - 1]] / D[order[m]]
    c = certify(ops, A, method="dense")
    Shat, edge_mask = features_static(ops)
    pf = prefix_features(n, R, N0s)
    rec = dict(n=n, seed=seed, m=m, R=R.astype(np.int32), A=A.astype(np.int32), theta=float(c["theta"]),
               theta_ipm=float(st["theta"]), certified=bool(c["ok"]), gap=float(gap), gmin=float(c["gmin"]),
               lmin=float(c["lmin"]), x=c["x"].astype(np.float64), lam=c["lam"].astype(np.float64),
               Shat=Shat.astype(np.float32), edge_mask=edge_mask.astype(np.float32),
               newton_ref=int(st["newton_steps"]), time=time.time() - t0)
    for N0 in N0s:
        rec[f"g{N0}"] = pf[N0]["g"]; rec[f"cost{N0}"] = int(pf[N0]["fft_cost"]); rec[f"x{N0}"] = pf[N0]["x"]
    return rec

if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1].split(",")]
    s0, s1 = int(sys.argv[2]), int(sys.argv[3]); out = sys.argv[4]
    recs = []
    for n in ns:
        for seed in range(s0, s1):
            r = gen_instance(n, seed)
            recs.append(r)
            print(f"n={n} seed={seed} m={r['m']} theta={r['theta']:.6f} certified={r['certified']} gap={r['gap']:.1e} "
                  f"newton={r['newton_ref']} {r['time']:.1f}s", flush=True)
    import pickle
    with open(out, "wb") as f:
        pickle.dump(recs, f)
    print("saved", out, len(recs))
