"""Singular-value law of the optimal cosine minor J_A across n (uses certified test instances)."""
import sys, glob, pickle, json, numpy as np
out = open(sys.argv[2] if len(sys.argv) > 2 else "results/minor_spectrum.jsonl", "a")
for fn in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "data/test_n*.pkl")):
    for r in pickle.load(open(fn, "rb")):
        if not r["certified"]: continue
        n, m = r["n"], r["m"]
        JA = 2.0 * np.cos(2 * np.pi * np.outer(r["A"], r["R"]) / n)
        sv = np.linalg.svd(JA, compute_uv=False) / np.sqrt(2 * n)      # normalise by ||J||
        q = np.quantile(sv, [0, .01, .05, .1, .25, .5, .75, .9, 1]).tolist()
        # active set statistics
        A = r["A"]; S = np.setdiff1d(np.arange(1, (n - 1) // 2 + 1), r["R"])
        rec = dict(n=n, m=m, seed=r["seed"], cond=float(sv[0] / sv[-1]), sv_quantiles=q,
                   frac_active_in_edges=float(np.isin(A, S).mean()), theta=r["theta"],
                   lam_min=float(r["lam"].min()), lam_max=float(r["lam"].max()), lam_mean=float(r["lam"].mean()),
                   x_absmax=float(np.abs(r["x"]).max()))
        out.write(json.dumps(rec) + "\n")
        print(f"n={n} m={m} seed={r['seed']} cond(J_A)={rec['cond']:.1f} sv_min={sv[-1]:.4f} sv_med={q[5]:.3f} "
              f"lam[min,mean,max]=[{rec['lam_min']:.3f},{rec['lam_mean']:.2f},{rec['lam_max']:.2f}]", flush=True)
