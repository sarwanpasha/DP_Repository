"""How predictable is the LP active set A (support of optimal dual) from cheap features?
Features per frequency k: (i) edge-set Fourier transform  S^(k) = sum_{s in S} cos(2 pi k s/n)
(ii) g_k at IPM iterates of increasing depth.  Report ROC-AUC and top-m precision."""
import sys, numpy as np
from theta_fft import random_circulant, solve_theta, ThetaOps, theta_reference

def auc(score, label):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(len(score))
    pos = label.astype(bool); npos = pos.sum(); nneg = len(label) - npos
    return (ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg)

ns = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [101, 211, 401]
for n in ns:
    for seed in range(3):
        R = random_circulant(n, rng=seed); ops = ThetaOps(n, R); m = ops.m
        nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
        # exact active set from a deep IPM solve (dense Newton, exact)
        s = solve_theta(n, R, tol=1e-9, precond="none", dense_newton=True)
        D = 1.0 / s["g"] ** 2
        Dr = D[reps]; order = np.argsort(-Dr)
        label = np.zeros(nprime, bool); label[order[:m]] = True
        gap = Dr[order[m - 1]] / Dr[order[m]]
        # feature 1: edge-set transform
        S = np.setdiff1d(reps, R)
        Shat = np.array([np.sum(2 * np.cos(2 * np.pi * k * S / n)) for k in reps])
        # feature 2: non-edge transform (equivalently -Shat - 1 since total sum = -1)
        # feature 3: early IPM iterates
        out = [f"n={n} m={m} seed={seed} |S|={len(S)} gap={gap:.1e}  AUC(Shat)={auc(Shat, label):.3f}"]
        topm = np.argsort(-Shat)[:m]; out.append(f"top-m prec(Shat)={label[topm].mean():.3f}")
        for tol in [1e2, 1e1, 1e0, 1e-1, 1e-2]:
            se = solve_theta(n, R, tol=tol, precond="none", dense_newton=True)
            De = (1.0 / se["g"] ** 2)[reps]
            topm = np.argsort(-De)[:m]
            out.append(f"tol={tol:.0e}[N={se['newton_steps']}] AUC={auc(De, label):.3f} prec={label[topm].mean():.3f}")
        print("  ".join(out), flush=True)
