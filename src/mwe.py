"""Minimal working example: certified Lovasz theta of a random circulant graph.

Pipeline (runs on CPU or GPU, a few seconds for the default n = 211):
  1. FFT-PDHG (restarted average PDHG with matrix-free J, J^T) to a loose tolerance,
  2. active-set identification from the m largest dual coordinates,
  3. certificate J_A x = -1, J_A^T lam = -2, g >= 0, lam >= 0 (simplex crossover if it fails),
  4. rigorous floating-point enclosure theta_lo <= theta(G) <= theta_up,
  5. optional cross-check against a dense LP solve with HiGHS (scipy).

usage: python mwe.py [n] [seed] [density]
"""
import sys, time, math, numpy as np, torch
from bign import Ops, simplex_gpu, dev
from pdlp_run import fft_pdhg
from rigorous import enclosure, dots_1d

torch.set_default_dtype(torch.float64)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 211
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dens = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    assert n % 2 == 1, "n must be odd"
    nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    rng = np.random.default_rng(seed)
    R = reps[~(rng.random(nprime) < dens)]            # non-edge generator representatives
    ops = Ops(n, R); m = ops.m
    print(f"device={dev}  n={n}  m={m} non-edge classes  (edge density {dens})")

    # 1. first-order identification
    t0 = time.time()
    info, x, y = fft_pdhg(ops, eps=1e-6, maxit=200000, log=lambda s: None, return_iter=True)
    print(f"FFT-PDHG: {info['iters']} iterations, {info['ffts']} FFTs, {time.time()-t0:.2f}s, "
          f"theta_pdhg={info['theta']:.6f} (KKT residual {info['kkt']:.1e})")

    # 2. top-m dual coordinates -> candidate basis A
    score = y[1:nprime + 1].cpu().numpy()
    A0 = np.sort(reps[np.argsort(-score)[:m]])

    # 3. certificate / crossover
    t0 = time.time()
    res = simplex_gpu(ops, A0, log=lambda s: print("  " + s))
    A = res.pop("A")
    print(f"crossover: {res['pivA']} Phase-A + {res['pivB']} Phase-B pivots, {time.time()-t0:.2f}s, "
          f"certified={res['ok']}, theta={res['theta']:.10f}")

    # 4. rigorous enclosure from the certified basis
    enc = enclosure(n, n, ops.R.long(), torch.as_tensor(A, device=dev).long(), dots_1d(n), dev)
    print(f"rigorous enclosure: {enc['lo']:.12f} <= theta(G) <= {enc['up']:.12f}  (width {enc['width']:.1e})")

    # 5. optional cross-check with HiGHS on the dense LP
    try:
        from scipy.optimize import linprog
        k = np.arange(n); Jd = 2 * np.cos(2 * np.pi * np.outer(k, R) / n)
        t0 = time.time()
        r = linprog(c=-2 * np.ones(m), A_ub=-Jd, b_ub=np.ones(n), bounds=[(None, None)] * m, method="highs")
        print(f"HiGHS (dense LP): theta={1 - r.fun:.10f}, {time.time()-t0:.2f}s, "
              f"|difference| = {abs(1 - r.fun - res['theta']):.1e}")
    except ImportError:
        print("scipy not installed; skipping the HiGHS cross-check")


if __name__ == "__main__":
    main()
