"""First-order (PDLP-style) baselines on the circulant theta LP  min -2 1'x  s.t.  J x >= -1.
(1) FFT-PDHG: restarted average PDHG (PDLP algorithm: adaptive restarts, primal weight) with
    matrix-free J / J^T by FFT on GPU (2 FFTs per iteration).
(2) Dense PDLP: MPAX (GPU, JAX) if installed, else OR-Tools PDLP (CPU), on the explicit dense matrix.
usage: python pdlp_run.py n seed out [--dense] [--eps 1e-4,1e-6,1e-8] [--maxit 200000]"""
import sys, time, math, json, numpy as np, torch
from bign import Ops, dev
torch.set_default_dtype(torch.float64)


def fft_pdhg(ops, eps, maxit, log=print, return_iter=False, x0=None, y0=None):
    n, m = ops.n, ops.m
    c = -2.0 * torch.ones(m, device=dev); q = -torch.ones(n, device=dev)
    normK = math.sqrt(2.0 * n); eta = 0.9 / normK; omega = 1.0
    x = torch.zeros(m, device=dev) if x0 is None else x0.clone(); y = torch.zeros(n, device=dev) if y0 is None else y0.clone()
    Kx = ops.J(x); KTy = ops.JT(y)
    xs = torch.zeros_like(x); ys = torch.zeros_like(y); cnt = 0
    x0, y0 = x.clone(), y.clone(); kkt_last = None; prev_cand = float('inf'); it_restart = 0; ffts = 0; t0 = time.time()
    nq = 1 + q.norm().item(); nc = 1 + c.norm().item()

    def kkt(xv, yv, Kxv, KTyv):
        pr = torch.clamp(q - Kxv, min=0).norm().item() / nq
        dr = (c - KTyv).norm().item() / nc
        po = (c @ xv).item(); do = (q @ yv).item()
        return math.sqrt(pr ** 2 + dr ** 2 + (abs(po - do) / (1 + abs(po) + abs(do))) ** 2), pr, dr, abs(po - do) / (1 + abs(po) + abs(do))

    for it in range(1, maxit + 1):
        tau = eta / omega; sigma = eta * omega
        xn = x - tau * (c - KTy); Kxn = ops.J(xn)
        yn = torch.clamp(y + sigma * (q - (2 * Kxn - Kx)), min=0); KTyn = ops.JT(yn); ffts += 2
        x, y, Kx, KTy = xn, yn, Kxn, KTyn
        xs += x; ys += y; cnt += 1
        if it % 64 == 0:
            xa, ya = xs / cnt, ys / cnt
            Kxa, KTya = ops.J(xa), ops.JT(ya); ffts += 2
            ka = kkt(xa, ya, Kxa, KTya); kc = kkt(x, y, Kx, KTy)
            if min(ka[0], kc[0]) <= eps:
                xf, yf = (xa, ya) if ka[0] <= kc[0] else (x, y)
                out = dict(theta=1 - (c @ xf).item(), theta_dual=1 - (q @ yf).item(), iters=it, ffts=ffts, time=time.time() - t0, kkt=min(ka[0], kc[0]), converged=True)
                return (out, xf, yf) if return_iter else out
            cand = ka if ka[0] <= kc[0] else kc
            if kkt_last is None: kkt_last = cand[0]
            do_restart = cand[0] <= 0.2 * kkt_last or (cand[0] <= 0.8 * kkt_last and cand[0] > prev_cand)
            if (it - it_restart) >= 0.36 * it: do_restart = True
            prev_cand = cand[0]
            if do_restart:
                if ka[0] <= kc[0]: x, y, Kx, KTy = xa, ya, Kxa, KTya
                dx = (x - x0).norm().item(); dy = (y - y0).norm().item()
                if dx > 1e-10 and dy > 1e-10: omega = math.exp(0.5 * math.log(dy / dx) + 0.5 * math.log(omega))
                x0, y0 = x.clone(), y.clone(); xs.zero_(); ys.zero_(); cnt = 0; kkt_last = cand[0]; it_restart = it
            if it % 5000 == 0: log(f"  it={it} kkt={cand[0]:.2e} (pr {cand[1]:.1e} dr {cand[2]:.1e} gap {cand[3]:.1e}) omega={omega:.2e} theta={1-(c@x).item():.8f} {time.time()-t0:.0f}s")
    xa, ya = (xs / cnt, ys / cnt) if cnt > 0 else (x, y)
    ka = kkt(xa, ya, ops.J(xa), ops.JT(ya))
    out = dict(theta=1 - (c @ xa).item(), theta_dual=1 - (q @ ya).item(), iters=maxit, ffts=ffts, time=time.time() - t0, kkt=ka[0], converged=False)
    return (out, xa, ya) if return_iter else out


def dense_pdlp(n, R, eps, time_limit=3600.0, log=print):
    m = len(R); k = np.arange(n); Jd = 2 * np.cos(2 * np.pi * np.outer(k, R) / n)
    try:
        import jax, jax.numpy as jnp
        from mpax import create_lp, raPDHG
        t0 = time.time()
        lp = create_lp(jnp.array(-2 * np.ones(m)), jnp.zeros((0, m)), jnp.zeros(0), jnp.array(Jd), jnp.array(-np.ones(n)), jnp.full(m, -jnp.inf), jnp.full(m, jnp.inf))
        res = raPDHG(eps_abs=eps, eps_rel=eps, verbose=False).optimize(lp)
        xv = np.asarray(res.primal_solution)
        return dict(backend="mpax", theta=1 + 2 * xv.sum(), time=time.time() - t0, iters=int(res.iteration_count), status=str(res.termination_status))
    except Exception as e:
        log(f"  mpax unavailable ({type(e).__name__}: {str(e)[:120]}); using OR-Tools PDLP")
    if n * m > 6_000_000: return dict(backend="none", theta=float("nan"), time=float("nan"), status="skipped (dense matrix too large for CPU PDLP build)")
    from ortools.linear_solver import pywraplp
    t0 = time.time(); s = pywraplp.Solver.CreateSolver("PDLP")
    s.SetSolverSpecificParametersAsString(f"termination_criteria {{ eps_optimal_absolute: {eps} eps_optimal_relative: {eps} }}")
    s.SetTimeLimit(int(time_limit * 1000))
    xv = [s.NumVar(-s.infinity(), s.infinity(), f"x{i}") for i in range(m)]
    for i in range(n):
        ct = s.Constraint(-1.0, s.infinity())
        row = Jd[i]
        for j in range(m): ct.SetCoefficient(xv[j], float(row[j]))
    obj = s.Objective()
    for j in range(m): obj.SetCoefficient(xv[j], 2.0)
    obj.SetMaximization(); tb = time.time() - t0
    st = s.Solve()
    return dict(backend="ortools_pdlp", theta=1 + obj.Value(), time=time.time() - t0, build_time=tb, status=int(st), iters=int(s.iterations()))


def main():
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]
    epss = [float(e) for e in (sys.argv[sys.argv.index("--eps") + 1] if "--eps" in sys.argv else "1e-4,1e-6,1e-8").split(",")]
    maxit = int(sys.argv[sys.argv.index("--maxit") + 1]) if "--maxit" in sys.argv else 200000
    rng = np.random.default_rng(seed); nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    R = reps[~(rng.random(nprime) < 0.5)]; m = len(R); ops = Ops(n, R)
    rec = dict(n=n, m=m, seed=seed, fft_pdhg={}, dense={})
    for eps in epss:
        r = fft_pdhg(ops, eps, maxit, log=lambda s: print(s, flush=True)); rec["fft_pdhg"][str(eps)] = r
        print(f"n={n} FFT-PDHG eps={eps:g}: theta={r['theta']:.8f} dual={r['theta_dual']:.8f} iters={r['iters']} ffts={r['ffts']} time={r['time']:.1f}s converged={r['converged']}", flush=True)
        with open(out, "a") as f: f.write(json.dumps(dict(rec, partial=True)) + chr(10))
    if "--dense" in sys.argv:
        for eps in epss:
            r = dense_pdlp(n, R, eps, log=lambda s: print(s, flush=True)); rec["dense"][str(eps)] = r
            print(f"n={n} dense PDLP [{r['backend']}] eps={eps:g}: theta={r['theta']:.8f} time={r['time']:.1f}s iters={r.get('iters')} status={r.get('status')}", flush=True)
    with open(out, "a") as f: f.write(json.dumps(rec) + chr(10))


if __name__ == "__main__":
    main()
