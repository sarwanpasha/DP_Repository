"""Primal-dual Mehrotra predictor-corrector interior-point method on GPU for the circulant theta LP,
written as  min c'x  s.t.  J x - s = q, s >= 0  with c = -2*1, q = -1.  The normal matrix J' D J
(D = y/s) is formed by the FFT identity of Lemma 2(c) and factored by Cholesky.
usage: python pdipm.py n seed out [density]"""
import sys, time, json, numpy as np, torch
from bign import Ops, simplex_gpu, dev
torch.set_default_dtype(torch.float64)


def pdipm(ops, tol=1e-9, maxit=200, log=print):
    n, m = ops.n, ops.m; R = ops.R
    rr = (R[:, None] - R[None, :]) % n; rs = (R[:, None] + R[None, :]) % n
    c = -2.0 * torch.ones(m, device=dev); q = -torch.ones(n, device=dev)
    x = torch.zeros(m, device=dev); s = torch.ones(n, device=dev); y = torch.ones(n, device=dev)
    t0 = time.time(); t_fact = 0.0; it = 0
    nq = 1 + q.norm().item(); nc = 1 + c.norm().item()
    while it < maxit:
        rp = q - ops.J(x) + s; rd = c - ops.JT(y); mu = (s @ y).item() / n
        gap = abs((c @ x).item() - (q @ y).item()) / (1 + abs((c @ x).item()))
        if rp.norm().item() / nq < tol and rd.norm().item() / nc < tol and gap < tol: break
        it += 1
        D = y / s
        t1 = time.time(); a = 2.0 * torch.fft.fft(D).real; M = a[rr] + a[rs]
        L = torch.linalg.cholesky(M); torch.cuda.synchronize(); t_fact += time.time() - t1

        def solve(rc):
            rhs = ops.JT((rc + y * rp) / s) - rd
            dx = torch.cholesky_solve(rhs[:, None], L)[:, 0]
            ds = ops.J(dx) - rp; dy = (rc - y * ds) / s
            return dx, ds, dy

        def maxstep(v, dv):
            neg = dv < 0
            return min(1.0, (0.99 * (-v[neg] / dv[neg]).min().item()) if neg.any() else 1.0)

        dxa, dsa, dya = solve(-s * y)
        ap, ad = maxstep(s, dsa), maxstep(y, dya)
        mua = ((s + ap * dsa) @ (y + ad * dya)).item() / n
        sigma = (mua / mu) ** 3
        dx, ds, dy = solve(sigma * mu - s * y - dsa * dya)
        ap, ad = maxstep(s, ds), maxstep(y, dy)
        x = x + ap * dx; s = s + ap * ds; y = y + ad * dy
        if it % 10 == 0: log(f"  it={it} mu={mu:.2e} rp={rp.norm().item():.1e} rd={rd.norm().item():.1e} gap={gap:.1e} ({time.time()-t0:.0f}s)")
    theta = 1 - (c @ x).item()
    yr = y[1:(n - 1) // 2 + 1]
    A = torch.topk(yr, m).indices.cpu().numpy() + 1
    return dict(theta=theta, theta_dual=1 - (q @ y).item(), iters=it, time=time.time() - t0, t_fact=t_fact, A=np.sort(A), converged=it < maxit)


def main():
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]; dens = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    rng = np.random.default_rng(seed); nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    R = reps[~(rng.random(nprime) < dens)]; m = len(R); ops = Ops(n, R)
    res = pdipm(ops, log=lambda s: print(s, flush=True))
    print(f"n={n} m={m} PD-IPM theta={res['theta']:.10f} dual={res['theta_dual']:.10f} iters={res['iters']} time={res['time']:.1f}s (chol {res['t_fact']:.1f}s) converged={res['converged']}", flush=True)
    t1 = time.time(); cert = simplex_gpu(ops, res["A"], log=lambda s: print("[cert]" + s, flush=True)); tc = time.time() - t1
    rec = dict(n=n, m=m, seed=seed, density=dens, pdipm={k: v for k, v in res.items() if k != "A"})
    if cert is not None:
        cert.pop("A"); cert["time"] = tc; rec["certify"] = cert
        print(f"certified theta={cert['theta']:.10f} ok={cert['ok']} pivots={cert['pivA']}+{cert['pivB']} time={tc:.1f}s", flush=True)
    with open(out, "a") as f: f.write(json.dumps(rec) + chr(10))


if __name__ == "__main__":
    main()
