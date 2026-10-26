"""Dense direct-Newton barrier method on GPU (baseline). H = J^T D J is formed by Lemma 2(c)
(Toeplitz+Hankel from one FFT of D) and solved by Cholesky. usage: python dense_ipm.py n seed out [density]"""
import sys, time, math, json, numpy as np, torch
from bign import Ops, simplex_gpu, dev
torch.set_default_dtype(torch.float64)


def dense_ipm(ops, gap_tol=1e-9, mu=10.0, log=print):
    n, m = ops.n, ops.m; R = ops.R
    rr = (R[:, None] - R[None, :]) % n; rs = (R[:, None] + R[None, :]) % n
    c = 2.0 * torch.ones(m, device=dev); x = torch.zeros(m, device=dev); g = 1.0 + ops.J(x)
    t = 1.0; steps = 0; t0 = time.time(); t_form = t_fact = 0.0
    while True:
        while True:
            D = 1.0 / g ** 2
            t1 = time.time(); a = 2.0 * torch.fft.fft(D).real; H = a[rr] + a[rs]; torch.cuda.synchronize(); t_form += time.time() - t1
            grad = -t * c - ops.JT(1.0 / g)
            t1 = time.time(); L = torch.linalg.cholesky(H); dx = -torch.cholesky_solve(grad[:, None], L)[:, 0]; torch.cuda.synchronize(); t_fact += time.time() - t1
            steps += 1; lam2 = -(grad @ dx).item()
            Jdx = ops.J(dx); s = 1.0
            f0 = (-t * (c @ x) - torch.log(g).sum()).item()
            while True:
                gn = g + s * Jdx
                if (gn > 0).all() and (-t * (c @ (x + s * dx)) - torch.log(gn).sum()).item() <= f0 + 1e-4 * s * (grad @ dx).item(): break
                s *= 0.5
                if s < 1e-12: break
            x = x + s * dx; g = gn
            if lam2 < 1e-10: break
        if n / t < gap_tol: break
        t *= mu
    theta = 1.0 + (c @ x).item()
    D = (1.0 / g ** 2)[1:(n - 1) // 2 + 1]
    A = torch.topk(D, m).indices.cpu().numpy() + 1
    return dict(theta=theta, newton=steps, time=time.time() - t0, t_form=t_form, t_fact=t_fact, gap=n / t, A=np.sort(A))


def main():
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]; dens = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    rng = np.random.default_rng(seed); nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    R = reps[~(rng.random(nprime) < dens)]; m = len(R)
    ops = Ops(n, R)
    res = dense_ipm(ops, log=lambda s: print(s, flush=True))
    print(f"n={n} m={m} dense IPM theta={res['theta']:.10f} newton={res['newton']} time={res['time']:.1f}s (form {res['t_form']:.1f}s, chol {res['t_fact']:.1f}s)", flush=True)
    t1 = time.time(); cert = simplex_gpu(ops, res["A"], log=lambda s: print("[cert]" + s, flush=True))
    tc = time.time() - t1
    rec = dict(n=n, m=m, seed=seed, density=dens, dense_ipm={k: v for k, v in res.items() if k != "A"})
    if cert is not None:
        cert.pop("A"); cert["time"] = tc; rec["certify"] = cert
        print(f"certified theta={cert['theta']:.10f} ok={cert['ok']} pivots={cert['pivA']}+{cert['pivB']} time={tc:.1f}s", flush=True)
    with open(out, "a") as f: f.write(json.dumps(rec) + chr(10))


if __name__ == "__main__":
    main()
