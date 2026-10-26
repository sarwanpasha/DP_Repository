import sys, time, math, json, numpy as np, torch
torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ops:
    def __init__(self, n, R):
        self.n = n; self.R = torch.as_tensor(R, device=dev); self.nR = n - self.R; self.m = len(R)
    def J(self, v):
        vt = torch.zeros(self.n, device=dev); vt.index_add_(0, self.R, v); vt.index_add_(0, self.nR, v)
        return torch.fft.fft(vt).real
    def JT(self, u):
        return 2.0 * torch.fft.fft(u).real[self.R]
    def H(self, D, v): return self.JT(D * self.J(v))


def cg(matvec, b, tol=1e-8, maxiter=5000):
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rr = r @ r; bn = math.sqrt(rr.item()) + 1e-300
    for it in range(maxiter):
        Ap = matvec(p); a = rr / (p @ Ap); x += a * p; r -= a * Ap; rr_new = r @ r
        if math.sqrt(rr_new.item()) / bn < tol: return x, it + 1
        p = r + (rr_new / rr) * p; rr = rr_new
    return x, maxiter


def prefix(ops, N0, mu=10.0):
    m = ops.m; c = 2.0 * torch.ones(m, device=dev); x = torch.zeros(m, device=dev); g = 1.0 + ops.J(x); t = 1.0; steps = 0; ffts = 0
    while steps < N0:
        for _ in range(200):
            D = 1.0 / g ** 2; grad = -t * c - ops.JT(1.0 / g); rhs = -grad
            dx, it = cg(lambda v: ops.H(D, v), rhs); ffts += 2 * it + 4
            lam2 = (rhs @ dx).item(); steps += 1
            if lam2 / 2 < 1e-9 or steps >= N0: break
            Jdx = ops.J(dx); neg = Jdx < 0
            smax = torch.min(-g[neg] / Jdx[neg]).item() if neg.any() else float("inf"); s = min(1.0, 0.99 * smax)
            f0 = (-t * (c @ x) - torch.log(g).sum()).item()
            while True:
                gn = g + s * Jdx
                if (gn > 0).all() and (-t * (c @ (x + s * dx)) - torch.log(gn).sum()).item() <= f0 + 1e-4 * s * (grad @ dx).item(): break
                s *= 0.5
                if s < 1e-12: break
            x = x + s * dx; g = gn
        if steps >= N0: break
        t *= mu
    return x, g, ffts


def predict(ckpt, n, R, Shat, g10, x10):
    sys.path.insert(0, "."); from evaluate import load_model, predict_learned
    model, a = load_model(ckpt, dev)
    rec = dict(n=n, m=len(R), R=np.asarray(R), Shat=Shat.astype(np.float32), A=np.array([1]),
               g10=g10.astype(np.float32), x10=x10.astype(np.float32), cost10=0)
    return predict_learned(model, a, rec, dev)


def simplex_gpu(ops, A0, max_pivots=None, feas_tol=1e-9, eps=1e-9, log=print):
    n, m = ops.n, ops.m; nprime = (n - 1) // 2; max_pivots = max_pivots or 6 * m
    R = ops.R.double()
    A = torch.as_tensor(np.sort(np.asarray(A0)), device=dev)
    def row(k): return 2.0 * torch.cos(2 * math.pi * k * R / n)
    JA = 2.0 * torch.cos(2 * math.pi * torch.outer(A.double(), R) / n)
    Binv = torch.linalg.inv(JA); nupd = 0
    c = 2.0 * torch.ones(m, device=dev); ones = torch.ones(m, device=dev)
    def swap(k_out, k_in):
        nonlocal A, JA, Binv, nupd
        i = int((A == k_out).nonzero()[0]); u = row(k_in) - JA[i]
        Bei = Binv[:, i].clone(); uB = u @ Binv; den = 1.0 + uB[i].item()
        if nupd < 30 and abs(den) > 1e-10:
            Binv -= torch.outer(Bei, uB) / den; JA[i] = row(k_in); A = A.clone(); A[i] = k_in; nupd += 1
        else:
            A = A.clone(); A[i] = k_in; JA = 2.0 * torch.cos(2 * math.pi * torch.outer(A.double(), R) / n); Binv = torch.linalg.inv(JA); nupd = 0
    x = -(Binv @ ones); lam = -2.0 * (Binv.T @ ones); g = 1.0 + ops.J(x)
    pivA = pivB = 0; t0 = time.time()
    lam_t = torch.clamp(lam, min=eps)
    inA = torch.zeros(n, dtype=torch.bool, device=dev)
    while True:
        inA.zero_(); inA[A] = True; inA[n - A] = True
        viol = (g < -feas_tol) & ~inA; viol[0] = False
        if not viol.any(): break
        if pivA + pivB >= max_pivots: return None
        kf = int(torch.argmin(torch.where(viol, g, torch.full_like(g, float("inf"))))); k_in = min(kf, n - kf)
        w = Binv.T @ row(k_in); pos = w > 1e-14
        if pos.any():
            ratios = torch.where(pos, lam_t / torch.where(pos, w, torch.ones_like(w)), torch.full_like(w, float("inf"))); i_out = int(torch.argmin(ratios))
        else: i_out = int(torch.argmin(lam_t))
        t = (lam_t[i_out] / w[i_out]).item() if w[i_out].item() > 1e-14 else 0.0
        lam_t = lam_t - t * w; lam_t[i_out] = t; lam_t = torch.clamp(lam_t, min=0.0)
        swap(int(A[i_out]), k_in)
        x = -(Binv @ ones); g = 1.0 + ops.J(x); pivA += 1
        if pivA % 200 == 0: log(f"   phaseA pivots={pivA} viol={int(viol.sum())} gmin={g.min().item():.2e} ({time.time()-t0:.0f}s)")
    lam = -2.0 * (Binv.T @ ones)
    while True:
        neg = lam < -feas_tol
        if not neg.any(): break
        if pivA + pivB >= max_pivots: return None
        i = int(torch.argmin(torch.where(neg, lam, torch.full_like(lam, float("inf")))))
        d = Binv[:, i].clone(); Jd = ops.J(d)
        inA.zero_(); inA[A] = True; inA[n - A] = True
        blocking = (Jd < -1e-14) & ~inA; blocking[0] = False
        if not blocking.any(): return None
        ratios = torch.where(blocking, -g / torch.where(blocking, Jd, torch.ones_like(Jd)), torch.full_like(g, float("inf")))
        kf = int(torch.argmin(ratios)); tt = ratios[kf].item(); k_in = min(kf, n - kf)
        x = x + tt * d; g = g + tt * Jd
        swap(int(A[i]), k_in); lam = -2.0 * (Binv.T @ ones); pivB += 1
        if pivB % 200 == 0: log(f"   phaseB pivots={pivB} lam_min={lam.min().item():.2e} ({time.time()-t0:.0f}s)")
    theta = 1.0 + (c @ x).item(); g = 1.0 + ops.J(x)
    ok = g.min().item() >= -1e-7 and lam.min().item() >= -1e-7 and abs(theta - (1 + lam.sum().item())) < 1e-6 * theta
    return dict(theta=theta, theta_dual=1 + lam.sum().item(), gmin=g.min().item(), lmin=lam.min().item(), ok=bool(ok), pivA=pivA, pivB=pivB, A=A.cpu().numpy())


def main():
    n = int(sys.argv[1]); seed = int(sys.argv[2]); N0 = sys.argv[3]; ckpt = sys.argv[4]; out = sys.argv[5]
    rng = np.random.default_rng(seed); nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    R = reps[~(rng.random(nprime) < 0.5)]; m = len(R); S = np.setdiff1d(reps, R)
    ops = Ops(n, R); T = {}; t0 = time.time()
    s_ind = torch.zeros(n, device=dev); s_ind[torch.as_tensor(S, device=dev)] = 1; s_ind[n - torch.as_tensor(S, device=dev)] = 1
    Shat = torch.fft.fft(s_ind).real[1:nprime + 1].cpu().numpy()
    x10, g10, ffts10 = prefix(ops, 10); T["prefix10"] = time.time() - t0; t1 = time.time()
    g10r = g10[1:nprime + 1].cpu().numpy()
    results = dict(n=n, m=m, seed=seed, ffts_prefix10=ffts10, t_prefix10=T["prefix10"], methods={})
    cands = {}
    if ckpt != "none":
        cands["equiv_p10_L"] = predict(ckpt, n, R, Shat, g10r, x10.cpu().numpy()); T["predict"] = time.time() - t1
    for N in [int(a) for a in str(N0).split("+")]:
        t1 = time.time(); xN, gN, fftsN = prefix(ops, N); T[f"prefix{N}"] = time.time() - t1
        D = (1.0 / gN ** 2)[1:nprime + 1].cpu().numpy(); cands[f"prefix{N}"] = np.sort(reps[np.argsort(-D)[:m]])
        results[f"ffts_prefix{N}"] = fftsN; results[f"t_prefix{N}"] = T[f"prefix{N}"]
        print(f"n={n} m={m} prefix{N} {T[f'prefix{N}']:.0f}s ({fftsN} FFTs)", flush=True)
    for name, A0 in cands.items():
        t1 = time.time(); res = simplex_gpu(ops, A0, log=lambda s: print(f"[{name}]" + s, flush=True))
        if res is None: results["methods"][name] = dict(ok=False); print(f"[{name}] FAILED", flush=True); continue
        res["time"] = time.time() - t1; res.pop("A"); results["methods"][name] = res
        print(f"[{name}] theta={res['theta']:.8f} ok={res['ok']} pivots={res['pivA']}+{res['pivB']} gmin={res['gmin']:.1e} lmin={res['lmin']:.1e} time={res['time']:.0f}s", flush=True)
    if "--highs" in sys.argv:
        from scipy.optimize import linprog
        t1 = time.time(); k = np.arange(n); Jd = 2 * np.cos(2 * np.pi * np.outer(k, R) / n)
        r = linprog(c=-2 * np.ones(m), A_ub=-Jd, b_ub=np.ones(n), bounds=[(None, None)] * m, method="highs")
        results["highs"] = dict(theta=1 - r.fun, time=time.time() - t1); print(f"HiGHS theta={1-r.fun:.8f} time={time.time()-t1:.0f}s", flush=True)
    with open(out, "a") as f: f.write(json.dumps(results) + chr(10))


if __name__ == "__main__":
    main()
