"""Theta of Cayley graphs of Z_n x Z_n (odd n) with the same pipeline: FFT-PDHG identification,
certificate, simplex crossover, rigorous enclosure.  Constraints are indexed by k in Z_n^2 (N = n^2,
rows k and -k coincide), variables by the non-edge generator classes {r, -r}.
usage: python cayley2d.py n seed out [--density p] [--eps 1e-6] [--rigorous]"""
import sys, time, math, json, numpy as np, torch
torch.set_default_dtype(torch.float64)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ops2D:
    def __init__(self, n, R):  # R: long tensor [m,2] of generator representatives
        self.n0 = n; self.N = n * n; self.n = self.N; self.m = len(R)
        self.R = torch.as_tensor(R, device=dev).long(); self.nR = (-self.R) % n
        self.ridx = self.R[:, 0] * n + self.R[:, 1]; self.nridx = self.nR[:, 0] * n + self.nR[:, 1]

    def J(self, v):
        vt = torch.zeros(self.N, device=dev); vt.index_add_(0, self.ridx, v); vt.index_add_(0, self.nridx, v)
        return torch.fft.fft2(vt.view(self.n0, self.n0)).real.reshape(-1)

    def JT(self, u):
        return 2.0 * torch.fft.fft2(u.view(self.n0, self.n0)).real.reshape(-1)[self.ridx]


def reps2d(n):
    """Representatives of the classes {k, -k} of Z_n^2 \\ {0}: all k=(a,b) with a < n/2, or a==0 and 0<b<n/2."""
    a, b = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = (a < (n + 1) // 2) & ~((a == 0) & (b >= (n + 1) // 2)) & ~((a == 0) & (b == 0))
    return torch.stack([a[keep], b[keep]], 1)


def dots_2d(n):
    def f(kidx, Rvec):  # kidx: flattened indices into Z_n^2; Rvec: [m,2]
        ka, kb = kidx // n, kidx % n
        return (ka[:, None] * Rvec[None, :, 0] + kb[:, None] * Rvec[None, :, 1]) % n
    return f


def simplex2d(ops, A0, feas_tol=1e-9, eps=1e-9, max_pivots=None, log=print):
    """Phase A dual simplex on the perturbed cost, Phase B primal simplex; A0 = flattened indices of m constraints."""
    n, N, m = ops.n0, ops.N, ops.m; max_pivots = max_pivots or 6 * m
    dots = dots_2d(n); Rl = ops.R
    def rows(kidx): return 2.0 * torch.cos(2 * math.pi * dots(kidx, Rl).double() / n)
    A = torch.as_tensor(np.sort(np.asarray(A0)), device=dev).long()
    JA = rows(A); Binv = torch.linalg.inv(JA); nupd = 0
    ones = torch.ones(m, device=dev); c = 2.0 * ones
    negidx = lambda k: ((-(k // n)) % n) * n + ((-(k % n)) % n)

    def swap(i, k_in):
        nonlocal A, JA, Binv, nupd
        u = rows(torch.tensor([k_in], device=dev))[0] - JA[i]
        Bei = Binv[:, i].clone(); uB = u @ Binv; den = 1.0 + uB[i].item()
        if nupd < 30 and abs(den) > 1e-10:
            Binv -= torch.outer(Bei, uB) / den; JA[i] = JA[i] + u; A = A.clone(); A[i] = k_in; nupd += 1
        else:
            A = A.clone(); A[i] = k_in; JA = rows(A); Binv = torch.linalg.inv(JA); nupd = 0

    def inmask():
        msk = torch.zeros(N, dtype=torch.bool, device=dev); msk[A] = True; msk[negidx(A)] = True; msk[0] = True; return msk

    x = -(Binv @ ones); lam = -2.0 * (Binv.T @ ones); g = 1.0 + ops.J(x)
    pivA = pivB = 0; t0 = time.time()
    if g.min().item() >= -feas_tol and lam.min().item() >= -eps:
        return dict(theta=1 + (c @ x).item(), theta_dual=1 + lam.sum().item(), gmin=g.min().item(), lmin=lam.min().item(), ok=True, pivA=0, pivB=0, A=A.cpu().numpy())
    lam_t = torch.clamp(lam, min=eps)
    while g.min().item() < -feas_tol:  # Phase A
        if pivA + pivB >= max_pivots: return None
        k = int(torch.argmin(g)); w = Binv.T @ rows(torch.tensor([k], device=dev))[0]
        pos = w > 1e-12
        if not pos.any(): return None
        ratios = torch.where(pos, lam_t / torch.where(pos, w, torch.ones_like(w)), torch.full_like(w, float("inf")))
        i = int(torch.argmin(ratios)); t = ratios[i].item()
        lam_t = lam_t - t * w; lam_t[i] = t
        swap(i, k); x = -(Binv @ ones); g = 1.0 + ops.J(x); pivA += 1
        if pivA % 100 == 0: log(f" phaseA pivots={pivA} gmin={g.min().item():.2e} ({time.time()-t0:.0f}s)")
    lam = -2.0 * (Binv.T @ ones)
    while lam.min().item() < -eps:  # Phase B
        if pivA + pivB >= max_pivots: return None
        i = int(torch.argmin(lam)); d = Binv[:, i].clone(); Jd = ops.J(d)
        if (c @ d).item() < 0: d = -d; Jd = -Jd
        blocking = (Jd < -1e-14) & ~inmask()
        if not blocking.any(): return None
        ratios = torch.where(blocking, -g / torch.where(blocking, Jd, torch.ones_like(Jd)), torch.full_like(g, float("inf")))
        kf = int(torch.argmin(ratios)); tt = ratios[kf].item(); k_in = min(kf, negidx(kf))
        x = x + tt * d; g = g + tt * Jd
        swap(i, k_in); lam = -2.0 * (Binv.T @ ones); pivB += 1
        if pivB % 100 == 0: log(f" phaseB pivots={pivB} lmin={lam.min().item():.2e} ({time.time()-t0:.0f}s)")
    theta = 1.0 + (c @ x).item(); g = 1.0 + ops.J(x)
    ok = g.min().item() >= -1e-7 and lam.min().item() >= -1e-7 and abs(theta - (1 + lam.sum().item())) < 1e-6 * theta
    return dict(theta=theta, theta_dual=1 + lam.sum().item(), gmin=g.min().item(), lmin=lam.min().item(), ok=bool(ok), pivA=pivA, pivB=pivB, A=A.cpu().numpy())


def main():
    n = int(sys.argv[1]); seed = int(sys.argv[2]); out = sys.argv[3]
    dens = float(sys.argv[sys.argv.index("--density") + 1]) if "--density" in sys.argv else 0.5
    epss = [float(e) for e in (sys.argv[sys.argv.index("--eps") + 1] if "--eps" in sys.argv else "1e-6").split(",")]
    reps = reps2d(n); rng = np.random.default_rng(seed)
    edge = rng.random(len(reps)) < dens; R = reps[~torch.as_tensor(edge)]
    ops = Ops2D(n, R); m = ops.m; N = ops.N
    kreps = torch.arange(N, device=dev); kmin = torch.minimum(kreps, ((-(kreps // n)) % n) * n + ((-(kreps % n)) % n))
    rep_mask = (kreps == kmin) & (kreps > 0)   # one representative per class {k,-k}, k != 0
    rec = dict(n=n, N=N, m=m, seed=seed, density=dens, methods={})
    print(f"Z_{n}xZ_{n}: N={N} constraints, m={m} generators, density={dens} seed={seed}", flush=True)
    from pdlp_run import fft_pdhg
    for eps in epss:
        t0 = time.time(); info, xp, yp = fft_pdhg(ops, eps, 200000, log=lambda s: None, return_iter=True); tp = time.time() - t0
        sc = torch.where(rep_mask, yp, torch.full_like(yp, -1.0)); A0 = torch.topk(sc, m).indices.cpu().numpy()
        print(f"pdhg{eps:g}: {tp:.1f}s {info['iters']} iters {info['ffts']} FFTs theta_pdhg={info['theta']:.8f}", flush=True)
        t1 = time.time()
        try: res = simplex2d(ops, A0, log=lambda s: print(f"[pdhg{eps:g}]" + s, flush=True))
        except Exception as e: print(f"exception {e}", flush=True); res = None
        tc = time.time() - t1
        entry = dict(t_prefix=tp, ffts_prefix=info["ffts"], iters=info["iters"], theta_pdhg=info["theta"], kkt=info["kkt"], time=tc)
        if res is None: entry["ok"] = False; print(f"[pdhg{eps:g}] FAILED", flush=True)
        else:
            A = res.pop("A"); entry.update(res)
            print(f"[pdhg{eps:g}] theta={res['theta']:.10f} ok={res['ok']} pivots={res['pivA']}+{res['pivB']} time={tc:.0f}s theta/n={res['theta']/n:.5f}", flush=True)
            if "--rigorous" in sys.argv:
                from rigorous import enclosure
                t2 = time.time(); enc = enclosure(n, N, ops.R, torch.as_tensor(A, device=dev).long(), dots_2d(n), dev); enc["time"] = time.time() - t2
                entry["rigorous"] = enc; print(f"[pdhg{eps:g}] rigorous: {enc['lo']:.12f} <= theta <= {enc['up']:.12f} (width {enc['width']:.1e})", flush=True)
        rec["methods"][f"pdhg{eps:g}"] = entry
    with open(out, "a") as f: f.write(json.dumps(rec) + chr(10))


if __name__ == "__main__":
    main()
