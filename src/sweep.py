"""Certified pipeline on one circulant instance: identification (barrier prefix, PDHG dual, or
learned-basis warm-started PDHG) followed by the GPU simplex crossover, optional rigorous enclosure.
usage: python sweep.py n seed N0s out [--density p] [--paley] [--pdhg eps1,eps2] [--warm ckpt]
       [--rigorous] [--highs]
N0s: '+'-separated barrier prefix depths, or 0 to skip the barrier prefix."""
import sys, time, math, json, numpy as np, torch
from bign import Ops, prefix, simplex_gpu, dev
torch.set_default_dtype(torch.float64)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def run_crossover(ops, A0, tag):
    t1 = time.time()
    try: res = simplex_gpu(ops, A0, log=lambda s: print(f"[{tag}]" + s, flush=True))
    except Exception as e: print(f"[{tag}] exception {e}", flush=True); res = None
    tc = time.time() - t1
    if res is None: return dict(ok=False, time=tc), None
    A = res.pop("A"); res["time"] = tc
    print(f"[{tag}] theta={res['theta']:.10f} ok={res['ok']} pivots={res['pivA']}+{res['pivB']} time={tc:.0f}s", flush=True)
    return res, A


def add_rigorous(entry, ops, A, tag):
    from rigorous import enclosure, dots_1d
    t1 = time.time()
    enc = enclosure(ops.n, ops.n, ops.R.long(), torch.as_tensor(np.asarray(A), device=dev).long(), dots_1d(ops.n), dev)
    enc["time"] = time.time() - t1; entry["rigorous"] = enc
    print(f"[{tag}] rigorous: {enc['lo']:.12f} <= theta <= {enc['up']:.12f} (width {enc['width']:.1e}, {enc['time']:.1f}s)", flush=True)


def main():
    n = int(sys.argv[1]); seed = int(sys.argv[2]); N0s = [int(a) for a in sys.argv[3].split("+")]; out = sys.argv[4]
    dens = float(arg("--density", 0.5)); paley = "--paley" in sys.argv; rig = "--rigorous" in sys.argv
    nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
    if paley:
        qr = np.unique((np.arange(1, n) ** 2) % n); S = qr[qr <= nprime]; R = np.setdiff1d(reps, S)
    else:
        rng = np.random.default_rng(seed); R = reps[~(rng.random(nprime) < dens)]
    m = len(R); ops = Ops(n, R)
    rec = dict(n=n, m=m, seed=seed, density=dens, paley=paley, methods={})
    print(f"n={n} m={m} paley={paley} density={dens} seed={seed}", flush=True)
    for N0 in [a for a in N0s if a > 0]:
        t0 = time.time(); xN, gN, ffts = prefix(ops, N0); tp = time.time() - t0
        D = (1.0 / gN ** 2)[1:nprime + 1].cpu().numpy(); order = np.argsort(-D)
        A0 = np.sort(reps[order[:m]]); gap = D[order[m - 1]] / D[order[m]] if m < nprime else float("inf")
        print(f"prefix{N0}: {tp:.0f}s {ffts} FFTs, weight ratio D(m)/D(m+1)={gap:.2e}, theta_barrier={1+2*xN.sum().item():.8f}", flush=True)
        entry = dict(t_prefix=tp, ffts_prefix=ffts, weight_ratio=gap, theta_barrier=1 + 2 * xN.sum().item())
        res, A = run_crossover(ops, A0, f"prefix{N0}"); entry.update(res)
        if rig and A is not None: add_rigorous(entry, ops, A, f"prefix{N0}")
        rec["methods"][f"prefix{N0}"] = entry
    if "--pdhg" in sys.argv:
        from pdlp_run import fft_pdhg
        starts = [("", None, None)]
        if "--warm" in sys.argv:
            from bign import predict
            S = np.setdiff1d(reps, R); s_ind = torch.zeros(n, device=dev)
            s_ind[torch.as_tensor(S, device=dev)] = 1; s_ind[n - torch.as_tensor(S, device=dev)] = 1
            Shat = torch.fft.fft(s_ind).real[1:nprime + 1].cpu().numpy()
            t1 = time.time(); Aw = predict(arg("--warm"), n, R, Shat, np.zeros(nprime), np.zeros(m)); tw = time.time() - t1
            JA = 2.0 * torch.cos(2 * math.pi * torch.outer(torch.as_tensor(Aw, device=dev).double(), ops.R.double()) / n)
            ones = torch.ones(m, device=dev); xw = torch.linalg.solve(JA, -ones); lw = torch.linalg.solve(JA.T, -2.0 * ones)
            yw = torch.zeros(n, device=dev); Ai = torch.as_tensor(Aw, device=dev)
            yw[Ai] = torch.clamp(lw, min=0) / 2; yw[n - Ai] = torch.clamp(lw, min=0) / 2
            print(f"warm start from learned basis: predict {tw:.1f}s", flush=True)
            starts.append(("warm", xw, yw))
            resw, Aw2 = run_crossover(ops, np.sort(np.asarray(Aw)), "learned")
            rec["methods"]["learned"] = dict(t_prefix=tw, **resw)
        for eps in [float(e) for e in arg("--pdhg").split(",")]:
            for sname, x0, y0 in starts:
                tag = f"pdhg{eps:g}{sname}"
                t0 = time.time(); info, xp, yp = fft_pdhg(ops, eps, 200000, log=lambda s: None, return_iter=True, x0=x0, y0=y0); tp = time.time() - t0
                sc = yp[1:nprime + 1].cpu().numpy(); order = np.argsort(-sc); A0 = np.sort(reps[order[:m]])
                gap = sc[order[m - 1]] / max(sc[order[m]], 1e-300) if m < nprime else float("inf")
                print(f"{tag}: {tp:.1f}s {info['iters']} iters {info['ffts']} FFTs theta_pdhg={info['theta']:.8f} dual ratio y(m)/y(m+1)={gap:.2e}", flush=True)
                entry = dict(t_prefix=tp, ffts_prefix=info["ffts"], iters=info["iters"], weight_ratio=gap, theta_pdhg=info["theta"], kkt=info["kkt"])
                res, A = run_crossover(ops, A0, tag); entry.update(res)
                if rig and A is not None: add_rigorous(entry, ops, A, tag)
                rec["methods"][tag] = entry
    if "--highs" in sys.argv:
        from scipy.optimize import linprog
        t1 = time.time(); k = np.arange(n); Jd = 2 * np.cos(2 * np.pi * np.outer(k, R) / n)
        r = linprog(c=-2 * np.ones(m), A_ub=-Jd, b_ub=np.ones(n), bounds=[(None, None)] * m, method="highs")
        rec["highs"] = dict(theta=1 - r.fun, time=time.time() - t1, status=int(r.status))
        print(f"HiGHS theta={1-r.fun:.10f} time={time.time()-t1:.0f}s status={r.status}", flush=True)
    with open(out, "a") as f: f.write(json.dumps(rec) + chr(10))


if __name__ == "__main__":
    main()
