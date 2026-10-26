"""Learned active-set refinement: the certificate residuals (g, lam) of a candidate basis are fed back
to an equivariant network that re-predicts the optimal active set; iterated until certified."""
import argparse, glob, pickle, math, time, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import EquivariantPredictor, topm
from train import BatchedOps, hamming


def minor_rows(n, A, R, device):
    A = torch.as_tensor(np.asarray(A), device=device, dtype=torch.float64)
    Rt = torch.as_tensor(np.asarray(R), device=device, dtype=torch.float64)
    return 2.0 * torch.cos(2 * math.pi * torch.outer(A, Rt) / n)


def certificate(n, R, A, ops_scale_J, device):
    JA = minor_rows(n, A, R, device)
    m = len(R)
    ones = torch.ones(m, device=device, dtype=torch.float64)
    x = -torch.linalg.solve(JA, ones)
    lam = -2.0 * torch.linalg.solve(JA.T, ones)
    return x, lam


def build_batch(recs, prefix, device, corrupt="mix", A_hats=None, rng=None, pred_fn=None):
    n = recs[0]["n"]; nprime = (n - 1) // 2; B = len(recs)
    ops = BatchedOps(n, [r["R"] for r in recs], device)
    rng = rng or np.random.default_rng()
    ff = []; gf = []; lab = torch.zeros(B, nprime); ds = []
    for b, r in enumerate(recs):
        m = r["m"]; A = np.asarray(r["A"]); reps = np.arange(1, nprime + 1)
        if A_hats is not None:
            Ah = np.asarray(A_hats[b])
        else:
            mode = rng.random()
            if pred_fn is not None and mode < 0.4:
                Ah = np.asarray(pred_fn(r)); mode = 1.0
            elif mode < 0.15:
                Ah = A.copy()
            elif mode < 0.55:
                d = rng.integers(1, max(2, int(0.15 * m)))
                out_idx = rng.choice(m, d, replace=False)
                notA = np.setdiff1d(reps, A); in_new = rng.choice(notA, d, replace=False)
                Ah = A.copy(); Ah[out_idx] = in_new
            else:
                score = np.asarray(r["Shat"], dtype=np.float64) / math.sqrt(n)
                sig = rng.uniform(0.05, 1.0) * score.std()
                Ah = reps[np.argsort(-(score + sig * rng.standard_normal(nprime)))[:m]]
        Ah = np.sort(Ah)
        try:
            x, lam = certificate(n, r["R"], Ah, None, device)
            if not (torch.isfinite(x).all() and torch.isfinite(lam).all()): raise RuntimeError
        except RuntimeError:
            Ah = np.sort(A); x, lam = certificate(n, r["R"], Ah, None, device)
        ds.append(int(m - len(np.intersect1d(Ah, A))))
        ops1 = BatchedOps(n, [r["R"]], device)
        xt = x.reshape(1, 1, m)
        g = 1.0 + math.sqrt(2 * n) * ops1.J(xt)[0, 0]
        inA = torch.zeros(nprime, device=device); inA[torch.as_tensor(Ah) - 1] = 1.0
        lam_s = torch.zeros(nprime, device=device, dtype=torch.float64); lam_s[torch.as_tensor(Ah) - 1] = lam
        f = [torch.as_tensor(r["Shat"], device=device) / math.sqrt(n), torch.ones(nprime, device=device), inA,
             torch.asinh(g * 10).float(), torch.asinh(lam_s).float(), (g < -1e-7).float(), (lam_s < -1e-7).float()]
        if prefix > 0:
            gp = torch.as_tensor(r[f"g{prefix}"], device=device).double().clamp_min(1e-300)
            f.append(torch.log(gp).float().clamp(-40, 5) / 10.0)
        ff.append(torch.stack(f, 0))
        gvec = [torch.ones(ops.mmax, device=device)]
        xg = torch.zeros(ops.mmax, device=device); xg[:m] = torch.asinh(x * math.sqrt(n)).float(); gvec.append(xg)
        gf.append(torch.stack(gvec, 0))
        lab[b, torch.as_tensor(A).long() - 1] = 1.0
    return torch.stack(ff), torch.stack(gf), lab.to(device), ops, ds


def feat_dims(prefix):
    return 7 + (1 if prefix > 0 else 0), 2


def refine_rounds(model, rec, A0, device, rounds=5, prefix=0):
    n = rec["n"]; m = rec["m"]
    A = np.sort(np.asarray(A0)); hist = []
    for t in range(rounds + 1):
        ff, gf, lab, ops, ds = build_batch([rec], prefix, device, A_hats=[A])
        viol = (ff[0, 5].sum() + ff[0, 6].sum()).item()
        hist.append(dict(round=t, d=ds[0], certified=viol == 0))
        if viol == 0 or t == rounds:
            break
        with torch.no_grad():
            logits = model(ff, gf, ops)
        A = np.sort(torch.topk(logits[0], m).indices.cpu().numpy() + 1)
    return A, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("--data", nargs="+"); ap.add_argument("--test", nargs="+", default=[])
    ap.add_argument("--prefix", type=int, default=0); ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--C", type=int, default=64); ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--out", default="runs/refine"); ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); rng = np.random.default_rng(0)
    recs = []
    for pat in args.data:
        for fn in glob.glob(pat): recs += pickle.load(open(fn, "rb"))
    recs = [r for r in recs if r["certified"]]
    test = []
    for pat in args.test:
        for fn in glob.glob(pat): test += pickle.load(open(fn, "rb"))
    test = [r for r in test if r["certified"]]
    byn = {}
    for r in recs: byn.setdefault(r["n"], []).append(r)
    f_in, g_in = feat_dims(args.prefix)
    model = EquivariantPredictor(f_in, g_in, C=args.C, layers=args.layers).to(dev)
    init = None
    from evaluate import load_model, predict_learned
    if args.init_ckpt:
        init = load_model(args.init_ckpt, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    nb = sum(math.ceil(len(v) / args.bs) for v in byn.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * nb)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"train {len(recs)} test {len(test)}", flush=True)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0; cnt = 0; din = []; dout = []
        batches = []
        for n, v in byn.items():
            perm = rng.permutation(len(v))
            for i in range(0, len(v), args.bs): batches.append([v[j] for j in perm[i:i + args.bs]])
        rng.shuffle(batches)
        for bt in batches:
            pred_fn = (lambda r: predict_learned(init[0], init[1], r, dev)) if init is not None else None
            ff, gf, lab, ops, ds = build_batch(bt, args.prefix, dev, rng=rng, pred_fn=pred_fn)
            logits = model(ff, gf, ops)
            loss = F.binary_cross_entropy_with_logits(logits, lab)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(bt); cnt += len(bt); din += ds
            dout += hamming(logits.detach(), lab, [r["m"] for r in bt])
        rec = dict(epoch=ep, loss=tot / cnt, d_in=float(np.mean(din)), d_out=float(np.mean(dout)), time=time.time() - t0)
        if test and (ep % 10 == 9 or ep == args.epochs - 1):
            model.eval(); res = {}
            with torch.no_grad():
                for n in sorted(set(r["n"] for r in test)):
                    tv = [r for r in test if r["n"] == n][:12]
                    d_start = []; d_end = []; rounds_used = []; cert = []
                    for r in tv:
                        nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
                        if init is not None:
                            A0 = predict_learned(init[0], init[1], r, dev)
                        else:
                            A0 = reps[np.argsort(-np.asarray(r["Shat"]))[:r["m"]]]
                        A, hist = refine_rounds(model, r, A0, dev, rounds=args.rounds, prefix=args.prefix)
                        d_start.append(hist[0]["d"]); d_end.append(hist[-1]["d"]); rounds_used.append(hist[-1]["round"])
                        cert.append(hist[-1]["certified"])
                    res[n] = dict(d_start=float(np.mean(d_start)), d_end=float(np.mean(d_end)), d_end_max=int(np.max(d_end)),
                                  frac_cert=float(np.mean(cert)), rounds=float(np.mean(rounds_used)))
            rec["test"] = res
        print(json.dumps(rec), flush=True)
        torch.save(dict(model=model.state_dict(), args=vars(args)), args.out + ".pt")


if __name__ == "__main__":
    main()
