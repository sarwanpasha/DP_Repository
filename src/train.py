"""Train the active-set predictor.

usage: python train.py --data data/train_*.pkl --test data/test_*.pkl --model equiv|cnn --prefix 0|5|10|15|25
"""
import argparse, glob, pickle, math, time, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import EquivariantPredictor, CNNBaseline, topm


class BatchedOps:
    """J / J^T for a batch of instances sharing n but with different R (padded)."""

    def __init__(self, n, R_list, device):
        self.n = n; self.nprime = (n - 1) // 2; self.device = device
        self.B = len(R_list); self.ms = [len(R) for R in R_list]; self.mmax = max(self.ms)
        Rp = torch.zeros(self.B, self.mmax, dtype=torch.long)
        self.mask = torch.zeros(self.B, self.mmax, dtype=torch.bool)
        for b, R in enumerate(R_list):
            Rp[b, :len(R)] = torch.as_tensor(R); self.mask[b, :len(R)] = True
        self.R = Rp.to(device); self.mask = self.mask.to(device)
        self.nR = (n - self.R) % n
        self.scale = 1.0 / math.sqrt(2.0 * n)

    def J(self, z):                                   # z [B,C,mmax] -> [B,C,n']
        B, C, _ = z.shape
        z = z * self.mask[:, None, :]
        vt = torch.zeros(B, C, self.n, device=z.device, dtype=z.dtype)
        idx = self.R[:, None, :].expand(B, C, self.mmax)
        vt.scatter_add_(2, idx, z); vt.scatter_add_(2, self.nR[:, None, :].expand(B, C, self.mmax), z)
        f = torch.fft.fft(vt, dim=2).real
        return f[..., 1:self.nprime + 1] * self.scale

    def JT(self, h):                                  # h [B,C,n'] -> [B,C,mmax]
        B, C, _ = h.shape
        u = torch.zeros(B, C, self.n, device=h.device, dtype=h.dtype)
        u[..., 1:self.nprime + 1] = h; u[..., self.nprime + 1:] = h.flip(-1)
        f = torch.fft.fft(u, dim=2).real
        out = torch.gather(f, 2, self.R[:, None, :].expand(B, C, self.mmax)) * (2.0 * self.scale)
        return out * self.mask[:, None, :]


def make_features(recs, prefix, device):
    """Return (ff [B,f_in,n'], gf [B,g_in,mmax], labels [B,n'], ops)."""
    n = recs[0]["n"]; nprime = (n - 1) // 2
    ops = BatchedOps(n, [r["R"] for r in recs], device)
    B = len(recs)
    ff = []; gf = []; lab = torch.zeros(B, nprime)
    for b, r in enumerate(recs):
        f = [torch.as_tensor(r["Shat"]) / math.sqrt(n), torch.ones(nprime)]
        if prefix > 0:
            g = torch.as_tensor(r[f"g{prefix}"]).double().clamp_min(1e-300)
            f.append(torch.log(g).float().clamp(-40, 5) / 10.0)
            f.append((g < 1e-3).float())
        ff.append(torch.stack(f, 0))
        gvec = [torch.ones(ops.mmax)]
        if prefix > 0:
            x = torch.zeros(ops.mmax); xr = torch.as_tensor(r.get(f"x{prefix}", np.zeros(r["m"])), dtype=torch.float32)
            x[:r["m"]] = xr * math.sqrt(n); gvec.append(x)
        gf.append(torch.stack(gvec, 0))
        lab[b, torch.as_tensor(r["A"]).long() - 1] = 1.0
    return torch.stack(ff).to(device), torch.stack(gf).to(device), lab.to(device), ops


def hamming(logits, lab, ms):
    d = []
    for b in range(logits.shape[0]):
        pred = topm(logits[b:b + 1], ms[b])[0]
        d.append(int(ms[b] - (pred & (lab[b] > 0.5)).sum().item()))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--test", nargs="+", default=[])
    ap.add_argument("--model", default="equiv"); ap.add_argument("--prefix", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6); ap.add_argument("--out", default="runs/run")
    ap.add_argument("--max_train_n", type=int, default=10 ** 9)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    recs = []
    for pat in args.data:
        for fn in glob.glob(pat): recs += pickle.load(open(fn, "rb"))
    recs = [r for r in recs if r["certified"] and r["n"] <= args.max_train_n]
    test = []
    for pat in args.test:
        for fn in glob.glob(pat): test += pickle.load(open(fn, "rb"))
    test = [r for r in test if r["certified"]]
    byn = {}
    for r in recs: byn.setdefault(r["n"], []).append(r)
    print(f"train instances: {len(recs)} over n={sorted(byn)}; test: {len(test)} over n={sorted(set(r['n'] for r in test))}", flush=True)
    f_in = 2 + (2 if args.prefix > 0 else 0); g_in = 1 + (1 if args.prefix > 0 else 0)
    from model_lpgnn import LPGNN; Model = {"equiv": EquivariantPredictor, "cnn": CNNBaseline, "lpgnn": LPGNN}[args.model]
    model = Model(f_in, g_in, C=args.C, layers=args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    nb_per_epoch = sum(math.ceil(len(v) / args.bs) for v in byn.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * nb_per_epoch)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0; cnt = 0; ds = []
        batches = []
        for n, v in byn.items():
            perm = np.random.permutation(len(v))
            for i in range(0, len(v), args.bs): batches.append([v[j] for j in perm[i:i + args.bs]])
        np.random.shuffle(batches)
        for bt in batches:
            ff, gf, lab, ops = make_features(bt, args.prefix, dev)
            logits = model(ff, gf, ops)
            # class-balanced BCE (positives are ~ m/n' ~ 1/2 of frequencies)
            loss = F.binary_cross_entropy_with_logits(logits, lab)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(bt); cnt += len(bt)
            ds += hamming(logits.detach(), lab, [r["m"] for r in bt])
        rec = dict(epoch=ep, loss=tot / cnt, train_d_mean=float(np.mean(ds)), time=time.time() - t0)
        if test and (ep % 5 == 4 or ep == args.epochs - 1):
            model.eval(); res = {}
            with torch.no_grad():
                for n in sorted(set(r["n"] for r in test)):
                    tv = [r for r in test if r["n"] == n]
                    dd = []
                    for i in range(0, len(tv), 8):
                        ff, gf, lab, ops = make_features(tv[i:i + 8], args.prefix, dev)
                        dd += hamming(model(ff, gf, ops), lab, [r["m"] for r in tv[i:i + 8]])
                    res[n] = dict(d_mean=float(np.mean(dd)), d_med=float(np.median(dd)), d_max=int(np.max(dd)),
                                  frac_exact=float(np.mean(np.array(dd) == 0)), m_mean=float(np.mean([r["m"] for r in tv])))
            rec["test"] = res
        log.append(rec); print(json.dumps(rec), flush=True)
        torch.save(dict(model=model.state_dict(), args=vars(args)), args.out + ".pt")
    json.dump(log, open(args.out + ".json", "w"))


if __name__ == "__main__":
    main()
