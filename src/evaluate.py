"""End-to-end evaluation of certified crossover with different active-set predictors.

For each test instance (from gen_data pickles):
  predictors: static heuristic (top-m Shat), prefix-N0 heuristic (top-m D at N0 Newton steps),
              learned models (checkpoints; static or prefix features)
  -> Hamming error d, two-phase simplex correction (pivots), certified theta, FFT count, time
  baselines: full FFT-IPM (FFT count, time), HiGHS (time) for n <= 3203.
usage: python evaluate.py --test data/test_n*.pkl --ckpt runs/x.pt runs/y.pt --out results/eval.jsonl [--solver dense|cgnr]
"""
import argparse, glob, pickle, json, time, math, os
import numpy as np, torch
from theta_fft import ThetaOps, solve_theta, theta_reference
from simplex_fix import simplex_correct
from crossover import FFTCounter
from train import make_features, BatchedOps
from model import EquivariantPredictor, CNNBaseline, topm


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False); a = ck["args"]
    f_in = 2 + (2 if a["prefix"] > 0 else 0); g_in = 1 + (1 if a["prefix"] > 0 else 0)
    from model_lpgnn import LPGNN; M = {"equiv": EquivariantPredictor, "cnn": CNNBaseline, "lpgnn": LPGNN}[a["model"]]
    model = M(f_in, g_in, C=a["C"], layers=a["layers"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
    return model, a


def predict_learned(model, a, rec, dev):
    with torch.no_grad():
        ff, gf, lab, ops = make_features([rec], a["prefix"], dev)
        logits = model(ff, gf, ops)
        idx = torch.topk(logits[0], rec["m"]).indices.cpu().numpy() + 1
    return np.sort(idx)


def run_pipeline(rec, A0, solver, fft_prefix=0):
    ops = ThetaOps(rec["n"], rec["R"])
    cnt = FFTCounter(); t0 = time.time()
    res = simplex_correct(ops, A0, solver=solver, counter=cnt)
    d = int(rec["m"] - len(set(A0.tolist()) & set(rec["A"].tolist())))
    return dict(d=d, ok=bool(res["ok"]), pivots=int(res["pivots"]), phaseA=int(res["phaseA"]), phaseB=int(res["phaseB"]),
                theta=float(res["theta"]), err=float(abs(res["theta"] - rec["theta"])), ffts=int(cnt.n_fft + fft_prefix),
                time=time.time() - t0, inner_iters=[int(i) for i in res.get("inner_iters", [])][:50])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", required=True); ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--solver", default="dense")
    ap.add_argument("--baselines", type=int, default=1); ap.add_argument("--max_n_highs", type=int, default=3203)
    ap.add_argument("--max_n_ipm", type=int, default=1601); ap.add_argument("--prefixes", default="5,10,15,25")
    ap.add_argument("--limit", type=int, default=10 ** 9); ap.add_argument("--no_static", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [(os.path.basename(p).replace(".pt", ""),) + load_model(p, dev) for p in args.ckpt]
    recs = []
    for pat in args.test:
        for fn in sorted(glob.glob(pat)): recs += pickle.load(open(fn, "rb"))
    recs = [r for r in recs if r["certified"]]
    keep = []; cnt_n = {}
    for r in recs:                                  # per-n limit
        if cnt_n.get(r["n"], 0) < args.limit: keep.append(r); cnt_n[r["n"]] = cnt_n.get(r["n"], 0) + 1
    recs = keep
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    prefixes = [int(p) for p in args.prefixes.split(",") if p]
    with open(args.out, "a") as f:
        for r in recs:
            n, m = r["n"], r["m"]; nprime = (n - 1) // 2; reps = np.arange(1, nprime + 1)
            out = dict(n=n, m=m, seed=r["seed"], theta=r["theta"], newton_ref=r["newton_ref"], methods={})
            # static heuristic
            if not args.no_static:
                A0 = np.sort(reps[np.argsort(-r["Shat"])[:m]]); out["methods"]["static"] = run_pipeline(r, A0, args.solver)
            # prefix heuristics
            for N0 in prefixes:
                D = 1.0 / np.asarray(r[f"g{N0}"], dtype=np.float64) ** 2
                A0 = np.sort(reps[np.argsort(-D)[:m]])
                out["methods"][f"prefix{N0}"] = run_pipeline(r, A0, args.solver, fft_prefix=r[f"cost{N0}"])
            # learned
            for name, model, a in models:
                A0 = predict_learned(model, a, r, dev)
                out["methods"][name] = run_pipeline(r, A0, args.solver, fft_prefix=r[f"cost{a['prefix']}"] if a["prefix"] > 0 else 0)
            # baselines
            if args.baselines:
                if n <= args.max_n_ipm:
                    t0 = time.time(); st = solve_theta(n, r["R"], tol=1e-8, precond="none")
                    out["ipm"] = dict(theta=float(st["theta"]), err=float(abs(st["theta"] - r["theta"])), newton=int(st["newton_steps"]),
                                      ffts=int(2 * st["total_cg"] + 4 * st["newton_steps"]), time=time.time() - t0)
                if n <= args.max_n_highs:
                    t0 = time.time(); th, _ = theta_reference(n, r["R"])
                    out["highs"] = dict(theta=float(th), err=float(abs(th - r["theta"])), time=time.time() - t0)
            f.write(json.dumps(out) + "\n"); f.flush()
            summ = " ".join(f"{k}:d={v['d']},p={v['pivots']},ok={int(v['ok'])}" for k, v in out["methods"].items())
            print(f"n={n} m={m} seed={r['seed']} {summ} | ipm={out.get('ipm', {}).get('ffts', '-')}ffts highs={out.get('highs', {}).get('time', float('nan')):.1f}s", flush=True)


if __name__ == "__main__":
    main()
