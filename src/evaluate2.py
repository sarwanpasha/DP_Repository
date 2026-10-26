"""Evaluate the full pipeline with learned refinement:
  stage-1 predictor -> R rounds of learned refinement (each = 2 minor solves + 1 FFT) -> simplex correction.
FFT accounting: each dense minor solve is charged 2*m FFTs (exact-arithmetic CG bound).
"""
import argparse, glob, pickle, json, time, os
import numpy as np, torch
from theta_fft import ThetaOps
from simplex_fix import simplex_correct
from crossover import FFTCounter
from evaluate import load_model, predict_learned
from refine import refine_rounds, feat_dims
from model import EquivariantPredictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", required=True); ap.add_argument("--stage1", required=True)
    ap.add_argument("--refine", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--solver", default="dense")
    ap.add_argument("--limit", type=int, default=10 ** 9)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s1, a1 = load_model(args.stage1, dev)
    ck = torch.load(args.refine, map_location=dev, weights_only=False); ar = ck["args"]
    f_in, g_in = feat_dims(ar["prefix"])
    rf = EquivariantPredictor(f_in, g_in, C=ar["C"], layers=ar["layers"]).to(dev); rf.load_state_dict(ck["model"]); rf.eval()
    recs = []
    for pat in args.test:
        for fn in sorted(glob.glob(pat)): recs += pickle.load(open(fn, "rb"))
    recs = [r for r in recs if r["certified"]]
    keep = []; cnt_n = {}
    for r in recs:
        if cnt_n.get(r["n"], 0) < args.limit: keep.append(r); cnt_n[r["n"]] = cnt_n.get(r["n"], 0) + 1
    recs = keep
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        for r in recs:
            n, m = r["n"], r["m"]; ops = ThetaOps(n, r["R"])
            t0 = time.time()
            A0 = predict_learned(s1, a1, r, dev)
            A, hist = refine_rounds(rf, r, A0, dev, rounds=args.rounds, prefix=ar["prefix"])
            t_ref = time.time() - t0
            cnt = FFTCounter(); t1 = time.time()
            res = simplex_correct(ops, A, solver=args.solver, counter=cnt)
            t_simplex = time.time() - t1
            rounds_used = hist[-1]["round"]
            ffts_ref = (rounds_used + 1) * (2 * 2 * m + 1)
            ffts_simplex = cnt.n_fft + (2 + 2 * res["pivots"]) * 2 * m if args.solver == "dense" else cnt.n_fft
            out = dict(n=n, m=m, seed=r["seed"], theta=r["theta"], d_hist=[h["d"] for h in hist], cert_hist=[h["certified"] for h in hist],
                       d0=hist[0]["d"], d_final=hist[-1]["d"], rounds=rounds_used, certified_by_refine=bool(hist[-1]["certified"]),
                       pivots=int(res["pivots"]), ok=bool(res["ok"]), err=float(abs(res["theta"] - r["theta"])),
                       ffts_total=int(ffts_ref + ffts_simplex), t_refine=t_ref, t_simplex=t_simplex)
            f.write(json.dumps(out) + chr(10)); f.flush()
            print(f"n={n} m={m} seed={r['seed']} d_hist={out['d_hist']} cert={out['certified_by_refine']} pivots={out['pivots']} "
                  f"ok={out['ok']} err={out['err']:.1e} ffts={out['ffts_total']} t={t_ref + t_simplex:.2f}s", flush=True)


if __name__ == "__main__":
    main()
