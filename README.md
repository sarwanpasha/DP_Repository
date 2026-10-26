# Certified Lovász theta of circulant graphs at scale

Code for computing the Lovász theta function θ(G) of circulant graphs (Cayley graphs of Z_n) with
n up to 10^5 vertices, with an exact optimality certificate and a rigorous floating-point
enclosure. The approach is

1. **Matrix-free first-order identification.** The theta LP of a circulant graph has an n × m
   cosine constraint matrix J that is applied with one FFT. A restarted average PDHG (PDLP-style)
   solves the LP to a loose tolerance with two FFTs per iteration and the m largest dual
   coordinates identify the optimal active set.
2. **Certified crossover.** From the candidate basis A the certificate J_A x = -1,
   J_A^T λ = -2, g = 1 + J x ≥ 0, λ ≥ 0 is checked on the GPU. If it fails, a two-phase
   simplex with Sherman–Morrison basis updates repairs it in a handful of pivots.
3. **Rigorous enclosure.** From the certified basis, a priori rounding bounds give an interval
   θ_lo ≤ θ(G) ≤ θ_up (typical width 1e-10 for n = 10^5).
4. **Learned and barrier identification** (ablations): an equivariant active-set predictor,
   CNN and LP-GNN baselines, and a barrier prefix with the FFT-applicable active-set
   preconditioner.

Baselines included: dense Newton barrier (Cholesky of J^T D J formed by one FFT), Mehrotra
predictor-corrector interior point, FFT-PDLP to high accuracy, MPAX/OR-Tools PDLP on the dense
matrix, and HiGHS. The pipeline also runs on Cayley graphs of Z_n × Z_n (`cayley2d.py`).

## Installation

```
python -m pip install -r requirements.txt
```

Python ≥ 3.9, NumPy, SciPy and PyTorch are required. A CUDA GPU is optional; everything runs on
the CPU for moderate n (up to a few thousand). The optional dense PDLP baseline needs
`jax` + `mpax` (GPU) or `ortools` (CPU).

## Minimal working example

```
cd src
python mwe.py            # random circulant, n = 211, seed 0, edge density 1/2
python mwe.py 1601 3     # n = 1601, seed 3
```

The script generates a random circulant graph, runs FFT-PDHG, identifies the active set,
certifies it, prints the rigorous enclosure and cross-checks θ against HiGHS on the dense LP:

```
device=cpu  n=211  m=58 non-edge classes  (edge density 0.5)
FFT-PDHG: 1024 iterations, 2080 FFTs, 0.12s, theta_pdhg=15.321304 (KKT residual 4.9e-07)
crossover: 0 Phase-A + 0 Phase-B pivots, 0.02s, certified=True, theta=15.3213028285
rigorous enclosure: 15.321302828545 <= theta(G) <= 15.321302828565  (width 2.0e-11)
HiGHS (dense LP): theta=15.3213028285, 0.03s, |difference| = 5.2e-14
```

For n = 1601 (seed 3) the same script needs 2 s for PDHG and 0.25 s for the certificate on a CPU,
and HiGHS on the dense 1601 × 395 LP takes 11 s:

```
rigorous enclosure: 39.391535553213 <= theta(G) <= 39.391535554738  (width 1.5e-09)
HiGHS (dense LP): theta=39.3915355533, 11.25s, |difference| = 2.1e-14
```

## Main scripts (run from `src/`)

| script | purpose |
|---|---|
| `sweep.py n seed N0s out [--density p] [--paley] [--pdhg eps,...] [--warm ckpt] [--rigorous] [--highs]` | full certified pipeline on one circulant instance (barrier prefix of depth `N0s`, or `0` for none; PDHG identification at the given tolerances; optional learned warm start; rigorous enclosure; HiGHS reference). Appends one JSON line to `out`. |
| `pdlp_run.py n seed out [--dense] [--eps 1e-4,1e-6,1e-8]` | FFT-PDHG and dense PDLP baselines |
| `pdipm.py n seed out [density]` | Mehrotra predictor-corrector IPM (normal matrix by FFT, Cholesky) + certificate |
| `dense_ipm.py n seed out [density]` | dense Newton barrier method + certificate |
| `cayley2d.py n seed out [--density p] [--eps ...] [--rigorous]` | same pipeline for Cayley graphs of Z_n × Z_n |
| `rigorous.py` | rigorous enclosure from a certified basis (library) |
| `bign.py` | FFT operators, barrier prefix, GPU simplex crossover (library + CLI) |
| `gen_data.py n_list seed_start seed_end out.pkl` | labelled instances (certified optimal active sets) for training |
| `train.py --data ... --test ... --model equiv/cnn/lpgnn --prefix 0/5/10/25` | train an active-set predictor |
| `evaluate.py --test ... --ckpt ... --out ...` | predictor + crossover evaluation with baselines |
| `refine.py`, `evaluate2.py` | learned refinement rounds |
| `active_precond.py`, `probe_minor.py`, `minor_spectrum.py`, `probe_activeset.py`, `probe_optW.py` | analysis of the active-set preconditioner and of the cosine minors |
| `theta_fft.py`, `crossover.py`, `simplex_fix.py` | NumPy reference implementations (FFT barrier method, certificate, simplex correction) |

Examples:

```
# certified theta of a random circulant with n = 12841 by PDHG identification and rigorous enclosure
python sweep.py 12841 0 0 results/out.jsonl --pdhg 1e-6 --rigorous

# Paley graph on 1601 vertices (theta = sqrt(n))
python sweep.py 1601 0 0 results/paley.jsonl --paley --pdhg 1e-6 --rigorous

# baselines on the same instance
python pdipm.py 12841 0 results/pdipm.jsonl
python dense_ipm.py 12841 0 results/dense.jsonl
python pdlp_run.py 12841 0 results/pdlp.jsonl --eps 1e-8

# Z_101 x Z_101 Cayley graph
python cayley2d.py 101 0 results/cayley2d.jsonl --eps 1e-6 --rigorous

# learned predictor: data, training, evaluation
python gen_data.py 211,401,809 0 200 data/train_small.pkl
python gen_data.py 211,401,809 200 240 data/test_small.pkl
python train.py --data data/train_small.pkl --test data/test_small.pkl --model equiv --prefix 0 --out runs/equiv_p0
python evaluate.py --test data/test_small.pkl --ckpt runs/equiv_p0.pt --out results/eval.jsonl
```

All scripts append JSON lines to the output file given on the command line, so runs can be
aggregated with a few lines of Python.

## Problem conventions

For an odd n and the set R of non-edge generator classes {r, -r}, the theta LP is

    maximize  1 + 2 Σ_r x_r   subject to   g_k = 1 + Σ_r 2 cos(2π k r / n) x_r ≥ 0,  k = 0..n-1,

with dual  minimize 1 + Σ_k y_k  subject to  J^T y = -2·1, y ≥ 0. The code stores one
representative k ∈ {1, …, (n-1)/2} per constraint class and one r per generator class.

## License

MIT, see `LICENSE`.
