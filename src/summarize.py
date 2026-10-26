import json,glob,numpy as np,collections
rows=[json.loads(l) for f in glob.glob('results/eval_dense_*.jsonl') for l in open(f)]
byn=collections.defaultdict(list)
for r in rows: byn[r['n']].append(r)
for n in sorted(byn):
    rs=byn[n]; print(f'n={n} N={len(rs)} m={np.mean([r["m"] for r in rs]):.0f}')
    for k in rs[0]['methods']:
        d=[r['methods'][k]['d'] for r in rs]; p=[r['methods'][k]['pivots'] for r in rs]; ok=[r['methods'][k]['ok'] for r in rs]; e=[r['methods'][k]['err'] for r in rs]; t=[r['methods'][k]['time'] for r in rs]
        print(f'   {k:10s} d={np.mean(d):6.1f} piv={np.mean(p):6.1f} ok={np.mean(ok):.2f} maxerr={max(e):.1e} t={np.mean(t):.2f}s')
    if 'ipm' in rs[0]: print('   IPM ffts=%.0f time=%.1fs newton=%.0f'%(np.mean([r['ipm']['ffts'] for r in rs]),np.mean([r['ipm']['time'] for r in rs]),np.mean([r['ipm']['newton'] for r in rs])))
    if 'highs' in rs[0]: print('   HiGHS time=%.1fs'%np.mean([r['highs']['time'] for r in rs]))
rows=[json.loads(l) for f in glob.glob('results/eval_cgnr_*.jsonl') for l in open(f)]
byn=collections.defaultdict(list)
for r in rows: byn[r['n']].append(r)
for n in sorted(byn):
    rs=byn[n]; print(f'CGNR n={n} N={len(rs)}', {k:(round(np.mean([r['methods'][k]['ffts'] for r in rs])), round(np.mean([r['methods'][k]['pivots'] for r in rs]),1)) for k in rs[0]['methods']})
