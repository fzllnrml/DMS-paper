"""Hierarchical bootstrap summaries and paired differences without p-value fishing."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import write_json
from .plots import summary_plots
METRICS=['nfe','candidate_count','pixel_knn_mean','feature_knn_mean','pixel_support_percentile','feature_support_percentile','class_coverage','class_entropy','mean_confidence','stability_return_rate','curvature_fraction_negative','displacement_rms','movement_efficiency','search_wall_seconds']

def bootstrap_summary(values,model_seeds,seed=123,repeats=2000):
    values=np.asarray(values,dtype=float); ids=np.asarray(model_seeds); blocks=[values[ids==s] for s in np.unique(ids)]
    rng=np.random.default_rng(seed); boot=[]
    for _ in range(repeats):
        chosen=rng.integers(len(blocks),size=len(blocks)); boot.append(np.mean([rng.choice(blocks[i],len(blocks[i]),replace=True).mean() for i in chosen]))
    return dict(n_runs=len(values),n_model_seeds=len(blocks),mean=float(np.mean([b.mean() for b in blocks])),std=float(values.std(ddof=1)) if len(values)>1 else 0.,median=float(np.median(values)),ci95=np.quantile(boot,[.025,.975]).tolist(),uncertainty_scope='across model and search seeds' if len(blocks)>1 else 'conditional on one checkpoint')

def summarize(rows,repeats=2000):
    groups=defaultdict(list); suites=defaultdict(dict); differences=defaultdict(list); summaries=[]; paired=[]
    for r in rows:
        groups[(r.get('condition_id',r['variant']),r['method'],r['space'],r['dimension'])].append(r)
        if r['method'] in suites[r['suite_id']]: raise ValueError('Duplicate suite/method')
        suites[r['suite_id']][r['method']]=r
    for (condition,method,space,dim),items in groups.items():
        for metric in METRICS:
            valid=[r for r in items if isinstance(r.get(metric),(int,float))]
            if valid: summaries.append(dict(condition=condition,variant=items[0]['variant'],method=method,space=space,dimension=dim,metric=metric,**bootstrap_summary([r[metric] for r in valid],[r['model_seed'] for r in valid],repeats=repeats)))
    for suite in suites.values():
        original=suite.get('original')
        if original is None: continue
        for method,r in suite.items():
            if method=='original': continue
            if any(r[k]!=original[k] for k in ('model_seed','search_seed','initial_noise_seed')): raise ValueError('Unpaired seeds')
            for metric in METRICS:
                if isinstance(r.get(metric),(int,float)) and isinstance(original.get(metric),(int,float)):
                    differences[(r.get('condition_id',r['variant']),method,r['space'],r['dimension'],metric)].append((r[metric]-original[metric],r['model_seed']))
    for (condition,method,space,dim,metric),values in differences.items():
        paired.append(dict(condition=condition,method=method,reference='original',direction='method minus original',space=space,dimension=dim,metric=metric,**bootstrap_summary([v[0] for v in values],[v[1] for v in values],repeats=repeats)))
    return summaries,paired

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',default='runs/mnist_experiments'); p.add_argument('--output',default='runs/mnist_experiments_aggregate'); p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args(); rows=[]
    for path in Path(a.root).rglob('summary.json'):
        data=json.loads(path.read_text())
        if isinstance(data,dict) and 'method' in data and (path.parent/'complete.json').exists(): rows.append(data)
    if not rows: raise SystemExit('No completed runs found')
    summary,paired=summarize(rows,a.bootstrap); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for name,items in [('individual_runs',rows),('summary',summary),('paired_differences',paired)]:
        write_json(out/f'{name}.json',items)
        if items:
            with (out/f'{name}.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in items)))); writer.writeheader(); writer.writerows(items)
    summary_plots(rows,out/'plots')
if __name__=='__main__': main()
