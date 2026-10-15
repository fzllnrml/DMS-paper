"""Publication summaries: explicit contrasts, hierarchical seed bootstrap, no tests of significance."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from .aggregate import bootstrap_summary
from .common import write_json, timestamp

METRICS = ['nfe','diagnostic_nfe','wall_time','final_candidate_count','compression_ratio',
           'pixel_knn_distance','feature_knn_distance','pixel_support_percentile','feature_support_percentile',
           'classifier_confidence','represented_class_count','class_entropy','clipping_fraction',
           'stability_return_rate','mean_nearest_train_distance','mean_nearest_test_distance',
           'fraction_unusually_close_train','mean_train_minus_test_distance','train_test_gap_percentile']
QUALITY = ['pixel_knn_distance','feature_knn_distance','pixel_support_percentile','feature_support_percentile',
           'represented_class_count','stability_return_rate']
CONTRASTS = [('rms_v3f2','ddim'),('rms_v3f2','matched_ddim'),
             ('rms_v3f2_merge','ddim_cluster'),('rms_v3f2_merge','matched_cluster'),
             ('rms_v3f2','original_v3f2'),('rms_v3f2_merge','rms_v3f2'),
             ('rms_v3f2_merge','rms_v3f2_mean'),('rms_v3f2_merge','rms_v3f2_every_level')]


def condition(row):
    return (row['publication_series'], row['population_size'],row['definition_hash'],
            row['evaluation_definition_hash'],row.get('evaluator_sha256'))


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def statistics(values, seeds, repeats, seed):
    result=bootstrap_summary(values,seeds,seed=seed,repeats=repeats)
    result['ci95_low'],result['ci95_high']=result.pop('ci95')
    return result


def summarize(rows, repeats=2000, seed=123):
    if repeats < 20:
        raise ValueError('Use at least 20 bootstrap replicates (2000 by default)')
    groups=defaultdict(list)
    pair_groups=defaultdict(dict)
    for row in rows:
        key=condition(row)
        groups[(key,row['method'])].append(row)
        pair=(key,row['model_seed'],row['search_seed'])
        if row['method'] in pair_groups[pair]:
            raise ValueError('Duplicate condition/model_seed/search_seed/method')
        pair_groups[pair][row['method']]=row
    summaries=[]
    paired_values=defaultdict(list)
    for (key,method),group in sorted(groups.items(),key=str):
        for metric in METRICS:
            valid=[r for r in group if finite(r.get(metric))]
            if valid:
                summaries.append(dict(publication_series=key[0],population_size=key[1],definition_hash=key[2],
                    evaluation_definition_hash=key[3],evaluator_sha256=key[4],method=method,metric=metric,
                    **statistics([r[metric] for r in valid],[r['model_seed'] for r in valid],repeats,seed)))
    for (key,model_seed,search_seed),methods in pair_groups.items():
        for method,reference in CONTRASTS:
            if method not in methods or reference not in methods:
                continue
            a,b=methods[method],methods[reference]
            for field in ('initial_noise_seed','checkpoint_sha256','initial_gaussian_prefix_sha256','initial_ddim_prefix_sha256'):
                if a[field]!=b[field]:
                    raise ValueError(f'Unpaired {field}: {method} vs {reference}')
            for metric in QUALITY:
                if finite(a.get(metric)) and finite(b.get(metric)):
                    paired_values[(key,method,reference,metric)].append((a[metric]-b[metric],model_seed,search_seed))
    comparisons=[]
    for (key,method,reference,metric),values in sorted(paired_values.items(),key=str):
        comparisons.append(dict(publication_series=key[0],population_size=key[1],definition_hash=key[2],
            evaluation_definition_hash=key[3],evaluator_sha256=key[4],method=method,reference=reference,metric=metric,
            direction='method minus reference',
            **statistics([r[0] for r in values],[r[1] for r in values],repeats,seed)))
    return summaries,comparisons


def read_rows(roots):
    rows=[]
    seen=set()
    for root in roots:
        for path in sorted(Path(root).rglob('summary.json')):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            if not (path.parent/'complete.json').exists():
                continue
            data=json.loads(path.read_text())
            if not isinstance(data,list) or not data or not data[0].get('protocol'):
                continue
            marker=json.loads((path.parent/'complete.json').read_text())
            metadata=json.loads((path.parent/'metadata.json').read_text())
            if marker['signature'] != metadata['signature'] or {r['method'] for r in data} != set(marker['methods']):
                raise ValueError('Corrupt completed suite')
            for row in data:
                folder=path.parent/row['method']
                if not all((folder/name).exists() for name in ('arrays.pt','metrics.json','summary.json','complete.json')):
                    raise ValueError('Completed suite has missing method artifacts')
                saved=json.loads((folder/'complete.json').read_text())
                if saved['signature']!=marker['signature'] or saved['summary']!=row:
                    raise ValueError('Method summary disagrees with completed suite')
                rows.append(dict(row,source_directory=str(folder)))
    if not rows:
        raise ValueError('No complete publication suites found')
    return rows


def csv_write(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['n_runs']
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def aggregate(roots,output,repeats=2000,seed=123,plots=True):
    rows=read_rows(roots)
    summary,paired=summarize(rows,repeats,seed)
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('raw_runs',rows),('summary',summary),('paired_comparisons',paired)]:
        csv_write(out/(name+'.csv'),data)
        write_json(out/(name+'.json'),data)
    write_json(out/'aggregation_config.json',dict(roots=[str(r) for r in roots],bootstrap_repeats=repeats,bootstrap_seed=seed,
        timestamp=timestamp(),n_method_runs=len(rows),n_suites=len({r['paired_suite_id'] for r in rows}),contrasts=CONTRASTS))
    lines=['# Publication numerical summary','',
           'Completed suites only. Mode candidates; digit classes are external semantic diagnostics.',
           'Hierarchical bootstrap: model seeds, then paired search seeds within each model. No p-values.',
           'Differences are method minus reference. Lower kNN distance and higher support are favorable.',
           'Intervals with one model seed are conditional on that checkpoint. Three model seeds give limited model-level precision.',
           'Missing methods/metrics are not imputed. n_runs shows the available paired sample count.','',
           '| Series | P | Method | Metric | n | Mean | Std | Median | 95% CI |',
           '|---|---:|---|---|---:|---:|---:|---:|---|']
    for r in summary:
        if r['metric'] in QUALITY:
            lines.append(f"| {r['publication_series']} | {r['population_size']} | {r['method']} | {r['metric']} | {r['n_runs']} | {r['mean']:.5g} | {r['std']:.5g} | {r['median']:.5g} | [{r['ci95_low']:.5g}, {r['ci95_high']:.5g}] |")
    lines+=['','All compute, memorization, clipping and paired results are in CSV/JSON. No scientific conclusion is generated.']
    (out/'publication_summary.md').write_text('\n'.join(lines)+'\n')
    if plots:
        from .publication_plots import final_plots
        final_plots(rows,paired,out/'plots')
    return summary,paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',nargs='+',default=['runs/mnist_experiments/publication_main'])
    p.add_argument('--output',default='runs/mnist_experiments/publication_summary')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    aggregate(a.root,a.output,a.bootstrap,a.seed,not a.no_plots)
if __name__=='__main__': main()
