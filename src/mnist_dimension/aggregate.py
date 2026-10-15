"""Neutral dimension tables, paired intervals and visible missing/failed points."""
from collections import defaultdict
from pathlib import Path
import json,math
import numpy as np
from src.mnist_experiments.aggregate import bootstrap_summary
from src.mnist_experiments.publication_aggregate import csv_write
from src.mnist_experiments.common import write_json
from .artifacts import validate_complete
from .design import METHODS

METRICS=['nfe','wall_time','representation_knn_rms','representation_knn_relative_scale','representation_support_percentile',
         'pixel_knn_distance','pixel_support_percentile','feature_knn_distance','feature_support_percentile',
         'classifier_confidence','represented_class_count','class_entropy','decoded_fraction_outside',
         'regular_proposed_step_rms','regular_actual_step_rms','regular_actual_pixel_step_rms','actual_to_expected_step_ratio',
         'displacement_rms','displacement_pixel_rms','cumulative_rms_path','cumulative_pixel_rms_path',
         'displacement_relative_to_rep_scale','clipping_fraction_mean']
PAIRED_METRICS=['representation_support_percentile','pixel_knn_distance','pixel_support_percentile',
                'feature_knn_distance','feature_support_percentile','regular_actual_step_rms']


def stat(values,seeds,repeats):
    result=bootstrap_summary(values,seeds,seed=123,repeats=repeats)
    result['ci95_low'],result['ci95_high']=result.pop('ci95')
    return result


def strata(rows,replicated):
    return [('curve_seed42',[r for r in rows if r['model_seed']==42]),
            ('replicated_anchors',[r for r in rows if r['dimension'] in replicated or r['representation']=='pixel'])]


def summarize(rows,replicated,repeats=2000):
    if repeats<20:raise ValueError('At least 20 bootstrap replicates required')
    summaries=[];paired=[]
    for stratum,subset in strata(rows,replicated):
        groups=defaultdict(list);suites=defaultdict(dict)
        for r in subset:
            key=(r['representation'],r['dimension'],r['method'])
            groups[key].append(r)
            pairkey=(r['representation'],r['dimension'],r['model_seed'],r['search_seed'])
            if r['method'] in suites[pairkey]:raise ValueError('Duplicate dimension/model/search/method')
            suites[pairkey][r['method']]=r
        for (rep,d,method),group in sorted(groups.items()):
            for metric in METRICS:
                valid=[r for r in group if isinstance(r.get(metric),(int,float)) and math.isfinite(r[metric])]
                if valid:summaries.append(dict(stratum=stratum,representation=rep,dimension=d,method=method,metric=metric,
                    **stat([r[metric] for r in valid],[r['model_seed'] for r in valid],repeats)))
        diffs=defaultdict(list)
        for (rep,d,model,seed),methods in suites.items():
            for reference in ('original_v3f2','matched_ddim'):
                if 'rms_v3f2' not in methods or reference not in methods:continue
                a,b=methods['rms_v3f2'],methods[reference]
                for key in ('checkpoint_sha256','representation_sha256','initial_noise_seed','initial_noise_prefix_sha256','initial_ddim_prefix_sha256','evaluation_key'):
                    if a[key]!=b[key]:raise ValueError('Unpaired '+key)
                for metric in PAIRED_METRICS:
                    if metric in a and metric in b:diffs[(rep,d,reference,metric)].append((a[metric]-b[metric],model))
        for (rep,d,reference,metric),items in sorted(diffs.items()):
            paired.append(dict(stratum=stratum,representation=rep,dimension=d,method='rms_v3f2',reference=reference,
                               metric=metric,direction='RMS minus reference',
                               **stat([x[0] for x in items],[x[1] for x in items],repeats)))
    return summaries,paired


def inventory(root,plan):
    root=Path(root);rows=[];status=[];qualities=[];errors=[]
    for name in ('train','search'):
        path=root/(name+'_last_invocation.json')
        if path.exists():errors+=json.loads(path.read_text())['errors']
    def state(path):
        if validate_complete(path):return 'completed'
        if (path/'complete.json').exists():return 'corrupt_artifact'
        if (path/'failure.json').exists():return 'failed'
        if (path/'started.json').exists():return 'incomplete'
        return 'missing'
    fit_errors=[e['error'] for e in errors if e['job']=='representation fit']
    fit_status=state(root/'representation')
    if fit_errors and fit_status!='completed':fit_status='failed'
    status.append(dict(kind='representation_fit',id='shared_pca',representation='pca',dimension=None,model_seed=None,status=fit_status,errors=fit_errors))
    for model in plan['models']:
        for kind,folder in [('training',root/'models'/model['id']),('quality',root/'quality'/model['id'])]:
            if kind=='training' and model['representation']=='pixel':continue
            value=state(folder)
            reasons=[e['error'] for e in errors if model['id'] in e['job']]
            if value in {'missing','incomplete'} and reasons:value='blocked_or_failed'
            status.append(dict(kind=kind,id=model['id'],representation=model['representation'],dimension=model['dimension'],model_seed=model['model_seed'],status=value,errors=reasons))
            if kind=='quality' and value=='completed':qualities.append(json.loads((folder/'quality.json').read_text()))
    for job in plan['jobs']:
        folder=Path(job['output']);value=state(folder)
        if value=='completed' and not all(validate_complete(folder/m) for m in METHODS):value='corrupt_method_artifact'
        reasons=[e['error'] for e in errors if job['id']==e['job'] or (f'{job["representation"]}_d{job["dimension"]}_m{job["model_seed"]}') in e['job']]
        if value in {'missing','incomplete'} and reasons:value='blocked_or_failed'
        status.append(dict(kind='search',id=job['id'],representation=job['representation'],dimension=job['dimension'],model_seed=job['model_seed'],search_seed=job['search_seed'],status=value,errors=reasons))
        for method in METHODS:
            if validate_complete(folder/method):
                row=json.loads((folder/method/'summary.json').read_text());row['suite_status']=value;rows.append(row)
    return rows,qualities,status


def aggregate(root,repeats=2000,plots=True):
    root=Path(root);plan=json.loads((root/'study_plan.json').read_text());cfg=plan['resolved_config']
    rows,qualities,status=inventory(root,plan)
    # Partial method outputs stay in raw_runs, but not in primary paired inference.
    accepted=[r for r in rows if r['suite_status']=='completed']
    summary,paired=summarize(accepted,cfg['replicated_dimensions'],repeats)
    out=root/'summary';out.mkdir(parents=True,exist_ok=True)
    quality_rows=[]
    for q in qualities:
        r={k:q[k] for k in ('representation','dimension','model_seed','parameter_count','diagnostic_epsilon_mse','diagnostic_prefix16_mse','best_training_validation_loss','quality_diagnostic_nfe','flags')}
        r.update({k:v for k,v in q['representation_quality'].items() if k!='validation_per_image_reconstruction_mse'})
        r.update({'ddim_'+k:v for k,v in q['sanity'].items()})
        r.update({'reconstructed_real_'+k:v for k,v in q['reconstructed_real'].items()})
        quality_rows.append(r)
    table=[]
    for stratum in ('curve_seed42','replicated_anchors'):
        points=[('pca',d) for d in cfg['dimensions'] if stratum=='curve_seed42' or d in cfg['replicated_dimensions']]+[('pixel',784)]
        for rep,d in points:
            relevant=[s for s in status if s['kind']=='search' and s['representation']==rep and s['dimension']==d and (stratum!='curve_seed42' or s['model_seed']==42)]
            row=dict(stratum=stratum,representation=rep,dimension=d,expected_jobs=len(relevant),completed_jobs=sum(s['status']=='completed' for s in relevant),
                     statuses=sorted({s['status'] for s in relevant}))
            q=[r for r in quality_rows if r['representation']==rep and r['dimension']==d and (stratum!='curve_seed42' or r['model_seed']==42)]
            row['quality_flags']=sorted({flag for r in q for flag in r['flags']})
            for k in ('retained_variance','validation_reconstruction_rms'):
                row[k]=q[0].get(k) if q else None
            for s in summary:
                if (s['stratum'],s['representation'],s['dimension'])==(stratum,rep,d) and s['metric'] in ('feature_knn_distance','representation_support_percentile','regular_actual_step_rms','nfe'):
                    for k in ('mean','median','std','ci95_low','ci95_high','n_runs','n_model_seeds'):
                        row[f'{s["method"]}_{s["metric"]}_{k}']=s[k]
            for s in paired:
                if (s['stratum'],s['representation'],s['dimension'])==(stratum,rep,d) and s['metric']=='feature_knn_distance':
                    for k in ('mean','median','ci95_low','ci95_high','n_runs'):
                        row[f'RMS_minus_{s["reference"]}_{k}']=s[k]
            table.append(row)
    for name,data in [('raw_runs',rows),('dimension_summary',summary),('paired_comparisons',paired),('model_representation_quality',quality_rows),('job_status',status),('scientific_table',table)]:
        write_json(out/(name+'.json'),data);csv_write(out/(name+'.csv'),data)
    write_json(out/'aggregation_config.json',dict(bootstrap_seed=123,bootstrap_repeats=repeats,
        strata=['seed42 complete curve','replicated anchor dimensions'],partial_runs='retained in raw table, excluded from primary summaries',
        expected_search_jobs=plan['search_jobs'],completed_search_jobs=sum(s['kind']=='search' and s['status']=='completed' for s in status)))
    lines=['# Dimension-study numerical summary','',
        'Mode candidates; semantic digit classes are external diagnostics.',
        'Primary curve uses seed42 throughout. Independent-model replication is shown separately at fixed anchors.',
        '95% CIs are bootstrap intervals conditional on the fixed representation. No automatic hypothesis conclusion.',
        'RMS minus reference: negative distance difference is favorable. Missing/failed points remain in this table.',
        '', '| Stratum | Representation | d | Completed / expected | Reconstruction RMS | RMS feature kNN, 95% CI | Flags/status |',
        '|---|---|---:|---:|---:|---|---|']
    for r in table:
        prefix='rms_v3f2_feature_knn_distance_'
        result=(f"{r[prefix+'mean']:.5g} [{r[prefix+'ci95_low']:.5g}, {r[prefix+'ci95_high']:.5g}]" if prefix+'mean' in r else 'missing')
        lines.append(f"| {r['stratum']} | {r['representation']} | {r['dimension']} | {r['completed_jobs']}/{r['expected_jobs']} | {r['validation_reconstruction_rms']} | {result} | {', '.join(r['quality_flags']+r['statuses'])} |")
    problems=[s for s in status if s['status']!='completed']
    if problems:
        lines+=['','## Incomplete or failed jobs','']
        for s in problems:lines.append(f"- {s['kind']} {s['id']}: {s['status']}; {'; '.join(s['errors'])}")
    lines+=['','Full method/paired metrics, uncertainty, actual steps, compute and failure inventory are in CSV/JSON.',
            'PCA truncation changes retained information. PCA-784 and pixel-CNN-784 are distinct representation/model controls.',
            'The theoretical step ratio is algebraic; it is not itself evidence that support gains are caused by dimension.']
    (out/'dimension_summary.md').write_text('\n'.join(lines)+'\n')
    if plots:
        from .plots import make_plots
        make_plots(summary,paired,qualities,status,cfg,out/'plots')
    return summary,paired,status
