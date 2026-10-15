"""Complete expected inventory; failed/missing runs remain explicit."""
import csv
from pathlib import Path
import numpy as np
from src.mnist_experiments.common import write_json
from .protocol import QUALITY, DIAGNOSTICS, BASELINES, METRICS
from .artifacts import inspect_job, read_json


def write_table(path, rows, columns=()):
    keys = list(dict.fromkeys(list(columns)+[k for r in rows for k in r]))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    write_json(Path(path).with_suffix('.json'),rows)


def flat(row):
    return {k:v for k,v in row.items() if not isinstance(v,(dict,list))}


def compatible(a,b):
    for k in ('checkpoint_sha256','evaluator_sha256','evaluation_cache_key',
              'initial_gaussian_prefix_sha256','initial_ddim_prefix_sha256','context_signature'):
        if not a.get(k) or a.get(k) != b.get(k):
            return False
    return True


def guardrail(rows, references):
    if len(rows)!=len(references) or not rows:
        return None
    return (any(r['class_coverage']<8 for r in rows)
        or np.mean([b['class_coverage']-a['class_coverage'] for a,b in zip(rows,references)])>1
        or np.mean([b['class_entropy']-a['class_entropy'] for a,b in zip(rows,references)])>.10)


def aggregate(root, *, read_only=False):
    root = Path(root)
    manifest = read_json(root/'manifest.json')
    sig = manifest['signature']; c = manifest['context']['config']
    seeds = c['search_seeds']; output = root/'summary'
    model_seed = c['model_seed']
    if not read_only: output.mkdir(exist_ok=True)
    inventory, raw, stability, per_candidate = [], [], [], []
    lookup = {}
    for seed in seeds:
        p = root/f'model{model_seed}_seed{seed}/inputs'
        status, error = inspect_job(p,sig)
        inventory.append(dict(model_seed=model_seed,search_seed=seed,method='inputs',kind='dependency',status=status,error=error))
        if status=='complete':
            baseline_rows = read_json(p/'summary.json')['baseline_rows']
            for r in baseline_rows:
                r = dict(r,status='complete',kind='baseline')
                raw.append(flat(r)); lookup[(seed,r['method'])]=r
        else:
            for method in BASELINES:
                raw.append(dict(model_seed=model_seed,search_seed=seed,method=method,status='blocked',kind='baseline',error=error))
    for job in manifest['jobs']:
        p = root/job['id']; status,error = inspect_job(p,sig)
        entry = dict(job,status=status,error=error)
        inventory.append(entry)
        if status!='complete':
            if job['kind']=='quality':
                raw.append(flat(entry))
            else:
                for scale in (.02,.05):
                    stability.append(dict(model_seed=model_seed,search_seed=job['search_seed'],method=job['method'],scale=scale,status=status,error=error))
            continue
        row = read_json(p/'summary.json')
        if job['kind']=='quality':
            row['status']='complete';raw.append(flat(row));lookup[(row['search_seed'],row['method'])]=row
        else:
            for r in row['stability_summary']:
                stability.append(dict(model_seed=model_seed,search_seed=job['search_seed'],method=job['method'],
                    status='complete',diagnostic_nfe=row['diagnostic_nfe'],**r))
            details = read_json(p/'metrics.json')['per_candidate']
            per_candidate += [dict(model_seed=model_seed,search_seed=job['search_seed'],method=job['method'],**r) for r in details]
    summaries, pairs = [], []
    for method in BASELINES+QUALITY:
        observations = [lookup[(s,method)] for s in seeds if (s,method) in lookup]
        for metric in METRICS+('nfe',):
            values = [r[metric] for r in observations]
            summaries.append(dict(model_seed=model_seed,method=method,metric=metric,n_expected=len(seeds),n_complete=len(values),
                status='complete' if len(values)==len(seeds) else 'incomplete',
                mean=float(np.mean(values)) if values else None,
                std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                search_seeds=','.join(map(str,seeds))))
    controls = {}
    for method in QUALITY:
        all_ok=True
        for metric in METRICS:
            differences=[]; gains=[]; numerators=[]
            for seed in seeds:
                a=lookup.get((seed,method)); b=lookup.get((seed,'rms_v3f2')); m=lookup.get((seed,'matched_ddim'))
                ok=bool(a and b and m and compatible(a,b) and compatible(b,m))
                all_ok &= ok
                d=a[metric]-b[metric] if ok else None
                pairs.append(dict(record_type='pair',model_seed=model_seed,search_seed=seed,method=method,reference='rms_v3f2',
                    metric=metric,status='complete' if ok else 'missing_or_incompatible',
                    control_value=a.get(metric) if a else None,search_value=b.get(metric) if b else None,
                    difference=d,pairing_type=a.get('pairing_type') if a else None))
                if ok:
                    differences.append(d); gains.append(m[metric]-b[metric]); numerators.append(m[metric]-a[metric])
            margin={'pixel_knn_mean':.0069,'feature_knn_mean':.196}.get(metric)
            denominator=float(np.mean(gains)) if gains else None
            f=(float(np.mean(numerators))/denominator if margin is not None and len(gains)==len(seeds) and denominator>margin else None)
            pairs.append(dict(record_type='mean',model_seed=model_seed,method=method,reference='rms_v3f2',metric=metric,
                n_complete=len(differences),n_expected=len(seeds),status='complete' if len(differences)==len(seeds) else 'incomplete',
                difference=float(np.mean(differences)) if differences else None,practical_margin=margin,
                gain_fraction=f,gain_fraction_note='ratio of paired means' if f is not None else 'undefined, unstable, incomplete, or not a primary distance'))
        if all_ok:
            cr=[lookup[(s,method)] for s in seeds]; sr=[lookup[(s,'rms_v3f2')] for s in seeds]; dr=[lookup[(s,'ddim')] for s in seeds]
            excess = (np.mean([s['class_coverage']-r['class_coverage'] for r,s in zip(cr,sr)])>1 or
                      np.mean([s['class_entropy']-r['class_entropy'] for r,s in zip(cr,sr)])>.1)
            eligible=not guardrail(cr,dr) and not excess
            ds=[(r[k]-s[k],e) for r,s in zip(cr,sr) for k,e in [('pixel_knn_mean',.0069),('feature_knn_mean',.196)]]
            controls[method]=dict(eligible=bool(eligible),
                pilot_practically_close=bool(eligible and all(abs(d)<=e for d,e in ds)),
                pilot_control_not_worse=bool(eligible and all(d<=e for d,e in ds)),
                pilot_search_clear_win=bool(not guardrail(sr,dr) and all(d>e for d,e in ds)))
        else:
            controls[method]=dict(status='incomplete')
    complete=all(x['status']=='complete' for x in inventory)
    decision=dict(status='complete' if complete else 'incomplete',controls=controls,
                  interpretation='Operational pilot rules only; three seeds do not establish statistical equivalence',
                  next_action='No automatic launch; review the frozen NEXT_EXPERIMENTS.md')
    if complete:
        sr=[lookup[(s,'rms_v3f2')] for s in seeds]; dr=[lookup[(s,'ddim')] for s in seeds]
        collapse=guardrail(sr,dr)
        sampling_tie=any(controls[m]['pilot_control_not_worse'] for m in ('ddim_105','ddim_pool_representatives'))
        decision.update(search_diversity_guardrail_violated=bool(collapse),
                        branch='narrow_claim_stop_after_pilot' if collapse or sampling_tie else 'one_confirmation_all_three_controls')
        stable=[x for x in stability if x['method']=='stability_v2_rms']
        stable_ok=all(x['valid_fraction']>=.9 and x['median_rho'] is not None and x['median_rho']<=.9
                      and x['fraction_rho_lt_1']>=.75 for x in stable)
        drift_ok=all(x['median_reference_drift']<=.01 for x in stable)
        decision.update(stability_pilot_positive=bool(stable_ok),stability_drift_small=bool(drift_ok),
                        stability_confirmation_eligible=bool(stable_ok and drift_ok and not(collapse or sampling_tie)))
    tables = dict(raw_runs=raw, summary=summaries, paired_comparisons=pairs,
                  stability_v2=stability, stability_v2_per_candidate=per_candidate,
                  job_status=inventory)
    if read_only:
        return dict(complete=complete, tables=tables, signature=sig, context=manifest['context'])
    if model_seed != 42:
        decision = dict(status='complete' if complete else 'incomplete',
                        interpretation='Per-model descriptive tables; confirmation inference is in the series summary')
    write_table(output/'raw_runs.csv',raw)
    write_table(output/'summary.csv',summaries)
    write_table(output/'paired_comparisons.csv',pairs)
    write_table(output/'stability_v2.csv',stability)
    write_table(output/'stability_v2_per_candidate.csv',per_candidate,('model_seed','search_seed','method','scale','candidate','direction'))
    write_table(output/'job_status.csv',inventory)
    write_json(output/'decisions.json',decision)
    executed = 0
    for entry in inventory:
        if entry['status']=='complete':
            p=root/f"model{model_seed}_seed{entry['search_seed']}"/entry['method']/'summary.json'
            executed += read_json(p).get('newly_executed_nfe',0)
    write_json(output/'compute.json',dict(completed_artifacts_executed_nfe=executed,
        note='Excludes uncheckpointed work in failed/interrupted attempts; per-job status/runtime records identify failures'))
    from .plots import plot
    if model_seed == 42: plot(raw,output)
    (output/'controls_summary.md').write_text(('# Frozen pilot status\n\n' if model_seed == 42 else '# Frozen controls status\n\n')+
        f"Expected quality jobs: {len(seeds)*3}; diagnostics: {len(seeds)*2}; shared-input dependencies: {len(seeds)}.\n\n"+
        f"Complete: {complete}. All expected seeds: {seeds}.\n\n"+
        'See raw_runs.csv for raw values, paired_comparisons.csv for paired differences, '
        'job_status.csv for failures. Stability is a projected-map diagnostic, not proof of a mode.\n')
    return dict(complete=complete,quality_jobs=len(seeds)*3,stability_jobs=len(seeds)*2,
                failed_or_missing=[r for r in inventory if r['status']!='complete'])
