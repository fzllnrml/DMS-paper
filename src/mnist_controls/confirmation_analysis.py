"""Read existing artifacts; report all expected runs and the frozen confirmation rule."""
from pathlib import Path
import numpy as np
from src.mnist_experiments.common import digest, fingerprint, write_json
from .aggregate import aggregate, write_table, guardrail
from .artifacts import read_json
from .protocol import QUALITY, DIAGNOSTICS, BASELINES, METRICS, FROZEN, EVALUATION, job_specs
from .confirmation import INFERENCE, model_configs, validate_paths

TABLES = ('raw_runs','summary','paired_comparisons','stability_v2',
          'stability_v2_per_candidate','job_status')
PRIMARY = {'pixel_knn_mean':.0069, 'feature_knn_mean':.196}


def missing_report(c, status='missing', error='No model manifest'):
    tables = {key:[] for key in TABLES}
    for seed in c['search_seeds']:
        tables['job_status'].append(dict(model_seed=c['model_seed'],search_seed=seed,
            method='inputs',kind='dependency',status=status,error=error))
        for method in BASELINES:
            tables['raw_runs'].append(dict(model_seed=c['model_seed'],search_seed=seed,
                method=method,kind='baseline',status=status,error=error))
    for job in job_specs(c):
        row=dict(job,status=status,error=error)
        tables['job_status'].append(row)
        if job['kind']=='quality': tables['raw_runs'].append(row)
        else:
            tables['stability_v2'] += [dict(row,scale=scale) for scale in (.02,.05)]
    for method in BASELINES+QUALITY:
        for metric in METRICS+('nfe',):
            tables['summary'].append(dict(model_seed=c['model_seed'],method=method,
                metric=metric,n_expected=len(c['search_seeds']),n_complete=0,status=status,mean=None))
    for method in QUALITY:
        for metric in METRICS:
            for seed in c['search_seeds']:
                tables['paired_comparisons'].append(dict(record_type='pair',model_seed=c['model_seed'],
                    search_seed=seed,method=method,reference='rms_v3f2',metric=metric,status=status,
                    difference=None))
    return dict(complete=False,tables=tables)


def collect(c):
    root=Path(c['output_root'])
    if not (root/'manifest.json').exists(): return missing_report(c)
    try:
        manifest=read_json(root/'manifest.json')
        if manifest['signature'] != fingerprint(manifest['context']):
            raise ValueError('Manifest context/signature mismatch')
        if fingerprint(manifest['context']['config']) != fingerprint(c):
            raise ValueError('Unexpected per-model resolved config')
        if manifest['jobs'] != list(job_specs(c)):
            raise ValueError('Unexpected job inventory')
        return aggregate(root,read_only=True)
    except (ValueError,KeyError,OSError) as error:
        return missing_report(c,'incompatible',str(error))


def intervals(pairs, models, seeds):
    """Run-level paired bootstrap, stratified by fixed model, equal model weights."""
    rows=[]
    for method in QUALITY:
        for metric,margin in PRIMARY.items():
            lookup={(r['model_seed'],r['search_seed']):r for r in pairs
                    if r['record_type']=='pair' and r['method']==method and r['metric']==metric}
            ordered=[lookup.get((m,s)) for m in models for s in seeds]
            good=[r for r in ordered if r and r['status']=='complete' and r.get('difference') is not None]
            row=dict(record_type='confirmation_interval',method=method,reference='rms_v3f2',
                metric=metric,practical_margin=margin,n_expected=len(ordered),n_complete=len(good),
                status='complete' if len(good)==len(ordered) else 'incomplete',
                interval_level=1-INFERENCE['family_alpha']/INFERENCE['family_contrasts'],
                bootstrap_samples=INFERENCE['bootstrap_samples'],bootstrap_seed=INFERENCE['bootstrap_seed'])
            if len(good)==len(ordered):
                values=np.array([r['difference'] for r in ordered],dtype=float).reshape(len(models),len(seeds))
                rng=np.random.default_rng(INFERENCE['bootstrap_seed'])
                # Reuse the same resampling plan for the six primary contrasts.
                draws=np.zeros(INFERENCE['bootstrap_samples'])
                means=values.mean(axis=1)
                for v in values:
                    indices=rng.integers(len(seeds),size=(INFERENCE['bootstrap_samples'],len(seeds)))
                    draws += v[indices].mean(axis=1)/len(models)
                tail=INFERENCE['family_alpha']/(2*INFERENCE['family_contrasts'])
                lo,hi=np.quantile(draws,[tail,1-tail])
                row.update(difference=float(means.mean()),ci_lower=float(lo),ci_upper=float(hi),
                    model_means={str(m):float(v) for m,v in zip(models,means)},
                    search_win_quality=bool(np.all(means>margin) and lo>0),
                    equivalent_quality=bool(np.all(abs(means)<=margin) and lo>=-margin and hi<=margin),
                    control_noninferior_quality=bool(hi<=margin),
                    control_win_quality=bool(np.all(means < -margin) and hi<0))
            rows.append(row)
    return rows


def decisions(tables, models, seeds, complete):
    ci=intervals(tables['paired_comparisons'],models,seeds)
    lookup={(r['model_seed'],r['search_seed'],r['method']):r for r in tables['raw_runs'] if r['status']=='complete'}
    result=dict(status='complete' if complete else 'incomplete',controls={},
        inference=INFERENCE,interpretation='Confirmation only; fixed models, run-level paired inference. '
        'Pilot excluded from confidence intervals. Projected-map contraction is not proof of a mode.',
        next_action='Stop experimental series after this confirmation; no automatic further runs.')
    for method in QUALITY:
        rows=[r for r in ci if r['method']==method]
        guards=[]
        for model in models:
            required=[lookup.get((model,s,m)) for s in seeds for m in (method,'rms_v3f2','ddim')]
            if not all(required): continue
            cr=[lookup[(model,s,method)] for s in seeds]
            sr=[lookup[(model,s,'rms_v3f2')] for s in seeds]
            dr=[lookup[(model,s,'ddim')] for s in seeds]
            excess=(np.mean([s['class_coverage']-c['class_coverage'] for c,s in zip(cr,sr)])>1 or
                    np.mean([s['class_entropy']-c['class_entropy'] for c,s in zip(cr,sr)])>.1)
            guards.append(dict(model_seed=model,search_guardrail=bool(guardrail(sr,dr)),
                control_guardrail=bool(guardrail(cr,dr)),control_excess_diversity_loss=bool(excess)))
        eligible=complete and len(guards)==len(models) and all(r['status']=='complete' for r in rows)
        result['controls'][method]=dict(status='complete' if eligible else 'incomplete',diversity=guards)
        if eligible:
            search_ok=not any(g['search_guardrail'] for g in guards)
            control_ok=not any(g['control_guardrail'] or g['control_excess_diversity_loss'] for g in guards)
            result['controls'][method].update(
                search_confirmed=bool(search_ok and all(r['search_win_quality'] for r in rows)),
                practical_equivalence=bool(search_ok and control_ok and all(r['equivalent_quality'] for r in rows)),
                control_noninferior=bool(control_ok and all(r['control_noninferior_quality'] for r in rows)),
                control_confirmed=bool(control_ok and all(r['control_win_quality'] for r in rows)))
    stability=[]
    for model in models:
        rows=[r for r in tables['stability_v2'] if r['model_seed']==model and r['method']=='stability_v2_rms']
        valid=all(r['status']=='complete' for r in rows) and len(rows)==2*len(seeds)
        if not valid:
            stability.append(dict(model_seed=model,status='incomplete'));continue
        by_seed={s:[r for r in rows if r['search_seed']==s] for s in seeds}
        passed=sum(all(r['valid_fraction']>=.9 and r['median_rho'] is not None and
                       r['median_rho']<=.9 and r['fraction_rho_lt_1']>=.75 for r in group)
                   for group in by_seed.values())
        small_drift=sum(all(r['median_reference_drift']<=.01 for r in group) for group in by_seed.values())
        stability.append(dict(model_seed=model,status='complete',passing_runs_both_scales=passed,
            small_drift_runs=small_drift,all_runs_valid=all(r['valid_fraction']>=.9 for r in rows),
            positive=bool(all(r['valid_fraction']>=.9 for r in rows) and passed>=4 and small_drift>=4)))
    result['stability_v2']=stability
    result['stability_confirmed']=bool(complete and all(r.get('positive',False) for r in stability))
    return result,ci


def pilot_report(c):
    root=Path(c['pilot_root'])
    placeholder=dict(FROZEN,output_root=str(root))
    if not (root/'manifest.json').exists(): return missing_report(placeholder)
    try:
        manifest=read_json(root/'manifest.json'); pc=manifest['context']['config']
        for key,value in FROZEN.items():
            if pc[key]!=value: raise ValueError('Unexpected pilot protocol: '+key)
        for key,value in EVALUATION.items():
            if pc['evaluation'][key]!=value: raise ValueError('Pilot evaluation mismatch: '+key)
        # Keep stored paths in the signed config; artifacts may have been relocated.
        if manifest['signature']!=fingerprint(manifest['context']):
            raise ValueError('Pilot signature mismatch')
        if manifest['jobs']!=list(job_specs(pc)): raise ValueError('Pilot inventory mismatch')
        return aggregate(root,read_only=True)
    except (KeyError,ValueError,OSError) as error:
        return missing_report(placeholder,'incompatible',str(error))


def aggregate_confirmation(c):
    validate_paths(c)
    root=Path(c['output_root']); manifest_path=root/'confirmation_manifest.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path)
        saved=manifest['config']
        # --aggregate without --device reuses the recorded execution device.
        candidate=dict(c,device=saved['device']) if c['device']=='auto' else c
        if fingerprint(candidate)!=fingerprint(saved): raise ValueError('Aggregation config mismatch')
        if manifest['signature']!=fingerprint(dict(config=saved,implementation=manifest['implementation'])):
            raise ValueError('Confirmation manifest signature mismatch')
        c=saved
    else:
        raise FileNotFoundError('No confirmation manifest; use --dry-run or --execute first')
    errors=read_json(root/'last_confirmation_invocation.json').get('errors',{}) if (root/'last_confirmation_invocation.json').exists() else {}
    tables={key:[] for key in TABLES}; reports=[]
    for child in model_configs(c):
        report=collect(child)
        error=errors.get(str(child['model_seed']))
        if error:
            if report['complete']:
                report=missing_report(child,'incompatible',error)
            elif not (Path(child['output_root'])/'manifest.json').exists():
                report=missing_report(child,'failed',error)
        reports.append(report)
        for key in TABLES: tables[key] += [dict(r,phase='confirmation') for r in report['tables'][key]]
    complete=all(r['complete'] for r in reports) and not errors
    decision,ci=decisions(tables,c['model_seeds'],c['search_seeds'],complete)
    tables['paired_comparisons'] += [dict(r,phase='confirmation') for r in ci]
    output=root/'summary'; output.mkdir(parents=True,exist_ok=True)
    for key,rows in tables.items(): write_table(output/(key+'.csv'),rows)
    write_json(output/'decisions.json',decision)
    write_json(output/'compute.json',dict(completed_artifacts_executed_nfe=sum(
        read_json(Path(child['output_root'])/f"model{child['model_seed']}_seed{r['search_seed']}"/r['method']/'summary.json').get('newly_executed_nfe',0)
        for child,report in zip(model_configs(c),reports) for r in report['tables']['job_status'] if r['status']=='complete'),
        note='Completed artifacts only; failed/interrupted work is not silently counted as zero-cost success.'))
    pilot=pilot_report(c)
    combined=output/'combined'; combined.mkdir(exist_ok=True)
    for key in TABLES:
        write_table(combined/(key+'.csv'),[dict(r,phase='pilot') for r in pilot['tables'][key]]+tables[key])
    # No aggregate(pilot_root) call in write mode, no copying over old CSVs.
    source_paths=[Path(c['pilot_root'])/'manifest.json']+[Path(x['output_root'])/'manifest.json' for x in model_configs(c)]
    cache_keys={r.get('evaluation_cache_key') for r in pilot['tables']['raw_runs']+tables['raw_runs']
                if r['status']=='complete' and r.get('evaluation_cache_key')}
    evaluator_hashes={r.get('evaluator_sha256') for r in pilot['tables']['raw_runs']+tables['raw_runs']
                      if r['status']=='complete' and r.get('evaluator_sha256')}
    comparable=len(cache_keys)==1 and len(evaluator_hashes)==1
    combined_complete=bool(complete and pilot['complete'] and comparable)
    write_json(combined/'provenance.json',dict(complete=combined_complete,pilot_complete=pilot['complete'],
        confirmation_complete=complete,same_evaluator_and_reference_cache=comparable,
        source_manifest_sha256={str(p):digest(p) for p in source_paths if p.exists()},
        note='Sources read only. Phases remain separate; pilot never enters confirmation confidence intervals. '
             'Missing/incompatible phases and failures remain explicit; no run-level values fabricated.'))
    (output/'controls_summary.md').write_text(
        '# Frozen confirmation\n\n30 quality jobs, 20 diagnostics, 10 input dependencies.\n\n'
        f'Complete: {complete}. Combined with read-only pilot: {combined_complete}.\n\n'
        'All paired raw values are retained. Confirmation intervals exclude pilot and use 10,000 '
        'stratified paired bootstrap resamples, equal model weights, six-contrast Bonferroni family 95%. '
        'See decisions.json, paired_comparisons.csv and job_status.csv. '
        'Stability is a diagnostic of the specified projected map, not proof of a mode.\n')
    return dict(complete=complete,combined_complete=combined_complete,quality_jobs=30,stability_jobs=20,
        dependency_jobs=10,failed_or_missing=[r for r in tables['job_status'] if r['status']!='complete'],
        summary_directory=str(output))
