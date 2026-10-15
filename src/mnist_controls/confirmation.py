"""Frozen 43/44 confirmation orchestration; all neural methods use the pilot engine."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import yaml

from src.mnist_experiments.common import fingerprint, merge_dict, write_json
from src.mnist_experiments.config import DEFAULT
from . import execution
from .artifacts import exclusive, read_json
from .protocol import FROZEN, EVALUATION, implementation_hash, plan


# Descriptive, validated contract: these values are NOT new algorithm arguments.
# The same unchanged controls.py and publication definitions execute both phases.
METHOD_PARAMETERS = dict(
    ddim_105=dict(steps=105, eta=0, timesteps='round(linspace(999,0,105))'),
    low_noise_only=dict(initialization_steps=50, t=0, steps=[45,10], alpha=[.03,.003],
                        step_mode='rms', projection='clamp', merge=False),
    ddim_pool_representatives=dict(pool_size=268, ddim_steps=50, count=128,
        distance='pixel_rms', first='global_medoid', selection='greedy_facility_location',
        tie_break='lowest_original_index', neural_nfe=0),
    stability_v2=dict(scales=[.02,.05], directions=2, steps=10, t=0, alpha=.003,
        step_mode='rms', projection='clamp', invalid_delta0_le=.01, resample=False,
        rho='RMS(yT-xT)/RMS(y0-x0)', drift='RMS(xT-x0)'))
INFERENCE = dict(pixel_margin=.0069, feature_margin=.196, bootstrap_samples=10000,
                 bootstrap_seed=20261013, family_contrasts=6, family_alpha=.05)
GRID = dict(protocol='mnist_controls_confirmation_v1', model_seeds=[43,44],
            search_seeds=[0,1,2,3,4], population_size=128, pool_size=268, batch_size=128,
            methods=FROZEN['methods'], diagnostics=FROZEN['diagnostics'],
            method_parameters=METHOD_PARAMETERS, inference=INFERENCE)


def validate_paths(c):
    target, pilot = Path(c['output_root']).resolve(), Path(c['pilot_root']).resolve()
    if target == pilot or target in pilot.parents or pilot in target.parents:
        raise ValueError('Confirmation and pilot roots must be disjoint')
    protected = [Path(c['publication_root']).resolve(),
                 Path(c['evaluation']['reference_cache']).resolve()]
    if any(target == p or target in p.parents or p in target.parents for p in protected):
        raise ValueError('Confirmation output must be separate from publication/cache roots')


def load_config(path='configs/mnist_controls_confirmation.yaml', device=None):
    supplied = yaml.safe_load(Path(path).read_text())
    allowed = set(GRID) | {'output_root','pilot_root','checkpoints','device','evaluation',
                           'publication_root','publication_suites'}
    if set(supplied)-allowed:
        raise ValueError('Unknown confirmation keys: '+str(set(supplied)-allowed))
    c = merge_dict(GRID, supplied)
    for key, expected in GRID.items():
        if c[key] != expected:
            raise ValueError('Frozen confirmation parameter: '+key)
    c['checkpoints'] = {int(k):str(v) for k,v in c['checkpoints'].items()}
    if set(c['checkpoints']) != {43,44}:
        raise ValueError('Exactly checkpoints 43 and 44 are required')
    c['evaluation'] = merge_dict(DEFAULT['evaluation'], c.get('evaluation', {}))
    for key, expected in EVALUATION.items():
        if c['evaluation'][key] != expected:
            raise ValueError('Publication evaluation must be unchanged: '+key)
    if device: c['device'] = device
    validate_paths(c)
    return c


def model_configs(c):
    for model_seed in c['model_seeds']:
        child = {k:deepcopy(c[k]) for k in ('protocol','search_seeds','population_size',
                 'pool_size','batch_size','methods','diagnostics','evaluation','device','publication_root')}
        sources = c.get('publication_suites', {})
        child.update(model_seed=model_seed, checkpoint=c['checkpoints'].get(model_seed, c['checkpoints'].get(str(model_seed))),
            publication_suites=deepcopy(sources.get(model_seed, sources.get(str(model_seed), {}))),
            output_root=str(Path(c['output_root'])/f'model{model_seed}'))
        yield child


def dry_run(c):
    plans = [plan(child) for child in model_configs(c)]
    jobs, preparation, sources = [], [], []
    for child, p in zip(model_configs(c), plans):
        jobs += p['jobs']
        for seed, source in p['publication_sources'].items():
            sources.append(dict(model_seed=child['model_seed'], search_seed=seed, **source))
            reused = source['complete_tensor_bundle']
            preparation.append(dict(model_seed=child['model_seed'], search_seed=seed,
                method='inputs', source=source['path'],
                newly_executed_nfe_upper=0 if reused else 20440,
                reused_nfe_upper=20440 if reused else 0))
    required = list(dict.fromkeys(x for p in plans for x in p['dependencies']))
    return dict(protocol=c['protocol'], model_seeds=c['model_seeds'], search_seeds=c['search_seeds'],
        quality_jobs=sum(p['quality_jobs'] for p in plans),
        stability_jobs=sum(p['stability_jobs'] for p in plans), dependency_jobs=len(preparation),
        jobs=jobs, preparation_jobs=preparation, publication_sources=sources,
        dependencies=required, missing_dependencies=[x for x in required if not Path(x).exists()],
        fair_controls_and_diagnostic_nfe=sum(p['fair_controls_and_diagnostic_nfe'] for p in plans),
        new_job_nfe=sum(p['new_job_nfe'] for p in plans),
        reused_nfe=sum(j['reused_nfe'] for j in jobs),
        dependency_nfe_upper=sum(p['dependency_nfe_upper'] for p in plans),
        total_new_nfe_upper=sum(p['total_new_nfe_upper'] for p in plans),
        pilot_analysis_available=(Path(c['pilot_root'])/'manifest.json').exists(),
        note='Presence-only dry-run: reuse is conditional on the unchanged content/hash checks. '
             'Preparation is charged once per model/search seed; shared reuse is not additive work. '
             'Missing old tensors are regenerated via unchanged publication definitions. No training.')


def execute(c, resume=False):
    validate_paths(c)
    root = Path(c['output_root'])
    identity = dict(config=c, implementation=implementation_hash())
    signature = fingerprint(identity)
    from .confirmation_analysis import aggregate_confirmation
    with exclusive(root):
        manifest = root/'confirmation_manifest.json'
        if manifest.exists():
            if read_json(manifest)['signature'] != signature:
                raise ValueError('Confirmation context changed; use a distinct output root')
            if not resume:
                raise FileExistsError('Confirmation exists; use --resume')
        else:
            if any(root.glob('model*/manifest.json')):
                raise ValueError('Existing model outputs without confirmation manifest')
            write_json(manifest, dict(signature=signature, **identity))
            write_json(root/'resolved_config.json',c)
            write_json(root/'plan.json',dry_run(c))
        errors = {}
        for child in model_configs(c):
            try:
                execution.execute(child, resume=resume)
            except Exception as error:
                errors[str(child['model_seed'])] = f'{type(error).__name__}: {error}'
        # Current invocation errors cannot be hidden by older complete artifacts.
        write_json(root/'last_confirmation_invocation.json',dict(errors=errors))
        result = aggregate_confirmation(c)
        result['execution_errors'] = errors
        result['complete'] &= not errors
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/mnist_controls_confirmation.yaml')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--aggregate',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'])
    args=parser.parse_args(argv)
    if args.resume and not args.execute: parser.error('--resume requires --execute')
    try:
        c=load_config(args.config,args.device)
        if args.aggregate:
            from .confirmation_analysis import aggregate_confirmation
            with exclusive(c['output_root']): result=aggregate_confirmation(c)
        elif args.execute:
            result=execute(c,args.resume)
        else:
            print(json.dumps(dry_run(c),indent=2)); return 0
        print(json.dumps(result,indent=2))
        return 0 if result['complete'] else 1
    except Exception as error:
        print(f'{type(error).__name__}: {error}')
        return 1


if __name__=='__main__': raise SystemExit(main())
