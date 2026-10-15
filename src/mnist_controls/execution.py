"""Shared inputs, strict provenance and isolated execution of the frozen pilot."""
from copy import deepcopy
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
import yaml
from src.mnist_experiments.common import (CountedModel, PixelSpace, gaussian_noise, digest,
    fingerprint, write_json, seed_all, synchronize, code_digest, device_for)
from src.mnist_experiments.publication import definitions, definition_hash
from src.mnist_experiments.run import load_model
from src.mnist_experiments.search import run_search
from src.mnist_experiments.data import ReferenceData
from src.mnist_experiments.evaluator import FrozenEvaluator
from src.mnist_experiments.publication_evaluation import PublicationSupport
from .protocol import (QUALITY, DIAGNOSTICS, BASELINES, METRICS, job_specs, plan,
                       discover_sources, usable_source_files, implementation_hash, tensor_hash)
from .controls import ddim_105, ddim_indices, low_noise_only, select_representatives, stability_v2
from .artifacts import exclusive, read_json, run_job, inspect_job


def validate_pixels(x, count):
    if x.shape != (count,784) or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError('Invalid candidate tensor')
    if (x.abs() > 1.00001).any():
        raise ValueError('Candidates outside publication pixel box')


def dependency_context(c):
    """Hash actual input bytes, not only filenames. No model evaluations."""
    files = [Path(c['checkpoint']), Path(c['evaluation']['evaluator_checkpoint'])]
    raw = Path(c['evaluation']['data_root'])/'MNIST/raw'
    files += [raw/n for n in ('train-images-idx3-ubyte','train-labels-idx1-ubyte',
                             't10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError('Required inputs missing (no implicit download/training): '+', '.join(missing))
    sources = discover_sources(c)
    used_sources = {}
    for seed, folder in sources.items():
        paths = usable_source_files(folder)
        if paths and all(p.exists() for p in paths):
            files += paths
            used_sources[seed] = folder
        else:
            used_sources[seed] = None
    device = str(device_for(c['device']))
    return dict(config=c, implementation=implementation_hash(), publication_definition=definition_hash(),
                publication_implementation=code_digest(), files={str(p):digest(p) for p in files},
                source_suites=used_sources, torch=str(torch.__version__), numpy=np.__version__,
                device=device, python=platform.python_version())


def load_backend(c, context):
    model, space, state = load_model(c['checkpoint'], context['device'])
    if space.name != 'pixel' or model.schedule.T != 1000:
        raise ValueError('Frozen controls require the original MNIST pixel model, T=1000')
    if state.get('config', {}).get('seed') != c['model_seed']:
        raise ValueError(f"Diffusion checkpoint must identify model seed{c['model_seed']}")
    if model.config.get('beta_start') != .0001 or model.config.get('beta_end') != .02:
        raise ValueError('Publication linear beta schedule mismatch')
    evaluator = FrozenEvaluator.load(c['evaluation']['evaluator_checkpoint'], context['device'], c['batch_size'])
    data = ReferenceData(state, c['evaluation']['data_root'], False, c['evaluation']['include_test'])
    for key in ('train','validation'):
        if not torch.equal(evaluator.metadata[key+'_indices'], data.indices[key]):
            raise ValueError('Evaluator/diffusion split mismatch')
    support = PublicationSupport(data, evaluator, c['evaluation'], digest(c['evaluation']['evaluator_checkpoint']))
    return model, space, evaluator, support


def load_publication_inputs(folder, c, context, seed, noise):
    """Import endpoints only after checking definitions, weights, starts and pairing."""
    root = Path(folder)
    metadata = read_json(root/'metadata.json')
    cfg = yaml.safe_load((root/'resolved_config.yaml').read_text())
    expected = dict(model_seed=c['model_seed'], search_seed=seed, population_size=c['population_size'],
                    initial_noise_seed=seed, batch_size=c['batch_size'], ddim_steps=50)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('Imported publication config mismatch: '+key)
    if metadata.get('definition_hash') != definition_hash():
        raise ValueError('Imported publication definitions differ')
    if metadata.get('implementation_sha256') != code_digest():
        raise ValueError('Imported publication implementation differs')
    for key, path in [('checkpoint_sha256',c['checkpoint']),('evaluator_sha256',c['evaluation']['evaluator_checkpoint'])]:
        if metadata.get(key) != context['files'][path]:
            raise ValueError('Imported '+key+' mismatch')
    for key in ('knn_k','train_limit','validation_limit','include_test','unusually_close_quantile'):
        if cfg['evaluation'].get(key) != c['evaluation'][key]:
            raise ValueError('Imported evaluation differs: '+key)
    arrays = {}
    for method in ('matched_ddim','rms_v3f2'):
        row = read_json(root/method/'summary.json')
        completed = read_json(root/method/'complete.json')
        if not metadata.get('signature') or completed.get('signature') != metadata['signature'] or completed.get('summary') != row:
            raise ValueError('Imported publication completion marker mismatch')
        for key, value in expected.items():
            if key in ('batch_size','ddim_steps'):
                continue
            if row.get(key) != value:
                raise ValueError('Imported row mismatch: '+key)
        if row.get('definition_hash') != definition_hash() or row.get('resolved_method_definition') != definitions()[method]:
            raise ValueError('Imported method definition mismatch')
        if row.get('checkpoint_sha256') != metadata['checkpoint_sha256'] or row.get('evaluator_sha256') != metadata['evaluator_sha256']:
            raise ValueError('Imported row checkpoint mismatch')
        a = torch.load(root/method/'arrays.pt', map_location='cpu', weights_only=True)
        count = c['pool_size'] if method == 'matched_ddim' else c['population_size']
        if not torch.equal(a['initial_noise'], noise[:count]):
            raise ValueError('Imported Gaussian noise mismatch')
        validate_pixels(a['initial'], count); validate_pixels(a['candidates'], count)
        if row.get('initial_gaussian_prefix_sha256') != tensor_hash(noise[:c['population_size']]):
            raise ValueError('Imported noise prefix hash mismatch')
        if row.get('initial_ddim_prefix_sha256') != tensor_hash(a['initial'][:c['population_size']]):
            raise ValueError('Imported image prefix hash mismatch')
        if method == 'matched_ddim' and (row['nfe'] != c['pool_size']*50 or not torch.equal(a['initial'], a['candidates'])):
            raise ValueError('Imported matched pool mismatch')
        arrays[method] = (a,row)
    pool = arrays['matched_ddim'][0]['candidates']
    initial = arrays['rms_v3f2'][0]['initial']
    if not torch.equal(initial, pool[:c['population_size']]):
        raise ValueError('Imported DDIM prefix is not the search initialization')
    return pool, initial, arrays['rms_v3f2'][0]['candidates'], arrays['rms_v3f2'][1]['nfe']


def evaluate(support, evaluator, pixels, count):
    before = evaluator.images_evaluated
    started = time.perf_counter()
    full = support.evaluate(pixels.reshape(-1,1,28,28), count)
    row = {}
    for kind in ('pixel','feature'):
        row[kind+'_knn_mean'] = float(full[kind]['knn_mean_distance'].mean())
        row[kind+'_support_percentile'] = float(np.mean(full[kind]['knn_percentiles']['support_percentile']))
    for key in ('class_coverage','class_entropy'):
        row[key] = full['semantics'][key]
    if not all(np.isfinite(row[k]) for k in METRICS):
        raise FloatingPointError('Nonfinite publication metrics')
    row.update(evaluation_wall_seconds=time.perf_counter()-started,
               evaluator_images=evaluator.images_evaluated-before,
               evaluation_cache_key=support.metadata['cache_key'])
    return row, full


def provenance(context, seed, noise, initial, support):
    return dict(model_seed=context['config']['model_seed'], search_seed=seed, checkpoint_sha256=context['files'][context['config']['checkpoint']],
        evaluator_sha256=context['files'][context['config']['evaluation']['evaluator_checkpoint']],
        publication_definition_hash=definition_hash(), implementation=context['implementation'],
        initial_gaussian_prefix_sha256=tensor_hash(noise[:len(initial)]),
        initial_ddim_prefix_sha256=tensor_hash(initial), reference_metadata=support.metadata,
        evaluation_cache_key=support.metadata['cache_key'], device=context['device'],
        torch=context['torch'], context_signature=fingerprint(context))


def execute(c, resume=False):
    root = Path(c['output_root'])
    context = dependency_context(c)
    signature = fingerprint(context)
    specs = list(job_specs(c))
    with exclusive(root):
        manifest = root/'manifest.json'
        if manifest.exists():
            if read_json(manifest)['signature'] != signature:
                raise ValueError('Output context changed: config/code/checkpoint/evaluator/data/source hashes; use a distinct output root')
            if not resume:
                raise FileExistsError('Pilot exists; use --resume')
        else:
            write_json(manifest, dict(signature=signature, context=context, jobs=specs))
            write_json(root/'plan.json', plan(c))
        if (root/'reference.json').exists():
            ref = read_json(root/'reference.json')
            cache = Path(c['evaluation']['reference_cache'])/(ref['metadata']['cache_key']+'.pt')
            if not cache.exists() or digest(cache) != ref['cache_sha256']:
                raise ValueError('Reference cache changed or is missing')
        all_paths = [root/j['id'] for j in specs] + [root/f"model{c['model_seed']}_seed{s}/inputs" for s in c['search_seeds']]
        if resume and all(inspect_job(p, signature)[0]=='complete' for p in all_paths):
            from .aggregate import aggregate
            return aggregate(root)
        try:
            model, space, evaluator, support = load_backend(c, context)
        except Exception as error:
            paths = [root/j['id'] for j in specs] + [root/f"model{c['model_seed']}_seed{s}/inputs" for s in c['search_seeds']]
            for path in paths:
                if inspect_job(path, signature)[0] == 'complete':
                    continue
                def blocked(message=str(error)):
                    raise RuntimeError('Backend dependency failed: '+message)
                try: run_job(path, signature, c, blocked, resume)
                except Exception: pass
            write_json(root/'last_invocation.json',dict(backend_error=str(error)))
            from .aggregate import aggregate
            aggregate(root)
            raise
        reference = dict(metadata=support.metadata,
                         checkpoint=context['files'][c['checkpoint']],
                         evaluator=context['files'][c['evaluation']['evaluator_checkpoint']])
        cache_path = Path(c['evaluation']['reference_cache'])/(support.metadata['cache_key']+'.pt')
        reference['cache_sha256'] = digest(cache_path)
        if (root/'reference.json').exists() and read_json(root/'reference.json') != reference:
            raise ValueError('Reference cache or reference definition changed')
        write_json(root/'reference.json', reference)
        errors = []
        for seed in c['search_seeds']:
            seed_all(seed)
            suite = root/f"model{c['model_seed']}_seed{seed}"
            noise = gaussian_noise(c['pool_size'], 784, seed)
            source = context['source_suites'].get(seed, context['source_suites'].get(str(seed)))
            def prepare():
                counted = CountedModel(model, c['batch_size'])
                synchronize(model.device); started = time.perf_counter()
                if source:
                    pool, initial, full, full_nfe = load_publication_inputs(source, c, context, seed, noise)
                    method = 'validated publication tensors'
                else:
                    ddim_indices(model.schedule.T, 50)
                    boundary = c['population_size']
                    initial = counted.ddim(noise[:boundary], 50)
                    pool = torch.cat((initial, counted.ddim(noise[boundary:],50)))
                    before = counted.nfe
                    found = run_search(counted, initial, definitions()['rms_v3f2'], space)
                    full = found['candidates']
                    full_nfe = boundary*50+counted.nfe-before
                    method = 'regenerated unchanged publication protocol'
                for x, n in ((pool,c['pool_size']),(initial,c['population_size']),(full,c['population_size'])):
                    validate_pixels(x,n)
                synchronize(model.device)
                elapsed = time.perf_counter()-started
                prov = provenance(context,seed,noise,initial,support)
                baseline_rows, baseline_metrics = [], {}
                for name, x, cost in [('ddim',initial,c['population_size']*50),
                                      ('matched_ddim',pool,c['pool_size']*50),('rms_v3f2',full,full_nfe)]:
                    values, metrics = evaluate(support,evaluator,x.cpu(),len(x))
                    baseline_rows.append(dict({**{k:v for k,v in prov.items() if k!='reference_metadata'}, **values},
                        method=name,model_seed=c['model_seed'],search_seed=seed,output_count=len(x),nfe=cost))
                    baseline_metrics[name] = metrics
                summary = dict(method='inputs',model_seed=c['model_seed'],search_seed=seed,newly_executed_nfe=counted.nfe,
                               forward_calls=counted.forward_calls,source=method,source_directory=source,
                               generation_wall_seconds=elapsed,full_search_nfe=full_nfe,baseline_rows=baseline_rows)
                return summary, baseline_metrics, dict(noise=noise,initial=initial,pool=pool,full=full), prov
            try:
                inputs_row = run_job(suite/'inputs',signature,c,prepare,resume)
                data = torch.load(suite/'inputs/arrays.pt',map_location='cpu',weights_only=True)
                prov = read_json(suite/'inputs/metadata.json')
            except Exception as e:
                errors.append(dict(seed=seed,job='inputs',error=str(e)))
                # Preserve completed jobs, explicitly mark unexecuted dependents.
                for j in [j for j in specs if j['search_seed']==seed]:
                    if inspect_job(root/j['id'],signature)[0] != 'complete':
                        def blocked(error=str(e)):
                            raise RuntimeError('Shared input dependency failed: '+error)
                        try: run_job(root/j['id'],signature,c,blocked,resume)
                        except Exception: pass
                continue
            for spec in [j for j in specs if j['search_seed']==seed]:
                name = spec['method']
                def action(name=name, spec=spec):
                    counted = CountedModel(model,c['batch_size'])
                    initial = data['initial'].to(model.device)
                    arrays, extra, metrics = {}, {}, {}
                    synchronize(model.device); started = time.perf_counter()
                    if name == 'ddim_105':
                        pixels = ddim_105(counted,data['noise'][:c['population_size']])
                        pairing = 'same Gaussian prefix; different DDIM trajectory'
                        reused = 0
                        arrays['initial_noise'] = data['noise'][:c['population_size']]
                    elif name == 'low_noise_only':
                        pixels = low_noise_only(counted,initial)
                        pairing = 'identical DDIM-50 initialization'
                        reused = c['population_size']*50
                    elif name == 'ddim_pool_representatives':
                        pixels, indices, extra = select_representatives(data['pool'],c['population_size'])
                        arrays['selected_indices'] = indices
                        pairing = 'same 268-sample generated pool; set-level comparison, no pointwise pairing to 128 search outputs'
                        reused = c['pool_size']*50
                    else:
                        x = data['full'] if name=='stability_v2_rms' else data['initial']
                        metrics, arrays = stability_v2(counted,x.to(model.device),seed)
                        pairing = 'same direction RNG; reference and perturbed trajectories use identical ten-step map'
                        reused = 0
                    synchronize(model.device)
                    generation_seconds = time.perf_counter()-started
                    if name in QUALITY:
                        validate_pixels(pixels,c['population_size'])
                        arrays['candidates'] = pixels
                        values, metrics = evaluate(support,evaluator,pixels.cpu(),c['population_size'])
                    else:
                        values = {'stability_summary':metrics['summary']}
                    if counted.nfe+reused != spec['expected_nfe']:
                        raise RuntimeError('Actual NFE does not match frozen control budget')
                    summary = dict({**{k:v for k,v in prov.items() if k!='reference_metadata'},**values,**extra},
                        method=name,model_seed=c['model_seed'],search_seed=seed,output_count=c['population_size'],
                        kind=spec['kind'],nfe=counted.nfe+reused,newly_executed_nfe=counted.nfe,reused_nfe=reused,
                        diagnostic_nfe=counted.nfe if name in DIAGNOSTICS else 0,
                        forward_calls=counted.forward_calls,generation_wall_seconds=generation_seconds,
                        pairing_type=pairing,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
                    return summary,metrics,arrays,dict(prov,pairing_type=pairing,shared_inputs_sha256=digest(suite/'inputs/arrays.pt'))
                try:
                    run_job(root/spec['id'],signature,c,action,resume)
                except Exception as e:
                    errors.append(dict(seed=seed,job=name,error=str(e)))
        write_json(root/'last_invocation.json',dict(errors=errors))
        from .aggregate import aggregate
        result = aggregate(root)
        if errors:
            raise RuntimeError(f'{len(errors)} failed/incompatible jobs: {errors}; see summary/job_status.csv')
        return result
