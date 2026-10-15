"""Frozen pilot design, metadata-only planning and input discovery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import yaml
from src.mnist_experiments.common import digest, fingerprint, merge_dict, code_digest
from src.mnist_experiments.config import DEFAULT
from src.mnist_experiments.publication import definitions, definition_hash

QUALITY = ('ddim_105', 'low_noise_only', 'ddim_pool_representatives')
DIAGNOSTICS = ('stability_v2_rms', 'stability_v2_ddim')
BASELINES = ('ddim', 'matched_ddim', 'rms_v3f2')
METRICS = ('pixel_knn_mean', 'feature_knn_mean', 'pixel_support_percentile',
           'feature_support_percentile', 'class_coverage', 'class_entropy')
FROZEN = dict(protocol='mnist_controls_pilot_v1', model_seed=42, search_seeds=[0,1,2],
              population_size=128, pool_size=268, batch_size=128,
              methods=list(QUALITY), diagnostics=list(DIAGNOSTICS))
EVALUATION = dict(enabled=True, require_evaluator=True, knn_k=5, train_limit=None,
                  validation_limit=1024, include_test=True, unusually_close_quantile=.01)


def load_config(path='configs/mnist_controls.yaml', device=None):
    supplied = yaml.safe_load(Path(path).read_text())
    allowed = set(FROZEN) | {'output_root','checkpoint','device','evaluation','publication_root','publication_suites'}
    if set(supplied)-allowed:
        raise ValueError('Unknown config keys: '+str(set(supplied)-allowed))
    c = merge_dict(FROZEN, supplied)
    for key, value in FROZEN.items():
        if c[key] != value:
            raise ValueError('Frozen pilot parameter: '+key)
    c['evaluation'] = merge_dict(DEFAULT['evaluation'], c.get('evaluation', {}))
    for key, value in EVALUATION.items():
        if c['evaluation'][key] != value:
            raise ValueError('Publication evaluation must be unchanged: '+key)
    if device:
        c['device'] = device
    return c


def implementation_hash():
    h = hashlib.sha256(code_digest().encode())
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def discover_sources(c):
    """Only small JSON/YAML files; no checkpoints, dataset or neural calls."""
    explicit = {int(k):str(v) for k,v in c.get('publication_suites', {}).items()}
    discovered = {}
    root = Path(c.get('publication_root', 'runs/mnist_experiments/publication_main'))
    for p in sorted(root.glob('**/resolved_config.yaml')) if root.exists() else []:
        cfg = yaml.safe_load(p.read_text())
        if (cfg.get('model_seed') == c['model_seed'] and cfg.get('population_size') == 128
                and cfg.get('search_seed') in c['search_seeds']):
            seed = cfg['search_seed']
            if seed in explicit:
                continue
            if seed in discovered:
                raise ValueError(f'Ambiguous publication suites for seed {seed}; set publication_suites explicitly')
            discovered[seed] = str(p.parent)
    return {seed:explicit.get(seed, discovered.get(seed)) for seed in c['search_seeds']}


def usable_source_files(folder):
    if not folder:
        return []
    p = Path(folder)
    required = [p/'metadata.json',p/'resolved_config.yaml']
    for method in ('rms_v3f2','matched_ddim'):
        required += [p/method/n for n in ('arrays.pt','summary.json','complete.json')]
    return required


def job_specs(c):
    p = c['population_size']
    for seed in c['search_seeds']:
        for method in QUALITY+DIAGNOSTICS:
            diagnostic = method in DIAGNOSTICS
            cost = p*50 if diagnostic else c['pool_size']*50 if method==QUALITY[2] else p*105
            yield dict(id=f"model{c['model_seed']}_seed{seed}/{method}", model_seed=c['model_seed'], search_seed=seed,
                       method=method, kind='diagnostic' if diagnostic else 'quality',
                       output_count=p, expected_nfe=cost)


def plan(c):
    sources = discover_sources(c)
    required = [c['checkpoint'], c['evaluation']['evaluator_checkpoint']]
    data_root = Path(c['evaluation']['data_root'])/'MNIST/raw'
    required += [str(data_root/n) for n in ('train-images-idx3-ubyte','train-labels-idx1-ubyte',
                                          't10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')]
    rows = []
    source_info = {}
    for seed, folder in sources.items():
        files = usable_source_files(folder)
        source_info[seed] = dict(path=folder, complete_tensor_bundle=bool(files) and all(p.exists() for p in files),
                                 missing=[str(p) for p in files if not p.exists()],
                                 validation='Metadata presence only; content compatibility is checked before reuse')
    for job in job_specs(c):
        j = deepcopy(job)
        j['reused_nfe'] = (c['population_size']*50 if j['method']=='low_noise_only' else
                           c['pool_size']*50 if j['method']==QUALITY[2] else 0)
        j['newly_executed_nfe'] = j['expected_nfe']-j['reused_nfe']
        j['reuse_from'] = 'shared inputs dependency' if j['reused_nfe'] else None
        rows.append(j)
    p, n = c['population_size'], len(c['search_seeds'])
    additional = sum(0 if x['complete_tensor_bundle'] else c['pool_size']*50+p*55 for x in source_info.values())
    return dict(protocol=c['protocol'], jobs=rows, quality_jobs=3*n, stability_jobs=2*n,
        dependency_jobs=n, dependencies=required, missing_dependencies=[x for x in required if not Path(x).exists()],
        publication_sources=source_info, fair_controls_and_diagnostic_nfe=sum(j['expected_nfe'] for j in rows),
        new_job_nfe=sum(j['newly_executed_nfe'] for j in rows),
        dependency_nfe_upper=additional, total_new_nfe_upper=sum(j['newly_executed_nfe'] for j in rows)+additional,
        note='Missing old tensors are optional: regenerate pool and full search by unchanged publication definitions. '
             'Source reuse costs are conditional on hash/content validation. No training.')
