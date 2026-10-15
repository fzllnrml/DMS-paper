"""Train independent seeds using the existing MNIST trainer and seed42 config."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
import yaml
from .common import digest, fingerprint, git_revision, timestamp, write_json


def seed_training_config(source, seed, output, device):
    if seed not in (43, 44):
        raise ValueError('Publication training seeds are 43 and 44')
    if source.get('format') != 'mnist_diffusion_v1':
        raise ValueError('Expected the original MNIST checkpoint')
    cfg = deepcopy(source['config'])
    if cfg.get('seed') != 42:
        raise ValueError('Training source must be seed42')
    for key in ('model', 'training', 'data'):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f'Seed42 checkpoint must contain its full {key} config')
    for key in ('epochs','batch_size','lr','weight_decay','grad_clip','num_workers'):
        if key not in cfg['training']:
            raise ValueError(f'Seed42 training config lacks {key}')
    if not isinstance(cfg['data'].get('split_seed'),int) or 'split_indices' not in source:
        raise ValueError('Seed42 must record a fixed split seed and split indices')
    if cfg['data'].get('train_limit') or cfg['data'].get('validation_limit'):
        raise ValueError('Refusing to replicate a smoke/subset checkpoint')
    for key, value in source['model_config'].items():
        if cfg['model'].get(key) != value:
            raise ValueError('Saved model and training configuration disagree')
    cfg.update(seed=seed, output_dir=str(output), device=device)
    return cfg


def verify_split(source, data_root):
    """Check the recorded split before spending any training compute."""
    from torchvision.datasets import MNIST
    cfg = source['config']['data']
    dataset = MNIST(data_root, train=True, download=cfg.get('download',False))
    indices = torch.cat([torch.as_tensor(source['split_indices'][key]).long() for key in ('train','validation')])
    if len(indices) != len(dataset.data) or len(indices.unique()) != len(indices):
        raise ValueError('Seed42 split is not a complete disjoint MNIST partition')
    if len(source['split_indices']['validation']) != cfg['validation_size']:
        raise ValueError('Seed42 validation size disagrees with saved config')


def verify_artifact(path, source, config):
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = deepcopy(config)
    actual = deepcopy(state['config'])
    # Trainer may resolve auto into an actual device; these fields are runtime only.
    for c in (actual, expected):
        c.pop('device', None)
        c.pop('output_dir', None)
    if actual != expected or state['model_config'] != source['model_config']:
        raise ValueError('New checkpoint settings differ from the seed42 training recipe')
    for key in ('train','validation'):
        if not torch.equal(torch.as_tensor(state['split_indices'][key]),torch.as_tensor(source['split_indices'][key])):
            raise ValueError('New checkpoint split differs from seed42')
    return state


def load_training_source(c):
    best = torch.load(c['source_checkpoint'], map_location='cpu', weights_only=True)
    if not c.get('source_latest_checkpoint'):
        return best
    latest = torch.load(c['source_latest_checkpoint'], map_location='cpu', weights_only=True)
    if latest['config']['seed'] != 42 or latest['model_config'] != best['model_config']:
        raise ValueError('Seed42 best/latest checkpoints disagree')
    for key in ('train','validation'):
        if not torch.equal(torch.as_tensor(latest['split_indices'][key]),torch.as_tensor(best['split_indices'][key])):
            raise ValueError('Seed42 best/latest splits disagree')
    if latest['epoch'] < latest['config']['training']['epochs']:
        raise ValueError('Seed42 training is incomplete; finish its declared budget first')
    return latest


def train_job(job, resume=False):
    c = job['config']
    source_path = Path(c['source_checkpoint'])
    source = load_training_source(c)
    config = seed_training_config(source,c['model_seed'],c['training_output'],c.get('device','auto'))
    root = Path(job['output'])
    output = Path(config['output_dir'])
    latest = output / 'checkpoints/latest.pt'
    best = output / 'checkpoints/best.pt'
    marker = root / 'complete.json'
    spec = importlib.util.find_spec('src.mnist.train')
    if spec is None or not spec.origin:
        raise RuntimeError('Install into your existing MNIST repository: src/mnist/train.py is required')
    # Include the full existing trainer package, not the publication orchestration.
    trainer_files = sorted(Path(spec.origin).parent.glob('*.py'))
    training_hash = fingerprint({p.name:digest(p) for p in trainer_files})
    signature_config = deepcopy(config)
    signature_config.pop('device',None)
    signature = fingerprint(dict(config=signature_config,source=digest(source_path),
        source_latest=digest(c['source_latest_checkpoint']) if c.get('source_latest_checkpoint') else None,trainer=training_hash))
    if marker.exists():
        record = json.loads(marker.read_text())
        if not resume:
            raise FileExistsError('Training completed; use --resume to skip')
        if record['signature'] != signature or not best.exists() or digest(best) != record['artifact_sha256'] or not latest.exists() or digest(latest) != record['latest_sha256']:
            raise ValueError('Completed training inputs or artifact changed')
        return str(best)
    metadata_path = root / 'metadata.json'
    if metadata_path.exists():
        if not resume:
            raise FileExistsError('Training already started; use --resume')
        if json.loads(metadata_path.read_text())['signature'] != signature:
            raise ValueError('Training config/source/code changed')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Unmanaged training output exists; refusing to overwrite it')
    verify_split(source,config['data']['root'])
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'training_config.yaml'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    if not metadata_path.exists():
        write_json(metadata_path,dict(signature=signature,source_checkpoint=str(source_path),source_sha256=digest(source_path),
            model_seed=c['model_seed'],resolved_config=config,trainer_sha256=training_hash,
            source_latest_checkpoint=c.get('source_latest_checkpoint'),
            source_latest_sha256=digest(c['source_latest_checkpoint']) if c.get('source_latest_checkpoint') else None,
            selection='Existing trainer: best validation loss, same fixed validation procedure',
            git=git_revision(),torch_version=str(torch.__version__),created_at=timestamp()))
    command = [sys.executable,'-m','src.mnist.train','--config',str(config_path)]
    if latest.exists():
        state = verify_artifact(latest,source,config)
        if state['epoch'] >= config['training']['epochs']:
            command = None
        else:
            command += ['--resume',str(latest)]
    elif resume and output.exists() and any(output.iterdir()):
        # A crash before the first epoch leaves no resumable checkpoint. Preserve
        # all such files and allow the existing trainer a clean output directory.
        archived = output.with_name(output.name + '.interrupted-' + timestamp().replace(':','-'))
        output.rename(archived)
    started = time.perf_counter()
    if command:
        subprocess.run(command,check=True)
    verify_artifact(best,source,config)
    final = verify_artifact(latest,source,config)
    if final['epoch'] < config['training']['epochs']:
        raise RuntimeError('Trainer returned before the required epoch budget')
    write_json(marker,dict(signature=signature,artifact=str(best),artifact_sha256=digest(best),
        latest_sha256=digest(latest),epochs=final['epoch'],finished_at=timestamp(),
        last_invocation_wall_seconds=time.perf_counter()-started))
    return str(best)
