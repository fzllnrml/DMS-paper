"""Atomic completion and immutable dependency signatures, isolated from old outputs."""
from pathlib import Path
import json
import platform
import importlib.metadata
import torch
from src.mnist_experiments.common import digest,write_json,fingerprint,timestamp,git_revision
from .design import study_hash


def signature(config,dependencies):
    return fingerprint(dict(config=config,code=study_hash(),dependencies={str(p):digest(p) for p in dependencies}))


def start(path,sig,resume):
    path=Path(path);meta=path/'started.json';done=path/'complete.json'
    if path.exists() and any(path.iterdir()):
        if not resume: raise FileExistsError(f'{path}: use --resume')
        if not meta.exists() or json.loads(meta.read_text())['signature']!=sig:
            raise ValueError(f'{path}: inputs/config/code changed, refusing to overwrite')
        if done.exists():
            result=json.loads(done.read_text())
            if result['signature']!=sig: raise ValueError('Completion signature mismatch')
            for name,sha in result['artifacts'].items():
                if not (path/name).exists() or digest(path/name)!=sha:
                    raise ValueError(f'Completed artifact missing or changed: {path/name}')
            return True
    path.mkdir(parents=True,exist_ok=True)
    if not meta.exists():
        write_json(meta,dict(signature=sig,created_at=timestamp(),git=git_revision(),implementation_sha256=study_hash(),
                            python=platform.python_version(),torch_version=str(torch.__version__),
                            packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','torchvision')}))
    return False


def complete(path,sig,names):
    path=Path(path)
    write_json(path/'complete.json',dict(signature=sig,finished_at=timestamp(),artifacts={name:digest(path/name) for name in names}))
    if (path/'failure.json').exists(): (path/'failure.json').unlink()


def failure(path,exc):
    path=Path(path)
    # Refused access/resume must not mark valid old outputs as failures.
    if (path/'complete.json').exists() or isinstance(exc,FileExistsError): return
    write_json(path/'failure.json',dict(error=repr(exc),at=timestamp()))


def validate_complete(path):
    path=Path(path)
    try:
        done=json.loads((path/'complete.json').read_text())
        meta=json.loads((path/'started.json').read_text())
        return done['signature']==meta['signature'] and all((path/n).exists() and digest(path/n)==sha for n,sha in done['artifacts'].items())
    except (OSError,KeyError,ValueError):return False
