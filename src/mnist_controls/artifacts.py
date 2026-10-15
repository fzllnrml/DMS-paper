"""Atomic job completion with checksums; never replace a valid completed job."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time
import traceback
from src.mnist_experiments.common import digest, save_pt, write_json, timestamp


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root/'.pilot.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('Another controls process holds this output root') from e
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def inspect_job(folder, signature=None):
    p = Path(folder)
    try:
        if (p/'complete.json').exists():
            state = read_json(p/'complete.json')
            if signature is not None and state['signature'] != signature:
                return 'incompatible', 'Signature changed'
            for name, sha in state['artifacts'].items():
                if not (p/name).exists() or digest(p/name) != sha:
                    return 'corrupt', 'Checksum mismatch: '+name
            return 'complete', None
        if (p/'status.json').exists():
            s = read_json(p/'status.json')
            if signature is not None and s['signature'] != signature:
                return 'incompatible', 'Signature changed'
            return s['status'], s.get('error')
        return 'missing', None
    except (ValueError, KeyError, OSError) as e:
        return 'corrupt', str(e)


def run_job(folder, signature, config, action, resume=False):
    p = Path(folder)
    state, error = inspect_job(p, signature)
    if state in ('corrupt','incompatible'):
        raise ValueError(f'{p}: {state}: {error}')
    if state == 'complete':
        if not resume:
            raise FileExistsError(f'{p}: completed job; use --resume')
        return read_json(p/'summary.json')
    if p.exists() and any(p.iterdir()) and not resume:
        raise FileExistsError(f'{p}: partial job; use --resume')
    p.mkdir(parents=True, exist_ok=True)
    write_json(p/'resolved_config.json', config)
    write_json(p/'status.json', dict(status='running', signature=signature, started_at=timestamp()))
    start = time.perf_counter()
    try:
        summary, metrics, arrays, provenance = action()
        runtime = time.perf_counter()-start
        summary['job_wall_seconds'] = runtime
        save_pt(p/'arrays.pt', arrays)
        write_json(p/'metrics.json', metrics)
        write_json(p/'metadata.json', provenance)
        write_json(p/'summary.json', summary)
        write_json(p/'runtime.json', dict(job_wall_seconds=runtime, nfe={k:v for k,v in summary.items() if 'nfe' in k}))
        write_json(p/'status.json', dict(status='complete', signature=signature, finished_at=timestamp()))
        names = ['resolved_config.json','arrays.pt','metrics.json','metadata.json','summary.json','runtime.json','status.json']
        write_json(p/'complete.json', dict(signature=signature, artifacts={n:digest(p/n) for n in names}))
        return summary
    except Exception as e:
        write_json(p/'status.json', dict(status='failed', signature=signature, error=f'{type(e).__name__}: {e}',
                                      traceback=traceback.format_exc(), runtime_seconds=time.perf_counter()-start))
        raise
