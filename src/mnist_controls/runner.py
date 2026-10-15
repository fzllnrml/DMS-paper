"""Explicit dry-run/execute/resume/aggregate for the frozen pilot only."""
import argparse
import json
from pathlib import Path
from .protocol import load_config, plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/mnist_controls.yaml')
    p.add_argument('--phase',choices=['pilot'],default='pilot')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--aggregate',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'])
    a=p.parse_args(argv)
    if a.resume and not a.execute: p.error('--resume requires --execute')
    c=load_config(a.config,a.device)
    if a.aggregate:
        from .aggregate import aggregate
        from .artifacts import exclusive
        with exclusive(c['output_root']): result=aggregate(c['output_root'])
        print(json.dumps(result,indent=2)); return 0 if result['complete'] else 1
    if not a.execute:
        print(json.dumps(plan(c),indent=2)); return 0
    from .execution import execute
    try:
        result=execute(c,a.resume)
        print(json.dumps(result,indent=2)); return 0 if result['complete'] else 1
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        return 1

if __name__=='__main__': raise SystemExit(main())
