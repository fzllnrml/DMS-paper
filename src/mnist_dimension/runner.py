"""Opt-in train/search orchestration with explicit expected-job inventory."""
import argparse,json
from pathlib import Path
from src.mnist_experiments.common import write_json,fingerprint
from .design import load_config,plan,models
from .artifacts import failure


def save_plan(cfg):
    root=Path(cfg['root']);path=root/'study_plan.json'
    data=plan(cfg);data['resolved_config']=cfg
    if path.exists():
        old=json.loads(path.read_text())
        if old['config_hash']!=data['config_hash']:raise ValueError('Study config changed; existing root is protected')
    else:write_json(path,data)
    return data


def execute(cfg,phase,resume=False):
    expected=save_plan(cfg);errors=[]
    def attempt(name,path,fn):
        print(name,flush=True)
        try:fn();return True
        except (FileExistsError,ValueError) as exc:
            # Artifact/input failures remain visible in the invocation inventory.
            errors.append(dict(job=name,path=str(path),error=repr(exc)));print(f'FAILED {name}: {exc}',flush=True);return False
        except Exception as exc:
            failure(path,exc);errors.append(dict(job=name,path=str(path),error=repr(exc)))
            print(f'FAILED {name}: {exc}',flush=True);return False
    if phase=='train':
        from .representation import prepare
        from .training import train_model
        if attempt('representation fit',Path(cfg['root'])/'representation',lambda:prepare(cfg,resume)):
            for model in expected['models']:
                if model['representation']=='pca':
                    attempt(model['id'],Path(cfg['root'])/'models'/model['id'],lambda m=model:train_model(cfg,m,resume))
    elif phase=='search':
        from .execution import quality_check,search_job
        from .evaluation import StudyEvaluator
        evaluator=None
        # Check existence/completion before instantiating data/evaluator on a pure resume.
        from .artifacts import validate_complete
        all_complete=all(validate_complete(Path(j['output'])) for j in expected['jobs'])
        if all_complete and resume:
            # Each job still validates signatures and dependencies through the entry point.
            for j in expected['jobs']:search_job(cfg,j,None,True)
            return
        for model in expected['models']:
            if not Path(model['checkpoint']).exists():
                errors.append(dict(job=model['id'],path=model['checkpoint'],error='missing checkpoint'));continue
            if evaluator is None:evaluator=StudyEvaluator(cfg)
            if not attempt('quality '+model['id'],Path(cfg['root'])/'quality'/model['id'],lambda m=model:quality_check(cfg,m,evaluator,resume)):
                continue
            for job in expected['jobs']:
                if job['representation']==model['representation'] and job['dimension']==model['dimension'] and job['model_seed']==model['model_seed']:
                    attempt(job['id'],Path(job['output']),lambda j=job:search_job(cfg,j,evaluator,resume))
    else:raise ValueError(phase)
    write_json(Path(cfg['root'])/(phase+'_last_invocation.json'),dict(phase=phase,errors=errors))
    if errors:raise RuntimeError(f'{len(errors)} failed/blocked jobs. Inspect invocation inventory; completed outputs preserved.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['train','search','aggregate','smoke'])
    p.add_argument('--config',default='configs/mnist_dimension.yaml')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'])
    action=p.add_mutually_exclusive_group();action.add_argument('--execute',action='store_true');action.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args();cfg=load_config(a.config,a.device,smoke=a.phase=='smoke')
    if a.phase=='aggregate':
        from .aggregate import aggregate
        aggregate(cfg['root'],a.bootstrap);return
    print(json.dumps(plan(cfg),indent=2))
    if not a.execute:
        print('DRY RUN ONLY: no data/checkpoint reads, fitting, training or search.');return
    if a.phase=='smoke':
        execute(cfg,'train',a.resume);execute(cfg,'search',a.resume)
        from .aggregate import aggregate
        aggregate(cfg['root'],50)
    else:execute(cfg,a.phase,a.resume)
if __name__=='__main__':main()
