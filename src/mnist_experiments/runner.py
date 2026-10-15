"""YAML matrix runner. No execution unless --execute is explicitly supplied."""
import argparse
from copy import deepcopy
import fnmatch,itertools,json,subprocess,sys
from pathlib import Path
import yaml
from .common import merge_dict,fingerprint,write_json,timestamp,digest
from .config import DEFAULT

def set_dotted(cfg,path,value):
    keys=path.split('.'); node=cfg
    for key in keys[:-1]: node=node.setdefault(key,{})
    node[keys[-1]]=value

def render_paths(value,context):
    if isinstance(value,dict): return {k:render_paths(v,context) for k,v in value.items()}
    if isinstance(value,list): return [render_paths(v,context) for v in value]
    if isinstance(value,str): return value.format(**context)
    return value

def expand_matrix(path,seen=None):
    path=Path(path).resolve(); seen=seen or set()
    if path in seen: raise ValueError('Recursive matrix include')
    doc=yaml.safe_load(path.read_text()); jobs=[]
    for include in doc.get('include',[]): jobs.extend(expand_matrix(path.parent/include,seen|{path}))
    for entry in doc.get('experiments',[]):
        grid=entry.get('grid',{})
        if any(not isinstance(v,list) or not v for v in grid.values()): raise ValueError('Grid axes must be nonempty lists')
        for values in itertools.product(*grid.values()):
            cfg=merge_dict(doc.get('defaults',{}),entry.get('config',{}))
            for key,value in zip(grid,values): set_dotted(cfg,key,value)
            if cfg.get('kind')=='publication_search':
                from .publication import configure
                cfg=configure(cfg)
            elif cfg.get('kind','search')=='search': cfg=merge_dict(DEFAULT,cfg)
            cfg.setdefault('kind','search'); cfg.setdefault('variant',entry['name'])
            if cfg['kind'] in {'search','publication_search'} and cfg['initial_noise_seed'] is None: cfg['initial_noise_seed']=cfg['search_seed']
            cfg=render_paths(cfg,{k:v for k,v in cfg.items() if isinstance(v,(str,int,float))})
            identifier=entry['name']+'--'+fingerprint(cfg)
            jobs.append(dict(id=identifier,group=entry['name'],config=cfg,output=str(Path(doc.get('output_root','runs/mnist_experiments'))/identifier)))
    if len({j['id'] for j in jobs})!=len(jobs): raise ValueError('Duplicate job IDs')
    return jobs

def estimate(job):
    c=job['config']; kind=c['kind']
    if kind=='publication_search':
        from .publication import estimate_publication
        return estimate_publication(c)
    if kind in {'pixel_train','pca_train','evaluator_train','publication_train'}: return dict(model_trainings=1,nfe_upper_bound=None)
    if kind=='pca_fit': return dict(model_trainings=0,pca_fits=1,nfe_upper_bound=0)
    if kind!='search': raise ValueError(kind)
    s=c['search']; schedule=s['schedule']; p=c['population_size']; init=p*c['ddim_steps']
    levels=len(schedule['timesteps']) if schedule['kind']=='explicit' else len(schedule.get('values',[])) or schedule['count']
    original=init+p*(levels*s['steps_per_level']+s['final_steps']); costs={}
    for m in c['methods']:
        costs[m]=init if m in {'ddim','ddim_cluster'} else init+p*s['final_steps'] if m=='final_refine' else (original//c['ddim_steps'])*c['ddim_steps'] if m.startswith('matched_') else original
    n=original//c['ddim_steps']; stability=c['stability']; curvature=c['curvature']; diagnostic=0
    if stability['enabled']: diagnostic+=stability['steps']*(1+len(stability['scales'])*stability['directions'])
    if curvature['enabled']: diagnostic+=2*curvature['directions']
    return dict(model_trainings=0,nfe_upper_bound=sum(costs.values()),per_method_nfe_upper=costs,matched_samples_upper=n,diagnostic_nfe_upper=sum((n if m.startswith('matched_') else p)*diagnostic for m in c['methods']),note='No-merge upper bounds. Evaluator work excluded; stopping/merges can reduce NFE.')

def training_job(job,resume):
    from .evaluator import train_evaluator
    from .pca import fit_pca,train_pca
    c=job['config']
    if c['kind']=='publication_train':
        from .publication_training import train_job
        return train_job(job,resume)
    path=Path(job['output']); marker=path/'complete.json'
    signature=fingerprint(dict(config=c,sources={k:digest(c[k]) for k in ('source_checkpoint','pca_path','base_config') if k in c}))
    if marker.exists():
        old=json.loads(marker.read_text())
        if not resume: raise FileExistsError('Training job already completed')
        if old['signature']!=signature or not Path(old['artifact']).exists() or digest(old['artifact'])!=old['artifact_sha256']: raise ValueError('Training inputs/artifact changed')
        return
    path.mkdir(parents=True,exist_ok=True)
    if c['kind']=='evaluator_train': artifact=train_evaluator(c['source_checkpoint'],c['training_output'],device=c.get('device','auto'),seed=c.get('seed',71),epochs=c.get('epochs',5),resume=resume)
    elif c['kind']=='pca_fit':
        artifact=Path(c['pca_output']); latents=artifact.with_name(artifact.stem+'_latents.pt')
        if resume and artifact.exists() and latents.exists():
            import torch
            old=torch.load(artifact,map_location='cpu',weights_only=True)
            if old['source_checkpoint_sha256']!=digest(c['source_checkpoint']) or old['dimensions']!=sorted(c['dimensions']): raise ValueError('Existing PCA differs')
        else:
            if resume and latents.exists() and not artifact.exists(): latents.unlink() # incomplete, regenerable fit output only
            artifact=fit_pca(c['source_checkpoint'],artifact,c['dimensions'])
    elif c['kind']=='pca_train': artifact=train_pca(c['pca_path'],c['training_output'],c['dimension'],device=c.get('device','auto'),seed=c.get('model_seed',42),epochs=c.get('epochs',30),resume=resume)
    elif c['kind']=='pixel_train':
        config=yaml.safe_load(Path(c['base_config']).read_text()); config.update(seed=c['model_seed'],device=c.get('device','auto'),output_dir=c['training_output'])
        filename=path/'training_config.yaml'; filename.write_text(yaml.safe_dump(config)); command=[sys.executable,'-m','src.mnist.train','--config',str(filename)]
        latest=Path(c['training_output'])/'checkpoints/latest.pt'
        if resume and latest.exists():
            import torch
            state=torch.load(latest,map_location='cpu',weights_only=True)
            if state['epoch']<config['training']['epochs']: command+=['--resume',str(latest)]
            else: command=None
        if command: subprocess.run(command,check=True)
        artifact=Path(c['training_output'])/'checkpoints/best.pt'
    else: raise ValueError(c['kind'])
    write_json(marker,dict(signature=signature,artifact=str(artifact),artifact_sha256=digest(artifact),finished_at=timestamp()))

def execute_jobs(jobs,resume=False):
    from .run import run_suite
    for i,job in enumerate(jobs,1):
        print(f"[{i}/{len(jobs)}] {job['id']}",flush=True)
        try:
            if job['config']['kind'] in {'search','publication_search'}: run_suite(dict(job['config'],suite_id=job['id']),job['output'],resume)
            else: training_job(job,resume)
        except Exception as exc:
            # A refused resume / existing run must not mutate protected results.
            if not isinstance(exc,(FileExistsError,ValueError)):
                write_json(Path(job['output'])/'failure.json',dict(error=repr(exc),timestamp=timestamp()))
            raise

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--matrix',default='configs/mnist_experiments/diagnostic.yaml')
    action=p.add_mutually_exclusive_group(); action.add_argument('--dry-run',action='store_true'); action.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true'); p.add_argument('--only',nargs='+'); p.add_argument('--device',choices=['auto','cpu','mps','cuda']); p.add_argument('--plan-output')
    a=p.parse_args(); jobs=expand_matrix(a.matrix)
    if a.only:
        jobs=[j for j in jobs if any(fnmatch.fnmatch(j['id'],pattern) or fnmatch.fnmatch(j['group'],pattern) for pattern in a.only)]
        if not jobs: p.error('--only matched no jobs')
    if a.device:
        for j in jobs: j['config']['device']=a.device
    plan=[dict(**j,estimate=estimate(j)) for j in jobs]
    print(json.dumps(dict(number_of_jobs=len(jobs),estimated_model_trainings=sum(j['estimate']['model_trainings'] for j in plan),
        search_nfe_upper_bound=sum(j['estimate'].get('nfe_upper_bound') or 0 for j in plan),
        diagnostic_nfe_upper_bound=sum(j['estimate'].get('diagnostic_nfe_upper',0) for j in plan),
        raw_tensor_bytes_estimate=sum(j['estimate'].get('raw_tensor_bytes_estimate',0) for j in plan),
        training_compute_note='Training forward/backward costs are separate and depend on the saved seed42 recipe.',jobs=plan),indent=2))
    if a.plan_output: write_json(a.plan_output,plan)
    if a.execute: execute_jobs(jobs,a.resume)
    else: print('DRY RUN ONLY. No data loading, training or search was performed.')
if __name__=='__main__': main()
