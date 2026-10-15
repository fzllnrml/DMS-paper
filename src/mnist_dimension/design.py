"""Scientific design and exact inference budgets; no checkpoint reads in dry-run."""
from copy import deepcopy
from pathlib import Path
import hashlib
import yaml
from src.mnist_experiments.common import fingerprint
from src.mnist_experiments.publication import definitions, definition_hash

PROTOCOL='mnist_dimension_v1'
DIMENSIONS=[16,32,64,128,256,512,784]
REPLICATED=[16,128,784]
METHODS=['ddim','original_v3f2','rms_v3f2','matched_ddim']


def load_config(path,device=None,smoke=False):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['protocol']!=PROTOCOL or cfg['dimensions']!=DIMENSIONS or cfg['replicated_dimensions']!=REPLICATED:
        raise ValueError('Dimension grid/protocol is frozen')
    if cfg['search_seeds']!=[0,1,2,3,4] or cfg['population_size']!=128 or cfg['model_seeds']!=[42,43,44]:
        raise ValueError('Replication design is frozen')
    if cfg['hidden']!=[256,256,256] or cfg['ddim_steps']!=50:
        raise ValueError('Architecture principle/DDIM protocol is frozen')
    if device: cfg['device']=device
    cfg['smoke']=bool(smoke)
    if smoke:
        cfg.update(dimensions=[2,4],replicated_dimensions=[],model_seeds=[42],search_seeds=[0],
                   population_size=2,hidden=[8,8],root=cfg['root']+'_smoke',pixel_model_seeds=[42],
                   train_limit=16,validation_limit=8,epochs_override=1,sanity_count=2,quality_validation_count=4)
    return cfg


def models(cfg):
    result=[]
    for d in cfg['dimensions']:
        seeds=cfg['model_seeds'] if d in cfg['replicated_dimensions'] else [42]
        for seed in seeds:
            result.append(dict(id=f'pca_d{d}_m{seed}',representation='pca',dimension=d,model_seed=seed,
                               checkpoint=str(Path(cfg['root'])/'models'/f'pca_d{d}_m{seed}'/'best.pt')))
    for seed in cfg['pixel_model_seeds']:
        result.append(dict(id=f'pixel_d784_m{seed}',representation='pixel',dimension=784,model_seed=seed,
                           checkpoint=cfg['pixel_checkpoints'][seed]))
    return result


def method_config(method,representation):
    if method not in METHODS: raise ValueError(method)
    c=deepcopy(definitions()[method])
    if representation=='pca': c['projection']='none'
    elif representation!='pixel': raise ValueError(representation)
    return c


def study_hash():
    h=hashlib.sha256()
    folder=Path(__file__).parent
    for p in sorted(folder.glob('*.py'))+sorted((folder.parent/'mnist_experiments').glob('*.py'))+[folder.parent/'mnist/model.py']:
        h.update(str(p.relative_to(folder.parent)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def plan(cfg):
    ms=models(cfg);jobs=[];p=cfg['population_size'];target=p*105;matched=target//50*50
    costs={'ddim':p*50,'original_v3f2':target,'rms_v3f2':target,'matched_ddim':matched}
    for m in ms:
        for seed in cfg['search_seeds']:
            jobs.append(dict(m,id=f'{m["id"]}_s{seed}',search_seed=seed,initial_noise_seed=seed,
                             output=str(Path(cfg['root'])/'search'/f'{m["id"]}_s{seed}'),
                             nfe_upper_bound=sum(costs.values()),per_method_nfe_upper=costs))
    qn=cfg['quality_validation_count']*len(cfg['quality_timesteps'])+cfg['sanity_count']*50
    return dict(protocol=PROTOCOL,publication_definition_hash=definition_hash(),config_hash=fingerprint(cfg),
                dimensions=cfg['dimensions'],replicated_dimensions=cfg['replicated_dimensions'],representation_fits=1,
                diffusion_trainings=sum(m['representation']=='pca' for m in ms),existing_pixel_models=sum(m['representation']=='pixel' for m in ms),
                quality_jobs=len(ms),search_jobs=len(jobs),search_seed_count=len(cfg['search_seeds']),
                search_nfe_upper_bound=sum(j['nfe_upper_bound'] for j in jobs),
                quality_nfe_per_model=qn,diagnostic_nfe_upper_bound=qn*len(ms),
                search_method_diagnostic_nfe=0,total_inference_nfe_upper_bound=sum(j['nfe_upper_bound'] for j in jobs)+qn*len(ms),
                training_compute='Separate forward/backward budget. Epochs/batch/lr/weight decay/grad clip read from completed seed42 latest.pt.',
                models=ms,jobs=jobs)
