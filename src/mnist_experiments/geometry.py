"""Complete linkage and explicit geometry choices; network evaluations are batched."""
import math
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

def step_delta(s,alpha,mode):
    if not math.isfinite(alpha) or alpha<0: raise ValueError('Invalid step size')
    if mode=='raw': return alpha*s
    if mode not in {'l2','rms'}: raise ValueError('Unknown step mode')
    norm=s.norm(dim=1,keepdim=True)
    return alpha*s/torch.where(norm>1e-8,norm,torch.ones_like(norm))*(math.sqrt(s.shape[1]) if mode=='rms' else 1.)

def project(x,mode,space):
    over=(space.decode(x).abs()-1).clamp_min(0)
    stats=dict(fraction_outside_range=(over>0).float().mean().item(),mean_overshoot=over.mean().item(),max_overshoot=over.max().item())
    if mode=='clamp':
        if space.name!='pixel': raise ValueError('PCA coefficients have no pixel box; use projection=none')
        x=x.clamp(-1,1)
    elif mode!='none': raise ValueError('Unknown projection')
    return x,stats

def distances(x,metric,space,evaluator=None):
    if metric.startswith('feature_'):
        if evaluator is None: raise ValueError('Independent evaluator required for feature geometry')
        if metric not in {'feature_l2','feature_cosine'}: raise ValueError(metric)
        z=evaluator.features(space.decode(x)).cpu().numpy().astype(np.float64)
        kind='cosine' if metric=='feature_cosine' else 'euclidean'
    else: z=x.detach().cpu().numpy().astype(np.float64); kind=metric
    if kind in {'euclidean','rms_euclidean'}:
        d=cdist(z,z)
        if kind=='rms_euclidean': d/=math.sqrt(x.shape[1])
    elif kind=='cosine':
        norm=np.linalg.norm(z,axis=1); unit=z/np.maximum(norm[:,None],1e-12)
        d=np.clip(1-unit@unit.T,0,2); d[np.ix_(norm<1e-12,norm<1e-12)]=0
    else: raise ValueError(metric)
    np.fill_diagonal(d,0); return d

def distance_stats(d):
    v=d[np.triu_indices(len(d),1)]
    if not len(v): return dict(count=0,mean=None,std=None,cv=None,quantiles=[])
    return dict(count=len(v),mean=float(v.mean()),std=float(v.std()),cv=float(v.std()/v.mean()) if v.mean() else 0.,quantiles=np.quantile(v,[0,.1,.5,.9,1]).tolist())

def radius_at(cfg,model,t):
    if cfg['radius_schedule']=='constant': radius=cfg['radius']
    elif cfg['radius_schedule']=='noise_scaled':
        sigma=model.schedule.sigma(t)
        if cfg['noise_units']=='legacy_sigma': scale=cfg['legacy_t0_sigma'] if t==0 else sigma
        elif cfg['noise_units']=='pairwise_sigma':
            if cfg['metric'] not in {'euclidean','rms_euclidean'}: raise ValueError('Use constant radius for cosine/feature metrics')
            scale=sigma*math.sqrt(2*(model.dim if cfg['metric']=='euclidean' else 1))
        else: raise ValueError('Unknown noise units')
        radius=max(cfg['min_radius'],cfg['coefficient']*scale)
    else: raise ValueError('Unknown radius schedule')
    if not math.isfinite(radius) or radius<0: raise ValueError('Invalid merge radius')
    return radius

def should_merge(schedule,index,t,selected,final=False):
    if schedule not in {'every_level','final_only','selected_levels','none'}: raise ValueError(schedule)
    return schedule!='none' and (final or schedule=='every_level' or schedule=='selected_levels' and t in selected)

def merge_particles(x,radius,cfg,space,evaluator=None):
    if len(x)>cfg['max_particles']: raise ValueError('Complete linkage is quadratic; raise max_particles deliberately or reduce budget')
    d=distances(x,cfg['metric'],space,evaluator)
    labels=np.arange(len(x)) if len(x)<2 or radius==0 else AgglomerativeClustering(n_clusters=None,distance_threshold=radius,linkage='complete',metric='precomputed').fit_predict(d)
    outputs=[]; sizes=[]; reps=[]
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label); sizes.append(len(ids))
        if cfg['representative']=='mean': outputs.append(x[torch.as_tensor(ids,device=x.device)].mean(0)); reps.append(None)
        elif cfg['representative']=='medoid':
            index=int(ids[np.argmin(d[np.ix_(ids,ids)].sum(1))]); outputs.append(x[index]); reps.append(index)
        else: raise ValueError('Unknown representative')
    return torch.stack(outputs),labels,dict(particles_before=len(x),particles_after=len(outputs),number_of_clusters=len(outputs),cluster_sizes=sizes,merge_radius=radius,distance_statistics=distance_stats(d),representative_indices=reps)
