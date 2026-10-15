"""Batched v3f2 mechanics, traceable irreversible merges and named baselines."""
from copy import deepcopy
import math
import torch
from .geometry import step_delta,project,merge_particles,radius_at,should_merge,distances,distance_stats

def resolve_levels(cfg,schedule):
    sigma=schedule.sqrt_one_minus_alphas_cumprod.cpu(); a=schedule.alphas_cumprod.cpu(); snr=a.log()-torch.log1p(-a)
    kind=cfg['kind']
    if kind=='explicit':
        requested=cfg['timesteps']; levels=requested
        if not levels or any(type(t)!=int or not 0<=t<schedule.T for t in levels) or levels!=sorted(set(levels),reverse=True): raise ValueError('Use distinct descending integer timesteps')
    elif kind in {'sigma','log_snr'}:
        axis=sigma if kind=='sigma' else snr
        if 'values' in cfg: requested=cfg['values']
        else:
            lo,hi,n=cfg['min'],cfg['max'],cfg['count']
            if lo>hi or n<1 or kind=='sigma' and lo<=0: raise ValueError('Invalid noise range')
            requested=(torch.logspace(math.log10(lo),math.log10(hi),n) if kind=='sigma' else torch.linspace(lo,hi,n)).tolist()
        if not requested or any(v<axis.min().item()-1e-6 or v>axis.max().item()+1e-6 for v in requested): raise ValueError('Levels outside trained schedule')
        levels=sorted({int((axis-v).abs().argmin()) for v in requested},reverse=True)
    else: raise ValueError('Unknown schedule')
    return levels,dict(parameterization=kind,requested=requested,resolved=levels,sigma=[sigma[t].item() for t in levels],log_snr=[snr[t].item() for t in levels],duplicates_collapsed=len(requested)-len(levels),t_zero_is_clean=False,sigma_at_zero=sigma[0].item())

def move(model,x,t,steps,alpha,mode,projection,space,tolerance=1e-6):
    if steps<0: raise ValueError('steps must be >=0')
    records=[]
    for index in range(steps):
        s=model.score(x,t); delta=step_delta(s,alpha,mode); proposed,clip=project(x+delta,projection,space); actual=proposed-x
        records.append(dict(step=index,score_norm=s.norm(dim=1).cpu(),step_l2=actual.norm(dim=1).cpu(),step_rms=actual.square().mean(1).sqrt().cpu(),proposed_step_l2=delta.norm(dim=1).cpu(),proposed_step_rms=delta.square().mean(1).sqrt().cpu(),clipping=clip,nfe=model.nfe))
        x=proposed
        if actual.norm(dim=1).max().item()<tolerance: break
    return x,records

def run_search(model,initial,cfg,space,evaluator=None):
    x=initial.clone(); levels,metadata=resolve_levels(cfg['schedule'],model.schedule)
    lineage=[[i] for i in range(len(x))]; stages=[]; trajectory=[]; merge=cfg['merge']; start=model.nfe
    plan=[('level',t,cfg['steps_per_level'],cfg['alpha']) for t in levels]+[('final',cfg['final_t'],cfg['final_steps'],cfg['alpha']*cfg['final_step_scale'])]
    for index,(kind,t,steps,alpha) in enumerate(plan):
        enabled=should_merge(merge['schedule'],index,t,merge['selected_levels'],kind=='final')
        if kind!='final' and steps==0 and not enabled: continue
        before=x.clone(); parents=deepcopy(lineage); nfe_before=model.nfe
        x,movement=move(model,x,t,steps,alpha,cfg['step_mode'],cfg['projection'],space,cfg['tolerance']); after=x.clone()
        labels=list(range(len(x)))
        info=dict(particles_before=len(x),particles_after=len(x),number_of_clusters=len(x),cluster_sizes=[1]*len(x),merge_radius=None)
        if enabled:
            x,labels,info=merge_particles(x,radius_at(merge,model,t),merge,space,evaluator)
            lineage=[sum((parents[i] for i,label in enumerate(labels) if label==group),[]) for group in range(len(x))]
        else:
            ids=torch.linspace(0,len(x)-1,min(len(x),512)).long().to(x.device)
            info['distance_statistics']=dict(**distance_stats(distances(x[ids],merge['metric'],space,evaluator)),sampled_particles=len(ids),approximate=len(ids)<len(x))
        anchors=torch.stack([initial[ids].mean(0) for ids in lineage])
        stages.append(dict(kind=kind,t=t,level_index=index,sigma=model.schedule.sigma(t),nfe_before=nfe_before,nfe=model.nfe,nfe_movement=model.nfe-nfe_before,movement=movement,merge=info,lineage=deepcopy(lineage),parent_to_cluster=torch.as_tensor(labels).long(),displacement_from_initial=(x-anchors).norm(dim=1).cpu()))
        if cfg['save_trajectory']: trajectory.append(dict(kind=kind,t=t,x_before=before.cpu(),x_after=after.cpu(),x_after_merge=x.cpu(),lineage_before=parents,lineage_after=deepcopy(lineage),parent_to_cluster=torch.as_tensor(labels).long()))
    return dict(candidates=x,stages=stages,trajectory=trajectory,lineage=lineage,schedule=metadata,movement_nfe=model.nfe-start)

def baseline_config(cfg,method):
    c=deepcopy(cfg)
    if method in {'ddim','matched_ddim'}: c['steps_per_level']=c['final_steps']=0; c['merge']['schedule']='none'
    elif method in {'ddim_cluster','matched_cluster'}: c['steps_per_level']=c['final_steps']=0; c['merge']['schedule']='final_only'
    elif method=='final_refine': c['steps_per_level']=0; c['merge']['schedule']='none'
    elif method=='no_intermediate_merge': c['merge']['schedule']='final_only'
    elif method=='low_noise_only':
        n=len(cfg['schedule']['timesteps']) if cfg['schedule']['kind']=='explicit' else cfg['resolved_level_count']
        c['schedule']={'kind':'explicit','timesteps':[cfg['final_t']]}; c['steps_per_level']*=n
    elif method!='original': raise ValueError(method)
    return c
