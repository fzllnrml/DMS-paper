"""Offline figures; raw data are also saved in PT/JSON/CSV."""
from copy import deepcopy
from pathlib import Path
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from .geometry import merge_particles

def grid(images,path,nmax=64):
    if len(images): save_image((images[:nmax].detach().cpu().clamp(-1,1)+1)/2,path,nrow=math.ceil(math.sqrt(min(len(images),nmax))))

def neighborhood(images,metrics,support,path,nmax=16):
    n=min(len(images),nmax)
    if not n or 'pixel' not in metrics: return
    ids=metrics['pixel']['reference_positions'][:n,:min(support.k,3)]
    columns=[images[:n].cpu()]+[support.train[ids[:,i]] for i in range(ids.shape[1])]
    if 'test_nearest' in metrics: columns.append(support.test[metrics['test_nearest']['test_indices'][:n]])
    save_image((torch.stack(columns,1).reshape(-1,1,28,28).clamp(-1,1)+1)/2,path,nrow=len(columns))

def trajectory(initial,records,space,path,nmax=4):
    rows=[]
    for particle in range(min(len(initial),nmax)):
        rows.append(space.decode(initial[particle:particle+1]).cpu()[0])
        for record in records:
            idx=next(i for i,ids in enumerate(record['lineage_after']) if particle in ids)
            rows.append(space.decode(record['x_after_merge'][idx:idx+1]).cpu()[0])
    if rows: save_image((torch.stack(rows).clamp(-1,1)+1)/2,path,nrow=len(records)+1)

def merge_comparison(record,cfg,space,evaluator,path):
    x=record['x_after']; images=[space.decode(x)]
    for representative in ('mean','medoid'):
        settings=deepcopy(cfg); settings['representative']=representative
        merged,_,_=merge_particles(x,settings['comparison_radius'],settings,space,evaluator); images.append(space.decode(merged))
    fig,axes=plt.subplots(3,1,figsize=(12,5))
    for ax,values,title in zip(axes,images,['Before merge','Mean','Medoid']):
        ax.imshow(torch.cat([x[0].clamp(-1,1) for x in values[:16]],1).cpu().numpy(),cmap='gray',vmin=-1,vmax=1); ax.set_title(title); ax.axis('off')
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)

def summary_plots(rows,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    metrics=['pixel_knn_mean','pixel_support_percentile','feature_support_percentile','class_coverage','stability_return_rate','candidate_count','movement_efficiency','pca_variance_retained','validation_reconstruction_rms']
    groups=sorted({(r['variant'],r['method'],r['space']) for r in rows})
    for x in ('nfe','population_size','dimension'):
        for y in metrics:
            if not any(isinstance(r.get(y),(int,float)) for r in rows): continue
            fig,ax=plt.subplots(figsize=(7,4.5))
            for group in groups:
                v=[r for r in rows if (r['variant'],r['method'],r['space'])==group and isinstance(r.get(y),(int,float))]
                if v: ax.scatter([r[x] for r in v],[r[y] for r in v],s=25,alpha=.7,label='/'.join(group))
            ax.set(xlabel=x,ylabel=y); ax.grid(alpha=.2); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(output/f'{y}_vs_{x}.png',dpi=140); plt.close(fig)
    for metric in ('pixel_support_percentile','feature_support_percentile','class_coverage'):
        if not any(isinstance(r.get(metric),(int,float)) for r in rows): continue
        fig,ax=plt.subplots(figsize=(8,5))
        for group in groups:
            v=[r for r in rows if (r['variant'],r['method'],r['space'])==group and isinstance(r.get(metric),(int,float))]
            if v: ax.scatter([r['nfe'] for r in v],[r[metric] for r in v],s=[15+8*math.sqrt(r['candidate_count']) for r in v],label='/'.join(group),alpha=.6)
        ax.set(xlabel='Search NFE (diagnostics excluded)',ylabel=metric,title='Pareto view: bubble area increases with candidate count')
        ax.legend(fontsize=6); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output/f'pareto_{metric}.png',dpi=140); plt.close(fig)
