"""Uncertainty plots, with PCA curves separate from the pixel-CNN anchor."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.mnist_experiments.aggregate import bootstrap_summary
from .design import METHODS


def save(fig,path):
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=160);fig.savefig(str(path)+'.pdf');plt.close(fig)


def errors(rows):
    means=np.array([r['mean'] for r in rows])
    return means,np.maximum(0,np.array([means-np.array([r['ci95_low'] for r in rows]),np.array([r['ci95_high'] for r in rows])-means]))


def dimension_axis(ax,cfg):
    ticks=sorted(set(cfg['dimensions']+[784]));ax.set_xscale('log',base=2);ax.set_xticks(ticks,[str(d) for d in ticks]);ax.set_xlabel('Representation dimension d');ax.grid(alpha=.2)


def make_plots(summary,paired,quality,status,cfg,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    metrics=['feature_knn_distance','feature_support_percentile','representation_support_percentile','representation_knn_relative_scale',
             'pixel_knn_distance','pixel_support_percentile','regular_actual_step_rms','regular_actual_pixel_step_rms',
             'cumulative_rms_path','displacement_rms']
    for stratum in ('curve_seed42','replicated_anchors'):
        for metric in metrics:
            values=[r for r in summary if r['stratum']==stratum and r['metric']==metric]
            fig,ax=plt.subplots(figsize=(9,5))
            for method in METHODS:
                for rep in ('pca','pixel'):
                    rows=sorted([r for r in values if r['method']==method and r['representation']==rep],key=lambda r:r['dimension'])
                    if not rows:continue
                    means,err=errors(rows)
                    ax.errorbar([r['dimension'] for r in rows],means,yerr=err,fmt='o-' if rep=='pca' else 's',
                        capsize=3,label=method+'/'+rep,alpha=.85)
            if metric=='regular_actual_step_rms':
                dims=np.array(cfg['dimensions']);ax.plot(dims,.03/np.sqrt(dims),'k--',lw=1,label='unclipped L2: .03/sqrt(d)')
                ax.plot(dims,np.full(len(dims),.03),'k:',lw=1,label='unclipped RMS: .03')
            if metric=='regular_actual_pixel_step_rms':
                dims=np.array(cfg['dimensions']);ax.plot(dims,.03*np.sqrt(dims/784),'k:',lw=1,label='PCA RMS decoded: .03 sqrt(d/784)')
                ax.plot(dims,np.full(len(dims),.03/28),'k--',lw=1,label='PCA L2 decoded: .03/28')
            for d in cfg['dimensions']:
                if not any(r['representation']=='pca' and r['dimension']==d for r in values):
                    ax.plot(d,.02,'rx',transform=ax.get_xaxis_transform())
            dimension_axis(ax,cfg);ax.set(ylabel=metric,title=stratum+'; 95% bootstrap CI; red x = no completed PCA data')
            handles,_=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=7)
            save(fig,out/f'{stratum}_{metric}_vs_dimension')
        for reference in ('original_v3f2','matched_ddim'):
            for metric in ('feature_knn_distance','representation_support_percentile','pixel_knn_distance'):
                values=[r for r in paired if r['stratum']==stratum and r['reference']==reference and r['metric']==metric]
                fig,ax=plt.subplots(figsize=(8,4.5))
                for rep in ('pca','pixel'):
                    rows=sorted([r for r in values if r['representation']==rep],key=lambda r:r['dimension'])
                    if rows:
                        mean,err=errors(rows);ax.errorbar([r['dimension'] for r in rows],mean,yerr=err,fmt='o-' if rep=='pca' else 's',capsize=3,label=rep)
                dimension_axis(ax,cfg);ax.axhline(0,color='gray',lw=1);ax.set(ylabel='RMS minus '+reference+': '+metric,title=stratum+'; paired 95% bootstrap CI')
                if values:ax.legend()
                save(fig,out/f'{stratum}_RMS_minus_{reference}_{metric}')
    # Model/representation diagnostics have one row per model, never duplicated by search seed.
    for metric in ('reconstruction','diffusion','sanity'):
        fig,ax=plt.subplots(figsize=(9,5))
        for seed in cfg['model_seeds']:
            for rep in ('pca','pixel'):
                qs=sorted([q for q in quality if q['model_seed']==seed and q['representation']==rep],key=lambda q:q['dimension'])
                if not qs:continue
                values=[];lo=[];hi=[]
                for q in qs:
                    if metric=='reconstruction':
                        v=q['representation_quality']['validation_reconstruction_rms'];ci=q['representation_quality']['reconstruction_rms_ci95']
                    elif metric=='diffusion':v=q['diagnostic_epsilon_mse'];ci=q['diagnostic_epsilon_mse_ci95']
                    else:
                        # Between-search uncertainty is shown in performance figures.
                        # Sanity uses one independent fixed seed, so no fabricated CI.
                        v=q['sanity']['feature_knn_distance'];ci=q['sanity_feature_knn_ci95']
                    values.append(v);lo.append(max(0,v-ci[0]));hi.append(max(0,ci[1]-v))
                ax.errorbar([q['dimension'] for q in qs],values,yerr=[lo,hi],fmt='o-' if rep=='pca' else 's',capsize=3,label=f'{rep} model{seed}')
        dimension_axis(ax,cfg);ax.set(ylabel=metric,title='Conditional 95% image bootstrap CI; fixed representation/model')
        if quality:ax.legend(fontsize=7)
        save(fig,out/f'{metric}_quality_vs_dimension')
    for rep in ('pca','pixel'):
        for d in (cfg['dimensions'] if rep=='pca' else [784]):
            for metric in ('feature_knn_distance','representation_support_percentile'):
                fig,ax=plt.subplots(figsize=(8,4.5))
                values=[r for r in summary if r['stratum']=='curve_seed42' and r['representation']==rep and r['dimension']==d and r['metric']==metric]
                for r in values:
                    nfe=next(s for s in summary if s['stratum']=='curve_seed42' and s['representation']==rep and s['dimension']==d and s['metric']=='nfe' and s['method']==r['method'])
                    ax.errorbar(nfe['mean'],r['mean'],xerr=[[max(0,nfe['mean']-nfe['ci95_low'])],[max(0,nfe['ci95_high']-nfe['mean'])]],
                        yerr=[[max(0,r['mean']-r['ci95_low'])],[max(0,r['ci95_high']-r['mean'])]],fmt='o',capsize=3,label=r['method'])
                ax.set(xlabel='Search NFE',ylabel=metric,title=f'{rep} d={d}; model42; 95% CI')
                if values:ax.legend(fontsize=7)
                ax.grid(alpha=.2);save(fig,out/f'{rep}_d{d}_{metric}_vs_NFE')
    # Full convergence histories stay visible, including worsening epochs.
    for q in quality:
        if not q['history']:continue
        fig,ax=plt.subplots(figsize=(7,4))
        for key in ('train_loss','validation_loss'):
            ax.plot([r['epoch'] for r in q['history']],[r[key] for r in q['history']],label=key)
        ax.set(xlabel='Epoch',ylabel='epsilon MSE / coordinate',title=q['id']);ax.legend();ax.grid(alpha=.2)
        save(fig,out/f'convergence_{q["id"]}')
