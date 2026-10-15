"""Headless PNG/PDF plots and fixed-index paired grids."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def save_figure(fig,path):
    fig.tight_layout()
    fig.savefig(str(path)+'.png',dpi=150)
    fig.savefig(str(path)+'.pdf')
    plt.close(fig)


def paired_grid(suite,methods,max_images=16):
    suite=Path(suite)
    arrays={m:torch.load(suite/m/'arrays.pt',map_location='cpu',weights_only=True) for m in methods}
    first=arrays[methods[0]]
    n=min(max_images,len(first['initial']))
    columns=[first['decoded_initial'][:n]]
    mappings={'initial':list(range(n))}
    for method,a in arrays.items():
        # Same ancestor index on every row. A merged representative may repeat.
        ids=[next(j for j,lineage in enumerate(a['lineage']) if i in lineage) for i in range(n)]
        columns.append(a['decoded_candidates'][ids])
        mappings[method]=ids
    grid=torch.stack(columns,1).reshape(-1,1,28,28)
    save_image((grid.clamp(-1,1)+1)/2,suite/'paired_grid.png',nrow=len(columns))
    (suite/'paired_grid_columns.json').write_text(json.dumps(dict(columns=['initial']+methods,source_indices=mappings,
        note='Rows follow initial ancestor index. Merged representatives repeat across their ancestors.'),indent=2))


def final_plots(rows,paired,output):
    root=Path(output)
    partitions=defaultdict(list)
    for row in rows:
        partitions[(row['publication_series'],row['definition_hash'],row['evaluation_definition_hash'],row.get('evaluator_sha256'))].append(row)
    for (series,definition,evaluation,evaluator),values in partitions.items():
        out=root/f'{series}_{definition}_{evaluation}_{str(evaluator)[:12]}'
        out.mkdir(parents=True,exist_ok=True)
        methods=sorted({r['method'] for r in values})
        colors={m:plt.get_cmap('tab10')(i%10) for i,m in enumerate(methods)}
        pairs=[('nfe',y) for y in ['pixel_support_percentile','feature_support_percentile','pixel_knn_distance',
                 'feature_knn_distance','final_candidate_count','represented_class_count']]
        pairs += [('final_candidate_count',y) for y in ('pixel_support_percentile','feature_support_percentile')]
        pairs += [('compression_ratio','pixel_support_percentile'),('compression_ratio','feature_support_percentile')]
        for x,y in pairs:
            if not any(isinstance(r.get(y),(int,float)) for r in values):
                continue
            fig,ax=plt.subplots(figsize=(8,5))
            for method in methods:
                group=[r for r in values if r['method']==method and isinstance(r.get(y),(int,float))]
                if group:
                    ax.scatter([r[x] for r in group],[r[y] for r in group],label=method,color=colors[method],alpha=.6)
            ax.set(xlabel=x,ylabel=y,title=series)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
            save_figure(fig,out/f'{y}_vs_{x}')
        for metric in ('pixel_support_percentile','feature_support_percentile','pixel_knn_distance','feature_knn_distance'):
            valid=[r for r in values if isinstance(r.get(metric),(int,float))]
            if not valid:
                continue
            fig,ax=plt.subplots(figsize=(9,5))
            for method in methods:
                group=[r for r in valid if r['method']==method]
                if not group:
                    continue
                populations=sorted({r['population_size'] for r in group})
                averages=[]
                for p in populations:
                    subset=[r for r in group if r['population_size']==p]
                    model_means=[np.mean([r[metric] for r in subset if r['model_seed']==s]) for s in sorted({r['model_seed'] for r in subset})]
                    averages.append(np.mean(model_means))
                ax.plot(populations,averages,'o-',label=method,color=colors[method])
            ax.set(xlabel='population_size',ylabel=metric,title='Means balanced across model seeds')
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
            save_figure(fig,out/f'population_curve_{metric}')
            # Do not pool P64 and P128 into one distribution by model seed.
            for p in sorted({r['population_size'] for r in valid}):
                fig,ax=plt.subplots(figsize=(12,5))
                groups=[];labels=[]
                for model in sorted({r['model_seed'] for r in valid}):
                    for method in methods:
                        group=[r[metric] for r in valid if r['model_seed']==model and r['method']==method and r['population_size']==p]
                        if group: groups.append(group);labels.append(f'{model}/{method}')
                ax.boxplot(groups)
                ax.set_xticks(range(1,len(labels)+1),labels)
                ax.tick_params(axis='x',labelrotation=70,labelsize=7)
                ax.set(ylabel=metric,title=f'Distribution by model seed, P={p}')
                save_figure(fig,out/f'by_model_seed_P{p}_{metric}')
            comparisons=[r for r in paired if r['publication_series']==series and r['definition_hash']==definition
                         and r['evaluation_definition_hash']==evaluation and r.get('evaluator_sha256')==evaluator and r['method']=='rms_v3f2'
                         and r['reference']=='matched_ddim' and r['metric']==metric]
            if comparisons:
                fig,ax=plt.subplots(figsize=(7,4))
                for i,r in enumerate(comparisons):
                    sign=-1 if 'distance' in metric else 1
                    mean=sign*r['mean'];lo,hi=sorted([sign*r['ci95_low'],sign*r['ci95_high']])
                    ax.plot([lo,hi],[i,i],color='tab:blue');ax.plot(mean,i,'o',color='tab:blue')
                ax.set_yticks(range(len(comparisons)),[f"P={r['population_size']} (n={r['n_runs']})" for r in comparisons])
                ax.axvline(0,color='gray',lw=1)
                ax.set(xlabel=f'{metric}: improvement over matched-DDIM (positive favorable)',title='Paired mean and 95% hierarchical bootstrap CI')
                save_figure(fig,out/f'paired_improvement_{metric}')
