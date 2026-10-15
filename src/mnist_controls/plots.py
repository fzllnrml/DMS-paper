"""Only the two frozen, existing semantic diversity metrics."""
from pathlib import Path

def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for method in dict.fromkeys(r['method'] for r in rows):
        good=[r for r in rows if r['method']==method and r['status']=='complete']
        if not good: continue
        for ax,metric,label in zip(axes,('class_coverage','class_entropy'),('Class coverage','Class entropy (nat)')):
            x=[r[metric] for r in good]; y=[r['feature_knn_mean'] for r in good]
            artist=ax.scatter(x,y,s=18,alpha=.5)
            ax.scatter(sum(x)/len(x),sum(y)/len(y),s=65,marker='D',color=artist.get_facecolor()[0],
                       label=f"{method} (n={len(good)}, NFE={good[0]['nfe']})")
            ax.set_xlabel(label); ax.set_ylabel('Feature kNN (lower is better)')
    if axes[1].get_legend_handles_labels()[0]: axes[1].legend(fontsize=6)
    fig.suptitle('Pilot: fixed checkpoints; partial results retain observed runs only')
    fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(Path(output)/('quality_diversity.'+suffix),dpi=140)
    plt.close(fig)
