"""Plot real loss records; phases and restart segments remain separate."""
import csv
from collections import Counter
import json
from pathlib import Path


def plot_smoke(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    groups=list(dict.fromkeys(r['group'] for r in rows))
    for group in groups:
        data=[r for r in rows if r['group']==group]
        for ax,phase in zip(axes,('pose','3d')):
            part=[r for r in data if r['phase']==phase]
            ax.plot([r['global_update'] for r in part],[r['loss_total'] for r in part],marker='o',label=group)
    for ax,phase in zip(axes,('pose','3D')):
        ax.set(title=phase+' fixture updates',xlabel='Actual fixture update count',ylabel='Weighted total loss')
    axes[-1].legend(ncol=2,fontsize=8)
    fig.suptitle('GPU smoke only: mature-stage gradient fixture, not a convergence experiment')
    fig.savefig(Path(out)/'smoke_loss.png',dpi=170);plt.close(fig)


def plot_all(stage,only_group=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(stage)/'training_curves';out.mkdir(exist_ok=True)
    diagnostics={}
    for run in sorted((Path(stage)/'runs').iterdir()):
        if only_group is not None and run.name!=only_group:continue
        segments=sorted((run/'segments').glob('*/metrics.csv'))
        if not segments:continue
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        seen=Counter();count=0
        for path in segments:
            with path.open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            seen.update(int(r['global_update']) for r in rows);count+=len(rows)
            for ax,phase in zip(axes[0],('pose','3d')):
                part=[r for r in rows if r['phase']==phase]
                for loss in ('total','rgb','perceptual','neighbor','normal','laplacian','uniform'):
                    ax.plot([int(r['global_update']) for r in part],[float(r['loss_'+loss]) for r in part],
                            label=loss if path==segments[0] else None,alpha=.75,lw=.7)
                ax.set(title=phase+' losses',xlabel='Global optimizer update',ylabel='Weighted loss')
            for alpha in ('shape','texture'):
                axes[1,0].plot([int(r['global_update']) for r in rows],[float(r['alpha_'+alpha]) for r in rows],
                               label=alpha if path==segments[0] else None,lw=.8)
        for path in sorted((run/'segments').glob('*/validation.csv')):
            with path.open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
            axes[1,1].plot([int(r['global_update']) for r in rows],[float(r['val_rgb_mse']) for r in rows],lw=.8,marker='.',markersize=3)
        axes[1,0].set(title='Actual residual coefficients',xlabel='Global optimizer update',ylabel='alpha')
        axes[1,1].set(title='Common validation image MSE',xlabel='Global optimizer update',ylabel='Mean MSE (64 validation objects)')
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
        fig.suptitle(run.name+': real logs, separate phases and restart segments')
        for ext in ('png','pdf'):fig.savefig(out/(run.name+'_loss.'+ext),dpi=210)
        plt.close(fig)
        diagnostics[run.name]={'raw_rows':count,'repeated_update_coordinates':sum(n-1 for n in seen.values() if n>1),
            'segments':[str(p.relative_to(stage)) for p in segments],
            'plot_policy':'All rows retained. Each restart segment and phase drawn separately; no smoothing/deduplication.'}
    metadata='curve_provenance.json' if only_group is None else only_group+'_curve_provenance.json'
    (out/metadata).write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    print('CURVES: '+str(out),flush=True)
