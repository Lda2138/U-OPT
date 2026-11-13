"""Regenerate seven-group loss and metric figures from public numerical CSVs."""
import argparse
from pathlib import Path


def plot(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    groups=('U','G','T','GT','GS','TS','GTS')
    colors=dict(zip(groups,('#414141','#2764a5','#de8a16','#9a509f','#219182','#ce554f','#4d7724')))
    epoch=pd.read_csv(data/'seven_epoch_phase_means.csv')
    validation=pd.read_csv(data/'seven_validation_records.csv')
    metrics=pd.read_csv(data/'seven_group_metrics_unrounded.csv').set_index('group')
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for group in groups:
        for axis,phase in zip(axes[:2],('pose','3d')):
            p=epoch[(epoch.group==group)&(epoch.phase==phase)]
            axis.plot(p.epoch,p.loss_total_mean,color=colors[group],label=group,lw=1)
        p=validation[validation.group==group]
        axes[2].plot(p.completed_epoch_fraction,p.val_rgb_mse,color=colors[group],label=group,lw=1)
    for axis,title in zip(axes,('Pose loss (epoch mean)','3D loss (epoch mean)','Validation RGB MSE')):
        axis.set(title=title,xlabel='Epoch');axis.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=7,frameon=False)
    fig.suptitle('Single paired seed; real logged values, no smoothing')
    for ext in ('png','pdf'):fig.savefig(output/('training_curves.'+ext),dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for axis,key in zip(axes.flat,('icp_chamfer_l1_x10','icp_fscore_01','icp_fscore_02','icp_normal_abs_cos','image_psnr_db','image_ssim')):
        scale=100 if 'fscore' in key else 1
        vals=metrics.loc[list(groups),key]*scale
        axis.axhline(vals.loc['U'],color='.6',ls='--',lw=.8)
        for i,g in enumerate(groups):axis.scatter(i,vals.loc[g],color=colors[g])
        axis.set(title=key+(' (%)' if scale==100 else ''),xticks=range(7),xticklabels=groups)
        axis.grid(axis='y',alpha=.2)
    for ext in ('png','pdf'):fig.savefig(output/('test_metrics.'+ext),dpi=220)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'results/chair')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.data,args.output)
