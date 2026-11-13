"""Real GPU acceptance and resumable single-GPU training."""
import csv
import datetime
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .engine import (ALPHAS,ChairImages,atomic_checkpoint,epoch_loader,load_checkpoint,
                     make_model,optimizer_for,seed_all,tensor_hash,update,validation_mse)
from .protocol import GROUPS,alpha_at,config_for,stable_hash
from .launcher import dump


def token():
    return os.environ.get('SLURM_JOB_ID','manual')+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def require_gpu():
    if not torch.cuda.is_available():raise RuntimeError('This job requires an allocated CUDA GPU')
    torch.set_num_threads(min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','4'))))
    return torch.device('cuda:0')


def alpha_values(model):return [getattr(model,k).item() for k in ALPHAS]


def write_metrics(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def initial_hashes(model):
    state=model.state_dict()
    common={k:v for k,v in state.items() if not k.startswith(('vert_fc.','uv_fc.')) and k not in ALPHAS}
    return {'common':tensor_hash(common),
            'geometry':tensor_hash({k:v for k,v in state.items() if k.startswith('vert_fc.')}) if hasattr(model,'vert_fc') else None,
            'texture':tensor_hash({k:v for k,v in state.items() if k.startswith('uv_fc.')}) if hasattr(model,'uv_fc') else None}


def phase_isolation(model,phase):
    pose_prefixes=('encoder_pose.','T_regressors.','rot_regressors.','proba_regressor.')
    inactive=[(n,p) for n,p in model.named_parameters() if p.requires_grad and
              ((not n.startswith(pose_prefixes)) if phase=='pose' else n.startswith(pose_prefixes))]
    wrong=[n for n,p in inactive if p.grad is not None]
    if wrong:raise RuntimeError('Unexpected gradient through detached phase: '+repr(wrong[:10]))


def smoke(workspace,stage,preparation,fingerprint):
    name=token();out=stage/'smoke'/name;out.mkdir()
    report_path=workspace/'reports'/('uopt_gts_smoke_'+name+'.json')
    report={'kind':'GPU training-path smoke, not paper results','fingerprint':fingerprint,
            'job_id':os.environ.get('SLURM_JOB_ID'),'all_passed':False,'cases':[],'output':str(out)}
    started=time.perf_counter();summary_rows=[]
    try:
        device=require_gpu()
        report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,
                           'gpu':torch.cuda.get_device_name(0),'node':os.environ.get('SLURMD_NODENAME'),
                           'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                           'cudnn_deterministic':torch.backends.cudnn.deterministic,
                           'exact_cuda_trajectory_claimed':False}
        dataset=ChairImages(preparation['nmr_chair_root'],preparation['smoke_objects'])
        images=torch.stack([dataset[i] for i in range(4)]).to(device)
        common_hash=None;geometry_hash=None;texture_hash=None
        for group in GROUPS:
            print('[CHECK] '+group,flush=True)
            case={'group':group,'ok':False};report['cases'].append(case)
            config=config_for(group);model=make_model(config,device)
            report['runtime']['cudnn_deterministic']=torch.backends.cudnn.deterministic
            optimizer,scheduler,manifest=optimizer_for(model,config)
            case['optimizer_groups']=manifest;case['initial_hashes']=initial_hashes(model)
            h=case['initial_hashes']
            if common_hash is None:common_hash=h['common']
            if h['common']!=common_hash:raise RuntimeError('Common network initialization differs across groups')
            if h['geometry']:
                if geometry_hash is None:geometry_hash=h['geometry']
                if h['geometry']!=geometry_hash:raise RuntimeError('Geometry initialization differs across active groups')
            if h['texture']:
                if texture_hash is None:texture_hash=h['texture']
                if h['texture']!=texture_hash:raise RuntimeError('Texture initialization differs across active groups')
            if any(n in dict(model.named_parameters()) for n in ALPHAS):raise RuntimeError('Alpha must not be trainable')
            if hasattr(model,'vert_fc')!=config['geometry'] or hasattr(model,'uv_fc')!=config['texture']:
                raise RuntimeError('Disabled branch still exists or active branch missing')
            case['schedule_boundary_checks']=[]
            for index in (0,29,30,31,127,224,237):
                model.sync_epoch(index);actual=alpha_values(model);expected=alpha_at(config,index)
                if not np.allclose(actual,expected,rtol=1e-6,atol=1e-9):raise RuntimeError('Alpha schedule boundary mismatch')
                case['schedule_boundary_checks'].append({'epoch_index':index,'alpha':actual})
            # Actual zero-alpha/noS early-phase backward checks, not just scalar formula tests.
            model.sync_epoch(0);early=[]
            for _ in range(2):
                row=update(model,optimizer,images);phase_isolation(model,row['phase']);early.append(row)
                if config['schedule'] and (row['geometry_gradient_entries'] or row['texture_gradient_entries']):
                    raise RuntimeError('Scheduled residual received gradients before activation')
            # Mature-stage fixture activates progressive shape/texture and neighborhood loss.
            model.sync_epoch(127);case_rows=[]
            for _ in range(5):
                if not model.pose_step:np.random.seed(0) # Ensure non-averaged texture for this gradient fixture only.
                before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith(('vert_fc.','uv_fc.'))}
                row=update(model,optimizer,images);phase_isolation(model,row['phase'])
                for label in ('common','residual'):
                    clip=config['training']['clip_'+label]
                    if row[label+'_grad_after_clip']>clip*(1+1e-5):raise RuntimeError('Clipping did not bound gradient')
                if row['phase']=='pose':
                    if any(not torch.equal(old,dict(model.named_parameters())[n]) for n,old in before.items()):
                        raise RuntimeError('Detached residual parameter changed during pose step')
                else:
                    for label in ('geometry','texture'):
                        if config[label] and row[label+'_grad_norm']<=0:
                            raise RuntimeError('Enabled residual has no nonzero 3D gradient: '+label)
                row.update({'group':group,'fixture':'mature','global_update':len(case_rows)+1,'epoch':128})
                case_rows.append(row)
            case['mature_neighbor_memory_size']=len(model.nbr_memory['img'])
            if case['mature_neighbor_memory_size']<=0:raise RuntimeError('Inherited neighborhood path was not exercised')
            if group in ('U','GTS'):
                from .resume_checks import resume_step
                case['resume_checks']=[]
                for fixture_step in range(2):
                    phase='pose' if model.pose_step else '3d'
                    cp=out/(group+'_resume_'+phase+'.pkl')
                    cursor={'epoch_index':127,'batch':5+fixture_step}
                    counters={'updates':7+fixture_step,'pose':4,'3d':3+fixture_step}
                    result={};case['resume_checks'].append(result)
                    model,optimizer,scheduler=resume_step(cp,model,optimizer,scheduler,config,cursor,
                                                         fingerprint,counters,images,result)
                    print('[RESUME] {} {} exact_load=True gradient_max_abs={:.3e} live_weight_max_abs={:.3e} fixed_gradient_Adam=True'.format(
                        group,phase,result['gradient_comparison']['max_abs'],result['live_weight_comparison']['max_abs']),flush=True)
                    for label,check in [('same_instance',result['same_instance_control']['gradient_vector_check']),
                                        ('reconstructed',result['gradient_vector_check'])]:
                        for name,metrics in check['groups'].items():
                            print('[GRADIENT_VECTOR] {} {} {} {} relative_L2={:.3e} cosine={:.9f} passed={}'.format(
                                group,phase,label,name,metrics['relative_l2'],metrics['cosine'],metrics['passed']),flush=True)
                before_validation=tensor_hash(model.state_dict())
                score=validation_mse(model,preparation['nmr_chair_root'],preparation['validation_objects'][:4])
                if not math.isfinite(score) or tensor_hash(model.state_dict())!=before_validation:
                    raise RuntimeError('Validation changed model state or produced nonfinite score')
                case['validation_path']={'objects':4,'rgb_mse':score,'state_unchanged':True}
            case['early_rows']=early
            write_metrics(out/(group+'_metrics.csv'),case_rows)
            summary_rows.extend(case_rows)
            if group=='GTS':
                gc.collect();torch.cuda.empty_cache()
                benchmark_images=torch.stack([dataset[i] for i in range(32)]).to(device)
                torch.cuda.reset_peak_memory_stats();benchmark=[]
                for _ in range(4):benchmark.append(update(model,optimizer,benchmark_images))
                case['batch32_calibration']={'rows':benchmark,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'scope':'Four real batch32 updates at mature fixture; not a training-time or quality result.'}
            case['ok']=True
            print('[PASS] '+group+': initialization, gates, gradients, clipping, stage isolation'+
                  (', two-phase restart' if 'resume_checks' in case else ''),flush=True)
            del model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
        report['all_passed']=len(report['cases'])==7 and all(c['ok'] for c in report['cases'])
        report['common_initial_hash']=common_hash
        from .plot_losses import plot_smoke
        plot_smoke(summary_rows,out)
        # This is a gate for the frozen stage only; future edits invalidate the fingerprint.
        if report['all_passed']:
            dump(stage/'smoke'/('PASS_'+out.name+'.json'),{'fingerprint':fingerprint,'report':str(report_path),'all_passed':True})
    except Exception as exc:
        report['all_passed']=False
        report['error']=str(exc);report['traceback']=traceback.format_exc()
        print('[FAIL] '+str(exc),flush=True);print(report['traceback'],flush=True)
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;dump(report_path,report)
        print('REPORT: '+str(report_path),flush=True)
        print('ARTIFACTS: '+str(out),flush=True)
        print('ALL_PASSED: '+str(report['all_passed']),flush=True)
    return 0 if report['all_passed'] else 1


def smoke_gate(stage,fingerprint):
    matches=[]
    for path in (stage/'smoke').glob('PASS_*.json'):
        value=json.loads(path.read_text(encoding='utf-8'))
        if value.get('all_passed') is True and value.get('fingerprint')==fingerprint:
            report_path=Path(value['report'])
            if report_path.is_file():
                report=json.loads(report_path.read_text(encoding='utf-8'))
                if report.get('all_passed') is True and report.get('fingerprint')==fingerprint:matches.append(str(path))
    if not matches:raise RuntimeError('No successful real GPU smoke for this exact frozen stage; long training is blocked')
    return sorted(matches)[-1]


def train(workspace,stage,preparation,fingerprint,group):
    device=require_gpu();gate=smoke_gate(stage,fingerprint)
    if shutil.disk_usage(workspace).free<16*1024**3:raise RuntimeError('At least16GiB free required before starting this training cohort')
    import fcntl
    run=stage/'runs'/group;run.mkdir(exist_ok=True)
    lock=(run/'training.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('This group is already being trained; duplicate job refused')
    name=token();segment=run/'segments'/name;segment.mkdir(parents=True)
    config=config_for(group);dump(segment/'config.json',config)
    info={'group':group,'fingerprint':fingerprint,'gate':gate,'job_id':os.environ.get('SLURM_JOB_ID','manual'),
          'node':os.environ.get('SLURMD_NODENAME'),'gpu':torch.cuda.get_device_name(0),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
          'exact_cuda_trajectory_claimed':False}
    dump(segment/'runtime.json',info)
    checkpoint=run/'checkpoint.pkl';started=time.perf_counter();all_passed=False
    try:
        if (run/'final.pkl').exists():raise RuntimeError('This configuration already finished; refusing duplicate training')
        model=make_model(config,device);optimizer,scheduler,manifest=optimizer_for(model,config)
        dump(segment/'optimizer_parameter_groups.json',manifest)
        dump(segment/'initial_hashes.json',initial_hashes(model))
        cursor={'epoch_index':0,'batch':0};counters={'updates':0,'pose':0,'3d':0}
        if checkpoint.exists():
            cursor,counters=load_checkpoint(checkpoint,model,optimizer,scheduler,config,fingerprint)
            print('RESUME: '+json.dumps({'cursor':cursor,'counters':counters}),flush=True)
        dataset=ChairImages(preparation['nmr_chair_root'],preparation['train_objects'])
        n_batches=(len(dataset)+config['dataset']['batch_size']-1)//config['dataset']['batch_size']
        log=(segment/'metrics.csv').open('x',encoding='utf-8',newline='');writer=None
        val=(segment/'validation.csv').open('x',encoding='utf-8',newline='')
        val_writer=csv.DictWriter(val,fieldnames=['global_update','epoch','epoch_index','val_rgb_mse','objects']);val_writer.writeheader()
        # Cooperative wall-time stop keeps a clean checkpoint before a96-hour Slurm limit.
        deadline=time.monotonic()+93*3600
        for epoch in range(cursor['epoch_index'],config['training']['epochs']):
            start_batch=cursor['batch'] if epoch==cursor['epoch_index'] else 0
            model.sync_epoch(epoch)
            loader=epoch_loader(dataset,config,epoch,start_batch)
            for batch,images in enumerate(loader,start=start_batch):
                result=update(model,optimizer,images.to(device,non_blocking=True))
                counters['updates']+=1;counters[result['phase']]+=1
                result.update({'group':group,'epoch':epoch+1,'batch':batch+1,'global_update':counters['updates'],
                               'pose_updates':counters['pose'],'geometry_updates':counters['3d'],
                               'segment':name,'segment_wall_seconds':time.perf_counter()-started})
                if writer is None:
                    writer=csv.DictWriter(log,fieldnames=list(result));writer.writeheader()
                writer.writerow(result)
                if counters['updates']%20==0:log.flush()
                if counters['updates']%200==0:
                    print('PROGRESS {} epoch={}/{} batch={}/{} update={} phase={} loss={:.6f} alpha={:.6f}/{:.6f}'.format(
                        group,epoch+1,config['training']['epochs'],batch+1,n_batches,counters['updates'],result['phase'],result['loss_total'],result['alpha_shape'],result['alpha_texture']),flush=True)
                if counters['updates']%config['training']['validation_updates']==0:
                    score=validation_mse(model,preparation['nmr_chair_root'],preparation['validation_objects'])
                    val_writer.writerow({'global_update':counters['updates'],'epoch':epoch+1,'epoch_index':epoch,
                                         'val_rgb_mse':score,'objects':len(preparation['validation_objects'])});val.flush()
                cursor={'epoch_index':epoch,'batch':batch+1}
                if batch+1==n_batches:
                    scheduler.step();model.sync_epoch(epoch+1);cursor={'epoch_index':epoch+1,'batch':0}
                if counters['updates']%config['training']['checkpoint_updates']==0 or batch+1==n_batches or time.monotonic()>deadline:
                    log.flush();atomic_checkpoint(checkpoint,model,optimizer,scheduler,config,cursor,fingerprint,counters)
                if time.monotonic()>deadline:
                    print('CLEAN_STOP: checkpoint saved before job time limit; resubmit same group to continue.',flush=True)
                    return 0
            print('EPOCH_DONE '+group+' '+str(epoch+1),flush=True)
        atomic_checkpoint(run/'final.pkl',model,optimizer,scheduler,config,cursor,fingerprint,counters)
        all_passed=True
        print('TRAINING_COMPLETED: '+group,flush=True)
    except Exception:
        print(traceback.format_exc(),flush=True);raise
    finally:
        if 'log' in locals():log.close()
        if 'val' in locals():val.close()
        plot_error=None
        if 'writer' in locals() and writer is not None:
            try:
                from .plot_losses import plot_all
                plot_all(stage,only_group=group)
            except Exception as exc:
                plot_error=str(exc)
                print('CURVE_ERROR: '+plot_error+'; raw per-update CSV is preserved.',flush=True)
        dump(segment/'segment_end.json',{'training_completed':all_passed,'elapsed_seconds':time.perf_counter()-started,
                                        'cursor':cursor if 'cursor' in locals() else None,
                                        'counters':counters if 'counters' in locals() else None,
                                        'curve_error':plot_error})
        lock.close()
    return 0
