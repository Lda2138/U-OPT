"""Seven-group final-weight evaluation. Preparation imports only the stdlib."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import time
import traceback

from geometry_metrics import PROTOCOL as GEOMETRY_PROTOCOL
from image_metrics import IMAGE_PROTOCOL

STAGE = 'chair_metrics_v1'
TRAIN_STAGE = 'chair_ablation_v1'
GROUPS = ('U', 'G', 'T', 'GT', 'GS', 'TS', 'GTS')
PROTOCOL = {
    'version': 'chair_seven_final_view0_10k_rgb_v1',
    'geometry': GEOMETRY_PROTOCOL,
    'images': IMAGE_PROTOCOL,
    'scope': 'Seven newly trained groups, one paired seed132422; final238-epoch checkpoints only. No training.',
    'full_selection': 'Every object in frozen softras_test.lst, in original order; one input view0 per object.',
    'pilot_selection': 'Eight evenly spaced test-list positions, including both endpoints; all seven models.',
    'checkpoint_selection': 'Fixed final.pkl, never monitored minimum or legacy model.pkl.',
    'self_intersection': 'Not computed; mesh connectivity and triangle quality are not self-intersection tests.',
}



def build_bundle(destination):
    import zipfile
    root = Path(__file__).resolve().parent
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('evaluation.py', 'geometry_metrics.py', 'image_metrics.py'):
            archive.write(root / name, name)
        archive.writestr('__main__.py', 'import sys\nsys.dont_write_bytecode=True\nfrom evaluation import main\nraise SystemExit(main())\n')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def valid_ids(values):
    if not values or len(values) != len(set(values)):
        raise ValueError('Empty or repeated object IDs')
    if any(Path(n).name != n or n in ('.', '..') or '/' in n or '\\' in n for n in values):
        raise ValueError('Unsafe object identifier')


def prepare(workspace, archive):
    workspace = Path(workspace).resolve()
    stage = workspace/'stages'/STAGE
    train = workspace/'stages'/TRAIN_STAGE
    if stage.exists() or stage.is_symlink() or not inside(stage, workspace):
        raise ValueError('Refusing to overwrite an evaluation stage. Use its existing sbatch files to resume.')
    for part in ('logs', 'reports'):
        if not inside(workspace/part, workspace):
            raise ValueError('Output path leaves personal workspace')
    preparation_path = train/'preparation.json'
    training = read_json(preparation_path)
    training_fingerprint = sha256(preparation_path)
    if training.get('version') != 'chair_gts_20261030_v3':
        raise ValueError('Expected the released seven-group training protocol')
    for rel, expected in {**training['source_hashes'], **training['package_hashes']}.items():
        if not inside(train/rel, train) or sha256(train/rel) != expected:
            raise ValueError('Frozen training source changed: '+rel)
    nmr = Path(training['nmr_chair_root'])
    splits = {}
    for name in ('train', 'val', 'test'):
        path = nmr/('softras_'+name+'.lst')
        if sha256(path) != training['split_sha256'][name]:
            raise ValueError('Frozen split changed: '+name)
        splits[name] = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        valid_ids(splits[name])
    if any(set(splits[a]) & set(splits[b]) for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Dataset splits overlap')
    if len(splits['test']) != 1356 or len(splits['test']) != training['test_object_count']:
        raise ValueError('Expected1356 chair test objects')
    objects = splits['test']
    pilot = [objects[i*(len(objects)-1)//7] for i in range(8)]
    checkpoints = {}
    for group in GROUPS:
        config = read_json(train/'configs'/(group+'.json'))
        if config['group'] != group or config['training']['epochs'] != 238:
            raise ValueError('Unexpected group config')
        config_hash = hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if config_hash != training['config_hashes'][group]:
            raise ValueError('Frozen training config changed: '+group)
        checkpoint = train/'runs'/group/'final.pkl'
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
            raise ValueError('Missing final checkpoint: '+group)
        checkpoints[group] = {'path': str(checkpoint), 'sha256': sha256(checkpoint),
                              'bytes': checkpoint.stat().st_size, 'config': config}
    vgg = Path(training['vgg_weights'])
    if sha256(vgg) != training['vgg_sha256']:
        raise ValueError('Offline VGG changed')
    data_files = {}
    for index, obj in enumerate(objects):
        data_files[obj] = {rel: sha256(nmr/obj/rel) for rel in ('pointcloud.npz','cameras.npz','image/0000.png')}
        if (index+1)%300 == 0:
            print('PREPARE_DATA: '+str(index+1)+'/'+str(len(objects)), flush=True)
    stage.mkdir()
    for part in ('runs','completed_objects','qualitative'):
        (stage/part).mkdir()
    copy_sources = dict(training['source_hashes'])
    for rel in ('ablation/__init__.py', 'ablation/protocol.py', 'ablation/engine.py'):
        copy_sources[rel] = training['package_hashes'][rel]
    for rel, expected in copy_sources.items():
        target = stage/rel
        if not inside(target, stage):
            raise ValueError('Unsafe copied source path')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(train/rel, target)
        if sha256(target) != expected:
            raise ValueError('Copied source mismatch')
    runner = stage/'evaluate.pyz'
    build_bundle(runner)
    prep = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'protocol': PROTOCOL, 'training_stage': str(train),
            'training_fingerprint': training_fingerprint,
            'runner_sha256': sha256(runner), 'source_hashes': copy_sources, 'checkpoint_selection': 'fixed_final_epoch238',
            'nmr_chair_root': str(nmr), 'split_path': str(nmr/'softras_test.lst'),
            'split_sha256': training['split_sha256']['test'], 'objects': objects,
            'pilot_objects': pilot, 'data_files': data_files, 'checkpoints': checkpoints,
            'vgg_weights': str(vgg), 'vgg_sha256': training['vgg_sha256']}
    write_json(stage/'preparation.json', prep)
    for part in ('logs','reports'):
        (workspace/part).mkdir(exist_ok=True)
    base = ['#!/bin/bash', '#SBATCH --partition=gpu', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
            '#SBATCH --gres=gpu:1', '#SBATCH --cpus-per-task=4', '#SBATCH --mem=24G']
    environment = ['set -eo pipefail',
                   'export CUBLAS_WORKSPACE_CONFIG=:4096:8',
                   'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"']
    command = shlex.quote(sys.executable)+' -B -u '+shlex.quote(str(runner))+' --workspace '+shlex.quote(str(workspace))
    pilot_job = base+['#SBATCH --job-name=uopt_seven_pilot', '#SBATCH --time=01:00:00',
                     '#SBATCH --output='+str(workspace/'logs/uopt_seven_pilot_%j.out')]+environment+[command+' --smoke']
    full_job = base+['#SBATCH --job-name=uopt_seven_eval', '#SBATCH --time=04:00:00', '#SBATCH --array=0-6%2',
                    '#SBATCH --output='+str(workspace/'logs/uopt_seven_eval_%A_%a.out')]+environment+[command+' --full']
    (stage/'pilot.sbatch').write_text('\n'.join(pilot_job)+'\n', encoding='utf-8')
    (stage/'evaluate_array.sbatch').write_text('\n'.join(full_job)+'\n', encoding='utf-8')
    print('PREPARED: '+str(stage), flush=True)
    print('FINAL_WEIGHTS: '+', '.join(GROUPS), flush=True)
    print('TEST_OBJECTS: '+str(len(objects))+'; PILOT: 8 objects x7 groups', flush=True)
    print('METRICS: CD, F-score@1%/2%, absolute normal cosine, mesh quality, RGB MSE/PSNR/SSIM', flush=True)
    print('NEXT_PILOT_ONLY: sbatch '+str(stage/'pilot.sbatch'), flush=True)


def verify(stage):
    prep = read_json(stage/'preparation.json')
    if prep['protocol'] != PROTOCOL or sha256(stage/'evaluate.pyz') != prep['runner_sha256']:
        raise ValueError('Frozen evaluator/protocol changed')
    if sha256(Path(prep['training_stage'])/'preparation.json') != prep['training_fingerprint']:
        raise ValueError('Training preparation changed')
    for rel, expected in prep['source_hashes'].items():
        if not inside(stage/rel, stage) or sha256(stage/rel) != expected:
            raise ValueError('Frozen copied source changed: '+rel)
    if sha256(prep['split_path']) != prep['split_sha256'] or sha256(prep['vgg_weights']) != prep['vgg_sha256']:
        raise ValueError('Dataset split or offline VGG changed')
    return prep, sha256(stage/'preparation.json')


def validate_checkpoint_metadata(checkpoint, config, fingerprint):
    from ablation.protocol import stable_hash, config_for
    if config != config_for(config['group']):
        raise ValueError('Config differs from frozen training protocol')
    if (checkpoint.get('format') != 'uopt_ablation_v1' or checkpoint.get('config') != config or
        checkpoint.get('config_hash') != stable_hash(config) or checkpoint.get('fingerprint') != fingerprint):
        raise ValueError('Checkpoint/config/source mismatch')
    if checkpoint.get('cursor') != {'epoch_index':238,'batch':0} or checkpoint.get('cur_epoch') != 238:
        raise ValueError('Checkpoint is not the fixed final epoch')
    if checkpoint.get('counters') != {'updates':846804,'pose':423402,'3d':423402}:
        raise ValueError('Incomplete checkpoint update counters')


def strict_restore_model(model, state):
    import torch
    current = model.state_dict()
    if set(current) != set(state) or any(current[k].shape != state[k].shape or current[k].dtype != state[k].dtype for k in current):
        raise ValueError('Checkpoint model schema mismatch')
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint tensor')
    if 'uvs' in current and not torch.allclose(current['uvs'].cpu(), state['uvs'].cpu(), rtol=1e-6, atol=1e-7):
        raise ValueError('Template UV indexing mismatch')
    torch.nn.Module.load_state_dict(model, state, strict=True)
    if any(not torch.equal(v.detach().cpu(), state[k].detach().cpu()) for k,v in model.state_dict().items()):
        raise ValueError('Restored model tensors differ from checkpoint')


def restore_final_model(group, prep, device):
    import torch
    from ablation.engine import make_model
    from ablation.protocol import ALPHAS, alpha_at
    item = prep['checkpoints'][group]
    if sha256(item['path']) != item['sha256']:
        raise ValueError('Final checkpoint changed: '+group)
    try:
        checkpoint = torch.load(item['path'], map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(item['path'], map_location='cpu')
    validate_checkpoint_metadata(checkpoint, item['config'], prep['training_fingerprint'])
    model = make_model(item['config'], device)
    strict_restore_model(model, checkpoint['model_state'])
    alpha = {name: checkpoint['model_state'][name].detach().cpu().tolist() for name in ALPHAS}
    model.sync_epoch(checkpoint['cur_epoch'])
    if any(not torch.equal(model.state_dict()[name].cpu(), checkpoint['model_state'][name].cpu()) for name in ALPHAS):
        raise ValueError('Final epoch synchronization changed saved alpha')
    if any(name in dict(model.named_parameters()) for name in ALPHAS):
        raise ValueError('Fusion alpha must remain a buffer')
    model.eval()
    check = {'group':group,'strict_state_entries':len(checkpoint['model_state']), 'checkpoint_sha256':item['sha256'],
             'saved_epoch':238,'saved_alpha':alpha,'expected_alpha':alpha_at(item['config'],238),
             'exact_tensor_restore':True,'constructor_uvs_match':True}
    del checkpoint
    gc.collect()
    return model, check


def validate_row(envelope, fingerprint, group, obj):
    from geometry_metrics import paired_seed
    if envelope.get('evaluation_fingerprint') != fingerprint:
        raise ValueError('Cached object belongs to a different evaluator')
    row = envelope['row']
    if (row.get('group') != group or row.get('object_id') != obj or row.get('view') != 0 or
        row.get('seed') != paired_seed(obj,0) or row.get('pred_points') != 10000):
        raise ValueError('Cached object identity/settings mismatch')
    required = ('gt_subset_sha256','raw_chamfer_l1_x10','icp_chamfer_l1_x10', 'icp_fscore_01',
                'icp_fscore_02','icp_normal_abs_cos','mesh_mean_triangle_quality','image_psnr_db','image_ssim')
    if any(key not in row for key in required):
        raise ValueError('Incomplete cached metrics')
    if any(not math.isfinite(v) for v in row.values() if isinstance(v,(int,float))):
        raise ValueError('Nonfinite cached metric')
    return row


def publish(path, value, token):
    if path.exists():
        raise ValueError('Refusing to overwrite a completed object')
    temporary = path.with_name(path.name+'.tmp.'+token)
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary),str(path))


def evaluate_object(model, obj, prep, device, qualitative_path=None):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    from utils.icp import gradient_icp
    from geometry_metrics import paired_seed, mesh_quality, surface_metrics, transform_normals
    from image_metrics import image_metrics
    root = Path(prep['nmr_chair_root'])/obj
    seed = paired_seed(obj,0)
    started = time.perf_counter()
    with np.load(root/'pointcloud.npz',allow_pickle=False) as arr:
        rotation = np.array(GEOMETRY_PROTOCOL['gt_coordinate_rotation'],dtype=np.float32)
        gt_all = np.array(arr['points'],dtype=np.float32) @ rotation
        gn_all = np.array(arr['normals'],dtype=np.float32) @ rotation
    if not np.isfinite(gt_all).all() or abs(float(np.abs(gt_all).max())-.5)>=.01:
        raise ValueError('GT does not fit expected unit cube: '+obj)
    diagonal = float(np.linalg.norm(np.ptp(gt_all.astype(np.float64),axis=0)))
    indices = np.random.default_rng(seed).choice(len(gt_all),min(len(gt_all),10000),replace=False)
    gt, gn = gt_all[indices],gn_all[indices]
    gt_tensor = torch.from_numpy(gt).unsqueeze(0).to(device)
    with np.load(root/'cameras.npz',allow_pickle=False) as cameras:
        P = np.array(cameras['world_mat_0'],dtype=np.float32)
    if P.shape!=(4,4) or not np.isfinite(P).all():
        raise ValueError('Invalid GT camera')
    row_adj = torch.tensor([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]],device=device)
    col_adj = torch.diag(torch.tensor([-1.,-1.,1.],device=device))
    R_gt = row_adj @ torch.from_numpy(P[:3,:3].copy()).to(device).T @ col_adj
    T_gt = torch.from_numpy(P[:3,3].copy()).to(device)
    with Image.open(root/'image/0000.png') as image:
        if image.size!=(64,64):
            raise ValueError('Input must remain64x64')
        imgs = to_tensor(image.convert('RGB')).unsqueeze(0).to(device)
    tick = time.perf_counter()
    with torch.no_grad():
        mesh,(R,T),_ = model.predict_mesh_pose_bkg(imgs)
        rgb = model.renderer(mesh,R,T)[:,:3]
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-tick
        v,f = mesh.verts_padded(),mesh.faces_padded()
        if not all(torch.isfinite(a).all() for a in (v,R,T,rgb)):
            raise ValueError('Nonfinite prediction')
        pred_image = rgb[0].permute(1,2,0).cpu().numpy()
        target = imgs[0].permute(1,2,0).cpu().numpy()
        image_values = image_metrics(pred_image,target)
        if qualitative_path is not None and not qualitative_path.exists():
            pair = np.concatenate([target,pred_image.clip(0,1)],axis=1)
            Image.fromarray(np.round(pair*255).clip(0,255).astype('uint8')).save(qualitative_path)
        quality = mesh_quality(v[0].cpu().numpy(),f[0].cpu().numpy())
        aligned = (v @ R + T[:,None] - T_gt[None,None]) @ R_gt.T
        eval_mesh = Meshes(verts=aligned,faces=f)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        pred,normals = sample_points_from_meshes(eval_mesh,10000,return_normals=True)
        raw = surface_metrics(pred[0].cpu().numpy(),normals[0].cpu().numpy(),gt,gn,diagonal)
        center = .5*(aligned.max(1)[0]+aligned.min(1)[0])
        scale_norm = (aligned-center[:,None]).abs().flatten(1).max(1)[0]*2
        if (scale_norm<=1e-8).any():
            raise ValueError('Collapsed prediction')
        normalized_points = ((pred-center[:,None])/scale_norm[:,None,None]).detach()
    tick = time.perf_counter()
    corrected,(Rf,Tf,sf) = gradient_icp(normalized_points,gt_tensor,estimate_scale=True,anisotropic_scale=True,
                                       lr=GEOMETRY_PROTOCOL['icp']['lr'],n_iter=GEOMETRY_PROTOCOL['icp']['iterations'])
    torch.cuda.synchronize()
    icp_seconds = time.perf_counter()-tick
    Rn,sn = Rf.detach()[0].cpu().numpy(),sf.detach()[0].cpu().numpy()
    post = surface_metrics(corrected.detach()[0].cpu().numpy(),transform_normals(normals[0].cpu().numpy(),Rn,sn),gt,gn,diagonal)
    row = {'object_id':obj,'view':0,'seed':seed,'gt_available':int(len(gt_all)),'gt_used':int(len(gt)),
           'gt_subset_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
           'pred_points':10000,'gt_diagonal':diagonal,'inference_seconds':float(inference_seconds),
           'icp_seconds':float(icp_seconds),'total_object_seconds':float(time.perf_counter()-started),
           'icp_scale_x':float(sn[0]),'icp_scale_y':float(sn[1]),'icp_scale_z':float(sn[2]),
           'icp_rotation_determinant':float(np.linalg.det(Rn)),'icp_has_negative_scale':bool((sn<0).any())}
    for prefix,values in (('raw',raw),('icp',post),('mesh',quality),('image',image_values)):
        row.update((prefix+'_'+key,value) for key,value in values.items())
    return row


def means(rows):
    import numpy as np
    keys = [k for k in rows[0] if k.startswith(('raw_','icp_','mesh_','image_')) and isinstance(rows[0][k],(int,float))]
    return {key:float(np.mean([r[key] for r in rows])) for key in keys}


def pilot_gate(stage, fingerprint):
    gate = read_json(stage/'pilot_pass.json')
    report = read_json(gate['report'])
    if (gate.get('evaluation_fingerprint') != fingerprint or sha256(gate['report']) != gate['report_sha256'] or
        report.get('evaluation_fingerprint') != fingerprint or report.get('mode') != 'pilot' or
        report.get('all_passed') is not True or len(report['rows']) != 56 or len(report['model_checks']) != 7):
        raise ValueError('A successful exact-stage pilot is required before full evaluation')


def run(workspace, mode, group=None):
    stage = workspace/'stages'/STAGE
    prep, fingerprint = verify(stage)
    if sha256(Path(sys.argv[0]).resolve()) != prep['runner_sha256']:
        raise ValueError('Execute the prepared evaluate.pyz bundle')
    job = os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():
        raise ValueError('Submit this GPU evaluation through its sbatch file')
    if mode=='full':
        pilot_gate(stage,fingerprint)
        if group is None:
            index = int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))
            if not 0<=index<len(GROUPS):
                raise ValueError('Missing valid Slurm array index')
            group = GROUPS[index]
        groups, objects = (group,),prep['objects']
    else:
        if (stage/'pilot_pass.json').exists():
            pilot_gate(stage,fingerprint)
            print('PILOT_ALREADY_PASSED: use evaluate_array.sbatch',flush=True)
            return 0
        groups, objects = GROUPS,prep['pilot_objects']
    import fcntl
    lock_path = stage/(('full_'+group if mode=='full' else 'pilot')+'.lock')
    with lock_path.open('a') as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This evaluation is already running; duplicate job refused')
        if mode=='full' and (stage/('final_'+group+'.json')).exists():
            previous = read_json(stage/('final_'+group+'.json'))
            if previous.get('all_passed') is not True or previous.get('evaluation_fingerprint') != fingerprint:
                raise ValueError('Unexpected existing final evaluation report')
            print('EVALUATION_ALREADY_COMPLETED: '+group,flush=True)
            return 0
        token = job+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out = stage/'runs'/token
        out.mkdir()
        report_path = workspace/'reports'/('uopt_seven_'+mode+'_'+token+'.json')
        report = {'job_id':job,'mode':mode,'groups':list(groups),'evaluation_fingerprint':fingerprint,
                  'all_passed':False,'protocol':PROTOCOL,'rows':[],'model_checks':[],
                  'checkpoint_sources':{g:prep['checkpoints'][g] for g in groups},'new_objects_evaluated':0,'objects_reused':0}
        started = time.perf_counter()
        csv_stream = None
        status = 1
        try:
            sys.path.insert(0,str(stage/'src'))
            sys.path.insert(0,str(stage))
            os.environ.pop('SCRATCH', None)
            os.environ['UOPT_VGG_WEIGHTS'] = prep['vgg_weights']
            import numpy as np
            import torch
            from PIL import Image
            from torchvision.transforms.functional import to_tensor
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required; do not run evaluation on the login node')
            torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
            device = torch.device('cuda:0')
            report['runtime'] = {'torch':str(torch.__version__),'cuda':torch.version.cuda,
                                 'gpu':torch.cuda.get_device_name(0),'node':os.environ.get('SLURMD_NODENAME')}
            writer = None
            for name in groups:
                print('[MODEL] '+name+': strictly restoring final.pkl',flush=True)
                model,check = restore_final_model(name,prep,device)
                warm_path = Path(prep['nmr_chair_root'])/objects[0]/'image/0000.png'
                with Image.open(warm_path) as im:
                    warm = to_tensor(im.convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    m1,(r1,t1),_ = model.predict_mesh_pose_bkg(warm)
                    m2,(r2,t2),_ = model.predict_mesh_pose_bkg(warm)
                    differences = [float((a-b).abs().max().item()) for a,b in ((m1.verts_padded(),m2.verts_padded()),(r1,r2),(t1,t2))]
                if max(differences)>1e-6:
                    raise ValueError('Repeated evaluation forward differs: '+name)
                check['repeat_mesh_pose_max_abs_difference'] = max(differences)
                report['model_checks'].append(check)
                del m1,m2,r1,r2,t1,t2,warm
                cache_dir = stage/'completed_objects'/name
                cache_dir.mkdir(exist_ok=True)
                for index,obj in enumerate(objects,1):
                    root = Path(prep['nmr_chair_root'])/obj
                    for rel,expected in prep['data_files'][obj].items():
                        if sha256(root/rel)!=expected:
                            raise ValueError('Frozen test input changed: '+obj+'/'+rel)
                    cache = cache_dir/(obj+'.json')
                    if cache.exists():
                        row = validate_row(read_json(cache),fingerprint,name,obj)
                        report['objects_reused'] += 1
                    else:
                        picture = stage/'qualitative'/(name+'_'+obj+'.png') if obj in prep['pilot_objects'] else None
                        row = evaluate_object(model,obj,prep,device,picture)
                        row = {'group':name,**row}
                        envelope = {'evaluation_fingerprint':fingerprint,'completed_by_job':job,'row':row}
                        validate_row(envelope,fingerprint,name,obj)
                        publish(cache,envelope,token)
                        report['new_objects_evaluated'] += 1
                    if writer is None:
                        csv_stream = (out/'per_object.csv').open('x',encoding='utf-8',newline='')
                        writer = csv.DictWriter(csv_stream,fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    csv_stream.flush()
                    report['rows'].append(row)
                    if index==1 or index%100==0 or index==len(objects):
                        print('[PASS] {} {}/{} CD={:.6f} F1={:.4f} PSNR={:.3f} SSIM={:.4f}'.format(
                            name,index,len(objects),row['icp_chamfer_l1_x10'],row['icp_fscore_01'],row['image_psnr_db'],row['image_ssim']),flush=True)
                if sha256(prep['checkpoints'][name]['path'])!=prep['checkpoints'][name]['sha256']:
                    raise ValueError('Final checkpoint changed during evaluation')
                del model
                gc.collect()
                torch.cuda.empty_cache()
            for obj in objects:
                paired = [r for r in report['rows'] if r['object_id']==obj]
                if len(paired)!=len(groups) or len({(r['seed'],r['gt_subset_sha256'],r['gt_used'],r['gt_diagonal']) for r in paired})!=1:
                    raise ValueError('Unpaired test object data/seed')
            report['metric_means'] = {g:means([r for r in report['rows'] if r['group']==g]) for g in groups}
            report['image_psnr_floor_applied_counts'] = {g:sum(bool(r['image_psnr_floor_applied']) for r in report['rows'] if r['group']==g) for g in groups}
            report['all_passed'] = len(report['rows'])==len(groups)*len(objects) and len(report['model_checks'])==len(groups)
            if not report['all_passed']:
                raise ValueError('Incomplete evaluation')
            status = 0
        except Exception as exc:
            report['error'] = str(exc)
            report['traceback'] = traceback.format_exc()
            print(report['traceback'],flush=True)
        finally:
            if csv_stream is not None:
                csv_stream.close()
            report['elapsed_seconds'] = time.perf_counter()-started
            report['artifacts'] = str(out)
            write_json(report_path,report)
        if report['all_passed']:
            if mode=='pilot':
                write_json(stage/'pilot_pass.json',{'evaluation_fingerprint':fingerprint,'report':str(report_path),
                                                   'report_sha256':sha256(report_path),'all_passed':True})
            else:
                write_json(stage/('final_'+group+'.json'),report)
        print('REPORT: '+str(report_path),flush=True)
        print('ARTIFACTS: '+str(out),flush=True)
        print('ALL_PASSED: '+str(report['all_passed']),flush=True)
        if mode=='pilot' and report['all_passed']:
            print('NEXT: sbatch '+str(stage/'evaluate_array.sbatch'),flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--smoke',action='store_true')
    modes.add_argument('--full',action='store_true')
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--group',choices=GROUPS)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.workspace,None)
        return 0
    return run(args.workspace.resolve(),'pilot' if args.smoke else 'full',args.group)
