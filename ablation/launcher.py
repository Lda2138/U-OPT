"""Prepare an isolated, fingerprinted seven-group workspace from public code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

from .protocol import GROUPS, STAGE_NAME, config_for, stable_hash, validate_configs


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def prepare(workspace, data_root, vgg_weights, partition='gpu'):
    repository = Path(__file__).resolve().parents[1]
    if data_root is None or vgg_weights is None:
        raise ValueError('--data-root and --vgg-weights are required when preparing')
    workspace, root, vgg = map(lambda p: Path(p).resolve(), (workspace, data_root, vgg_weights))
    stage = workspace/'stages'/STAGE_NAME
    if stage.exists() or stage.is_symlink() or not inside(stage, workspace):
        raise ValueError('Refusing to overwrite an existing stage; reuse its run.py to resume')
    if inside(workspace, root) or inside(root, workspace):
        raise ValueError('Keep the read-only dataset and writable workspace in separate directories')
    if not partition or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_,-' for c in partition):
        raise ValueError('Invalid Slurm partition name')
    names = {}
    for split in ('train', 'val', 'test'):
        names[split] = [s.strip() for s in (root/('softras_'+split+'.lst')).read_text().splitlines() if s.strip()]
        if not names[split] or len(names[split]) != len(set(names[split])):
            raise ValueError('Empty or duplicated split: '+split)
        if any(Path(n).name != n or '/' in n or '\\' in n or n in ('.','..') for n in names[split]):
            raise ValueError('Unsafe dataset object name')
    if any(set(names[a]) & set(names[b]) for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Train, validation and test splits overlap')
    if len(names['train']) != 4744 or len(names['test']) != 1356 or len(names['val']) < 64:
        raise ValueError('This fixed chair recipe requires4744 training,1356 test and at least64 validation objects')
    val = [names['val'][i*(len(names['val'])-1)//63] for i in range(64)]
    smoke = [names['train'][i*(len(names['train'])-1)//3] for i in range(4)]
    inputs, validation_inputs = [], []
    for obj in smoke:
        for view in (0,6,12,18,23):
            path = root/obj/'image'/('%04d.png'%view)
            inputs.append({'path':str(path),'sha256':sha256(path)})
    for obj in val:
        path = root/obj/'image/0000.png'
        validation_inputs.append({'path':str(path),'sha256':sha256(path)})
    vgg_hash = sha256(vgg)
    if vgg_hash != '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0':
        raise ValueError('Expected the full torchvision vgg16-397923af.pth weight file')
    # Validate all read-only inputs before creating the personal output stage.
    workspace.mkdir(parents=True, exist_ok=True)
    for part in ('logs','reports','stages'):
        if not inside(workspace/part,workspace):
            raise ValueError('Output symlink escapes workspace')
        (workspace/part).mkdir(exist_ok=True)
    stage.mkdir()
    for part in ('src','ablation','configs','smoke','runs'):
        (stage/part).mkdir()
    for directory in ('src','ablation'):
        for path in sorted((repository/directory).rglob('*.py')):
            target = stage/path.relative_to(repository)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    (stage/'run.py').write_text('import sys\nsys.dont_write_bytecode=True\nfrom ablation.launcher import main\nif __name__ == "__main__": main()\n',encoding='utf-8')
    configs = {g:config_for(g) for g in GROUPS}
    validate_configs(configs)
    for group, config in configs.items():
        dump(stage/'configs'/(group+'.json'),config)
    manifest = {'version':config_for('U')['version'],'workspace':str(workspace),'stage':str(stage),
                'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'nmr_chair_root':str(root),'vgg_weights':str(vgg),'vgg_sha256':vgg_hash,
                'train_objects':names['train'],'validation_objects':val,'test_object_count':len(names['test']),
                'smoke_objects':smoke,'smoke_inputs':inputs,'validation_inputs':validation_inputs,
                'split_sha256':{s:sha256(root/('softras_'+s+'.lst')) for s in names},
                'source_hashes':{p.relative_to(stage).as_posix():sha256(p) for p in sorted((stage/'src').rglob('*.py'))},
                'package_hashes':{p.relative_to(stage).as_posix():sha256(p) for p in sorted((stage/'ablation').rglob('*.py'))},
                'config_hashes':{g:stable_hash(c) for g,c in configs.items()},
                'training_scope':'Seven independently initialized chair groups; one paired seed132422; no pretrained task checkpoint.',
                'public_launcher_port':True}
    manifest['package_hashes']['run.py'] = sha256(stage/'run.py')
    dump(stage/'preparation.json',manifest)
    base = ['#!/bin/bash','#SBATCH --partition='+partition,'#SBATCH --nodes=1','#SBATCH --ntasks=1',
            '#SBATCH --gres=gpu:1','#SBATCH --cpus-per-task=4','#SBATCH --mem=24G']
    environment = ['set -eo pipefail','export CUBLAS_WORKSPACE_CONFIG=:4096:8',
                   'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"']
    command = shlex.quote(sys.executable)+' -B -u '+shlex.quote(str(stage/'run.py'))+' --workspace '+shlex.quote(str(workspace))
    smoke_job = base+['#SBATCH --job-name=uopt_smoke','#SBATCH --time=01:00:00',
                      '#SBATCH --output='+str(workspace/'logs/train_smoke_%j.out')]+environment+[command+' --smoke']
    train_job = base+['#SBATCH --job-name=uopt_train','#SBATCH --time=4-00:00:00','#SBATCH --array=0-6%2',
                      '#SBATCH --output='+str(workspace/'logs/train_%A_%a.out')]+environment+[command+' --train']
    (stage/'smoke.sbatch').write_text('\n'.join(smoke_job)+'\n',encoding='utf-8')
    (stage/'train_array.sbatch').write_text('\n'.join(train_job)+'\n',encoding='utf-8')
    print('PREPARED: '+str(stage),flush=True)
    print('NEXT: run the generated smoke.sbatch before train_array.sbatch',flush=True)


def verify(stage):
    prep = json.loads((stage/'preparation.json').read_text(encoding='utf-8'))
    for rel, expected in {**prep['source_hashes'],**prep['package_hashes']}.items():
        if not inside(stage/rel,stage) or sha256(stage/rel) != expected:
            raise RuntimeError('Frozen source changed: '+rel)
    for group, expected in prep['config_hashes'].items():
        cfg = json.loads((stage/'configs'/(group+'.json')).read_text(encoding='utf-8'))
        if stable_hash(cfg) != expected or cfg != config_for(group):
            raise RuntimeError('Frozen config changed: '+group)
    for split,expected in prep['split_sha256'].items():
        if sha256(Path(prep['nmr_chair_root'])/('softras_'+split+'.lst')) != expected:
            raise RuntimeError('Dataset split changed: '+split)
    if sha256(prep['vgg_weights']) != prep['vgg_sha256']:
        raise RuntimeError('Offline VGG changed')
    for item in prep['smoke_inputs']+prep['validation_inputs']:
        if sha256(item['path']) != item['sha256']:
            raise RuntimeError('Representative input changed')
    return prep,sha256(stage/'preparation.json')


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','smoke','train','plot'):
        mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,help='ShapeNet NMR chair category directory03001627')
    parser.add_argument('--vgg-weights',type=Path)
    parser.add_argument('--partition',default='gpu')
    parser.add_argument('--group',choices=list(GROUPS))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.prepare:
        prepare(workspace,args.data_root,args.vgg_weights,args.partition)
        return
    stage = workspace/'stages'/STAGE_NAME
    prep,fingerprint = verify(stage)
    os.environ.pop('SCRATCH',None)
    os.environ['UOPT_VGG_WEIGHTS'] = prep['vgg_weights']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0,str(stage/'src'))
    sys.path.insert(0,str(stage))
    # Require the prepared runner, so modified checkout code cannot bypass the
    # accepted stage fingerprint while writing its checkpoint/metrics directory.
    if Path(__file__).resolve() != (stage/'ablation/launcher.py').resolve():
        raise RuntimeError('Use the prepared workspace/stages/'+STAGE_NAME+'/run.py for execution')
    if args.plot:
        from .plot_losses import plot_all
        plot_all(stage)
        return
    from .runner import smoke,train
    if args.smoke:
        raise SystemExit(smoke(workspace,stage,prep,fingerprint))
    group = args.group
    if group is None:
        index = int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))
        if not 0<=index<len(GROUPS):
            parser.error('Supply --group or a valid Slurm array index')
        group = list(GROUPS)[index]
    raise SystemExit(train(workspace,stage,prep,fingerprint,group))
