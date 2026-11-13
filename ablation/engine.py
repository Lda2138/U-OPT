"""Model adapter, optimizer, loss logging and complete restart state."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from pytorch3d.renderer import TexturesUV
from model.unicorn import Unicorn
from utils.mesh import repeat

from .protocol import ALPHAS, LOSS_NAMES, alpha_at, stable_hash


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def tensor_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str((str(a.dtype),tuple(a.shape))).encode())
        digest.update(a.numpy().tobytes())
    return digest.hexdigest()


class AblationUnicorn(Unicorn):
    def __init__(self,config):
        self.ablation_config=deepcopy(config)
        super().__init__(**deepcopy(config['model']))
        # The fusion coefficients are deterministic state, never Adam parameters.
        for name in ALPHAS:
            value=getattr(self,name).detach().clone()
            delattr(self,name)
            self.register_buffer(name,value)
        # The adapter owns the gate; inherited residual code is not used.
        self.start_local_feat=0
        self.diagnostics={}
        self.sync_epoch(0)

    def _init_local_shape_layers(self):
        if self.ablation_config['geometry']:
            devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.ablation_config['seed']+101)
                super()._init_local_shape_layers()

    def _init_uv_texture_layers(self):
        if self.ablation_config['texture']:
            devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.ablation_config['seed']+202)
                super()._init_uv_texture_layers()

    def sync_epoch(self,index):
        shape,texture=alpha_at(self.ablation_config,index)
        self.cur_epoch=index
        self.local_shape_weight.fill_(shape);self.local_texture_weight.fill_(texture)
        modules=[(self.deform_field,self.ablation_config['model']['mesh']['deform_fields']['milestones']),
                 (self.txt_generator,self.ablation_config['model']['mesh']['texture_uv']['milestones'])]
        for module,milestones in modules:
            module.cur_milestone=index
            module.set_milestones(list(milestones))

    def set_cur_epoch(self,index):
        self.sync_epoch(int(index))

    def step(self):
        self.sync_epoch(self.cur_epoch+1)

    def predict_disp_verts(self,verts,features):
        global_disp=self.deform_field(verts,features).reshape(-1,3)
        if self.is_live('freeze_shape'):
            global_disp=global_disp*0
        if not hasattr(self,'vert_fc') or self.local_shape_weight.item()==0:
            return global_disp
        B,D=features.shape;N=len(verts)
        deformed=verts[None].expand(B,-1,-1)+global_disp.view(B,N,3)
        expanded=features[:,None].expand(-1,N,-1)
        residual=self.vert_fc(torch.cat([deformed,expanded],-1).reshape(B*N,3+D))
        effective=self.local_shape_weight*residual
        if torch.is_grad_enabled():
            self.diagnostics['geometry_raw_rms']=residual.detach().square().mean().sqrt().item()
            self.diagnostics['geometry_effective_rms']=effective.detach().square().mean().sqrt().item()
        return global_disp+effective

    def predict_textures(self,faces,features,average_txt=False):
        maps=self.txt_generator(features);B,C,H,W=maps.shape
        if average_txt:
            maps=maps.mean(dim=(2,3),keepdim=True).expand_as(maps)
        elif hasattr(self,'uv_fc') and self.local_texture_weight.item()!=0:
            expanded=features[:,:,None,None].expand(-1,-1,H,W)
            inp=torch.cat([maps,expanded],1).permute(0,2,3,1).reshape(B*H*W,C+self.n_features)
            residual=self.uv_fc(inp).view(B,H,W,C).permute(0,3,1,2)
            effective=self.local_texture_weight*residual
            maps=maps+effective
            if torch.is_grad_enabled():
                self.diagnostics['texture_raw_rms']=residual.detach().square().mean().sqrt().item()
                self.diagnostics['texture_effective_rms']=effective.detach().square().mean().sqrt().item()
        return TexturesUV(maps.permute(0,2,3,1),faces[None].expand(B,-1,-1),self.uvs.expand(B,-1,-1))

    def forward(self,inp,debug=False):
        # Inherited rendering, losses and pose selection; no clipping in forward.
        self._debug=debug;self.diagnostics={}
        imgs=inp['imgs'];K,B=self.n_candidates,len(imgs)
        perturbed=self.training and np.random.binomial(1,p=0.2)
        average=self.is_live('constant_txt') or (perturbed and self.use_mean_txt and self.is_live('mean_txt'))
        self.diagnostics['average_texture']=bool(average)
        mesh,(R,T),bkg=self.predict_mesh_pose_bkg(imgs,average)
        if self.alternate_optim:
            if self.pose_step:
                mesh=mesh.detach();bkg=bkg.detach() if self.pred_background else None
            else:
                R,T=R.detach(),T.detach()
        fgs,alpha=self.renderer(repeat(mesh,len(T)//len(mesh)),R,T).split([3,1],dim=1)
        reconstruction=fgs*alpha+(1-alpha)*bkg if self.pred_background else fgs
        losses,indices=self.compute_losses(mesh,imgs,reconstruction,R,T,average_txt=average)
        if not self.pred_background and self.nbr_memory['bg'].numel():
            # This unused memory was created with torch.empty by the inherited loss.
            self.nbr_memory['bg'].zero_()
        if debug:
            self._debug=False
            return reconstruction.view(K,B,*reconstruction.shape[1:]) if K>1 else reconstruction[None]
        return losses,(reconstruction.view(K,B,*reconstruction.shape[1:])[indices,torch.arange(B)] if K>1 else reconstruction)


def make_model(config,device):
    seed_all(config['seed'])
    return AblationUnicorn(config).to(device)


def optimizer_for(model,config):
    common,residual=[],[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            (residual if name.startswith(('vert_fc.','uv_fc.')) else common).append((name,p))
    t=config['training'];groups=[];manifest=[]
    for label,pairs,lr,clip in [('common',common,t['base_lr'],t['clip_common']),
                                ('residual',residual,t['residual_lr'],t['clip_residual'])]:
        if pairs:
            groups.append({'params':[p for _,p in pairs],'lr':lr,'group_name':label,'clip_norm':clip})
            manifest.append({'group':label,'lr':lr,'clip_norm':clip,'parameter_names':[n for n,_ in pairs],
                             'parameter_count':sum(p.numel() for _,p in pairs)})
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==sum(p.requires_grad for p in model.parameters())
    assert not any(n in ALPHAS for g in manifest for n in g['parameter_names'])
    optimizer=torch.optim.Adam(groups,betas=tuple(t['betas']),eps=t['eps'],weight_decay=t['weight_decay'])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=t['lr_milestones'],gamma=t['lr_gamma'])
    return optimizer,scheduler,manifest


def gradnorm(params):
    grads=[p.grad.detach().float().norm().item() for p in params if p.grad is not None]
    return math.sqrt(sum(v*v for v in grads))


def update(model,optimizer,images):
    model.train();optimizer.zero_grad(set_to_none=True)
    started=time.perf_counter();stage='pose' if model.pose_step else '3d'
    losses,_=model({'imgs':images})
    if not torch.isfinite(losses['total']).all():
        raise RuntimeError('Non-finite loss')
    losses['total'].mean().backward()
    named=dict(model.named_parameters())
    result={'phase':stage,'epoch_index':model.cur_epoch,
            'alpha_shape':model.local_shape_weight.item(),'alpha_texture':model.local_texture_weight.item()}
    for label in ('common','residual'):
        for suffix in ('grad_before_clip','grad_after_clip','lr'):result[label+'_'+suffix]=0.0
    for name in LOSS_NAMES:
        value=losses.get(name)
        result['loss_'+name]=0.0 if value is None else value.detach().mean().item()
    for g in optimizer.param_groups:
        grads=[p for p in g['params'] if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for p in grads):
            raise RuntimeError('Non-finite parameter gradient')
        label=g['group_name']
        result[label+'_grad_before_clip']=gradnorm(g['params'])
        torch.nn.utils.clip_grad_norm_(g['params'],g['clip_norm'])
        result[label+'_grad_after_clip']=gradnorm(g['params'])
        result[label+'_lr']=g['lr']
    for label,prefix in [('geometry','vert_fc.'),('texture','uv_fc.'),('pose','encoder_pose.')]:
        ps=[p for n,p in named.items() if n.startswith(prefix)]
        result[label+'_grad_norm']=gradnorm(ps)
        result[label+'_gradient_entries']=sum(p.grad is not None for p in ps)
    for key in ('geometry_raw_rms','geometry_effective_rms','texture_raw_rms','texture_effective_rms'):
        result[key]=model.diagnostics.get(key,0.0)
    result['average_texture']=model.diagnostics.get('average_texture',False)
    optimizer.step();model.iter_step()
    if torch.cuda.is_available():torch.cuda.synchronize()
    result['update_seconds']=time.perf_counter()-started
    result['shape_code_size']=int(model.sh_code_size);result['texture_code_size']=int(model.tx_code_size)
    result['neighbor_memory_size']=int(len(model.nbr_memory['img']))
    return result


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def atomic_checkpoint(path,model,optimizer,scheduler,config,cursor,fingerprint,counters):
    path=Path(path);temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    state={'format':'uopt_ablation_v1','config':deepcopy(config),'config_hash':stable_hash(config),
           'fingerprint':fingerprint,'cursor':deepcopy(cursor),'counters':deepcopy(counters),
           'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
           'scheduler_state':scheduler.state_dict(),'cur_epoch':model.cur_epoch,'cur_iter':model.cur_iter,
           'pose_step':model.pose_step,'neighbor_memory':{k:v.detach().cpu() for k,v in model.nbr_memory.items()},
           'rng':rng_state()}
    with temporary.open('xb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(str(temporary),str(path))


def load_checkpoint(path,model,optimizer,scheduler,config,fingerprint):
    try:state=torch.load(path,map_location='cpu',weights_only=False)
    except TypeError:state=torch.load(path,map_location='cpu')
    if state.get('format')!='uopt_ablation_v1' or state['config']!=config or state['config_hash']!=stable_hash(config) or state['fingerprint']!=fingerprint:
        raise RuntimeError('Checkpoint/config/source fingerprint mismatch; do not load historical model.pkl for these groups')
    nn.Module.load_state_dict(model,state['model_state'],strict=True)
    saved_alpha=[getattr(model,n).clone() for n in ALPHAS]
    model.sync_epoch(state['cur_epoch'])
    if not all(torch.equal(a,getattr(model,n)) for a,n in zip(saved_alpha,ALPHAS)):
        raise RuntimeError('Saved alpha differs from frozen schedule')
    optimizer.load_state_dict(state['optimizer_state']);scheduler.load_state_dict(state['scheduler_state'])
    model.cur_iter=int(state['cur_iter']);model.pose_step=bool(state['pose_step'])
    device=next(model.parameters()).device
    model.nbr_memory={k:v.to(device) for k,v in state['neighbor_memory'].items()}
    set_rng(state['rng'])
    return state['cursor'],state['counters']


class ChairImages(torch.utils.data.Dataset):
    def __init__(self,root,objects,views=24):
        self.root=Path(root);self.objects=list(objects);self.views=views

    def __len__(self):return len(self.objects)*self.views

    def __getitem__(self,index):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        obj=self.objects[index%len(self.objects)];view=index//len(self.objects) if self.views>1 else 0
        with Image.open(self.root/obj/'image'/('%04d.png'%view)) as image:
            if image.size!=(64,64):raise RuntimeError('Expected 64x64 training input')
            return to_tensor(image.convert('RGB'))


def epoch_loader(dataset,config,epoch_index,start_batch=0):
    generator=torch.Generator().manual_seed(config['seed']+1000003*epoch_index)
    permutation=torch.randperm(len(dataset),generator=generator).tolist()
    bs=config['dataset']['batch_size']
    sampler=[permutation[i:i+bs] for i in range(0,len(dataset),bs)][start_batch:]
    worker_generator=torch.Generator().manual_seed(config['seed']+epoch_index+7001)
    return torch.utils.data.DataLoader(dataset,batch_sampler=sampler,num_workers=config['dataset']['workers'],
                                      pin_memory=True,generator=worker_generator)


def validation_mse(model,root,objects):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    saved_rng=rng_state();was_training=model.training
    model.eval();values=[]
    try:
        with torch.no_grad():
            for obj in objects:
                with Image.open(Path(root)/obj/'image/0000.png') as im:
                    images=to_tensor(im.convert('RGB')).unsqueeze(0).to(next(model.parameters()).device)
                meshes,(R,T),_=model.predict_mesh_pose_bkg(images)
                rgb=model.renderer(meshes,R,T)[:,:3]
                values.append((rgb-images).square().mean().item())
    finally:
        model.train(was_training);set_rng(saved_rng)
    return float(np.mean(values))
