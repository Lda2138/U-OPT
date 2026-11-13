"""Frozen configuration and pure helpers; no ML imports."""
from copy import deepcopy
import hashlib
import json
import math

VERSION = 'chair_gts_20261030_v3'
STAGE_NAME = 'chair_ablation_v1'
GROUPS = {
    'U': (False, False, False), 'G': (True, False, False),
    'T': (False, True, False), 'GT': (True, True, False),
    'GS': (True, False, True), 'TS': (False, True, True),
    'GTS': (True, True, True),
}
ALPHAS = ('local_shape_weight', 'local_texture_weight')
BASE = {
    'version': VERSION, 'seed': 132422, 'resume': None,
    'geometry': False, 'texture': False, 'schedule': False,
    'residual_start_index': 30, 'residual_end_index': 224,
    'shape_alpha_max': 0.005, 'texture_alpha_max': 0.01,
    'index_definition': 'Zero-based epoch index = completed full training epochs before current epoch.',
    'dataset': {'category': '03001627', 'split': 'softras_train.lst',
                'views': 24, 'batch_size': 32, 'workers': 2,
                'validation_split': 'softras_val.lst', 'validation_objects': 64, 'validation_view': 0},
    'model': {
        'img_size': [64,64], 'encoder': {'name': 'resnet18', 'shared': False, 'pretrained': False},
        'mesh': {'init': 'ellipse', 'scale': 0.4, 'use_mean_txt': True,
                 'deform_fields': {'powers':[2,8,64], 'milestones':[14,84,154], 'n_field_units':512},
                 'texture_uv': {'img_size':64, 'min_features':64, 'powers':[2,8,64,512], 'milestones':[0,14,84,154]}},
        'renderer': {'cameras': {'name':'perspective','focal_length':3.732050895690918}, 'faces_per_pixel':50},
        'rend_predictor': {'alternate_optim': True, 'azim_range':[0,360], 'elev_range':[30,30],
                           'roll_range':[0,0], 'T_range':[0.2,0.2,0], 'scale_range':[0.5,0.5,0.5]},
        'milestones': {'freeze_scale_predictor':3,'constant_txt':14},
        'loss': {'name':'mse','rgb_weight':1,'perceptual_weight':10,'laplacian_weight':0.01,
                 'normal_weight':0.01,'uniform_weight':0.02,'swap_weight':1,'swap_memsize':1024,'swap_n_vpbins':5},
    },
    'training': {'epochs':238, 'base_lr':1e-4, 'residual_lr':2.5e-5,
                 'betas':[0.9,0.999], 'eps':1e-8, 'weight_decay':0,
                 'lr_milestones':[224], 'lr_gamma':0.2,
                 'clip_common':1.0, 'clip_residual':0.5,
                 'checkpoint_updates':2000, 'validation_updates':4000,
                 'selection':'Fixed last epoch checkpoint for comparisons; validation monitors only.',
                 'loss_logging':'Every optimizer update; scalar losses already multiplied by inherited weights.',
                 'zero_grad_set_to_none': True},
}


def config_for(group):
    if group not in GROUPS:
        raise ValueError('Unknown ablation group: '+str(group))
    config=deepcopy(BASE)
    config.update(zip(('geometry','texture','schedule'),GROUPS[group]))
    config['group']=group
    return config


def alpha_at(config, epoch_index):
    if not isinstance(epoch_index,int) or epoch_index < 0:
        raise ValueError('Epoch index must be a nonnegative integer')
    start,end=config['residual_start_index'],config['residual_end_index']
    if not (0 <= start < end):
        raise ValueError('Residual schedule endpoints must be explicit and ordered')
    fraction=min(1.0,max(0.0,(epoch_index-start)/(end-start))) if config['schedule'] else 1.0
    return (config['shape_alpha_max']*fraction if config['geometry'] else 0.0,
            config['texture_alpha_max']*fraction if config['texture'] else 0.0)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def batches(count, batch_size, start=0):
    total=(count+batch_size-1)//batch_size
    if count<=0 or batch_size<=0 or not 0<=start<=total:
        raise ValueError('Invalid batch cursor')
    return total


def validate_configs(configs):
    assert list(configs)==list(GROUPS)
    common=[]
    for name,config in configs.items():
        assert config==config_for(name)
        same=deepcopy(config)
        for key in ('group','geometry','texture','schedule'):
            same.pop(key)
        common.append(stable_hash(same))
    assert len(set(common))==1


LOSS_NAMES=('rgb','perceptual','normal','laplacian','uniform','neighbor','total')
