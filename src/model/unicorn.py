from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from toolz import valfilter

import numpy as np
from pytorch3d.loss import mesh_laplacian_smoothing as laplacian_smoothing
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, TexturesUV
from pytorch3d.structures import Meshes
import torch
import torch.nn as nn
from torch.nn import functional as F

from .encoder import Encoder
from .field import ProgressiveField
from .generator import ProgressiveGenerator
from .loss import get_loss
from .renderer import Renderer, save_mesh_as_gif
from .tools import create_mlp, init_rotations, convert_3d_to_uv_coordinates, safe_model_state_dict, N_UNITS, N_LAYERS
from .tools import azim_to_rotation_matrix, elev_to_rotation_matrix, roll_to_rotation_matrix, cpu_angle_between
from utils import path_mkdir, use_seed
from utils.image import convert_to_img
from utils.logger import print_warning
from utils.mesh import save_mesh_as_obj, repeat, get_icosphere, normal_consistency, normalize
from utils.metrics import MeshEvaluator, ProxyEvaluator
from utils.pytorch import torch_to


# POSE & SCALE DEFAULT（完全保留官方值）
N_POSES = 6
N_ELEV_AZIM = [1, 6]
SCALE_ELLIPSE = [1, 0.7, 0.7]
PRIOR_TRANSLATION = [0., 0., 2.732]

# NEIGHBOR REC LOSS DEFAULT (previously called swap loss)（完全保留官方值）
MIN_ANGLE = 20
N_VPBINS = 5
MEMSIZE = 1024


class Unicorn(nn.Module):
    name = 'unicorn'

    def __init__(self, img_size, **kwargs):
        super().__init__()
        self.init_kwargs = deepcopy(kwargs)
        self.init_kwargs['img_size'] = img_size

        # ==============================================
        # 局部特征权重：温和线性增长+严格上限
        # ==============================================
        self.local_shape_weight = nn.Parameter(torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        self.local_texture_weight = nn.Parameter(torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        # 启动时机：从config读取（默认30轮）
        self.start_local_feat = self.init_kwargs.get('milestones', {}).get('start_local_feat', 30)
        # 权重上限：从config读取，无则用默认值（提升灵活性）
        self.max_local_shape_weight = self.init_kwargs.get('milestones', {}).get('max_local_shape_weight', 0.005)
        self.max_local_texture_weight = self.init_kwargs.get('milestones', {}).get('max_local_texture_weight', 0.01)
        # 权重增长终止轮次：从config读取（优先scheduler的milestones），无则默认224
        scheduler_milestones = self.init_kwargs.get('training', {}).get('scheduler', {}).get('milestones', [224])
        self.weight_growth_end = scheduler_milestones[0] if len(scheduler_milestones) > 0 else 224
        # 线性增长步长：适配不同类别的训练轮次
        self.shape_weight_step = self.max_local_shape_weight / (self.weight_growth_end - self.start_local_feat) if (self.weight_growth_end - self.start_local_feat) > 0 else 0
        self.texture_weight_step = self.max_local_texture_weight / (self.weight_growth_end - self.start_local_feat) if (self.weight_growth_end - self.start_local_feat) > 0 else 0

        # ==============================================
        # 原有初始化函数调用（顺序不变，完全保留官方逻辑）
        # ==============================================
        self._init_encoder(img_size, **kwargs.get('encoder', {}))
        self._init_meshes(**kwargs.get('mesh', {}))
        self.renderer = Renderer(img_size, **kwargs.get('renderer', {}))
        self._init_rend_predictors(**kwargs.get('rend_predictor', {}))
        self._init_background_model(img_size, **kwargs.get('background', {}))
        self._init_milestones(**kwargs.get('milestones', {}))
        self._init_loss(**kwargs.get('loss', {}))
        
        # 其他原有属性（保持不变）
        self.prop_heads = torch.zeros(self.n_poses)
        self.cur_epoch, self.cur_iter = 0, 0
        self._debug = False

    @property
    def n_features(self):
        return self.encoder.out_ch if self.shared_encoder else self.encoder_sh.out_ch

    @property
    def tx_code_size(self):
        return self.txt_generator.current_code_size

    @property
    def sh_code_size(self):
        return self.deform_field.current_code_size

    def _init_encoder(self, img_size, **kwargs):
        # 完全保留官方独立编码器逻辑（适配config的shared=False）
        self.shared_encoder = kwargs.pop('shared', True)
        if self.shared_encoder:
            self.encoder = Encoder(img_size, **kwargs)
        else:
            self.encoder_sh = Encoder(img_size, **kwargs)
            self.encoder_tx = Encoder(img_size, **kwargs)
            self.encoder_pose = Encoder(img_size, **kwargs)
            if len(self.init_kwargs.get('background', {})) > 0:
                self.encoder_bg = Encoder(img_size, **kwargs)

    def _init_meshes(self, **kwargs):
        # 完全保留官方网格初始化逻辑（适配config的ellipse/scale=0.4）
        kwargs = deepcopy(kwargs)
        mesh_init = kwargs.pop('init', 'sphere')
        scale = kwargs.pop('scale', 1)
        if 'sphere' in mesh_init or 'ellipse' in mesh_init:
            mesh = get_icosphere(3)  # 官方固定3级icosphere，保留
            if 'ellipse' in mesh_init:
                scale = scale * torch.Tensor([SCALE_ELLIPSE])
        else:
            raise NotImplementedError
        self.mesh_src = mesh.scale_verts(scale)
        self.register_buffer('uvs', convert_3d_to_uv_coordinates(self.mesh_src.get_mesh_verts_faces(0)[0])[None])

        self.use_mean_txt = kwargs.pop('use_mean_txt', kwargs.pop('use_mean_text', False))
        dfield_kwargs = kwargs.pop('deform_fields', {})
        tgen_kwargs = kwargs.pop('texture_uv', {})
        assert len(kwargs) == 0

        self.deform_field = ProgressiveField(inp_dim=self.n_features, name='deformation', **dfield_kwargs)
        self.txt_generator = ProgressiveGenerator(inp_dim=self.n_features, **tgen_kwargs)
        
        # 初始化局部特征层：砍掉注意力/固定token，改为轻量线性层
        self._init_local_shape_layers()
        self._init_uv_texture_layers()

    def _init_local_shape_layers(self):
        """重构：局部形状层-轻量线性层，无注意力/固定token（修复错误1）"""
        if self.n_features <= 0:
            return
        # 输入：全局变形后的顶点坐标(3) + 全局图像embedding(n_features)
        self.vert_fc = nn.Sequential(
            nn.Linear(3 + self.n_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # 输出局部顶点位移（仅做微调）
        ).to(self.local_shape_weight.device)
        # 权重初始化：xavier_uniform适配回归任务
        for m in self.vert_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _init_uv_texture_layers(self):
        """重构：局部纹理层-轻量线性层，无注意力/固定token（修复错误3）"""
        if self.n_features <= 0 or not hasattr(self, 'txt_generator'):
            return
        # 输入：全局纹理特征(n_c) + 全局图像embedding(n_features)
        self.uv_fc = nn.Sequential(
            nn.Linear(self.txt_generator.out_ch + self.n_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.txt_generator.out_ch)  # 输出纹理通道微调
        ).to(self.local_texture_weight.device)
        # 权重初始化：xavier_uniform适配回归任务
        for m in self.uv_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _init_rend_predictors(self, **kwargs):
        # 完全保留官方逻辑，适配config的alternate_optim=True
        kwargs = deepcopy(kwargs)
        self.n_poses = kwargs.pop('n_poses', N_POSES)
        n_elev, n_azim = kwargs.pop('n_elev_azim', N_ELEV_AZIM)
        assert self.n_poses == n_elev * n_azim
        
        # 核心：兼容官方alternate_optim=True/整数，适配config的True
        self.alternate_optim = kwargs.pop('alternate_optim', True)
        self.alternate_step = kwargs.pop('alternate_step', self.alternate_optim if isinstance(self.alternate_optim, int) else 2)
        self.pose_step = True  # 初始为姿态优化阶段，和官方一致

        NF, NP = self.n_features, self.n_poses
        NU, NL = kwargs.pop('n_reg_units', N_UNITS), kwargs.pop('n_reg_layers', N_LAYERS)

        # Translation（完全保留官方逻辑）
        self.T_regressors = nn.ModuleList([create_mlp(NF, 3, NU, NL, zero_last_init=True) for _ in range(NP)])
        T_range = kwargs.pop('T_range', 1)
        T_range = [T_range] * 3 if isinstance(T_range, (int, float)) else T_range
        self.register_buffer('T_range', torch.Tensor(T_range))
        self.register_buffer('T_init', torch.Tensor(kwargs.pop('prior_translation', PRIOR_TRANSLATION)))

        # Rotation（完全保留官方逻辑，适配config的固定视角）
        self.rot_regressors = nn.ModuleList([create_mlp(NF, 3, NU, NL, zero_last_init=True) for _ in range(NP)])
        a_range, e_range, r_range = kwargs.pop('azim_range'), kwargs.pop('elev_range'), kwargs.pop('roll_range')
        azim, elev, roll = [(e[1] - e[0]) / n for e, n in zip([a_range, e_range, r_range], [n_azim, n_elev, 1])]
        R_init = init_rotations('uniform', n_elev=n_elev, n_azim=n_azim, elev_range=e_range, azim_range=a_range)
        self.register_buffer('R_range', torch.Tensor([azim * 0.52, elev * 0.52, roll * 0.52]) if self.n_poses > 1 else torch.Tensor([azim * 0.5, elev * 0.5, roll * 0.5]))
        self.register_buffer('R_init', R_init)
        self.azim_range, self.elev_range, self.roll_range = a_range, e_range, r_range

        # Scale（完全保留官方逻辑）
        self.scale_regressor = create_mlp(NF, 3, NU, NL, zero_last_init=True)
        scale_range = kwargs.pop('scale_range', 0.5)
        scale_range = [scale_range] * 3 if isinstance(scale_range, (int, float)) else scale_range
        self.register_buffer('scale_range', torch.Tensor(scale_range))
        self.register_buffer('scale_init', torch.ones(3))

        # Pose probabilities（完全保留官方逻辑）
        if NP > 1:
            self.proba_regressor = create_mlp(NF, NP, NU, NL)

        assert len(kwargs) == 0, kwargs

    @property
    def n_candidates(self):
        return 1 if self.hard_select else self.n_poses

    @property
    def hard_select(self):
        # 完全保留官方交替优化逻辑，适配config的alternate_optim=True
        if isinstance(self.alternate_optim, (int, bool)) and self.alternate_optim and not self._debug:
            return False if (self.training and self.pose_step) else True
        else:
            return False

    def _init_background_model(self, img_size, **kwargs):
        # 完全保留官方逻辑
        if len(kwargs) > 0:
            bkg_kwargs = deepcopy(kwargs)
            self.bkg_generator = ProgressiveGenerator(inp_dim=self.n_features, img_size=img_size, **bkg_kwargs)

    def _init_milestones(self, **kwargs):
        # 完全保留官方里程碑逻辑，优先读取config的start_local_feat
        kwargs = deepcopy(kwargs)
        self.milestones = {
            'constant_txt': kwargs.pop('constant_txt', kwargs.pop('contant_text', 0)),
            'freeze_T_pred': kwargs.pop('freeze_T_predictor', 0),
            'freeze_R_pred': kwargs.pop('freeze_R_predictor', 0),
            'freeze_s_pred': kwargs.pop('freeze_scale_predictor', 0),
            'freeze_shape': kwargs.pop('freeze_shape', 0),
            'mean_txt': kwargs.pop('mean_txt', kwargs.pop('mean_text', self.use_mean_txt)),
        }
        # 强制从config读取，覆盖默认值
        self.start_local_feat = kwargs.pop('start_local_feat', 30)
        assert len(kwargs) == 0

    def _init_loss(self, **kwargs):
        # 完全保留官方损失逻辑，适配config的损失权重（perceptual=10/uniform=0.02等）
        kwargs = deepcopy(kwargs)
        loss_weights = {
            'rgb': kwargs.pop('rgb_weight', 1.0),
            'normal': kwargs.pop('normal_weight', 0),
            'laplacian': kwargs.pop('laplacian_weight', 0),
            'perceptual': kwargs.pop('perceptual_weight', 0),
            'uniform': kwargs.pop('uniform_weight', 0.1),
            'neighbor': kwargs.pop('neighbor_weight', kwargs.pop('swap_weight', 0)),
        }
        name = kwargs.pop('name', 'mse')
        perceptual_kwargs = kwargs.pop('perceptual', {})
        self.nbr_memsize = kwargs.pop('nbr_memsize', kwargs.pop('swap_memsize', MEMSIZE))
        self.nbr_n_vpbins = kwargs.pop('nbr_n_vpbins', kwargs.pop('swap_n_vpbins', N_VPBINS))
        self.nbr_min_angle = kwargs.pop('nbr_min_angle', kwargs.pop('swap_min_angle', MIN_ANGLE))
        self.nbr_memory = {k: torch.empty(0) for k in ['sh', 'tx', 'S', 'R', 'T', 'bg', 'img']}
        kwargs.pop('swap_equal_bins', False)
        assert len(kwargs) == 0, kwargs

        self.loss_weights = valfilter(lambda v: v > 0, loss_weights)
        self.loss_names = [f'loss_{n}' for n in list(self.loss_weights.keys()) + ['total']]
        self.criterion = get_loss(name)(reduction='none')
        if 'perceptual' in self.loss_weights:
            self.perceptual_loss = get_loss('perceptual')(**perceptual_kwargs)

    @property
    def pred_background(self):
        return hasattr(self, 'bkg_generator')

    def is_live(self, name):
        # 完全保留官方逻辑
        milestone = self.milestones[name]
        if isinstance(milestone, bool):
            return milestone
        else:
            return True if self.cur_epoch < milestone else False

    def to(self, device):
        # 完善设备迁移：局部特征层同步迁移
        super().to(device)
        self.mesh_src = self.mesh_src.to(device)
        self.renderer = self.renderer.to(device)
        if hasattr(self, 'vert_fc'):
            self.vert_fc = self.vert_fc.to(device)
        if hasattr(self, 'uv_fc'):
            self.uv_fc = self.uv_fc.to(device)
        # 确保权重参数在指定设备
        self.local_shape_weight = self.local_shape_weight.to(device)
        self.local_texture_weight = self.local_texture_weight.to(device)
        return self

    def forward(self, inp, debug=False):
        self._debug = debug

        imgs, K, B = inp['imgs'], self.n_candidates, len(inp['imgs'])
        perturbed = self.training and np.random.binomial(1, p=0.2)
        average_txt = self.is_live('constant_txt') or (perturbed and self.use_mean_txt and self.is_live('mean_txt'))
        
        # ==============================================
        # 核心：删掉requires_grad动态切换（高危操作）
        # 仅保留官方detach实现交替优化，局部特征全程参与优化
        # ==============================================
        
        # 预测网格/姿态/背景（完全保留官方逻辑）
        meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(imgs, average_txt)
        if isinstance(self.alternate_optim, (int, bool)) and self.alternate_optim:
            if self.pose_step:
                meshes, bkgs = meshes.detach(), bkgs.detach() if self.pred_background else None
            else:
                R, T = R.detach(), T.detach()

        # 渲染与损失计算（完全保留官方逻辑）
        meshes_to_render = repeat(meshes, len(T) // len(meshes))
        fgs, alpha = self.renderer(meshes_to_render, R, T).split([3, 1], dim=1)
        rec = fgs * alpha + (1 - alpha) * bkgs if self.pred_background else fgs
        losses, select_idx = self.compute_losses(meshes, imgs, rec, R, T, average_txt=average_txt)

        # 梯度裁剪：全局+局部分离，局部梯度更严格缩放（避免爆炸，无需冻结）
        if self.training:
            local_params, global_params = [], []
            for name, param in self.named_parameters():
                if 'vert_fc' in name or 'uv_fc' in name:
                    local_params.append(param)
                else:
                    global_params.append(param)
            # 全局梯度裁剪（官方阈值）
            if len(global_params) > 0:
                torch.nn.utils.clip_grad_norm_(global_params, max_norm=1.0)
            # 局部梯度更严格裁剪（适配轻量微调）
            if len(local_params) > 0 and self.cur_epoch >= self.start_local_feat:
                torch.nn.utils.clip_grad_norm_(local_params, max_norm=0.5)

        # 调试逻辑（完全保留官方逻辑）
        if debug:
            out = rec.view(K, B, *rec.shape[1:]) if K > 1 else rec[None]
            self._debug = False
        else:
            rec = rec.view(K, B, *rec.shape[1:])[select_idx, torch.arange(B)] if K > 1 else rec
            out = losses, rec

        return out

    def predict_mesh_pose_bkg(self, imgs, average_txt=False):
        # 完全保留官方独立编码器逻辑（适配config的shared=False）
        if self.shared_encoder:
            features = self.encoder(imgs)
            meshes = self.predict_meshes(features, average_txt=average_txt)
            R, T = self.predict_poses(features)
            bkgs = self.predict_background(features) if self.pred_background else None
        else:
            features_sh, features_tx = self.encoder_sh(imgs), self.encoder_tx(imgs)
            meshes = self.predict_meshes(features_sh, features_tx, average_txt=average_txt)
            R, T = self.predict_poses(self.encoder_pose(imgs))
            bkgs = self.predict_background(self.encoder_bg(imgs)) if self.pred_background else None
        return meshes, (R, T), bkgs

    def predict_meshes(self, features, features_tx=None, average_txt=False):
        # 完全保留官方逻辑
        if features_tx is None:
            features_tx = features
        verts, faces = self.mesh_src.get_mesh_verts_faces(0)
        meshes = self.mesh_src.extend(len(features))
        meshes.offset_verts_(self.predict_disp_verts(verts, features))
        meshes.textures = self.predict_textures(faces, features_tx, average_txt)
        meshes.scale_verts_(self.predict_scales(features))
        return meshes

    def predict_disp_verts(self, verts, features):
        """重构：局部位移基于全局变形后的顶点,无直接image→vertex映射"""
        # 1. 全局位移：官方逻辑+强制维度对齐（保留）
        disp_verts_global = self.deform_field(verts, features)
        B, D = features.shape  # 批次B，特征维度D
        N = verts.shape[0]     # 顶点数N=642
        disp_verts_global = disp_verts_global.view(B, N, 3).reshape(-1, 3)
        if self.is_live('freeze_shape'):
            disp_verts_global = disp_verts_global * 0
        
        # 无局部特征/局部未启动：直接返回全局位移
        if not hasattr(self, 'vert_fc') or self.cur_epoch < self.start_local_feat:
            return disp_verts_global
        
        # 2. 局部位移：基于「全局变形后的顶点」+图像embedding，无直接映射
        # 步骤1：获取全局变形后的顶点坐标
        verts_original = verts.unsqueeze(0).repeat(B, 1, 1)  # [B, N, 3]
        verts_deformed = verts_original + disp_verts_global.view(B, N, 3)  # [B, N, 3]
        # 步骤2：图像embedding扩维，每个顶点共享全局图像特征（无空间直接映射）
        img_feat_expand = features.unsqueeze(1).repeat(1, N, 1)  # [B, N, D]
        # 步骤3：拼接特征，预测局部微位移
        vert_feat = torch.cat([verts_deformed, img_feat_expand], dim=-1).reshape(B*N, 3+D)  # [B*N, 3+D]
        disp_verts_local = self.vert_fc(vert_feat).reshape(-1, 3)  # [B*N, 3]，仅做微调
        
        # 3. 融合：局部权重温和缩放，仅做全局位移的微调
        disp_verts = disp_verts_global + self.local_shape_weight * disp_verts_local
        
        return disp_verts

    def predict_textures(self, faces, features, average_txt=False):
        """重构：局部纹理基于全局纹理特征，通道级微调，保留全局先验"""
        # 1. 全局纹理生成：完全保留官方逻辑（核心，保证category prior）
        B = len(features)
        maps_global = self.txt_generator(features)  # [B, C, H, W]
        C, H, W = maps_global.shape[1], maps_global.shape[2], maps_global.shape[3]
        
        # 平均纹理/局部未启动：直接返回全局纹理，保留官方逻辑
        if average_txt or not hasattr(self, 'uv_fc') or self.cur_epoch < self.start_local_feat:
            if average_txt:
                maps_global = maps_global.mean(dim=[2,3], keepdim=True).expand_as(maps_global)
            return TexturesUV(maps_global.permute(0, 2, 3, 1), faces[None].expand(B, -1, -1), self.uvs.expand(B, -1, -1))
        
        # 2. 局部纹理：通道级微调，无空间幻觉（基于全局纹理特征）
        # 步骤1：图像embedding扩维，和全局纹理空间维度对齐（无直接空间映射）
        img_feat_expand = features.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # [B, D, H, W]
        # 步骤2：拼接全局纹理特征+图像embedding（保留全局纹理的空间结构）
        uv_feat = torch.cat([maps_global, img_feat_expand], dim=1)  # [B, C+D, H, W]
        # 步骤3：通道级微调，不改变空间结构（核心，避免空间幻觉）
        uv_feat = uv_feat.permute(0, 2, 3, 1).reshape(B*H*W, C+self.n_features)  # [B*H*W, C+D]
        uv_local_feat = self.uv_fc(uv_feat).reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 3. 融合：局部权重温和缩放，仅做全局纹理的通道级微调
        maps = maps_global + self.local_texture_weight * uv_local_feat
    
        return TexturesUV(maps.permute(0, 2, 3, 1), faces[None].expand(B, -1, -1), self.uvs.expand(B, -1, -1))

    def predict_scales(self, features):
        # 完全保留官方逻辑
        s_pred = self.scale_regressor(features).tanh()
        if self.is_live('freeze_s_pred'):
            s_pred = s_pred * 0
        self._scales = s_pred * self.scale_range + self.scale_init
        return self._scales

    def predict_poses(self, features):
        # 完全保留官方姿态预测逻辑
        B = len(features)

        T_pred = torch.stack([p(features) for p in self.T_regressors], dim=0).tanh()
        if self.is_live('freeze_T_pred'):
            T_pred = T_pred * 0
        T = (T_pred * self.T_range + self.T_init).view(-1, 3)

        R_pred = torch.stack([p(features) for p in self.rot_regressors], dim=0)
        R_pred = R_pred.tanh()[..., [1, 0, 2]]
        if self.is_live('freeze_R_pred'):
            R_pred = R_pred * 0
        R_pred = (R_pred * self.R_range + self.R_init[:, None]).view(-1, 3)
        azim, elev, roll = map(lambda t: t.squeeze(1), R_pred.split([1, 1, 1], 1))
        R = azim_to_rotation_matrix(azim) @ elev_to_rotation_matrix(elev) @ roll_to_rotation_matrix(roll)

        if self.n_poses > 1:
            weights = self.proba_regressor(features.view(B, -1)).permute(1, 0)
            self._pose_proba = torch.softmax(weights, dim=0)
            if self.hard_select:
                indices = self._pose_proba.max(0)[1]
                select_fn = lambda t: t.view(self.n_poses, B, *t.shape[1:])[indices, torch.arange(B)]
                R, T = map(select_fn, [R, T])
        return R, T

    def predict_background(self, features):
        # 完全保留官方逻辑
        res = self.bkg_generator(features)
        return res.repeat(self.n_candidates, 1, 1, 1) if self.n_candidates > 1 else res

    def compute_losses(self, meshes, imgs, rec, R, T, average_txt=False):
        # 完全保留官方损失计算逻辑，修复姿态均匀损失的索引判断
        K, B = self.n_candidates, len(imgs)
        if K > 1:
            imgs = imgs.repeat(K, 1, 1, 1)
        losses = {k: torch.tensor(0.0, device=imgs.device) for k in self.loss_weights}
        if self.training:
            update_3d, update_pose = (not self.pose_step, self.pose_step) if isinstance(self.alternate_optim, (int, bool)) and self.alternate_optim else (True, True)
        else:
            update_3d, update_pose = (False, False)

        # RGB损失（官方逻辑）
        if 'rgb' in losses:
            losses['rgb'] = self.loss_weights['rgb'] * self.criterion(rec, imgs).flatten(1).mean(1)

        # 感知损失（适配config的perceptual_weight=10）
        if 'perceptual' in losses:
            losses['perceptual'] = self.loss_weights['perceptual'] * self.perceptual_loss(rec, imgs)

        # 网格正则化（官方逻辑，适配config的normal/laplacian权重）
        if update_3d:
            if 'normal' in losses:
                losses['normal'] = self.loss_weights['normal'] * normal_consistency(meshes)
            if 'laplacian' in losses:
                losses['laplacian'] = self.loss_weights['laplacian'] * laplacian_smoothing(meshes, method='uniform')

        # 邻居损失（官方逻辑，适配config的swap_weight=1）
        if update_3d and 'neighbor' in losses and (self.tx_code_size > 0 and self.sh_code_size > 0):
            B, dev = len(meshes), imgs.device
            verts, faces, textures = meshes.verts_padded(), meshes.faces_padded(), meshes.textures
            scales = self._scales[:, None]
            z_sh, z_tx = [m._latent for m in [self.deform_field, self.txt_generator]]
            z_bg = self.bkg_generator._latent if self.pred_background else torch.empty(B, 1, device=dev)
            for n, t in [('sh', z_sh), ('tx', z_tx), ('bg', z_bg), ('S', scales), ('R', R), ('T', T), ('img', imgs)]:
                self.nbr_memory[n] = torch.cat([self.nbr_memory[n].to(dev), t.detach()])[-self.nbr_memsize:]

            min_angle, nb_vpbins = self.nbr_min_angle, self.nbr_n_vpbins
            with torch.no_grad():
                sim_sh = (z_sh[None] - self.nbr_memory['sh'][:, None]).pow(2).sum(-1)
                sim_tx = (z_tx[None] - self.nbr_memory['tx'][:, None]).pow(2).sum(-1)
                angles = cpu_angle_between(self.nbr_memory['R'][:, None], R[None]).view(sim_sh.shape)
                angle_bins = torch.randint(0, nb_vpbins, (B,), device=dev).float()
                bin_size = (180. - min_angle) / nb_vpbins
                min_angles, max_angles = [(angle_bins + k) * bin_size + min_angle for k in range(2)]
                invalid_mask = (angles < min_angles).float() + (angles >= max_angles).float()
                idx_sh, idx_tx = map(lambda t: (t + t.max() * invalid_mask).argmin(0), [sim_sh, sim_tx])

            v_src, f_src = self.mesh_src.get_mesh_verts_faces(0)
            nbr_list, select = [], lambda n, indices: self.nbr_memory[n][indices]
            sh_imgs, tx_imgs = select('img', idx_sh), select('img', idx_tx)

            with torch.no_grad():
                if self.shared_encoder:
                    sh_features = self.encoder(sh_imgs)
                    sh_tx = self.predict_textures(f_src, sh_features, average_txt)
                    sh_S = self.predict_scales(sh_features)[:, None]
                    sh_R, sh_T = self.predict_poses(sh_features)
                    sh_bg = self.predict_background(sh_features) if self.pred_background else None
                else:
                    sh_tx = self.predict_textures(f_src, self.encoder_tx(sh_imgs), average_txt)
                    sh_S = self.predict_scales(self.encoder_sh(sh_imgs))[:, None]
                    sh_R, sh_T = self.predict_poses(self.encoder_pose(sh_imgs))
                    sh_bg = self.predict_background(self.encoder_bg(sh_imgs)) if self.pred_background else None
            sh_mesh = Meshes((verts / scales) * sh_S, faces, sh_tx)
            nbr_list.append([sh_mesh, sh_R, sh_T, sh_bg, sh_imgs])

            with torch.no_grad():
                if self.shared_encoder:
                    tx_features = self.encoder(tx_imgs)
                    tx_verts = v_src + self.predict_disp_verts(v_src, tx_features).view(B, -1, 3)
                    tx_S = self.predict_scales(tx_features)[:, None]
                    tx_R, tx_T = self.predict_poses(tx_features)
                    tx_bg = self.predict_background(tx_features) if self.pred_background else None
                else:
                    tx_feat_sh = self.encoder_sh(tx_imgs)
                    tx_verts = v_src + self.predict_disp_verts(v_src, tx_feat_sh).view(B, -1, 3)
                    tx_S = self.predict_scales(tx_feat_sh)[:, None]
                    tx_R, tx_T = self.predict_poses(self.encoder_pose(tx_imgs))
                    tx_bg = self.predict_background(self.encoder_bg(tx_imgs)) if self.pred_background else None
            tx_mesh = Meshes(tx_verts * tx_S, faces, textures)
            nbr_list.append([tx_mesh, tx_R, tx_T, tx_bg, tx_imgs])

            loss = 0.
            for nbr_inp in nbr_list:
                nbr_mesh, R, T, bkgs, imgs = nbr_inp
                rec_sw, alpha_sw = self.renderer(nbr_mesh, R, T).split([3, 1], dim=1)
                rec_sw = rec_sw * alpha_sw + (1 - alpha_sw) * bkgs[:B] if bkgs is not None else rec_sw
                if 'rgb' in losses:
                    loss += self.loss_weights['rgb'] * self.criterion(rec_sw, imgs).flatten(1).mean(1)
                if 'perceptual' in losses:
                    loss += self.loss_weights['perceptual'] * self.perceptual_loss(rec_sw, imgs)
            losses['neighbor'] = self.loss_weights['neighbor'] * loss

        # 姿态均匀性损失（修复：判断K>1，避免n_poses=1时的索引错误）
        if update_pose and 'uniform' in losses and K > 1:
            losses['uniform'] = self.loss_weights['uniform'] * (self._pose_proba.mean(1) - 1 / K).abs().mean()

        # 总损失计算（官方逻辑）
        dist = sum(losses.values())
        if K > 1:
            dist, select_idx = dist.view(K, B), self._pose_proba.max(0)[1]
            dist = (self._pose_proba * dist).sum(0)
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = (self._pose_proba * v.view(K, B)).sum(0).mean()

            # 姿态分布监控（官方逻辑）
            pose_proba_d = self._pose_proba.detach().cpu()
            self._prob_heads = pose_proba_d.mean(1).tolist()
            self._prob_max = pose_proba_d.max(0)[0].mean().item()
            self._prob_min = pose_proba_d.min(0)[0].mean().item()
            count = torch.zeros(K, B).scatter(0, select_idx[None].cpu(), 1).sum(1)
            self.prop_heads = count / B

        else:
            select_idx = torch.zeros(B).long()
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = v.mean()

        losses['total'] = dist.mean()
        return losses, select_idx

    def iter_step(self):
        # 完全保留官方交替优化切换逻辑，适配config的alternate_optim=True/step=2
        self.cur_iter += 1
        if isinstance(self.alternate_optim, (int, bool)) and self.alternate_optim:
            step = self.alternate_optim if isinstance(self.alternate_optim, int) else self.alternate_step
            if self.cur_iter % step == 0:
                self.pose_step = not self.pose_step

    def step(self):
        # 官方渐进式模块更新（保留）+ 局部特征权重温和线性增长
        self.cur_epoch += 1
        self.deform_field.step()
        self.txt_generator.step()
        if self.pred_background:
            self.bkg_generator.step()
        
        # 局部特征权重：启动后线性增长，不超过最大值（适配不同类别的终止轮次）
        if self.cur_epoch >= self.start_local_feat and self.cur_epoch < self.weight_growth_end:
            # 形状权重线性增长
            new_shape_w = self.local_shape_weight.data + self.shape_weight_step
            self.local_shape_weight.data = torch.clamp(new_shape_w, 0, self.max_local_shape_weight)
            # 纹理权重线性增长
            new_tex_w = self.local_texture_weight.data + self.texture_weight_step
            self.local_texture_weight.data = torch.clamp(new_tex_w, 0, self.max_local_texture_weight)
        # 终止轮次后固定最大值，不再增长
        elif self.cur_epoch >= self.weight_growth_end:
            self.local_shape_weight.data = torch.clamp(self.local_shape_weight.data, 0, self.max_local_shape_weight)
            self.local_texture_weight.data = torch.clamp(self.local_texture_weight.data, 0, self.max_local_texture_weight)
    
    def set_cur_epoch(self, epoch):
        # 完全保留官方逻辑，适配断点续训+局部权重同步
        self.cur_epoch = epoch
        self.deform_field.set_cur_milestone(epoch)
        self.txt_generator.set_cur_milestone(epoch)
        if self.pred_background:
            self.bkg_generator.set_cur_milestone(epoch)
        # 断点续训时同步局部权重（修复硬编码224问题）
        if self.cur_epoch >= self.start_local_feat:
            # 计算当前应有的权重值（使用self.weight_growth_end，适配所有类别）
            epoch_offset = min(self.cur_epoch - self.start_local_feat, self.weight_growth_end - self.start_local_feat)
            self.local_shape_weight.data = torch.tensor(min(epoch_offset * self.shape_weight_step, self.max_local_shape_weight), 
                                                    device=self.local_shape_weight.device)
            self.local_texture_weight.data = torch.tensor(min(epoch_offset * self.texture_weight_step, self.max_local_texture_weight),
                                                        device=self.local_texture_weight.device)

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        # 完全保留官方逻辑，兼容加载官方预训练模型（跳过局部特征参数）
        unloaded_params = []
        state = self.state_dict()
        for name, param in safe_model_state_dict(state_dict).items():
            if name in state and name != 'T_init':
                try:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                except RuntimeError:
                    print_warning(f'Error load_state_dict param={name}: {list(param.shape)}, {list(state[name].shape)}')
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found (含局部特征参数，属正常现象)')

    ########################
    # Visualization utils（完全保留官方逻辑，无修改）
    ########################
    def get_synthetic_textures(self, colored=False):
        verts = self.mesh_src.verts_packed()
        if colored:
            colors = (verts - verts.min(0)[0]) / (verts.max(0)[0] - verts.min(0)[0])
        else:
            colors = torch.ones(verts.shape, device=verts.device) * 0.8
        return TexturesVertex(verts_features=colors[None])

    def get_prototype(self):
        verts = self.mesh_src.get_mesh_verts_faces(0)[0]
        latent = torch.zeros(1, self.n_features, device=verts.device)
        meshes = self.mesh_src.offset_verts(self.deform_field(verts, latent).view(-1, 3))
        return meshes

    @use_seed()
    @torch.no_grad()
    def get_random_prototype_views(self, N=10):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        mesh.textures = self.get_synthetic_textures(colored=True)
        azim = torch.randint(*self.azim_range, size=(N,))
        elev = torch.randint(*self.elev_range, size=(N,)) if np.diff(self.elev_range)[0] > 0 else self.elev_range[0]
        R, T = look_at_view_transform(dist=self.T_init[-1], elev=elev, azim=azim, device=mesh.device)
        return self.renderer(mesh.extend(N), R, T).split([3, 1], dim=1)[0]

    @torch.no_grad()
    def save_prototype(self, path=None):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        path = path_mkdir(path or Path('.'))
        d, elev = self.T_init[-1], np.mean(self.elev_range)
        mesh.textures = self.get_synthetic_textures()
        save_mesh_as_obj(mesh, path / 'proto.obj')
        save_mesh_as_gif(mesh, path / 'proto_li.gif', dist=d, elev=elev, renderer=self.renderer, eye_light=True)
        mesh.textures = self.get_synthetic_textures(colored=True)
        save_mesh_as_gif(mesh, path / 'proto_uv.gif', dist=d, elev=elev, renderer=self.renderer)

    ########################
    # Evaluation utils（完全保留官方逻辑，无修改，确保指标和官方一致）
    ########################
    @torch.no_grad()
    def quantitative_eval(self, loader, device, evaluator=None):
        self.eval()
        if loader.dataset.name in ['cub_200']:
            if evaluator is None:
                evaluator = ProxyEvaluator()
            for inp, _ in loader:
                mask_gt = inp['masks'].to(device)
                meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(inp['imgs'].to(device))
                mask_pred = self.renderer(meshes, R, T, viz_purpose=True).split([3, 1], dim=1)[1]
                if mask_pred.shape != mask_gt.shape:
                    mask_pred = F.interpolate(mask_pred, mask_gt.shape[-2:], mode='bilinear', align_corners=False)
                mask_pred = (mask_pred > 0.5).float()
                evaluator.update(mask_pred, mask_gt)

        else:
            if loader.dataset.name == 'p3d_car':
                print_warning('make sure that the canonical axes of predicted shapes correspond to the GT shapes axes')
            if evaluator is None:
                evaluator = MeshEvaluator()
            for inp, labels in loader:
                if isinstance(labels, torch.Tensor) and torch.all(labels == -1):
                    break

                meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(inp['imgs'].to(device))
                if not torch.all(inp['poses'] == -1):
                    verts, faces = meshes.verts_padded(), meshes.faces_padded()
                    R_gt, T_gt = map(lambda t: t.squeeze(2), inp['poses'].to(device).split([3, 1], dim=2))
                    verts = (verts @ R + T[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
                    meshes = Meshes(verts=verts, faces=faces, textures=meshes.textures)
                evaluator.update(meshes, torch_to(labels, device))
        return OrderedDict(zip(evaluator.metrics.names, evaluator.metrics.values))

    @torch.no_grad()
    def qualitative_eval(self, loader, device, path=None, N=32):
        path = path or Path('.')
        self.eval()
        self.save_prototype(path / 'model')

        renderer = self.renderer
        n_zeros, NI = int(np.log10(N - 1)) + 1, max(N // loader.batch_size, 1)
        for j, (inp, _) in enumerate(loader):
            if j == NI:
                break
            imgs = inp['imgs'].to(device)
            meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(imgs)
            rec, alpha = renderer(meshes, R, T).split([3, 1], dim=1)
            if bkgs is not None:
                rec = rec * alpha + (1 - alpha) * bkgs

            B, NV = len(imgs), 50
            d, e = self.T_init[-1], np.mean(self.elev_range)
            for k in range(B):
                i = str(j*B+k).zfill(n_zeros)
                convert_to_img(imgs[k]).save(path / f'{i}_inpraw.png')
                convert_to_img(rec[k]).save(path / f'{i}_inprec_full.png')
                if self.pred_background:
                    convert_to_img(bkgs[k]).save(path / f'{i}_inprec_wbkg.png')

                mcenter = normalize(meshes[k])
                save_mesh_as_gif(mcenter, path / f'{i}_meshabs.gif', n_views=NV, dist=d, elev=e, renderer=renderer)
                save_mesh_as_obj(mcenter, path / f'{i}_mesh.obj')
                mcenter.textures = self.get_synthetic_textures(colored=True)
                save_mesh_as_obj(mcenter, path / f'{i}_meshuv.obj')
                save_mesh_as_gif(mcenter, path / f'{i}_meshuv_raw.gif', dist=d, elev=e, renderer=renderer)