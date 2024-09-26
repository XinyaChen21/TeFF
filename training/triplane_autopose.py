# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Model for PoF3D

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.networks_stylegan2 import BackgroundGenerator
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler, depth2pts_outside
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from camera_utils import LookAtPose
import dnnlib
import math
import torch.nn.functional as F
from einops import repeat, rearrange
from unet.pytorch_DPCN import PhaseCorr, Corr2Softmax
from training.loss_pose import SolveRS

@persistence.persistent_class
class TriPlaneGeneratorPose(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        device,
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        bg_2d_kwargs = None,
        gen_cond = False,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer(wrong=rendering_kwargs['wrong'])
        self.ray_sampler = RaySampler()
        self.ray_marcher = MipRayMarcher2()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3*2, mapping_kwargs=mapping_kwargs, gen_cond=gen_cond, **synthesis_kwargs)
        if rendering_kwargs['superresolution_module'] is None:
            self.superresolution = None
        else:
            self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.dino_decoder = OSGDinoDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': rendering_kwargs['dino_channals']})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self.clamp_h = rendering_kwargs.get('clamp_h', None)
        self.clamp_v = rendering_kwargs.get('clamp_v', None)
        self.h_mean = rendering_kwargs.get('h_mean', math.pi/2)
        self.r_mean = rendering_kwargs.get('r_mean', 2.7)
        self.fov_mean = rendering_kwargs.get('fov_mean', 20)
        self.v_mean = rendering_kwargs.get('v_mean', math.pi/2)
        self.lookat_h_mean = rendering_kwargs.get('lookat_h_mean', math.pi/2)
        self.lookat_v_mean = rendering_kwargs.get('lookat_v_mean', math.pi/2)
        self.lookat_radius_mean = rendering_kwargs.get('lookat_radius_mean', 0.3)
        self.bg_2d_kwargs = bg_2d_kwargs
        self.generator_bg = None
        if self.bg_2d_kwargs is not None:
            # self.generator_bg = SG2_syn(w_dim=w_dim, img_resolution=neural_rendering_resolution, img_channels=img_channels, **bg_2d_kwargs)
            if not self.rendering_kwargs.get('given_bg', False):
                self.generator_bg = BackgroundGenerator(z_dim=z_dim, c_dim=0, w_dim=w_dim, img_resolution=self.neural_rendering_resolution, img_channels=32, **bg_2d_kwargs)
            if not self.rendering_kwargs.get('given_bg_dino', True):
                self.generator_bg = BackgroundGenerator(z_dim=z_dim, c_dim=0, w_dim=w_dim, img_resolution=self.neural_rendering_resolution, img_channels=rendering_kwargs.get('dino_channals', 3), **bg_2d_kwargs)
        if self.rendering_kwargs.get('use_phase_correlation', False) and self.rendering_kwargs.get('online_phase_correlation', False):
            self.model_corr2softmax = Corr2Softmax(43.8413, 0.)
            h_discrete_num = rendering_kwargs.get('h_discrete_num', 36)
            v_discrete_num = rendering_kwargs.get('v_discrete_num', 1)
            template_num = h_discrete_num*v_discrete_num
            # self.solve_rs = SolveRS(template_num, 'cpu')
            self.solve_rs = SolveRS(template_num, device)
    
        self._last_planes = None
        # self.template = None
        self.template_360_dino = None
        self.all_parameters = None
        self.temperature=rendering_kwargs.get('temperature', 1.0)
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, swapping_prob=None, cam_pose=None):
        z_pose, z = torch.split(z, [self.z_dim, self.z_dim], dim=1)

        cam2world_matrix=c[:, :16].view(-1, 4, 4)
        c_new = c.clone()
        c_new[:,:16] = cam2world_matrix.reshape(c.shape[0], -1)
        if swapping_prob is not None:
            c_swapped = torch.roll(c_new.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c_new.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c_new)
        else:
            c_gen_conditioning = c_new
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c_gen_conditioning = torch.zeros_like(c_gen_conditioning)
        ws = self.backbone.mapping_obj(z,c=c_gen_conditioning, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)  
        ws_bg=None
        if self.generator_bg is not None:
            ws_bg = self.generator_bg.mapping(z,c=None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)  
                                                                                                 
        return {'ws': ws,
                'ws_bg': ws_bg,
                # 'before_repeat_w': w_before,
                'c2w': cam2world_matrix,
                # 'c2w': c[:,:16].view(-1, 4, 4),
                'cam_pose': cam_pose}

    def synthesis(self, ws, ws_bg, c, cam2world_matrix=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, flip=False, flip_type='flip', fov=None, test=False, **synthesis_kwargs):
        
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        # if fov is not None:
        #     focal_length = 1 / (torch.tan(fov * 3.14159 / 360) * 1.414)
        #     intrinsics = intrinsics.clone()
        #     intrinsics[:,0,0] = focal_length[:,0]
        #     intrinsics[:,1,1] = focal_length[:,0]

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
        cam2world_matrix_2 = None
        
        if flip:
            if flip_type=='flip':
            #(2,0) (0,1) (0,2) (0,3)
                cam2world_matrix_2 = cam2world_matrix.clone()
                cam2world_matrix_2[:, 2, 0]*=-1
                cam2world_matrix_2[:, 0, 1]*=-1
                cam2world_matrix_2[:, 0, 2]*=-1
                cam2world_matrix_2[:, 0, 3]*=-1
                ray_origins_flip, ray_directions_flip = self.ray_sampler(cam2world_matrix_2, intrinsics, neural_rendering_resolution)
                ray_origins = torch.cat((ray_origins, ray_origins_flip), dim=0)
                ray_directions = torch.cat((ray_directions, ray_directions_flip), dim=0)
            elif flip_type=='roll':
                cam2world_matrix_2 = torch.roll(cam2world_matrix.clone(), 1, 0)
                ray_origins_roll, ray_directions_roll = self.ray_sampler(cam2world_matrix_2, intrinsics, neural_rendering_resolution)
                ray_origins = torch.cat((ray_origins, ray_origins_roll), dim=0)
                ray_directions = torch.cat((ray_directions, ray_directions_roll), dim=0)


        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws[:,:self.backbone.synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), -1, 32, planes.shape[-2], planes.shape[-1])

        radiance_planes, dino_planes = torch.split(planes, [3, 3], dim=1)

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, bg_lambda, dino_samples, fg_mask, var_depth = self.renderer(planes, self.decoder, self.dino_decoder, ray_origins, ray_directions, self.rendering_kwargs, test=test) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        dino_image = dino_samples.permute(0, 2, 1).reshape(N, dino_samples.shape[-1], H, W).contiguous()
        if self.bg_2d_kwargs is not None:
            if self.generator_bg is not None:
                noise_mode = 'none'
                rgb_bg = self.generator_bg.synthesis(ws_bg, update_emas=update_emas, noise_mode=noise_mode)
                # rgb_bg = self.generator_bg(z_bg, c=None, truncation_psi=truncation_psi,\
                #                                             truncation_cutoff=truncation_cutoff, update_emas=update_emas, **synthesis_kwargs)

                # alpha compositing
                # rgb_bg = rgb_bg / 2 + 0.5  # [-1, 1] -> [0, 1]
            assert fg_mask.min() >= 0 and fg_mask.max() <= 1+1e-3             # add some offset due to precision in alpha computation
            fg_mask = fg_mask.permute(0,2,1).reshape(N,-1,H,W)
            if not self.rendering_kwargs.get('given_bg', False):
                feature_image = feature_image + (1 - fg_mask) * rgb_bg
            if not self.rendering_kwargs.get('given_bg_dino', True):
                dino_image = dino_image + (1 - fg_mask) * rgb_bg

        if depth_image.size(2) != 64:
            scale_factor = 64 / depth_image.size(2)
            input_depth = F.interpolate(depth_image.clone(), scale_factor=scale_factor, mode='bilinear').squeeze(1)
        else:
            input_depth = depth_image.clone().squeeze(1)
        normal_map = get_normal_from_depth(input_depth).permute(0,3,1,2)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        if self.superresolution is not None:
            sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_image = rgb_image

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'normal_map': normal_map, 'c2w': cam2world_matrix, 'c2w_flip': cam2world_matrix_2, 'intrinsic': intrinsics, 'dino_raw': dino_image, 'dino_planes': dino_planes, 'fg_mask':fg_mask, 'radiance_planes': radiance_planes, 'var_depth':var_depth}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        if z.ndim == 2:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)['ws']
        else:
            ws = z
        planes = self.backbone.synthesis(ws[:,:self.backbone.synthesis.num_ws], update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), -1, 32, planes.shape[-2], planes.shape[-1])
        planes, dino_planes = torch.split(planes, [3, 3], dim=1)
        return self.renderer.run_model(planes, dino_planes, self.decoder, self.dino_decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws[:,:self.backbone.synthesis.num_ws], update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), -1, 32, planes.shape[-2], planes.shape[-1])
        planes, dino_planes = torch.split(planes, [3, 3], dim=1)
        return self.renderer.run_model(planes, dino_planes, self.decoder, self.dino_decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, flip=False, cam_pose=None, **synthesis_kwargs):
        # Render a batch of generated images.

        mapping_results = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas, cam_pose=cam_pose)
        results = self.synthesis(mapping_results['ws'], mapping_results['ws_bg'], c, cam2world_matrix=mapping_results['c2w'], update_emas=update_emas, 
                neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, 
                flip=flip, **synthesis_kwargs)
        results.update({'cam_pose':mapping_results['cam_pose']})
        return results        

def add_noise_to_interval(di):
    di_mid  = .5 * (di[..., 1:] + di[..., :-1])
    di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
    di_low  = torch.cat([di[..., :1], di_mid], dim=-1)
    noise   = torch.rand_like(di_low)
    ti      = di_low + (di_high - di_low) * noise
    return ti


def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid

def depth_to_3d_grid(depth, inv_K):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=True).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(inv_K.to(depth.device).transpose(2,1)) * depth
        return grid_3d
    
def get_normal_from_depth(depth, res=2):
    fov=18.837
    R = [[[1.,0.,0.],
            [0.,1.,0.],
            [0.,0.,1.]]]
    R = torch.FloatTensor(R).to(depth.device)
    t = torch.zeros(1,3, dtype=torch.float32).to(depth.device)
    fx = 1/(math.tan(fov/2 *math.pi/180)) # TODO: Check.
    fy = 1/(math.tan(fov/2 *math.pi/180))
    cx = 0
    cy = 0
    K = [[fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.]]
    K = torch.FloatTensor(K).to(depth.device)
    inv_K = torch.inverse(K).unsqueeze(0)
    b, h, w = depth.shape
    grid_3d = depth_to_3d_grid(depth, inv_K)

    tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
    tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
    normal = tu.cross(tv, dim=3)

    zero = torch.FloatTensor([0,0,1]).to(depth.device)
    normal = torch.cat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
    normal = torch.cat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
    normal = normal / (torch.norm(normal, p=2, dim=3, keepdim=True) + 1e-7)
    return normal

from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class OSGDinoDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        self.decoder_output_dim = options['decoder_output_dim']
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        if self.decoder_output_dim >= 384:
            dino = x
        else:
            dino = torch.sigmoid(x)*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF

        return {'dino': dino}

def get_density(self, sigma_raw, sigma_type='softplus'):
    if sigma_type == 'relu':
        sigma_raw = sigma_raw + torch.randn_like(sigma_raw)
        sigma = F.relu(sigma_raw)
    elif sigma_type == 'softplus':  # https://arxiv.org/pdf/2111.11215.pdf
        sigma = F.softplus(sigma_raw - 1)       # 1 is the shifted bias.
    elif sigma_type == 'exp_truncated':    # density in the log-space
        sigma = torch.exp(5 - F.relu(5 - (sigma_raw - 1)))  # up-bound = 5, also shifted by 1
    else:
        sigma = sigma_raw
    return sigma