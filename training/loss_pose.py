# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import math
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from training.loss_utils import warp_img1_to_img0, get_K
from camera_utils import LookAtPose, FOV_to_intrinsics, LookAtPoseSampler
import torch.nn as nn
# import torchvision
# from torch.profiler import profile, record_function, ProfilerActivity

import os
import cv2
import random
import torch.nn.functional as F
import sys
sys.path.insert(0, 'DPCN')
from utils.utils import *
from unet.pytorch_DPCN import PhaseCorr, Corr2Softmax, fft2, LogPolar
from log_polar.log_polar import polar_transformer
from phase_correlation.phase_corr import phase_corr
import kornia
EPS = 1e-7
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------
class CoverageLoss(torch.nn.Module):
    def __init__(self, lambda_cvg_fg, min_cvg_fg, lambda_cvg_bg, min_cvg_bg):
        super(CoverageLoss, self).__init__()
        self.lambda_cvg_fg = lambda_cvg_fg
        self.min_cvg_fg = min_cvg_fg
        self.lambda_cvg_bg = lambda_cvg_bg
        self.min_cvg_bg = min_cvg_bg

    def forward(self, mask):
        assert mask.min() >=0 and mask.max() <= 1+1e-3
        assert mask.ndim == 4

        loss = 0
        if self.lambda_cvg_fg > 0:
            with torch.autograd.profiler.record_function('Greg_cvg_fg_forward'):
                fg_cvg = mask.flatten(1, -1).mean(dim=1)
                loss_fg = self.lambda_cvg_fg * (self.min_cvg_fg - fg_cvg).clamp(min=0)
                training_stats.report('Loss/G/loss_cvg_fg', loss_fg)

            loss = loss + loss_fg.sum()

        if self.lambda_cvg_bg > 0:
            with torch.autograd.profiler.record_function('Greg_cvg_bg_forward'):
                bg_cvg = (1-mask).flatten(1, -1).mean(dim=1)
                loss_bg = self.lambda_cvg_bg * (self.min_cvg_bg - bg_cvg).clamp(min=0)
                training_stats.report('Loss/G/loss_cvg_bg', loss_bg)

            loss = loss + loss_bg.sum()

        return loss

class SolveRS():
    def __init__(self, template_num, device):
        self.h = logpolar_filter((256,256), device)

    def fft_logpolar(self, image,device):
        image = image*0.5+0.5

        image = (image*255).to(torch.uint8)
        image = torch.round(0.299*image[:,0] + 0.587*image[:,1] + 0.114*image[:,2])[:,None]#RGB2GRAY
        image = torch.round(torch.nn.functional.interpolate(image, size=(256, 256), mode='bicubic', align_corners=False)).clamp(0,255)/255

        # from torchvision import transforms
        # trans = transforms.Compose([
        # transforms.ToTensor(),
        #     ])
        # image = image.permute(0,2,3,1).detach().cpu().numpy()
        # image = (image*255).astype(np.uint8)
        # image_list = []
        # for i in range(image.shape[0]):
        #     image_gray = cv2.cvtColor(image[i], cv2.COLOR_RGB2GRAY)
        #     image_gray = cv2.resize(image_gray, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
        #     np_image_data = np.asarray(image_gray)
        #     image_tensor = trans(np_image_data)
        #     image_list.append(image_tensor)
        # image = torch.cat(image_list,dim=0)[:,None].to(device)
        
        image = image.permute(0,2,3,1)
        image = image.squeeze(-1)

        fft = fft2(image) # [B,H,W,1]
        
        fft = fft.squeeze(-1) * self.h
        fft = fft.unsqueeze(-1)

        logpolar, logbase_rot = polar_transformer(fft, (fft.shape[1], fft.shape[2]), device) 

        logpolar = logpolar.squeeze(-1)

        return logpolar, logbase_rot


    def rotate_scale_template(self,real_dino, template_360_dino, template_logpolar, template_logbase_rot, model_corr2softmax, device):
        #template_360_dino: discrete_num, batchsize, c, h, w
        #real_dino: batchsize, c, h, w
        logpolar, logbase_rot = self.fft_logpolar(real_dino,device)


        b, bs, c, h, w = template_360_dino.shape
        center = torch.ones((b,2),device=device)
        center[:, 0] = h // 2
        center[:, 1] = w // 2

        angle_rot_list = []
        scale_rot_list = []
        for idx in range(logpolar.shape[0]):
            if bs!=1:
                rotation_cal, scale_cal, corr_result_rot = phase_corr(logpolar[idx][None], template_logpolar[:,idx], device, template_logbase_rot, model_corr2softmax)
            else:
                rotation_cal, scale_cal, corr_result_rot = phase_corr(logpolar[idx][None], template_logpolar, device, template_logbase_rot, model_corr2softmax)
            angle_rot = -rotation_cal
            scale_rot = 1/scale_cal
            angle_rot_list.append(angle_rot)
            scale_rot_list.append(scale_rot)
        angle_rot = torch.cat(angle_rot_list, dim=0).reshape(-1,b)
        scale_rot = torch.cat(scale_rot_list, dim=0).reshape(-1,b)

        batch_size = logpolar.shape[0]
        rot_mat = kornia.get_rotation_matrix2d(center.repeat(batch_size,1), angle_rot.reshape(-1), scale_rot.reshape(-1,1).repeat(1,2))
        if bs!=1:
            template_360_rotated_dino = kornia.warp_affine(template_360_dino.permute(1,0,2,3,4).reshape(batch_size*b,c,h,w)/2+0.5, rot_mat, dsize=(h, w), align_corners=False)*2-1
        else:
            template_360_rotated_dino = kornia.warp_affine(template_360_dino.repeat(batch_size,1,1,1,1)[:,0]/2+0.5, rot_mat, dsize=(h, w), align_corners=False)*2-1
        template_360_rotated_dino = template_360_rotated_dino.reshape(batch_size, b, c, h, w)
        template_360_rotated_dino = template_360_rotated_dino.permute(1,0,2,3,4)
        angle_rot = angle_rot.permute(1,0)
        scale_rot = scale_rot.permute(1,0)

        return template_360_rotated_dino, angle_rot, scale_rot


class StyleGAN2LossPose(Loss):
    def __init__(self, device, G, D, D_dino, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, 
                    blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, 
                    neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.D_dino             = D_dino
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.dis_cam_weight = self.G.rendering_kwargs.get('dis_cam_weight', 0)
        self.dis_cam_dim = self.G.rendering_kwargs.get('dis_cam_dim', 2)
        self.dis_cond_dim = self.G.rendering_kwargs.get('dis_cond_dim', 2)
        self.cvg_reg = CoverageLoss(lambda_cvg_fg=self.G.rendering_kwargs.get('lambda_cvg_fg', 0.1), min_cvg_fg=self.G.rendering_kwargs.get('min_cvg_fg', 0.4), lambda_cvg_bg=self.G.rendering_kwargs.get('lambda_cvg_bg', 10), min_cvg_bg=self.G.rendering_kwargs.get('min_cvg_bg', 0.2))
        self.no_reg_until = 5000

        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False, flip=False, flip_type='flip', cam_flip=False, cam_pose=None, rank=0):
        gen_c = c.clone()
        mapping_results = self.G.mapping(z, c, update_emas=update_emas, swapping_prob=swapping_prob, cam_pose=cam_pose)
        ws = mapping_results['ws']
        ws_bg = mapping_results['ws_bg']
        cam2world_matrix = mapping_results['c2w']
        cam_pose = mapping_results['cam_pose']
        if cam_flip:
            # flip/roll the camera outside, not in the synthesis function, to avoid the minibatch std problem
            assert flip==False
            if flip_type=='flip_both':
                if self.G.rendering_kwargs.get('use_phase_correlation', False):
                    c_new = cam2world_matrix.clone()
                    c_new[:, 2, 0]*=-1
                    c_new[:, 0, 1]*=-1
                    c_new[:, 0, 2]*=-1
                    c_new[:, 0, 3]*=-1
                    gen_c[:,:16]= c_new.reshape(-1,16)   
                else:
                    if random.random()<0.5:
                        c_new = cam2world_matrix.clone()
                        c_new[:, 2, 0]*=-1
                        c_new[:, 0, 1]*=-1
                        c_new[:, 0, 2]*=-1
                        c_new[:, 0, 3]*=-1
                    else:
                        c_new = cam2world_matrix.clone()
                        c_new[:, 2, 1]*=-1
                        c_new[:, 0, 1]*=-1
                        c_new[:, 1, 0]*=-1
                        c_new[:, 1, 2]*=-1
                        c_new[:, 1, 3]*=-1
            elif flip_type=='flip_both_shapenet':
                if random.random()<0.5:
                    c_new = cam2world_matrix.clone()
                    c_new[:, 0, 0]*=-1
                    c_new[:, 2, 1]*=-1
                    c_new[:, 2, 2]*=-1
                    c_new[:, 2, 3]*=-1
                else:
                    c_new = cam2world_matrix.clone()
                    c_new[:, 2, 1]*=-1
                    c_new[:, 0, 1]*=-1
                    c_new[:, 1, 0]*=-1
                    c_new[:, 1, 2]*=-1
                    c_new[:, 1, 3]*=-1
        else:
            c_new = cam2world_matrix
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)['ws'][:, cutoff:]
        gen_output = self.G.synthesis(ws, ws_bg, c, cam2world_matrix=c_new, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, flip=flip, flip_type=flip_type)
        return {'gen_output':gen_output, 
                'ws': ws, 
                'ws_bg': ws_bg,
                'cam': c_new,
                'cam_pose': cam_pose,
                'gen_c': gen_c}

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, cam_pred=False):
        img_new = {}
        img_new['image']=img['image'].clone()
        img_new['image_raw']=img['image_raw'].clone()
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img_new['image'].device).div(blur_sigma).square().neg().exp2()
                img_new['image'] = upfirdn2d.filter2d(img_new['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img_new['image'],
                                                    torch.nn.functional.interpolate(img_new['image_raw'], size=img_new['image'].shape[2:], mode='bilinear')],
                                                    dim=1))
            img_new['image'] = augmented_pair[:, :img_new['image'].shape[1]]
            img_new['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img_new['image'].shape[1]:], size=img_new['image_raw'].shape[2:], mode='bilinear')

        results = self.D(img_new, c, update_emas=update_emas, cam_pred=cam_pred)
        return {'logits': results['score'], 'cam': results['cam']}
    
    def run_D_dino(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, detach_img=True):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        results = self.D_dino(img, c, update_emas=update_emas, detach_img = detach_img)
        return {'logits': results['score'], 'cam': results['cam']}


    def accumulate_gradients(self, phase, real_img, real_dino, real_c, real_rotscale, gen_z, gen_c, gain, cur_nimg, batch_idx=None, training_set_num=1, all_batch_size=4, resume_kimg=0, rank=0):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'IRboth','Dmain_dino', 'Dreg_dino', 'Dboth_dino']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain', 'Dreg_dino': 'none', 'Dboth_dino':'Dmain_dino'}.get(phase, phase)
        has_mask_loss = (self.cvg_reg.lambda_cvg_bg > 0 or self.cvg_reg.lambda_cvg_fg > 0) and cur_nimg > self.no_reg_until

        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
        real_dino_raw = filtered_resizing(real_dino, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        device = real_img_raw.device
        batch_size = real_dino_raw.shape[0]
        h_discrete_num = self.G.rendering_kwargs.get('h_discrete_num', 36)
        v_discrete_num = self.G.rendering_kwargs.get('v_discrete_num', 1)
        discrete_num = h_discrete_num*v_discrete_num
        h_mean = self.G.rendering_kwargs.get('h_mean', 1.570796)
            
        # camera pose estimation
        with torch.no_grad():
            real_dino_raw_mask = real_dino_raw.clone()
            if self.G.rendering_kwargs.get('maskbody', True):
                rgb=(real_dino_raw_mask.permute(0,2,3,1) * 127.5 + 128).clip(0, 255)
                mask_1 = torch.norm(rgb-torch.tensor([[ 99,  99, 114]]).to(device).to(torch.float32),dim=-1)<60
                mask_2 = torch.norm(rgb-torch.tensor([[144, 186, 208]]).to(device).to(torch.float32),dim=-1)<60
                mask = mask_1 #| mask_2
                mask = mask[:,None,:,:].repeat((1,3,1,1))
                real_dino_raw_mask[mask]=1
            
            if self.G.rendering_kwargs.get('use_phase_correlation', False):
                if self.G.rendering_kwargs.get('online_phase_correlation', False):
                    template_360_rotated_dino, angle_rot, scale_rot = self.G.solve_rs.rotate_scale_template(real_dino_raw_mask, self.G.template_360_dino, self.G.template_logpolar, self.G.template_logbase_rot, self.G.model_corr2softmax, device)
                    error = torch.mean(((real_dino_raw_mask[None]-template_360_rotated_dino)**2).reshape(self.G.template_360_dino.shape[0],batch_size,-1),dim=-1)
                else:
                    error = real_rotscale[:,:v_discrete_num*h_discrete_num,0].permute(1,0)
                    angle_rot = real_rotscale[:,:v_discrete_num*h_discrete_num,1].permute(1,0)
                    scale_rot = real_rotscale[:,:v_discrete_num*h_discrete_num,2].permute(1,0)
            else:
                error = torch.mean(((real_dino_raw_mask[None]-self.G.template_360_dino)**2).reshape(self.G.template_360_dino.shape[0],batch_size,-1),dim=-1)
            
            pdf = torch.nn.functional.softmax(input= -error*self.G.temperature, dim=0)
            cdf = torch.cumsum(pdf, dim=0)
            if self.G.rendering_kwargs.get('use_argmin', False):
                best_indices = torch.min(error,dim=0)[1]
            else:
                uu = torch.rand(size = (batch_size,1)).to(device)
                best_indices = torch.searchsorted(cdf.permute(1,0), uu, right=True)[:,0]
                best_indices = torch.clamp(best_indices, min=0, max=cdf.shape[0]-1)

            best_fov = self.G.all_parameters[best_indices][:,0]
            best_h_angle = self.G.all_parameters[best_indices][:,1]
            angle_p = self.G.all_parameters[best_indices][:,2]
            if self.G.rendering_kwargs.get('add_noise_to_angle', False):
                h_discrete_num = self.G.rendering_kwargs.get('h_discrete_num', 36)
                h_sigma = math.pi * 2 * self.G.rendering_kwargs.get('rot_range_h', 1.0) / h_discrete_num / 6
                h_noise = torch.randn(size = (batch_size,)).to(device) * h_sigma
                best_h_angle = best_h_angle + h_noise

                v_discrete_num = self.G.rendering_kwargs.get('v_discrete_num', 1)
                v_start = self.G.rendering_kwargs.get('v_start', 0)
                v_end = self.G.rendering_kwargs.get('v_end', 0)
                # assert self.G.rendering_kwargs.get('uniform_sphere_sampling', False)==False
                if self.G.rendering_kwargs.get('uniform_sphere_sampling', False):
                    v_sigma = (v_end-v_start)/(v_discrete_num-1) / 6
                    v_noise = torch.randn(size = (batch_size,)).to(device) * v_sigma
                    v = 0.5 * (1 - torch.cos(angle_p)) + v_noise
                    v = torch.clamp(v, min=0, max=1)
                    angle_p = torch.arccos((1 - v * 2))
                else:
                    v_sigma = (v_end-v_start)/max((v_discrete_num-1),1) / 6
                    v_noise = torch.randn(size = (batch_size,)).to(device) * v_sigma
                    angle_p = angle_p + v_noise

            focal_length = 1 / (torch.tan(best_fov * 3.14159 / 360) * 1.414)
            cam_pivot = torch.tensor(self.G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
            if self.G.rendering_kwargs.get('use_phase_correlation', False):
                best_scale = scale_rot[best_indices, torch.arange(batch_size)]
                best_angle = angle_rot[best_indices, torch.arange(batch_size)]
                cam_radius = (cam_radius/best_scale)[:,None]
                assert cam_pivot.sum()==0
            else:
                best_angle=None
                best_scale=1

            intrinsics = torch.tensor([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 1]], device=device)[None]
            intrinsics = intrinsics.repeat(batch_size,1,1)
            intrinsics[:,0,0] = focal_length
            intrinsics[:,1,1] = focal_length
            if self.G.rendering_kwargs.get('use_intrinsic_label', False):
                intrinsics = gen_c[:, 16:25].view(-1,3,3)
            cam2world_matrix = LookAtPoseSampler.sample(best_h_angle[:,None], angle_p[:,None], cam_pivot, radius=cam_radius, batch_size=batch_size, device=device)
            if self.G.rendering_kwargs.get('use_phase_correlation', False):
                in_plane_rotation = torch.zeros(cam2world_matrix.shape).to(device)
                in_plane_rotation[:,0,0] = torch.cos(best_angle/180*np.pi)
                in_plane_rotation[:,0,1] = -torch.sin(best_angle/180*np.pi)
                in_plane_rotation[:,1,0] = torch.sin(best_angle/180*np.pi)
                in_plane_rotation[:,1,1] = torch.cos(best_angle/180*np.pi)
                in_plane_rotation[:,2,2] = 1
                in_plane_rotation[:,3,3] = 1
                cam2world_matrix = (cam2world_matrix).matmul(in_plane_rotation)
            
            gen_c = torch.cat([cam2world_matrix.reshape(batch_size, -1), intrinsics.reshape(batch_size, -1)], dim=-1)
            cam_dim = 2
            cam_pose_prior = torch.zeros(batch_size, cam_dim).float().to(device)
            cam_pose_prior[:,0] = best_h_angle
            cam_pose_prior[:,1] = angle_p
    
        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())
                real_dino_raw = upfirdn2d.filter2d(real_dino_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw, 'dino_raw': real_dino_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, 
                                flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), cam_pose=cam_pose_prior, rank=rank)
                gen_img, cam2world = G_outputs['gen_output'], G_outputs['cam']
                if self.dis_cam_weight > 0:
                    d_cond = self.run_D(gen_img, None, blur_sigma=blur_sigma, cam_pred=True)['cam']
                    d_results = self.run_D(gen_img, d_cond.detach()[:,:self.dis_cond_dim], blur_sigma=blur_sigma)
                    d_results_dino = self.run_D_dino(gen_img, None, blur_sigma=blur_sigma)
                    
                else:
                    if self.G.rendering_kwargs.get('dis_pose_cond',True):
                        d_cond = gen_c
                        d_results = self.run_D(gen_img, d_cond, blur_sigma=blur_sigma)
                    else:
                        d_results = self.run_D(gen_img, None, blur_sigma=blur_sigma)
                    d_results_dino = self.run_D_dino(gen_img, None, blur_sigma=blur_sigma)
                gen_logits = d_results['logits']
                gen_logits_dino = d_results_dino['logits']
                loss_Gmain = 0
                loss_Gmain_dino = 0

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = loss_Gmain + torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)

                training_stats.report('Loss/scores/fake', gen_logits_dino)
                training_stats.report('Loss/signs/fake', gen_logits_dino.sign())
                loss_Gmain_dino = loss_Gmain_dino + torch.nn.functional.softplus(-gen_logits_dino)
                training_stats.report('Loss/G/loss_dino', loss_Gmain_dino)

                # Mask loss needs to be added to loss_Gmain because we can only backprop once through the rendering
                mask_loss = 0
                if has_mask_loss:
                    gen_alpha = gen_img['fg_mask']
                    assert gen_alpha.ndim == 4
                    # Coverage Regularization
                    mask_loss = mask_loss + self.cvg_reg(gen_alpha)
                

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_Gmain_dino + mask_loss).mean().mul(gain).backward()

            if self.G.rendering_kwargs.get('flip_to_dis', False):
                
                with torch.autograd.profiler.record_function('Gmain_flip_forward'):
                    G_outputs_flip = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, 
                                    flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), cam_flip=True, cam_pose=cam_pose_prior, rank=rank)
                    gen_img_flip = G_outputs_flip['gen_output']
                    if self.dis_cam_weight > 0:
                        d_cond_flip = self.run_D(gen_img_flip, None, blur_sigma=blur_sigma, cam_pred=True)['cam']
                        d_results_flip = self.run_D(gen_img_flip, d_cond_flip.detach()[:,:self.dis_cond_dim], blur_sigma=blur_sigma)
                        d_results_flip_dino = self.run_D_dino(gen_img_flip, None, blur_sigma=blur_sigma)
                    else:
                        if self.G.rendering_kwargs.get('dis_pose_cond',True):
                            d_cond = G_outputs_flip['gen_c']
                            d_results_flip = self.run_D(gen_img_flip, d_cond, blur_sigma=blur_sigma)
                        else:
                            d_results_flip = self.run_D(gen_img_flip, None, blur_sigma=blur_sigma)
                        d_results_flip_dino = self.run_D_dino(gen_img_flip, None, blur_sigma=blur_sigma)
                    gen_logits_flip = d_results_flip['logits']
                    gen_logits_flip_dino = d_results_flip_dino['logits']

                    loss_Gmain_flip = 0
                    loss_Gmain_flip_dino = 0

                    training_stats.report('Loss/scores/fake_flip', gen_logits_flip)
                    training_stats.report('Loss/signs/fake_flip', gen_logits_flip.sign())
                    loss_Gmain_flip = loss_Gmain_flip + torch.nn.functional.softplus(-gen_logits_flip)
                    training_stats.report('Loss/G/loss_flip', loss_Gmain_flip)

                    training_stats.report('Loss/scores/fake_flip_dino', gen_logits_flip_dino)
                    training_stats.report('Loss/signs/fake_flip_dino', gen_logits_flip_dino.sign())
                    loss_Gmain_flip_dino = loss_Gmain_flip_dino + torch.nn.functional.softplus(-gen_logits_flip_dino)
                    training_stats.report('Loss/G/loss_flip_dino', loss_Gmain_flip_dino)

                with torch.autograd.profiler.record_function('Gmain_flip_backward'):
                    (loss_Gmain_flip+loss_Gmain_flip_dino).mean().mul(gain).backward()
            
        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':

            mapping_results = self.G.mapping(gen_z, gen_c, update_emas=False)
            ws = mapping_results['ws']
            cam2world = mapping_results['c2w']
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)['ws'][:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

            

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if self.G.rendering_kwargs.get('flip_to_disd', False):
            loss_dgen_weight = self.G.rendering_kwargs.get('flip_to_disd_weight', 0.5)
        else:
            loss_dgen_weight = 1.0
        if phase in ['Dmain', 'Dboth']:
            # Update camera head
            if self.dis_cam_weight > 0: # can change the pose to the flipped view?
                with torch.autograd.profiler.record_function('Dcam_forward'):
                    G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, cam_pose=cam_pose_prior)
                    gen_img = G_outputs['gen_output']
                    cam_pose = G_outputs['cam_pose'].detach()
                    cam_pose_pred = self.run_D(gen_img, None, blur_sigma=blur_sigma, update_emas=True, cam_pred=True)['cam']
                    loss_Dcam = torch.nn.functional.mse_loss(cam_pose[:,:self.dis_cam_dim], cam_pose_pred)
                    training_stats.report('Loss/D/loss_Dcam', loss_Dcam)
                with torch.autograd.profiler.record_function('Dcam_backward'):
                    (loss_Dcam * self.dis_cam_weight).mean().mul(gain).backward()
            

            with torch.autograd.profiler.record_function('Dgen_forward'):
                G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, cam_pose=cam_pose_prior, rank=rank)
                gen_img = G_outputs['gen_output']
                if self.dis_cam_weight > 0:
                    d_cond = self.run_D(gen_img, None, blur_sigma=blur_sigma, update_emas=True, cam_pred=True)['cam']
                    d_results = self.run_D(gen_img, d_cond.detach()[:,:self.dis_cond_dim], blur_sigma=blur_sigma, update_emas=True)
                    
                else:
                    if self.G.rendering_kwargs.get('dis_pose_cond',True):
                        d_cond = gen_c
                        d_results = self.run_D(gen_img, d_cond, blur_sigma=blur_sigma, update_emas=True)
                    else:
                        d_results = self.run_D(gen_img, None, blur_sigma=blur_sigma, update_emas=True)
                
                loss_Dgen = 0

                gen_logits = d_results['logits']
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = loss_Dgen + torch.nn.functional.softplus(gen_logits)
                training_stats.report('Loss/D/loss_Dgen', loss_Dgen)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen * loss_dgen_weight).mean().mul(gain).backward()

            if self.G.rendering_kwargs.get('flip_to_disd', False):
                with torch.autograd.profiler.record_function('Dgen_flip_forward'):
                    G_outputs_flip = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True,
                                                flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), cam_flip=True, cam_pose=cam_pose_prior, rank=rank)
                    gen_img_flip = G_outputs_flip['gen_output']
                    if self.dis_cam_weight > 0:
                        d_cond_flip = self.run_D(gen_img_flip, None, blur_sigma=blur_sigma, update_emas=True, cam_pred=True)['cam']
                        d_results_flip = self.run_D(gen_img_flip, d_cond_flip.detach()[:,:self.dis_cond_dim], blur_sigma=blur_sigma, update_emas=True)
                    else:
                        if self.G.rendering_kwargs.get('dis_pose_cond',True):
                            d_cond = G_outputs_flip['gen_c']
                            d_results_flip = self.run_D(gen_img_flip, d_cond, blur_sigma=blur_sigma, update_emas=True)
                        else:
                            d_results_flip = self.run_D(gen_img_flip, None, blur_sigma=blur_sigma, update_emas=True)

                    loss_Dgen_flip = 0
                    
                    gen_logits_flip = d_results_flip['logits']
                    training_stats.report('Loss/scores/fake_flip', gen_logits_flip)
                    training_stats.report('Loss/signs/fake_flip', gen_logits_flip.sign())
                    loss_Dgen_flip = loss_Dgen_flip + torch.nn.functional.softplus(gen_logits_flip)
                    training_stats.report('Loss/D/loss_Dgen_flip', loss_Dgen_flip)
                with torch.autograd.profiler.record_function('Dgen_flip_backward'):
                    (loss_Dgen_flip * loss_dgen_weight).mean().mul(gain).backward()
            

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_dino_tmp = real_img['dino_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'dino_raw': real_dino_tmp}
                if self.dis_cam_weight > 0:
                    d_cond_real = self.run_D(real_img_tmp, None, blur_sigma=blur_sigma, cam_pred=True)['cam']
                    d_results = self.run_D(real_img_tmp, d_cond_real.detach()[:,:self.dis_cond_dim], blur_sigma=blur_sigma)
                else:
                    if self.G.rendering_kwargs.get('dis_pose_cond',True):
                        d_cond = gen_c
                        d_results = self.run_D(real_img_tmp, d_cond, blur_sigma=blur_sigma)
                    else:
                        d_results = self.run_D(real_img_tmp, None, blur_sigma=blur_sigma)

                real_logits_nocond = None
                
                real_logits = d_results['logits']
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    if self.G.rendering_kwargs.get('flip_to_disd', False):
                        training_stats.report('Loss/D/loss', loss_Dgen*loss_dgen_weight + loss_Dreal + loss_Dgen_flip*loss_dgen_weight)
                    else:
                        training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])# + r1_grads_dino_raw.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                        loss_Dr1 = r1_penalty * r1_gamma
                    
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Dmain_dino: Minimize logits for generated images.
        loss_Dgen_dino = 0
        if self.G.rendering_kwargs.get('flip_to_disd', False):
            loss_dgen_weight = self.G.rendering_kwargs.get('flip_to_disd_weight', 0.5)
        else:
            loss_dgen_weight = 1.0
        if phase in ['Dmain_dino', 'Dboth_dino']:
            with torch.autograd.profiler.record_function('Dgen_forward_dino'):
                G_outputs = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, cam_pose=cam_pose_prior)
                gen_img = G_outputs['gen_output']
                d_results_dino = self.run_D_dino(gen_img, None, blur_sigma=blur_sigma, update_emas=True)

                gen_logits_dino = d_results_dino['logits']
                training_stats.report('Loss/scores/fake_dino', gen_logits_dino)
                training_stats.report('Loss/signs/fake_dino', gen_logits_dino.sign())
                loss_Dgen_dino = torch.nn.functional.softplus(gen_logits_dino)
                training_stats.report('Loss/D/loss_Dgen_dino', loss_Dgen_dino)
            with torch.autograd.profiler.record_function('Dgen_backward_dino'):
                (loss_Dgen_dino * loss_dgen_weight).mean().mul(gain).backward()

            if self.G.rendering_kwargs.get('flip_to_disd', False):
                with torch.autograd.profiler.record_function('Dgen_flip_forward'):
                    G_outputs_flip = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True,
                                                flip=False, flip_type=self.G.rendering_kwargs.get('flip_type', 'flip'), cam_flip=True, cam_pose=cam_pose_prior)
                    gen_img_flip = G_outputs_flip['gen_output']
                    d_results_flip_dino = self.run_D_dino(gen_img_flip, None, blur_sigma=blur_sigma, update_emas=True)
                    
                    gen_logits_flip_dino = d_results_flip_dino['logits']
                    training_stats.report('Loss/scores/fake_flip_dino', gen_logits_flip_dino)
                    training_stats.report('Loss/signs/fake_flip_dino', gen_logits_flip_dino.sign())
                    loss_Dgen_flip_dino = torch.nn.functional.softplus(gen_logits_flip_dino)
                    training_stats.report('Loss/D/loss_Dgen_flip_dino', loss_Dgen_flip_dino)
                with torch.autograd.profiler.record_function('Dgen_flip_backward_dino'):
                    (loss_Dgen_flip_dino * loss_dgen_weight).mean().mul(gain).backward()
            

        # Dmain_dino: Maximize logits for real dino images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain_dino', 'Dreg_dino', 'Dboth_dino']:
            name = 'Dreal_dino' if phase == 'Dmain_dino' else 'Dr1_dino' if phase == 'Dreg_dino' else 'Dreal_Dr1_dino'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(False)
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg_dino', 'Dboth_dino'])
                real_dino_tmp = real_img['dino_raw'].detach().requires_grad_(phase in ['Dreg_dino', 'Dboth_dino'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'dino_raw': real_dino_tmp}
                d_results_dino = self.run_D_dino(real_img_tmp, None, blur_sigma=blur_sigma, detach_img=False)
                
                real_logits_dino = d_results_dino['logits']
                training_stats.report('Loss/scores/real_dino', real_logits_dino)
                training_stats.report('Loss/signs/real_dino', real_logits_dino.sign())

                loss_Dreal_dino = 0
                if phase in ['Dmain_dino', 'Dboth_dino']:
                    loss_Dreal_dino = torch.nn.functional.softplus(-real_logits_dino)
                    if self.G.rendering_kwargs.get('flip_to_disd', False):
                        training_stats.report('Loss/D/loss_dino', loss_Dgen_dino*loss_dgen_weight + loss_Dreal_dino + loss_Dgen_flip_dino*loss_dgen_weight)
                    else:
                        training_stats.report('Loss/D/loss_dino', loss_Dgen_dino + loss_Dreal_dino)

                loss_Dr1_dino = 0
                if phase in ['Dreg_dino', 'Dboth_dino']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads_dino'), conv2d_gradfix.no_weight_gradients():
                            r1_grads_dino = torch.autograd.grad(outputs=[real_logits_dino.sum()], inputs=[ real_img_tmp['image_raw'], real_img_tmp['dino_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image_raw_dino = r1_grads_dino[0]
                            r1_grads_dino_raw_dino = r1_grads_dino[1]
                        r1_penalty_dino = r1_grads_image_raw_dino.square().sum([1,2,3]) + r1_grads_dino_raw_dino.square().sum([1,2,3])
                        loss_Dr1_dino = r1_penalty_dino * (r1_gamma / 2)
                    else: # single discrimination
                        assert self.dual_discrimination
                        
                    training_stats.report('Loss/r1_penalty_dino', r1_penalty_dino)
                    training_stats.report('Loss/D/reg_dino', loss_Dr1_dino)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal_dino + loss_Dr1_dino).mean().mul(gain).backward()






#----------------------------------------------------------------------------
