# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""
from camera_utils import LookAtPose, FOV_to_intrinsics, LookAtPoseSampler
from training.dual_discriminator import filtered_resizing
import kornia
from torch_utils.ops import upfirdn2d
import math

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
from camera_utils import LookAtPoseSampler

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True, D=None):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.D              = D
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        # local_ckp_path = '/input/qingyan/models/InceptionV3.pkl'
        # with open(local_ckp_path, 'rb') as pickle_file:
        #     content = pickle.load(pickle_file)
        #     _feature_detector_cache[key] = content.eval().to(device)
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, dinos, _labels, _ in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    # c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    item_subset = [np.random.randint(low=0,high=len(dataset)) for i in range((stats.max_items - 1) // opts.num_gpus + 1)]

    device = opts.device
    h_discrete_num = G.rendering_kwargs.get('h_discrete_num', 36)
    v_discrete_num = G.rendering_kwargs.get('v_discrete_num', 1)

    # Main loop.
    # while not stats.is_full():
    for reals, dinos, _labels, real_rotscale in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        images = []
        dinos = dinos.to(opts.device)
        dinos = 2*dinos - 1
        real_rotscale = real_rotscale.to(opts.device)
        if dinos.shape[0]<batch_size:
            break
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim*2], device=opts.device)
            if G.rendering_kwargs.get('use_gt_label_eval', False):
                gen_c=_labels[_i*batch_gen: (_i+1)*batch_gen].to(device)
                if dataset.dataset=='cars':
                    cam2world = _labels[_i*batch_gen: (_i+1)*batch_gen,:16].reshape(-1,4,4)
                    cam2world_new = cam2world.clone()
                    cam2world_new[:,1] = cam2world[:,2]
                    cam2world_new[:,2] = -cam2world[:,1]
                    cam2world_new[:,:3,3] = cam2world_new[:,:3,3]/1.3*1.7
                    gen_c[:,:16] = cam2world_new.reshape(-1,16)
            elif G.rendering_kwargs.get('uniform_sampling_test', False):
                gen_c=_labels[_i*batch_gen: (_i+1)*batch_gen].to(device)
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 1.7)
                h_mean = G.rendering_kwargs.get('h_mean', 1.570796)
                h = (torch.rand((batch_gen, 1), device=device) * 2 - 1) * np.pi + h_mean
                v=np.pi/2 + (torch.rand((batch_gen, 1), device=device) * 2 - 1) * 0.08726646259971647
                cam2world_matrix = LookAtPoseSampler.sample(h, v, cam_pivot, radius=cam_radius, batch_size=batch_gen, device=device)
                gen_c[:,:16] = cam2world_matrix.reshape(batch_gen, -1)
            else:
                resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
                real_dino_raw = filtered_resizing(dinos[_i*batch_gen: (_i+1)*batch_gen], size=G.template_360_dino.shape[-1], f=resample_filter, filter_mode="antialiased")
                real_dino_raw_mask = real_dino_raw.clone()
                if G.rendering_kwargs.get('maskbody', True):
                    rgb=(real_dino_raw_mask.permute(0,2,3,1) * 127.5 + 128).clip(0, 255)
                    mask_1 = torch.norm(rgb-torch.tensor([[ 99,  99, 114]]).to(device).to(torch.float32),dim=-1)<60
                    mask_2 = torch.norm(rgb-torch.tensor([[144, 186, 208]]).to(device).to(torch.float32),dim=-1)<60
                    mask = mask_1 #| mask_2
                    mask = mask[:,None,:,:].repeat((1,3,1,1))
                    real_dino_raw_mask[mask]=1

                if G.rendering_kwargs.get('use_phase_correlation', False):
                    if G.rendering_kwargs.get('online_phase_correlation', False):
                        template_360_rotated_dino, angle_rot, scale_rot = G.solve_rs.rotate_scale_template(real_dino_raw_mask, G.template_360_dino, G.template_logpolar, G.template_logbase_rot, G.model_corr2softmax, device)
                        error = torch.mean(((real_dino_raw_mask[None]-template_360_rotated_dino)**2).reshape(G.template_360_dino.shape[0],batch_gen,-1),dim=-1)
                    else:
                        error = real_rotscale[_i*batch_gen: (_i+1)*batch_gen,:v_discrete_num*h_discrete_num,0].permute(1,0)
                        angle_rot = real_rotscale[_i*batch_gen: (_i+1)*batch_gen,:v_discrete_num*h_discrete_num,1].permute(1,0)
                        scale_rot = real_rotscale[_i*batch_gen: (_i+1)*batch_gen,:v_discrete_num*h_discrete_num,2].permute(1,0)
                else:
                    error = torch.mean(((real_dino_raw_mask[None]-G.template_360_dino)**2).reshape(G.template_360_dino.shape[0],batch_gen,-1),dim=-1)
                if G.rendering_kwargs.get('use_argmin', False):
                    best_indices = torch.min(error,dim=0)[1]
                else:
                    # best_indices = torch.min(error,dim=0)[1]
                    pdf = torch.nn.functional.softmax(input= -error*G.temperature, dim=0)
                    cdf = torch.cumsum(pdf, dim=0)
                    uu = torch.rand(size = (batch_gen,1)).to(device)
                    best_indices = torch.searchsorted(cdf.permute(1,0), uu, right=True)[:,0]
                    best_indices = torch.clamp(best_indices, min=0, max=cdf.shape[0]-1)

                best_fov = G.all_parameters[best_indices][:,0]
                best_h_angle = G.all_parameters[best_indices][:,1]
                angle_p = G.all_parameters[best_indices][:,2]
                if G.rendering_kwargs.get('add_noise_to_angle', False):
                    h_discrete_num = G.rendering_kwargs.get('h_discrete_num', 36)
                    h_sigma = math.pi * 2 * G.rendering_kwargs.get('rot_range_h', 1.0) / h_discrete_num / 6
                    h_noise = torch.randn(size = (batch_gen,)).to(device) * h_sigma
                    best_h_angle = best_h_angle + h_noise

                    v_discrete_num = G.rendering_kwargs.get('v_discrete_num', 1)
                    v_start = G.rendering_kwargs.get('v_start', 0)
                    v_end = G.rendering_kwargs.get('v_end', 0)
                    # assert G.rendering_kwargs.get('uniform_sphere_sampling', False)==False
                    if G.rendering_kwargs.get('uniform_sphere_sampling', False):
                        v_sigma = (v_end-v_start)/(v_discrete_num-1) / 6
                        v_noise = torch.randn(size = (batch_gen,)).to(device) * v_sigma
                        v = 0.5 * (1 - torch.cos(angle_p)) + v_noise
                        v = torch.clamp(v, min=0, max=1)
                        angle_p = torch.arccos((1 - v * 2))
                    else:
                        v_sigma = (v_end-v_start)/max((v_discrete_num-1),1) / 6
                        v_noise = torch.randn(size = (batch_gen,)).to(device) * v_sigma
                        angle_p = angle_p + v_noise
                focal_length = 1 / (torch.tan(best_fov * 3.14159 / 360) * 1.414)
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                if G.rendering_kwargs.get('use_phase_correlation', False):
                    best_scale = scale_rot[best_indices, torch.arange(batch_gen)]
                    best_angle = angle_rot[best_indices, torch.arange(batch_gen)]
                    # focal_length = focal_length * best_scale
                    cam_radius = (cam_radius/best_scale)[:,None]
                    assert cam_pivot.sum()==0
                else:
                    best_angle=None

                intrinsics = torch.tensor([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 1]], device=device)[None]
                intrinsics = intrinsics.repeat(batch_gen,1,1)
                intrinsics[:,0,0] = focal_length
                intrinsics[:,1,1] = focal_length
                if G.rendering_kwargs.get('use_intrinsic_label', False):
                    intrinsics = _labels[_i*batch_gen: (_i+1)*batch_gen, 16:25].view(-1,3,3).to(device)
                h_mean = G.rendering_kwargs.get('h_mean', 1.570796)
                cam2world_matrix = LookAtPoseSampler.sample(best_h_angle[:,None], angle_p[:,None], cam_pivot, radius=cam_radius, batch_size=batch_gen, device=device)
                if G.rendering_kwargs.get('use_phase_correlation', False):
                    in_plane_rotation = torch.zeros(cam2world_matrix.shape).to(device)
                    in_plane_rotation[:,0,0] = torch.cos(best_angle/180*np.pi)
                    in_plane_rotation[:,0,1] = -torch.sin(best_angle/180*np.pi)
                    in_plane_rotation[:,1,0] = torch.sin(best_angle/180*np.pi)
                    in_plane_rotation[:,1,1] = torch.cos(best_angle/180*np.pi)
                    in_plane_rotation[:,2,2] = 1
                    in_plane_rotation[:,3,3] = 1
                    cam2world_matrix = (cam2world_matrix).matmul(in_plane_rotation)
            
                gen_c = torch.cat([cam2world_matrix.reshape(batch_gen, -1), intrinsics.reshape(batch_gen, -1)], dim=-1)
            
            img = G(z=z, c=gen_c, **opts.G_kwargs)['image']
            # img = G(z=z, c=next(c_iter), **opts.G_kwargs)['image']
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)
        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------
