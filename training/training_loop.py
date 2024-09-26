# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from metrics.pose_distribution import PoseDistribution#compute_pose_distribution_for_generator, compute_pose_distribution_for_dataset
from metrics.depth import compute_depth_for_generator
from camera_utils import LookAtPose, FOV_to_intrinsics, LookAtPoseSampler
from training.crosssection_utils import sample_cross_section
import cv2
import math
import kornia
# from torch.profiler import profile, record_function, ProfilerActivity
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    # gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    # gh = np.clip(4320 // training_set.image_shape[1], 4, 32)
    gw = np.clip(7680 // 512, 7, 32)
    gh = np.clip(4320 // 512, 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, dinos, labels, rot_scale = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(dinos), np.stack(labels), np.stack(rot_scale)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    dis_pose_cond           = True,
    dis_cond_dim             = 2,
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(c_dim=training_set.label_dim, device=device, **G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    if dis_pose_cond:
        dis_c_dim = dis_cond_dim
    else:
        dis_c_dim = 0
    D = dnnlib.util.construct_class_by_name(c_dim=dis_c_dim, **D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    dino_D_kwargs = copy.deepcopy(D_kwargs)
    dino_D_kwargs.class_name = 'training.dual_discriminator.DinoDiscriminator'
    dino_dis_c_dim = 0
    dino_common_kwargs = dict(c_dim=dino_dis_c_dim, img_resolution=loss_kwargs.neural_rendering_resolution_final, img_channels=training_set.num_channels+G_kwargs.rendering_kwargs['dino_channals'])
    D_dino = dnnlib.util.construct_class_by_name(**dino_D_kwargs, **dino_common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim*2], device=device)
        c = torch.empty([batch_gpu, training_set.label_dim], device=device)
        if dis_c_dim > 0:
            c_d = torch.empty([batch_gpu, dis_c_dim], device=device)
        else:
            c_d = None
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c_d])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, D_dino=D_dino, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    D_dino_opt_kwargs = copy.deepcopy(D_opt_kwargs)
    D_dino_opt_kwargs.lr = D_opt_kwargs.lr * 0.5
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval), ('D_dino', D_dino, D_dino_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            if 'dino' in name:
                phases += [dnnlib.EasyDict(name='Dboth_dino', module=module, opt=opt, interval=1)]
            else:
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            if 'dino' in name:
                phases += [dnnlib.EasyDict(name='Dmain_dino', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name='Dreg_dino', module=module, opt=opt, interval=reg_interval)]
            else:
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    pose_distribution = PoseDistribution(max_items=10000)
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, dinos, labels, rot_scales = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        save_image_grid(dinos[:,:3], os.path.join(run_dir, 'reals_dino.png'), drange=[(dinos[:,:3]).min(), (dinos[:,:3]).max()], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim*2], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        rot_scales = torch.from_numpy(rot_scales).to(device).split(batch_gpu)
        dinos = torch.from_numpy(dinos).to(device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_dino, phase_real_c, phase_real_rotscale = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_dino = (phase_real_dino.to(device).to(torch.float32) * 2 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            phase_real_rotscale = phase_real_rotscale.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim*2], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
        
        # render template
        if (cur_nimg%512==0 and cur_nimg<100000) or (cur_nimg%(len(training_set)//batch_size*batch_size)==0 and cur_nimg>=100000) or cur_nimg==resume_kimg * 1000:
            h_discrete_num = G.rendering_kwargs.get('h_discrete_num', 36)
            
            v_discrete_num = G.rendering_kwargs.get('v_discrete_num', 1)
            v_start = G.rendering_kwargs.get('v_start', 0)
            v_end = G.rendering_kwargs.get('v_end', 0)
            h_mean = G.rendering_kwargs.get('h_mean', 1.570796)

            if G_ema.rendering_kwargs.get('uniform_sphere_sampling', False):
                v_list = [np.arccos((1-((v_end-v_start)/(v_discrete_num-1)*i+v_start)*2)) for i in range(v_discrete_num)]
            else:
                v_list = [np.pi/2 + (v_end-v_start)/max((v_discrete_num-1),1)*i + v_start for i in range(v_discrete_num)]
            # angle_p = 0
            neural_rendering_resolution=64

            with torch.no_grad():
                all_dino_images = []
                all_dino_images_255 = []
                all_parameters = []
                if not G.rendering_kwargs.get('given_bg_dino', True):
                    ws_bg = G.generator_bg.mapping.w_avg[None,None].repeat([1, G.generator_bg.synthesis.num_ws, 1])
                    noise_mode = 'none'
                    update_emas=False
                    rgb_bg = G.generator_bg.synthesis(ws_bg, update_emas=update_emas, noise_mode=noise_mode)
                
                ws = G_ema.backbone.mapping_obj.w_avg[None,None].repeat([1, G_ema.backbone.synthesis.num_ws, 1])
                avg_planes = G_ema.backbone.synthesis(ws, update_emas=False)
                avg_planes = avg_planes.view(len(avg_planes), -1, 32, avg_planes.shape[-2], avg_planes.shape[-1])
                all_cam2world_matrix = []
                fov_deg = G.rendering_kwargs.get('fov_deg', 30)
                intrinsics = FOV_to_intrinsics(fov_deg, device=device)
                for angle_p in v_list:
                    for idx in range(h_discrete_num):
                        angle_y = (idx / h_discrete_num - 0.5) * math.pi * 2 * G.rendering_kwargs.get('rot_range_h', 1.0) + h_mean
                    
                        cam_pivot = torch.tensor(G_ema.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                        cam_radius = G_ema.rendering_kwargs.get('avg_camera_radius', 2.7)
                        cam2world_pose = LookAtPoseSampler.sample(angle_y, angle_p, cam_pivot, radius=cam_radius, device=device)
                        cam2world_matrix = cam2world_pose
                        all_cam2world_matrix.append(cam2world_matrix)

                        all_parameters.append(torch.tensor([fov_deg, angle_y, angle_p])[None])

                bs = min(h_discrete_num, 36) #todo, assert
                all_cam2world_matrix = torch.cat(all_cam2world_matrix)
                for jj in range(math.ceil(len(all_cam2world_matrix)/bs)):
                    cam2world_matrix = all_cam2world_matrix[jj*bs:jj*bs+bs]
                    cur_bs = cam2world_matrix.shape[0]
                    ray_origins, ray_directions =  G_ema.ray_sampler(cam2world_matrix, intrinsics[None], neural_rendering_resolution)
                    feature_samples, depth_samples, weights_samples, bg_lambda, dino_samples, fg_mask, _ = G_ema.renderer(avg_planes.repeat(cur_bs,1,1,1,1), G_ema.decoder, G_ema.dino_decoder, ray_origins, ray_directions, G_ema.rendering_kwargs, test=True) # channels last
                    H = W = neural_rendering_resolution
                    N, M, _ = ray_origins.shape
                    dino_image = dino_samples.permute(0, 2, 1).reshape(N, dino_samples.shape[-1], H, W).contiguous()
                    
                    if not G.rendering_kwargs.get('given_bg_dino', True):
                        assert fg_mask.min() >= 0 and fg_mask.max() <= 1+1e-3             # add some offset due to precision in alpha computation
                        fg_mask = fg_mask.permute(0,2,1).reshape(cur_bs,-1,rgb_bg.shape[2],rgb_bg.shape[3])
                        dino_image = dino_image + (1 - fg_mask) * rgb_bg

                    if G_ema.rendering_kwargs.get('maskbody', True):
                        rgb=(dino_image.permute(0,2,3,1) * 127.5 + 128).clip(0, 255)
                        mask_1 = torch.norm(rgb-torch.tensor([[ 99,  99, 114]]).to(device).to(torch.float32),dim=-1)<60
                        mask_2 = torch.norm(rgb-torch.tensor([[144, 186, 208]]).to(device).to(torch.float32),dim=-1)<60
                        mask = mask_1 #| mask_2
                        mask = mask[:,None,:,:].repeat((1,3,1,1))
                        dino_image[mask]=1
                    all_dino_images.append(dino_image)

            G.template_360_dino = torch.cat(all_dino_images, dim=0)[:,None]
            G_ema.template_360_dino = G.template_360_dino
            G.all_parameters = torch.cat(all_parameters, dim=0).to(device)
            G_ema.all_parameters = G.all_parameters
            if G_ema.rendering_kwargs.get('use_phase_correlation', False) and G_ema.rendering_kwargs.get('online_phase_correlation', False):
                logpolar, logbase_rot = G.solve_rs.fft_logpolar(G.template_360_dino[:,0],device)
                G.template_logpolar = logpolar
                G_ema.template_logpolar = G.template_logpolar
                G.template_logbase_rot = logbase_rot
                G_ema.template_logbase_rot = G.template_logbase_rot

        if G_ema.rendering_kwargs.get('cache_pose', False):
            pose_distribution.compute_pose_distribution(batch_size=64, batch_gen=None, G=G_ema,
                            dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
            return

        temperature_end_kimg = G.rendering_kwargs.get('temperature_end_kimg', 0.0) 
        temperature_start_kimg = G.rendering_kwargs.get('temperature_start_kimg', 0.0) 
        temperature_start = G.rendering_kwargs.get('temperature_init', 1.0)
        temperature_end = G.rendering_kwargs.get('temperature', 1.0)
        assert temperature_start_kimg<=temperature_end_kimg
        assert temperature_start<=temperature_end
        G.temperature = max(min((cur_nimg-temperature_start_kimg*1e3) / ((temperature_end_kimg-temperature_start_kimg) * 1e3 + 1e-5), 1),0) * (temperature_end-temperature_start) + temperature_start if temperature_end_kimg>0 else temperature_end
        G_ema.temperature = G.temperature

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_dino, real_c, real_rotscale, gen_z, gen_c in zip(phase_real_img, phase_real_dino, phase_real_c, phase_real_rotscale, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_dino=real_dino, real_c=real_c, real_rotscale=real_rotscale, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg, training_set_num=len(training_set), all_batch_size=batch_size, resume_kimg=resume_kimg, rank=rank)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        fields += [f"Dloss {stats_collector['Loss/D/loss']:<4.3f}"]
        fields += [f"Gloss {stats_collector['Loss/G/loss']:<4.3f}"]
        fields += [f"Dloss_dino {stats_collector['Loss/D/loss_dino']:<4.3f}"]
        fields += [f"Gloss_dino {stats_collector['Loss/G/loss_dino']:<4.3f}"]
        fields += [f"Gloss_cvg_fg {stats_collector['Loss/G/loss_cvg_fg']:<4.3f}"]
        fields += [f"Gloss_cvg_bg {stats_collector['Loss/G/loss_cvg_bg']:<4.3f}"]
        fields += [f"temperature {G_ema.temperature:<4.3f}"]
        
        
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            dino_image_255 = (G_ema.template_360_dino.squeeze().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            stack_images = np.hstack(dino_image_255[:,:,:,:3].cpu().numpy())[:,:,::-1]
            cv2.imwrite(os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_360.png'), stack_images)

            cam_pose = torch.zeros(z.shape[0], 2).float().to(z.device)
            cam_pose[:,0:1] = torch.pi / 2
            cam_pose[:,1:2] = torch.pi / 2
            cam2world_matrix = LookAtPose.sample(cam_pose[:,0:1], cam_pose[:,1:2], torch.tensor(G_ema.rendering_kwargs['avg_camera_pivot'], device=z.device), 
                                                            radius=G_ema.rendering_kwargs['avg_camera_radius'], device=z.device)
            out = [G_ema(z=z, c=torch.cat([cam2world_matrix.reshape(-1,16), c[:,16:]], dim=1), noise_mode='const', cam_pose=cam_pose) for z, c in zip(grid_z, grid_c)]
            images = torch.cat([o['image'].cpu() for o in out]).numpy()
            images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            dinos = torch.cat([o['dino_raw'].cpu() for o in out]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_front.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_front.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_front.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            save_image_grid(dinos[:,:3], os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_dino_front.png'), drange=[(dinos[:,:3]).min(), (dinos[:,:3]).max()], grid_size=grid_size)

            cam_pose = torch.zeros(z.shape[0], 2).float().to(z.device)
            cam_pose[:,0:1] = 0
            cam_pose[:,1:2] = torch.pi / 2
            cam2world_matrix = LookAtPose.sample(cam_pose[:,0:1], cam_pose[:,1:2], torch.tensor(G_ema.rendering_kwargs['avg_camera_pivot'], device=z.device), 
                                                            radius=G_ema.rendering_kwargs['avg_camera_radius'], device=z.device)

            out = [G_ema(z=z, c=torch.cat([cam2world_matrix.reshape(-1,16), c[:,16:]], dim=-1), noise_mode='const', cam_pose=cam_pose) for z, c in zip(grid_z, grid_c)]
            images = torch.cat([o['image'].cpu() for o in out]).numpy()
            images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            dinos = torch.cat([o['dino_raw'].cpu() for o in out]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_side.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_side.png'), drange=[-1,1], grid_size=grid_size)
            save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_side.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            save_image_grid(dinos[:,:3], os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_dino_side.png'), drange=[(dinos[:,:3]).min(), (dinos[:,:3]).max()], grid_size=grid_size)
            if cur_tick % image_snapshot_ticks == 0:
                pose_distribution.compute_pose_distribution_for_generator(save_path=os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_pose.png'), batch_size=64, batch_gen=None, G=G_ema,
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device, D=D)


        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0 or cur_tick==25) and cur_tick>0:
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print(run_dir)
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
