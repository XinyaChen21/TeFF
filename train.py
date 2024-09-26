# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."

Code adapted from
"Alias-Free Generative Adversarial Networks"."""

from email.policy import default
import os
import click
import re
import json
import tempfile
import torch
# print(torch.cuda.is_available())
import math

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
import sys 
sys.path.insert(0, 'DPCN')

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir, cache_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    if cache_dir is not None:
        if rank == 0:
            print(f'Setting cache directory as `{cache_dir}`...')
            print()
        dnnlib.util.set_cache_dir(cache_dir)
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run, cache_dir):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir, cache_dir=cache_dir)
        else:
            # torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir, cache_dir), nprocs=c.num_gpus)
            processes = []
            for rank in range(c.num_gpus):
                p = torch.multiprocessing.Process(target=subprocess_fn, args=(rank, c, temp_dir, cache_dir))
                p.start()     
                processes.append(p)
            for p in processes:
                p.join()

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, dino_data, mask_data, resolution, create_label_fov=None, pad_long=False, dino_channals=None, black_bg_dino=False,black_bg=False, load_rotscale=False, rot_scale_path='',shapenet_multipeak=False):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, dino_path=dino_data, use_labels=True, max_size=None, xflip=False, resolution=resolution, create_label_fov=create_label_fov, pad_long=pad_long, dino_channals =dino_channals, black_bg_dino=black_bg_dino, black_bg=black_bg, load_rotscale=load_rotscale, rot_scale_path=rot_scale_path, shapenet_multipeak=shapenet_multipeak, mask_path=mask_data)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=str, required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--dino_data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--mask_data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, default='datasets/ffhq/ffhg_sam_bbox.zip', show_default=True)
@click.option('--rot_scale_data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, default='', show_default=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)
@click.option('--create_label_fov',        help='Fov if labels are not available', metavar='FLOAT',               type=click.FloatRange(min=0), default=30)
@click.option('--cache_dir', help='Cache directory', type=str, metavar='DIR')

@click.option('--ray_start',        help='ray_start', metavar='FLOAT',               type=click.FloatRange(min=0), default=0.1)
@click.option('--ray_end',        help='ray_end', metavar='FLOAT',               type=click.FloatRange(min=0), default=3.3)
@click.option('--avg_camera_radius',        help='ray_end', metavar='FLOAT',               type=click.FloatRange(min=0), default=2.7)
@click.option('--box_warp',        help='box_warp', metavar='FLOAT',               type=click.FloatRange(min=0), default=1.0)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--use_intrinsic_label',         help='whether use intrinsic label', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--lr_multiplier',          help='Learning rate multiplier for pose layers.', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.01, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

@click.option('--num_bg_pts',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--wrong',       help='whether use the wrong projection', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--pad_long',       help='whether model the background separately', metavar='BOOL',                  type=bool, default=False, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
# @click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=1, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# @click.option('--sr_module',    help='Superresolution module', metavar='STR',  type=str, required=True)
@click.option('--dataset_resolution', help='Resolution of dataset', metavar='INT',  type=click.IntRange(min=1), default=128, required=False)
@click.option('--neural_rendering_resolution_initial', help='Resolution to render at', metavar='INT',  type=click.IntRange(min=1), default=64, required=False)
@click.option('--neural_rendering_resolution_final', help='Final resolution to render at, if blending', metavar='INT',  type=click.IntRange(min=1), required=False, default=None)
@click.option('--neural_rendering_resolution_fade_kimg', help='Kimg to blend resolution over', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000, show_default=True)
@click.option('--flip_to_dis', help='If true, images rendered from the flipped camera pose will also be used to update G through D', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--flip_to_disd', help='If true, images rendered from the flipped camera pose will also be used to update D through G', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--flip_to_disd_weight', help='weight for discriminator loss of original image and the flipped one', metavar='FLOAT', type=click.FloatRange(min=0), required=False, default=1.0)
@click.option('--flip_type', help='flip or roll for camera pose', metavar='STR',  type=click.Choice(['flip_both','flip_both_shapenet']), required=False, default='flip_both')
@click.option('--dis_linear_pose', help='If true, use a linear layer to extract pose in the discriminator. Otherwise, use two linear layers with lrelu in between', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--cnn', help='Blur over how many', metavar='INT',  type=click.IntRange(min=0), required=False, default=0) # (0: not use cnn, 1: use kernel 1x1, 3: use kernel 3x3)
@click.option('--activation_cnn', help='Type of activation for CNNs', metavar='STR',  type=click.Choice(['lrelu', 'linear', 'relu']), required=False, default=None)
@click.option('--dis_cam_weight', help='The weight for camera prediction loss of D. If it is zero, then dis_pose_cond should be false.', metavar='FLOAT', type=click.FloatRange(min=0), required=False, default=0)
@click.option('--min_cvg_fg',        help='Minimum foreground coverage threshold', metavar='FLOAT',               type=click.FloatRange(min=0), default=0.4)
@click.option('--min_cvg_bg',        help='Minimum background coverage threshold', metavar='FLOAT',               type=click.FloatRange(min=0), default=0.2)
@click.option('--lambda_cvg_fg',        help='Strength of foreground coverage loss', metavar='FLOAT',               type=click.FloatRange(min=0), default=0.1)
@click.option('--lambda_cvg_bg',        help='Strength of background coverage loss', metavar='FLOAT',               type=click.FloatRange(min=0), default=10)

# Main TeFF hyperparameters.
@click.option('--temperature_init',        help='Initial temperature for pose PDF', metavar='FLOAT',               type=click.FloatRange(min=0), default=1.0)
@click.option('--temperature',        help='Terminal temperature for pose PDF', metavar='FLOAT',               type=click.FloatRange(min=0), default=1.0)
@click.option('--temperature_start_kimg', help='Kimg to start increasing temperature', metavar='INT',  type=click.FloatRange(min=0), required=False, default=0, show_default=True)
@click.option('--temperature_end_kimg', help='Kimg to stop increasing temperature', metavar='INT',  type=click.FloatRange(min=0), required=False, default=0, show_default=True)
@click.option('--h_discrete_num', help='Discrete number of horizonal angles', metavar='INT',  type=click.IntRange(min=0), required=False, default=36, show_default=True)
@click.option('--v_discrete_num', help='Discrete number of vertical angles', metavar='INT',  type=click.IntRange(min=0), required=False, default=1, show_default=True)
@click.option('--v_start', help='Start value of vertical angle', metavar='FLOAT',  type=click.FloatRange(min=-3.14), required=False, default=0.0, show_default=True)
@click.option('--v_end', help='End value of vertical angle', metavar='FLOAT',  type=click.FloatRange(min=-3.14), required=False, default=0.0, show_default=True)
@click.option('--rot_range_h', help='The range of horizontal angle', metavar='FLOAT', type=click.FloatRange(min=0), required=False, default=1.0)
@click.option('--h_mean', help='Mean value of horizonal angles', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=1.570796, show_default=True)
@click.option('--uniform_sphere_sampling', help='If true, discretize angles by uniform sphere sampling.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--add_noise_to_angle', help='If true, add noise to the solved discrete angle',  metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--use_phase_correlation', help='If true, use phase correlation',  metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--online_phase_correlation', help='If true, use online phase correlation',  metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--use_argmin', help='If true, select pose by argmin',  metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--bg_modeling_2d',       help='If true, model the background by 2D generator', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--black_bg_dino',       help='If true, set the background of DINO features as black', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--black_bg',       help='If true, set the background of RGB images as black', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--dino_channals',        help='the channel amount of DINO features', metavar='INT',                      type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--maskbody',       help='If true, mask body when matching (ffhq dataset)', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--shapenet_multipeak', help='If true, use multipeak shapenet',  metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--uniform_sampling_test',       help='If true, sample pose uniformly for rendering when evaluate FID, i.e. FID_360', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--use_gt_label_eval', help='If true, use gt pose for rendering when evaluate FID, i.e. FID_gt',  metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--cache_pose', help='If true, cache the solved pose',  metavar='BOOL',                  type=bool, default=False, show_default=True)

@click.option('--blur_fade_kimg', help='Blur over how many', metavar='INT',  type=click.IntRange(min=1), required=False, default=200)
@click.option('--gen_pose_cond', help='If true, enable generator pose conditioning.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--empty_gen_cond', help='If not enable generator pose conditioning, whether use zero-filled condition or remove the condition. If True, remove the condition.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--dis_pose_cond', help='If true, enable discriminator pose conditioning.', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--dis_cam_dim',        help='Discriminator camera params prediction dimension', metavar='INT',                      type=click.IntRange(min=2), default=2, show_default=True)
@click.option('--dis_cond_dim',        help='Discriminator condition dimension', metavar='INT',                      type=click.IntRange(min=-1), default=-1, show_default=True)
@click.option('--superres', help='If true, enable super-resolution module.', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--dual_discrimination', help='If true, enable dual discrimination.', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--c-scale', help='Scale factor for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=1)
@click.option('--c-noise', help='Add noise for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--gpc_reg_prob', help='Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0.5)
@click.option('--gpc_reg_fade_kimg', help='Length of swapping prob fade', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000)
@click.option('--disc_c_noise', help='Strength of discriminator pose conditioning regularization, in standard deviations.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--sr_noise_mode', help='Type of noise for superresolution', metavar='STR',  type=click.Choice(['random', 'none']), required=False, default='none')
@click.option('--resume_blur', help='Enable to blur even on resume', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--sr_num_fp16_res',    help='Number of fp16 layers in superresolution', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--g_num_fp16_res',    help='Number of fp16 layers in generator', metavar='INT', type=click.IntRange(min=0), default=0, required=False, show_default=True)
@click.option('--d_num_fp16_res',    help='Number of fp16 layers in discriminator', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--sr_first_cutoff',    help='First cutoff for AF superresolution', metavar='INT', type=click.IntRange(min=2), default=2, required=False, show_default=True)
@click.option('--sr_first_stopband',    help='First cutoff for AF superresolution', metavar='FLOAT', type=click.FloatRange(min=2), default=2**2.1, required=False, show_default=True)
@click.option('--style_mixing_prob',    help='Style-mixing regularization probability for training.', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0, required=False, show_default=True)
@click.option('--sr-module',    help='Superresolution module override', metavar='STR',  type=str, required=False, default=None)
@click.option('--density_reg',    help='Density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.25, required=False, show_default=True)
@click.option('--density_reg_every',    help='lazy density reg', metavar='int', type=click.FloatRange(min=1), default=4, required=False, show_default=True)
@click.option('--density_reg_p_dist',    help='density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.004, required=False, show_default=True)
@click.option('--reg_type', help='Type of regularization', metavar='STR',  type=click.Choice(['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation']), required=False, default='l1')
@click.option('--decoder_lr_mul',    help='decoder learning rate multiplier.', metavar='FLOAT', type=click.FloatRange(min=0), default=1, required=False, show_default=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_pose.StyleGAN2LossPose')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    if opts.use_phase_correlation and not opts.online_phase_correlation:
        load_rotscale = True
    else:
        load_rotscale = False
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, dino_data = opts.dino_data, resolution=opts.dataset_resolution, create_label_fov=opts.create_label_fov, pad_long=opts.pad_long, dino_channals=opts.dino_channals, black_bg_dino=opts.black_bg_dino, black_bg=opts.black_bg, load_rotscale=load_rotscale, rot_scale_path=opts.rot_scale_data, shapenet_multipeak=opts.shapenet_multipeak, mask_data=opts.mask_data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    if opts.bg_modeling_2d:
        bg_2d_kwargs = dnnlib.EasyDict(channel_base=2048, channel_max=64, fused_modconv_default='inference_only')
    else:
        bg_2d_kwargs = None
    c.G_kwargs.bg_2d_kwargs = bg_2d_kwargs

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    c.G_kwargs.mapping_kwargs.lr_multiplier = opts.lr_multiplier
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.D_kwargs.epilogue_kwargs.dis_cam_weight = opts.dis_cam_weight
    c.D_kwargs.epilogue_kwargs.dis_cam_dim = opts.dis_cam_dim
    if opts.dis_cond_dim == -1:
        dis_cond_dim = opts.dis_cam_dim
    else:
        dis_cond_dim = opts.dis_cond_dim
    c.dis_cond_dim = dis_cond_dim
    c.D_kwargs.epilogue_kwargs.dis_linear_pose = opts.dis_linear_pose
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    if opts.dis_cam_weight==0 and opts.dis_pose_cond:
        c.dis_cond_dim=25


    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = 'training.triplane_autopose.TriPlaneGeneratorPose'
    c.G_kwargs.gen_cond = not opts.empty_gen_cond
    if opts.gen_pose_cond:
        assert opts.empty_gen_cond==False, f"when gen_pose_cond is True, empty_gen_cond must be False"
    if opts.superres or opts.dual_discrimination:
        c.D_kwargs.class_name = 'training.dual_discriminator.DualDiscriminator'
    else:
        c.D_kwargs.class_name = 'training.dual_discriminator.SingleDiscriminator_1'
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = 'antialiased' # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    c.D_kwargs.disc_c_noise = opts.disc_c_noise # Regularization for discriminator pose conditioning
    c.dis_pose_cond = opts.dis_pose_cond

    if opts.superres:
        if opts.cnn == 0:
            if c.training_set_kwargs.resolution == 512:
                sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
            elif c.training_set_kwargs.resolution == 256:
                sr_module = 'training.superresolution.SuperresolutionHybrid4X'
            elif c.training_set_kwargs.resolution == 128:
                sr_module = 'training.superresolution.SuperresolutionHybrid2X'
            else:
                assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"
        elif opts.cnn == 1:
            sr_module = 'training.superresolution.CNN1x1'
        elif opts.cnn == 3:
            sr_module = 'training.superresolution.CNN3x3'
    else:
        sr_module = None

    if opts.sr_module != None:
        sr_module = opts.sr_module
    
    rendering_options = {
        'image_resolution': c.training_set_kwargs.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not opts.gen_pose_cond, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale': opts.c_scale, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': opts.sr_noise_mode, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': opts.density_reg, # strength of density regularization
        'density_reg_p_dist': opts.density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': opts.reg_type, # for experimenting with variations on density regularization
        'decoder_lr_mul': opts.decoder_lr_mul, # learning rate multiplier for decoder
        'sr_antialias': True,
        'flip_to_dis': opts.flip_to_dis,
        'flip_to_disd': opts.flip_to_disd,
        'flip_to_disd_weight': opts.flip_to_disd_weight,
        'flip_type': opts.flip_type, 
        'rot_range_h': opts.rot_range_h, 
        'dis_cam_weight': opts.dis_cam_weight,
        'wrong': opts.wrong,
        'dis_cam_dim': opts.dis_cam_dim,
        'dis_cond_dim': dis_cond_dim,
        'dino_channals': opts.dino_channals,
        'dis_pose_cond': opts.dis_pose_cond,
        'temperature': opts.temperature,
        'temperature_init': opts.temperature_init,
        'temperature_end_kimg': opts.temperature_end_kimg,
        'temperature_start_kimg': opts.temperature_start_kimg,
        'use_intrinsic_label': opts.use_intrinsic_label,
        'use_gt_label_eval': opts.use_gt_label_eval,
        'maskbody': opts.maskbody,
        'h_discrete_num': opts.h_discrete_num,
        'v_discrete_num': opts.v_discrete_num,
        'v_start': opts.v_start,
        'v_end': opts.v_end,
        'fov_deg': opts.create_label_fov,
        'h_mean': opts.h_mean,
        'uniform_sphere_sampling': opts.uniform_sphere_sampling,
        'lambda_cvg_fg': opts.lambda_cvg_fg,
        'lambda_cvg_bg': opts.lambda_cvg_bg,
        'min_cvg_fg': opts.min_cvg_fg,
        'min_cvg_bg': opts.min_cvg_bg,
        'black_bg_dino': opts.black_bg_dino,
        'black_bg': opts.black_bg,
        'use_phase_correlation': opts.use_phase_correlation,
        'online_phase_correlation': opts.online_phase_correlation,
        'use_argmin': opts.use_argmin,
        'shapenet_multipeak': opts.shapenet_multipeak,
        'cache_pose': opts.cache_pose,
        'add_noise_to_angle': opts.add_noise_to_angle,
        'uniform_sampling_test':opts.uniform_sampling_test
    }

    if opts.cfg == 'ffhq':
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 2.25, # near point along each ray to start taking samples.
            'ray_end': 3.3, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
            # 'h_mean': math.pi/2,
        })
    elif opts.cfg == 'elephant':
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 2.25, # near point along each ray to start taking samples.
            'ray_end': 3.3, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0], # used only in the visualizer to control center of camera rotation.
            # 'h_mean': math.pi/2,
        })
    elif opts.cfg == 'lsun':
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 2.25, # near point along each ray to start taking samples.
            'ray_end': 3.3, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0], # used only in the visualizer to control center of camera rotation.
            'given_bg': True,
            'given_bg_dino': True,
            # 'h_mean': math.pi/2,
        })
    elif opts.cfg == 'comp_shapenet':
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            'ray_start': 0.1,
            'ray_end': 2.6,
            'box_warp': 1.6,
            'given_bg': True,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],
            # 'h_mean': math.pi/2,
        })
    elif opts.cfg == 'afhq':
        rendering_options.update({
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 2.25,
            'ray_end': 3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
    elif opts.cfg == 'cats':
        rendering_options.update({
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 2.25,
            'ray_end': 3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, 0],
        })
    elif opts.cfg == 'shapenet':
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            'ray_start': 0.1,
            'ray_end': 2.6,
            'box_warp': 1.6,
            'given_bg': True,
            'given_bg_dino': False,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],
            # 'h_mean': math.pi,
        })
    else:
        assert False, "Need to specify config"



    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.num_fp16_res = 0
    c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32 # Fade out the blur during the first N kimg.

    c.loss_kwargs.gpc_reg_prob = opts.gpc_reg_prob if opts.gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = opts.gpc_reg_fade_kimg
    c.loss_kwargs.dual_discrimination = opts.dual_discrimination # for r1 regularization
    c.loss_kwargs.neural_rendering_resolution_initial = opts.neural_rendering_resolution_initial
    c.loss_kwargs.neural_rendering_resolution_final = opts.neural_rendering_resolution_final
    c.loss_kwargs.neural_rendering_resolution_fade_kimg = opts.neural_rendering_resolution_fade_kimg
    c.G_kwargs.sr_num_fp16_res = opts.sr_num_fp16_res

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=opts.cbase, channel_max=opts.cmax, fused_modconv_default='inference_only')

    c.loss_kwargs.style_mixing_prob = opts.style_mixing_prob

    if opts.activation_cnn is not None:
        c.G_kwargs.sr_kwargs.activation = opts.activation_cnn

    # Augmentation.
    if opts.aug != 'noaug':
        # c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=0, rotate90=0, xint=1, scale=1, rotate=0, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=0, rotate90=0, xint=0, scale=0, rotate=0, aniso=0, xfrac=0, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
            c.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup
        c.resume_kimg = int(opts.resume.split('-')[-1].split('.')[0])

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.G_kwargs.bg_2d_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.bg_2d_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name[:4]:s}-gpu{c.num_gpus:d}-bs{c.batch_size:d}-gm{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run, cache_dir=opts.cache_dir)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
