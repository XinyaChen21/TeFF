from . import metric_utils

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import copy
import torch
import dnnlib
import numpy as np
np.random.seed(seed=10086)
import cv2
import os


def compute_depth_for_generator(save_dir, batch_size=64, batch_gen=None, max_items=10000, **kwargs):
    opts = metric_utils.MetricOptions(**kwargs)

    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=batch_gen)

    # Main loop.
    cur_items = 0
    images = []
    images_depth = []
    while cur_items < max_items:
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, G.z_dim*2], device=opts.device)
            results = G(z=z, c=next(c_iter), **opts.G_kwargs)
            img = results['image']
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            img_depth = results['image_depth'].detach().cpu().numpy()
            images.append(img)
            images_depth.append(img_depth)
        cur_items += batch_size
    images = np.concatenate(images).transpose(0,2,3,1)
    images_depth = np.concatenate(images_depth).transpose(0,2,3,1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx in range(images.shape[0]):
        cv2.imwrite(os.path.join(save_dir,f'{idx}.png'),images[idx][:,:,::-1])
        np.save(os.path.join(save_dir,f'{idx}_depth.npy'),images_depth[idx])

    return 