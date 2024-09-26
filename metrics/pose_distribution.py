from . import metric_utils
from camera_utils import LookAtPose, FOV_to_intrinsics, LookAtPoseSampler
from training.dual_discriminator import filtered_resizing
from torch_utils.ops import upfirdn2d
import math

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import copy
import torch
import dnnlib
import numpy as np
import math
import os
np.random.seed(seed=10086)
import cv2

def visualize_axis(RTs, radius=10.0, vertices=None, save_path='poses.png'):
   """
   params:
   RTs: camera poses, Nx4x4
   radius: control the side lens of the axes for each camera
   vertices: optional, for visualization of the points
   """
   fig = plt.figure()
   ax_3d = fig.add_subplot(1, 1, 1, projection='3d')
   ax_3d.grid(False)
   ax_3d.set_xlabel('X')
   ax_3d.set_ylabel('Y')
   ax_3d.set_zlabel('Z')

   radius = radius
   sphere = np.random.randn(3, 100)
   sphere = radius * sphere / np.linalg.norm(sphere, axis=0, keepdims=True)
   if vertices is None:
       ax_3d.scatter(*sphere, c='k', alpha=0.1)  # random points
   else:
       for vertex in vertices:
           ax_3d.scatter(*vertex.T, alpha=0.1)

   s = 0.1 * radius
   for RT in RTs:
       R = RT[:3, :3]
       T = RT[:3, 3]
       e1, e2, e3 = s * R.transpose(1, 0) + T.reshape(1, 3)
       ax_3d.plot(*np.stack([e1, T], axis=1), c='r')  # a line connecting point e1 and T, red
       ax_3d.plot(*np.stack([e2, T], axis=1), c='g')
       ax_3d.plot(*np.stack([e3, T], axis=1), c='b')
   ax_3d.set_xlim(-5, 5)  # set the axis limits
   ax_3d.set_ylim(-5, 5)
   # plt.show()
   plt.savefig(save_path)

class PoseDistribution():
    def __init__(self, max_items=1000):
        self.gt_h = None
        self.gt_v = None
        self.max_items = max_items


    def compute_pose_distribution_for_dataset(self, save_path, batch_size=64, data_loader_kwargs=None, **kwargs):#stats_kwargs
        max_items = self.max_items
        opts = metric_utils.MetricOptions(**kwargs)
        
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        if data_loader_kwargs is None:
            data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

        num_items = len(dataset)
        if max_items is not None:
            num_items = min(num_items, max_items)

        # Main loop.
        X = []
        Y = [] 
        Z = []
        cam2worlds = []
        h = []
        v = []
        item_subset = np.random.randint(0, len(dataset), num_items).tolist() #[i for i in range(num_items)]
        for images, dinos, _labels, rot_scale in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
            cam2world = _labels[:,:16].reshape(-1,4,4).detach().cpu().numpy()
            if dataset.dataset=='cars':
                cam2world_new = copy.deepcopy(cam2world)
                cam2world_new[:,1] = cam2world[:,2]
                cam2world_new[:,2] = -cam2world[:,1]
                cam2world=cam2world_new
            cam2worlds.append(cam2world)

            rr=R.from_matrix(cam2world[:,:3,:3])
            euler = rr.as_euler('YXZ', degrees=False)
            Y += euler[:,0].tolist()
            X += euler[:,1].tolist()
            Z += euler[:,2].tolist()

            trans = cam2world[:,:3,3:][...,0]
            trans_norm = np.linalg.norm(trans, axis=-1)
            phi = np.arccos(trans[:,1]/trans_norm)
            theta = np.pi-(np.arccos(trans[:,0]/(trans_norm*np.sin(phi))) * np.sign(trans[:,2]))
            v += phi.tolist()
            h += theta.tolist()

        cam2worlds = np.concatenate(cam2worlds, axis=0)  # (N, 4, 4)
        visualize_axis(cam2worlds, radius=2.0, save_path=save_path)  # visualize camera poses with rgb axes 

        plt.figure(figsize=(8,6), dpi=80)

        plt.subplot(1, 2, 1)
        plt.hist(h, bins=30, color='skyblue', alpha=0.8)
        plt.title('gt horizion')

        plt.subplot(1, 2, 2)
        plt.hist(v, bins=30, color='skyblue', alpha=0.8)
        plt.title('gt vertical')

        plt.savefig(save_path.replace('pose.png', 'pose_hv.png'))

        self.gt_h = h
        self.gt_v = v

        return 


    def compute_pose_distribution_for_generator(self, save_path, batch_size=64, batch_gen=None, **kwargs):
        max_items = self.max_items
        opts = metric_utils.MetricOptions(**kwargs)

        if batch_gen is None:
            batch_gen = min(batch_size, 4)
        assert batch_size % batch_gen == 0

        # Setup generator and labels.
        G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
        # c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=batch_gen)
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
        # item_subset = [np.random.randint(low=0,high=len(dataset)) for i in range((max_items - 1) // opts.num_gpus + 1)]
        item_subset = [np.random.randint(low=0,high=len(dataset)) for i in range(max_items)]

        device = opts.device
        h_discrete_num = G.rendering_kwargs.get('h_discrete_num', 36)
        v_discrete_num = G.rendering_kwargs.get('v_discrete_num', 1)
        v_start = G.rendering_kwargs.get('v_start', 0)
        v_end = G.rendering_kwargs.get('v_end', 0)
        h_mean = G.rendering_kwargs.get('h_mean', 1.570796)


        gt_h = []
        gt_v = []
        pred_h_min = []
        pred_v_min = []
        pred_h_pdf = []
        pred_v_pdf = []
        all_pdf = []

        num_items=max_items
        batch_size = batch_gen
        item_subset = np.random.randint(0, len(dataset), num_items).tolist()
        for image, dino_image, _labels, real_rotscale in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, pin_memory=True, prefetch_factor=2, num_workers=1):
            batch_size = image.shape[0]
            image = image.to(device).to(torch.float32) / 127.5 - 1
            dino_image = dino_image.to(device).to(torch.float32) * 2 - 1
            real_rotscale = real_rotscale.to(opts.device)
            image_raw = filtered_resizing(image, size=G.neural_rendering_resolution, f=upfirdn2d.setup_filter([1,3,3,1], device=device), filter_mode='antialiased')
            dino_image = filtered_resizing(dino_image, size=G.neural_rendering_resolution, f=upfirdn2d.setup_filter([1,3,3,1], device=device), filter_mode='antialiased')
            real_img = {'image': image, 'image_raw': image_raw}
            

            if G.rendering_kwargs.get('maskbody', True):
                rgb=(dino_image.permute(0,2,3,1) * 127.5 + 128).clip(0, 255)
                mask_1 = torch.norm(rgb-torch.tensor([[ 99,  99, 114]]).to(device).to(torch.float32),dim=-1)<60
                mask_2 = torch.norm(rgb-torch.tensor([[144, 186, 208]]).to(device).to(torch.float32),dim=-1)<60
                mask = mask_1 #| mask_2
                mask = mask[:,None,:,:].repeat((1,3,1,1))
                dino_image[mask]=1
            
            if G.rendering_kwargs.get('use_phase_correlation', False):
                if G.rendering_kwargs.get('online_phase_correlation', False):
                    template_360_rotated_dino, angle_rot, scale_rot = G.solve_rs.rotate_scale_template(dino_image, G.template_360_dino, G.template_logpolar, G.template_logbase_rot, G.model_corr2softmax, device)
                    error = torch.mean(((dino_image-template_360_rotated_dino)**2).reshape(G.template_360_dino.shape[0],batch_size,-1),dim=-1)
                else:
                    error = real_rotscale[:,:v_discrete_num*h_discrete_num,0].permute(1,0)
                    angle_rot = real_rotscale[:,:v_discrete_num*h_discrete_num,1].permute(1,0)
                    scale_rot = real_rotscale[:,:v_discrete_num*h_discrete_num,2].permute(1,0)
            else:
                error = torch.mean(((dino_image-G.template_360_dino)**2).reshape(len(G.template_360_dino),batch_size,-1),dim=-1)
                
            temperature=G.temperature
            pdf = torch.nn.functional.softmax(input= -error*temperature, dim=0)
            cdf = torch.cumsum(pdf, dim=0)
            uu = torch.rand(size = (batch_size,1)).to(device)
            all_pdf.append(pdf.permute(1,0))

            best_indices = torch.searchsorted(cdf.permute(1,0), uu, right=True)[:,0]
            best_indices = torch.clamp(best_indices, min=0, max=cdf.shape[0]-1)
            best_h_angle = G.all_parameters[best_indices][:,1] 
            angle_p = G.all_parameters[best_indices][:,2] 
            pred_h_pdf += best_h_angle.detach().cpu().numpy().tolist()
            pred_v_pdf += angle_p.detach().cpu().numpy().tolist()
            
            best_indices_by_min = torch.min(error,dim=0)[1]
            best_h_angle = G.all_parameters[best_indices_by_min][:,1] 
            angle_p = G.all_parameters[best_indices_by_min][:,2] 
            pred_h_min += best_h_angle.detach().cpu().numpy().tolist()
            pred_v_min += angle_p.detach().cpu().numpy().tolist()
            
            if dataset._use_labels:
                cam2world = _labels[:,:16].reshape(-1,4,4).detach().cpu().numpy()
                if dataset.dataset=='cars':
                    cam2world_new = copy.deepcopy(cam2world)
                    cam2world_new[:,1] = cam2world[:,2]
                    cam2world_new[:,2] = -cam2world[:,1]
                    cam2world=cam2world_new
                trans = cam2world[:,:3,3:][...,0]
                trans_norm = np.linalg.norm(trans, axis=-1)
                phi = np.arccos(trans[:,1]/trans_norm)
                theta = np.pi-(np.arccos(trans[:,0]/(trans_norm*np.sin(phi))) * np.sign(trans[:,2]))
                gt_v += phi.tolist()
                gt_h += theta.tolist()
                focal_length = _labels[:,16].detach().cpu().numpy()

        
        plt.figure(figsize=(4*2,6), dpi=80)
        plt.subplot(1, 2, 1)
        if dataset._use_labels:
            plt.hist(gt_h, bins=24, color='skyblue', alpha=0.8, range=(0,2*np.pi), label='gt_h')
        plt.hist(pred_h_min, bins=24, color='orange', alpha=0.5, range=(0,2*np.pi), label='pred_h_min')
        plt.hist(pred_h_pdf, bins=24, color='red', alpha=0.4, range=(0,2*np.pi), label='pred_h_pdf')
        plt.title('horizion')
        plt.legend(loc=1, prop = {'size':6})

        #----------------v
        plt.subplot(1, 2, 2)
        if dataset._use_labels:
            plt.hist(gt_v, bins=12, color='skyblue', alpha=0.8, range=(0,np.pi), label='gt_v')
        plt.hist(pred_v_min, bins=12, color='orange', alpha=0.5, range=(0,np.pi), label='pred_v_min')
        plt.hist(pred_v_pdf, bins=12, color='red', alpha=0.4, range=(0,np.pi), label='pred_v_pdf')
        plt.title('vertical')
        plt.legend(loc=1, prop = {'size':6})


        plt.savefig(save_path)

        return
    
    def compute_pose_distribution(self, batch_size=4, batch_gen=4, **kwargs):
        opts = metric_utils.MetricOptions(**kwargs)
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2, drop_last=False)
        G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

        # Initialize.
        num_items = len(dataset)

        # Main loop.
        item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
        
        
        idx = 0
        ori_batch_size=batch_size
        save_dir = f'datasets/dino_feature/{dataset.dataset}_rot_scale_3v/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for images, dinos, _labels, real_rotscale in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=ori_batch_size, **data_loader_kwargs):
            batch_size = dinos.shape[0]
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            dinos = dinos.to(opts.device)
            dinos = 2*dinos - 1#[-1,1]

            device = opts.device
            if G.rendering_kwargs.get('use_phase_correlation', False):
                template_360_dino=G.template_360_dino
                template_360_rotated_dino, angle_rot, scale_rot = G.solve_rs.rotate_scale_template(dinos, template_360_dino, G.template_logpolar, G.template_logbase_rot, G.model_corr2softmax, device)
                dinos = torch.nn.functional.interpolate(dinos, size=(64, 64), mode='bilinear', align_corners=False)
                error = torch.mean(((dinos[None]-template_360_rotated_dino)**2).reshape(template_360_rotated_dino.shape[0],batch_size,-1),dim=-1)
            else:
                error = torch.mean(((dinos[None]-G.template_360_dino)**2).reshape(G.template_360_dino.shape[0],batch_size,-1),dim=-1)
            rot_scale = torch.stack([error, angle_rot, scale_rot], dim=-1)
            for i in range(batch_size):
                filename=os.path.basename(dataset._image_fnames[item_subset[idx*ori_batch_size+i]].replace('jpg','npy'))
                path=f'{os.path.join(save_dir,filename)}'
                np.save(path,rot_scale[:,i].detach().cpu().numpy())
            idx += 1
            
        return 

import numpy as np 
import scipy

def KL2(P, Q):
    return np.sum(P * np.log(P / Q))


def KL_divergence(gt, pred, is_h=False):
    if isinstance(gt, list):
        gt = np.array(gt)
    if isinstance(pred, list):
        pred = np.array(pred)    
    if is_h:
        num_bins = int(24)
        range_max = 2
    else:
        num_bins = int(12)
        range_max = 1

    gt_counts, gt_bins = np.histogram(gt, bins=num_bins, range=(0, np.pi*range_max))
    gt_counts = gt_counts / gt_counts.sum()
    gt_counts = np.clip(gt_counts, 1e-20, np.inf)

    pred_counts, pred_bins = np.histogram(pred, bins=num_bins, range=(0, np.pi*range_max))
    pred_counts = pred_counts / pred_counts.sum()
    pred_counts = np.clip(pred_counts, 1e-20, np.inf)

    assert (gt_bins == pred_bins).all()
    # print(f'KL gt pred: {KL2(gt_counts, pred_counts)}')
    # print(f'KL pred gt: {KL2(pred_counts, gt_counts)}')
    
    return KL2(gt_counts, pred_counts)

    