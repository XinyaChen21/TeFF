from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import copy
from unet.pytorch_DPCN import FFT2, UNet, LogPolar, PhaseCorr, Corr2Softmax
from data.dataset_DPCN import *
import numpy as np
import shutil
from utils.utils import *
import kornia
from data.dataset import *
from utils.detect_utils import *
from shutil import copyfile


def detect_model(template_path, source_path, model_template, model_source, model_corr2softmax,\
             model_trans_template, model_trans_source, model_trans_corr2softmax, use_unet=True):
    batch_size_inner = 1

    since = time.time()

    # Each epoch has a training and validation phase
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    phase = "val"

    model_template.eval()   # Set model to evaluate mode
    model_source.eval()
    model_corr2softmax.eval()
    model_trans_template.eval()
    model_trans_source.eval()
    model_trans_corr2softmax.eval()
    best_mse_loss = 10086
 
    source_dino_list = []
    source_list =[]
    for i in range(72, 7*36):
        source_path = os.path.join(f'/home/chenxinya/eg3d_pose_clean/training-runs/00435-comp-comp-gpu8-bs32-gm1-DC1-ali0.0-pc3-2D-bgTrue-sed10086-unfFalse-lvrFalse-vth0.0-gpcFalse-rdalFalse-itgrl0.0-itth0.0-flpTrue-lktfvFalse-dc2-avgFalse-dnTrue-sel-mulFalse-tem1.0t10.0-h1.0-DpFalse-dnspFalse-dtchFalse-var0.0-min0.01-tv0.0-probFalse/network-snapshot-001000_{i}.npy')
        source_dino, _, _, _, _, _ = default_loader(source_path, 256, mask=False,gray=False,channal_num=3)
        source, _, _, _, _, _ = default_loader(source_path, 256, mask=False,gray=True)
        source_dino_list.append(source_dino[None])
        source_list.append(source)
    source_dino = torch.cat(source_dino_list, dim=0)
    source = torch.cat(source_list, dim=0)
    with torch.no_grad():

        iters = 0
        acc = 0.
        template_dino, _, _, _, _, _ = default_loader(template_path, 256,  mask=True,gray=False,channal_num=3)
        template, _, _, _, _, _ = default_loader(template_path, 256,  mask=True,gray=True)

        template = template.to(device)
        source = source.to(device)
        template = template.unsqueeze(0)
        template = template.permute(1,0,2,3)
        source = source.unsqueeze(0)
        source = source.permute(1,0,2,3)

        iters += 1    
        since = time.time()
        rotation_cal, scale_cal = detect_rot_scale(template, source,\
                                    model_template, model_source, model_corr2softmax, device, use_unet=use_unet )
        # tranformation_y, tranformation_x, image_aligned, source_rotated = detect_translation(template, source, rotation_cal, scale_cal, \
        #                                     model_trans_template, model_trans_source, model_trans_corr2softmax, device, use_unet=use_unet)
        time_elapsed = time.time() - since
        # print('in detection time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        b, c, h, w = source.shape
        center = torch.ones(b,2).to(device)
        center[:, 0] = h // 2
        center[:, 1] = w // 2
        angle_rot = torch.ones(b).to(device) * (-rotation_cal.to(device))
        scale_rot = torch.ones(b).to(device) * (1/scale_cal.to(device))
        rot_mat = kornia.get_rotation_matrix2d(center, angle_rot, scale_rot)
        source_rotated = kornia.warp_affine(source.to(device), rot_mat, dsize=(h, w))
        source_rotated_dino = kornia.warp_affine(source_dino.to(device), rot_mat, dsize=(h, w))
    
        print("in detection time", time_elapsed)
        mse_loss = torch.mean(((source_rotated_dino-template_dino.to(device))**2).reshape(b,-1), dim=-1)
        mse_loss_orimg = torch.mean(((source_dino-template_dino)**2).reshape(b,-1), dim=-1)

        best_tempalate_idx = torch.min(mse_loss,dim=0)[1]
        best_tempalate_idx_orimg = torch.min(mse_loss_orimg,dim=0)[1]

        image_aligned = align_image(template[0,:,:], source_rotated[best_tempalate_idx,:,:])
        plot_and_save_result(template[0,:,:], source[best_tempalate_idx,:,:], source_rotated[best_tempalate_idx,:,:], image_aligned, save_path=os.path.join('demo', f'network-snapshot-001000_{best_tempalate_idx}.png'))

        image_aligned = align_image(template[0,:,:], source_rotated[best_tempalate_idx_orimg,:,:])
        plot_and_save_result(template[0,:,:], source[best_tempalate_idx_orimg,:,:], source_rotated[best_tempalate_idx_orimg,:,:], image_aligned, save_path=os.path.join('demo', f'network-snapshot-001000_{best_tempalate_idx_orimg}.png'))

    return best_tempalate_idx, best_tempalate_idx_orimg





checkpoint_path = "./checkpoints/checkpoint_simulation_hetero.pt"
template_path = "./demo/temp_1.jpg"
source_path = "./demo/src_1.jpg"

load_pretrained =True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The devices that the code is running on:", device)
# device = torch.device("cpu")
batch_size = 1
num_class = 1
start_epoch = 0
model_template = UNet(num_class).to(device)
model_source = UNet(num_class).to(device)
model_corr2softmax = Corr2Softmax(200., 0.).to(device)
model_trans_template = UNet(num_class).to(device)
model_trans_source = UNet(num_class).to(device)
model_trans_corr2softmax = Corr2Softmax(200., 0.).to(device)

optimizer_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
optimizer_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
optimizer_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)
optimizer_trans_ft_temp = optim.Adam(filter(lambda p: p.requires_grad, model_template.parameters()), lr=2e-4)
optimizer_trans_ft_src = optim.Adam(filter(lambda p: p.requires_grad, model_source.parameters()), lr=2e-4)
optimizer_trans_c2s = optim.Adam(filter(lambda p: p.requires_grad, model_corr2softmax.parameters()), lr=1e-1)

if load_pretrained:
    model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
    _, _, _, _, _, _,\
        start_epoch = load_checkpoint(\
                                    checkpoint_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,\
                                    optimizer_ft_temp, optimizer_ft_src, optimizer_c2s, optimizer_trans_ft_temp, optimizer_trans_ft_src, optimizer_trans_c2s, device)

# for i in range(1):
#     # template_path = os.path.join('/nas3/vilab/xychen/dino_feature/comprehensive_cars',str(i).zfill(10)+'.jpg')
#     # source_path = os.path.join('/nas3/vilab/xychen/dino_feature/comprehensive_cars',str(i+100).zfill(10)+'.jpg')
#     template_path = os.path.join('/nas3/vilab/xychen/dino_feature/comprehensive_cars',str(100).zfill(10)+'.jpg')
#     source_path = os.path.join('/nas3/vilab/xychen/dino_feature/comprehensive_cars',str(6+100).zfill(10)+'.jpg')
#     # template_path = os.path.join('/nas3/vilab/xychen/dino_feature/comprehensive_cars',str(28).zfill(10)+'.jpg')
#     # source_path = os.path.join('/nas3/vilab/xychen/dino_feature/comprehensive_cars',str(28+100).zfill(10)+'.jpg')
#     if not os.path.exists(template_path) or not os.path.exists(source_path):
#         continue
#     detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax,use_unet=False)
        
best_image = []
for idx in range(50):
    # template_path = os.path.join(f'/home/chenxinya/eg3d_pose_clean/training-runs/00410-comp-comp-gpu8-bs32-gm1-DC1-ali0.0-pc3-2D-bgTrue-sed10086-unfFalse-lvrFalse-vth0.0-gpcFalse-rdalFalse-itgrl0.0-itth0.0-flpTrue-lktfvFalse-dc2-avgFalse-dnTrue-sel-mulFalse-tem1.0t10.0-h1.0-DpFalse-dnspFalse-dtchFalse-var0.0-min0.01-tv0.0/network-snapshot-002000_{i}.npy')
    source_path = None
    template_path = os.path.join('/nas3/vilab/xychen/dino_feature/compcars_dinov1_stride4_pca16',str(idx).zfill(10)+'.npy')

    if not os.path.exists(template_path):
        continue
    best_tempalate_idx, best_tempalate_idx_orimg=detect_model(template_path, source_path, model_template, model_source, model_corr2softmax, model_trans_template, model_trans_source, model_trans_corr2softmax, use_unet=False)

    img=cv2.imread(f'demo/network-snapshot-001000_{best_tempalate_idx}.png')
    img_ori=cv2.imread(f'demo/network-snapshot-001000_{best_tempalate_idx_orimg}.png')
    best_image.append(np.hstack([img,img_ori]))
cv2.imwrite(f'demo/best_result.png',np.vstack(best_image))



                                     





