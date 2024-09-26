#bin/bash

ln -s /YOUR/DATASET/PATH datasets
ln -s /YOUR/OUTPUT/PATH training-runs

python train.py  --outdir=training-runs --cfg=shapenet --data=datasets/in_the_wild/cars_128.zip \
  --gpus=8 --batch=32 --gamma=0.3 --gen_pose_cond=false --dis_pose_cond=True --dis_cam_weight=2 \
  --dataset_resolution=128    --dis_linear_pose=False \
  --gpc_reg_prob=0.5 \
  --flip_to_dis=true --flip_to_disd=true  --gpc_reg_fade_kimg=1000 \
  --neural_rendering_resolution_final 64 \
  --dino_data=datasets/in_the_wild/shapenetcars_dinov1_stride4_pca16_nomask_5k.zip --dino_channals 3\
  --bg_modeling_2d=true --seed 10086  --temperature=100.0 \
  --use_intrinsic_label=True --maskbody=False \
  --v_start=0 --v_end=1 --v_discrete_num=18 --uniform_sphere_sampling=True \
  --h_discrete_num=36 --h_mean=3.1415926 --flip_type=flip_both_shapenet \
  --dis_cam_dim=2 --lambda_cvg_fg=100 \
  --temperature_init=10.0 --temperature_start_kimg=1500 --temperature_end_kimg=2500 \
  --shapenet_multipeak=True \
  --create_label_fov=69.1882