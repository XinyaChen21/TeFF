# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import math
import cv2
from camera_utils import FOV_to_intrinsics

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        load_rotscale = False,
        black_bg_dino=False,
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self.load_rotscale = load_rotscale
        self.black_bg_dino=black_bg_dino

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels or self.create_label_fov else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx]).astype(np.uint8)
        dino = self._load_raw_dino(self._raw_idx[idx])
        if self.load_rotscale:
            rot_scale = self._load_rot_scale(self._raw_idx[idx])
        else:
            rot_scale = -1

        assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
            dino = dino[:, :, ::-1]
        return image.copy(), dino.copy(), self.get_label(idx), rot_scale

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    # @property
    # def resolution(self):
    #     assert len(self.image_shape) == 3 # CHW
    #     assert self.image_shape[1] == self.image_shape[2]
    #     return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        create_label_fov    = None,
        pad_long = False,
        dino_path=None,
        dino_channals = None,
        rot_scale_path=None,
        black_bg=False,
        shapenet_multipeak=False,
        mask_path='datasets/ffhq/ffhg_sam_bbox.zip',
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._dino_path = dino_path
        self._rot_scale_path = rot_scale_path
        self._zipfile = None
        self.resize = None
        self.create_label_fov = create_label_fov
        self.pad_long = pad_long
        self.dino_channals = dino_channals
        self._mask_path = mask_path
        self._mask_zipfile = None
        self.dataset = os.path.basename(self._path).split('_')[0]
        self.black_bg=black_bg

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        if self.dataset=='comprehensive':
            valid_file = open(os.path.join(os.path.dirname(path), "valid_pc_file.txt"), 'r')
            self._image_fnames = sorted('comprehensive_cars/'+fname.strip() for fname in valid_file.readlines())
        elif self.dataset=='cars' and shapenet_multipeak:
            valid_file = open(os.path.join(os.path.dirname(path), "multi_peak_v.txt"), 'r')
            self._image_fnames = sorted(fname.strip() for fname in valid_file.readlines() if fname!='')
        elif self.dataset=='plane':
            valid_file = open(os.path.join(os.path.dirname(path), "valid_plane_rot_scale_3v_v2.txt"), 'r')
            self._image_fnames = sorted(fname.strip() for fname in valid_file.readlines())
        else:
            self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION and not fname.endswith('_depth.png') and not fname.endswith('_mask.png') and not fname.endswith('_normals.png'))
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        # self.dino_features = np.load('/nas3/vilab/xychen/dino_feature/ffhq/dino_dinov2_vitb14.npy')

        self._dino_zipfile = None
        if self.dataset=='comprehensive':
            self._dino_dir = 'compcars_dinov1_stride4_pca16'
        else:
            self._dino_dir = ''
        self._rot_scale_zipfile = None

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is None:
            self.resize = False
            self.resolution_req = raw_shape[2]
        else:
            self.resolution_req = resolution
            if resolution == raw_shape[2] or resolution == raw_shape[3]:
                self.resize = False
            else:
                self.resize = True
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile
    
    def _get_dino_zipfile(self):
        assert self._type == 'zip'
        if self._dino_zipfile is None:
            self._dino_zipfile = zipfile.ZipFile(self._dino_path)
        return self._dino_zipfile
    
    def _get_rot_scale_zipfile(self):
        assert self._type == 'zip'
        if self._rot_scale_zipfile is None:
            self._rot_scale_zipfile = zipfile.ZipFile(self._rot_scale_path)
        return self._rot_scale_zipfile
    
    def _get_mask_zipfile(self):
        assert self._type == 'zip'
        if self._mask_zipfile is None:
            self._mask_zipfile = zipfile.ZipFile(self._mask_path)
        return self._mask_zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
            if self._dino_zipfile is not None:
                self._dino_zipfile.close()
            if self._mask_zipfile is not None:
                self._mask_zipfile.close()   
            if self._rot_scale_zipfile is not None:
                self._rot_scale_zipfile.close()   
        finally:
            self._zipfile = None
            self._dino_zipfile = None
            self._mask_zipfile = None
            self._rot_scale_zipfile = None
    
    @property
    def resolution(self):
        return self.resolution_req

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None, _mask_zipfile=None, _dino_zipfile=None, _rot_scale_zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if self.black_bg:
            fname = self._image_fnames[raw_idx]
            mask_name = os.path.join(fname.split('.')[0]+'_mask.png')
            if self.dataset=='plane':
                with self._get_mask_zipfile().open(mask_name, 'r') as f:
                    if pyspng is not None and self._file_ext(fname) == '.png':
                        mask = pyspng.load(f.read())
                    else:
                        mask = np.array(PIL.Image.open(f))
            else:
                with self._open_file(mask_name) as f:
                    if pyspng is not None and self._file_ext(fname) == '.png':
                        mask = pyspng.load(f.read())
                    else:
                        mask = np.array(PIL.Image.open(f))

            mask = mask 
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)// 255
            if len(mask.shape)==2:
                mask=mask[:,:,None]
            bgcolor = (np.zeros(3)).astype('uint8')
            image = image * mask + (1-mask) * bgcolor
        if self.create_label_fov is not None:
            if self.pad_long:
                image = pad_long_image(image)
            else:
                image = crop_short_image(image)
        if self.resize:
            img = PIL.Image.fromarray(image, 'RGB') # type of image need be uint8
            img = img.resize((self.resolution_req, self.resolution_req), PIL.Image.LANCZOS)
            image = np.array(img)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    
    def _load_rot_scale(self, raw_idx):
        if self.dataset=='elephant':
            fname = os.path.join(self._image_fnames[raw_idx].split('.')[0]+'.npy')
        elif self.dataset=='plane':
            fname = os.path.join('plane_rot_scale_3v_v2', os.path.basename(self._image_fnames[raw_idx].split('.')[0]+'.png.npy'))
        else:
            # fname = os.path.join(os.path.basename(self._image_fnames[raw_idx].split('.')[0]+'.npy'))
            fname = os.path.join(os.path.basename(self._rot_scale_path).replace('.zip',''), os.path.basename(self._image_fnames[raw_idx].split('.')[0]+'.npy'))
        with self._get_rot_scale_zipfile().open(fname, 'r') as f:
            rot_scale = np.load(f)
        return rot_scale


    def _load_raw_dino(self, raw_idx):
        fname = self._image_fnames[raw_idx].split('.')[0]+'.npy'
        if self.dataset=='comprehensive':
            fname = os.path.join(self._dino_dir,os.path.basename(fname))
        else:
            fname = os.path.join(self._dino_dir,fname)
        with self._get_dino_zipfile().open(fname, 'r') as f:
            image = np.load(f)
        
        if self.black_bg_dino:
            fname = self._image_fnames[raw_idx]
            mask_name = os.path.join(fname.split('.')[0]+'_mask.png')
            if self.dataset=='plane':
                with self._get_mask_zipfile().open(mask_name, 'r') as f:
                    if pyspng is not None and self._file_ext(fname) == '.png':
                        mask = pyspng.load(f.read())
                    else:
                        mask = np.array(PIL.Image.open(f))
            else:
                with self._open_file(mask_name) as f:
                    if pyspng is not None and self._file_ext(fname) == '.png':
                        mask = pyspng.load(f.read())
                    else:
                        mask = np.array(PIL.Image.open(f))

            mask = mask 
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)// 255
            if len(mask.shape)==2:
                mask=mask[:,:,None]
            else:
                mask=mask[:,:,:1]
            image = image * mask 


        if self.create_label_fov is not None:
            if self.pad_long:
                image = pad_long_image(image)
            else:
                image = crop_short_image(image)

        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:self.dino_channals]

        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            if self.create_label_fov is not None:
                temp = np.zeros([self._raw_shape[0], 16], dtype=np.float32)
                focal_length = float(1 / (math.tan(self.create_label_fov * 3.14159 / 360) * 1.414))
                intrinsics = np.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
                intrinsics = np.reshape(intrinsics, (-1, 9))
                intrinsics = np.repeat(intrinsics, self._raw_shape[0], axis=0)
                labels = np.concatenate([temp, intrinsics], axis=1)
                labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
                return labels
            else:
                return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            if self.create_label_fov is not None:
                temp = np.zeros([self._raw_shape[0], 16], dtype=np.float32)
                focal_length = float(1 / (math.tan(self.create_label_fov * 3.14159 / 360) * 1.414))
                intrinsics = np.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
                intrinsics = np.reshape(intrinsics, (-1, 9))
                intrinsics = np.repeat(intrinsics, self._raw_shape[0], axis=0)
                labels = np.concatenate([temp, intrinsics], axis=1)
                labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
                return labels
            else:
                return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------


def crop_short_image(image):
    """Crops a square patch and then resizes it to the given size.

    Args:
        image: The input image to crop and resize.
        size: An integer, indicating the target size.

    Returns:
        An image with target size.

    Raises:
        TypeError: If the input `image` is not with type `numpy.ndarray`.
        ValueError: If the input `image` is not with shape [H, W, C].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Input image should be with type `numpy.ndarray`, '
                        f'but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [H, W, C], '
                         f'but `{image.shape}` is received!')

    height, width, channel = image.shape
    short_side = min(height, width)
    image = image[(height - short_side) // 2:(height + short_side) // 2,
                  (width - short_side) // 2:(width + short_side) // 2]
    # if channel == 3:
    #     pil_image = PIL.Image.fromarray(image)
    #     pil_image = pil_image.resize((size, size), PIL.Image.ANTIALIAS)
    #     image = np.asarray(pil_image)
    # elif channel == 1:
    #     image = cv2.resize(image, (size, size))
    #     if image.ndim == 2:
    #         image = image[:,:,np.newaxis]
    return image

def pad_long_image(image):
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Input image should be with type `numpy.ndarray`, '
                        f'but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [H, W, C], '
                         f'but `{image.shape}` is received!')

    height, width, channel = image.shape
    long_side = max(height, width)
    image_new = np.zeros((long_side, long_side, channel))+255.0
    image_new = image_new.astype(np.uint8)
    image_new[(long_side-height)//2:(long_side+height)//2, (long_side-width)//2:(long_side+width)//2] = image
    # image = image[(height - short_side) // 2:(height + short_side) // 2,
                #   (width - short_side) // 2:(width + short_side) // 2]
    return image_new

def centercrop(img, size):
    crop_height, crop_width = size, size
    img_height, img_width = img.shape[:2]

    y1 = max(0, int(round((img_height - crop_height) / 2.)))
    x1 = max(0, int(round((img_width - crop_width) / 2.)))
    y2 = min(img_height, y1 + crop_height) - 1
    x2 = min(img_width, x1 + crop_width) - 1

    # crop the image
    img = img[y1:y2, x1:x2]
    return img