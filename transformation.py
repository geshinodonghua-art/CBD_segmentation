# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 20:06:41 2026

@author: owner
"""

from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityd, 
    RandCropByPosNegLabeld, 
    RandRotate90d, 
    CopyItemsd,
    MedianSmoothd,
    ToTensord
)
from monai.data import Dataset
from create_dictionary import create_dictionary


def transform():
    train_file, val_file, test_file, _ = create_dictionary(
        r"",
        r"",
        r"",
        r"")
    
    
    train_trans = Compose([
        LoadImaged(keys = ["image","mask"]),
        EnsureChannelFirstd(keys = ["image","mask"],channel_dim='no_channel'),
        ScaleIntensityd(keys = ["image","mask"]),
        RandCropByPosNegLabeld(
            keys = ["image","mask"],
            label_key="mask",
            image_key = "image",
            spatial_size = (32,128,128),
            pos = 4,
            neg = 1,
            num_samples = 4
            ),
        MedianSmoothd(keys=["image"], radius=1),
        #spatialは(height,width,depth)で0,1ならx-y平面で回転したりする（反転させたり）
        #90度単位での回転(90,180,・・・)１度単位だとvoxelの再計算で画像がボケる恐れがある
        #RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=[1, 2]),
        ToTensord(keys=["image", "mask"])
        ])
    
    test_trans =  Compose([
        LoadImaged(keys = ["image","mask"]),
        CopyItemsd(keys=["image"], times=1, names=["orig_image"]),
        EnsureChannelFirstd(keys = ["image","mask"],channel_dim='no_channel'),
        MedianSmoothd(keys=["image"], radius=1),
        ScaleIntensityd(keys = ["image","mask"]),
        ToTensord(keys = ["image","mask","orig_image"])
        ])
    
    train_data = Dataset(data = train_file, transform = train_trans)
    val_data = Dataset(data = val_file, transform = test_trans)
    test_data = Dataset(data = test_file, transform = test_trans)
    
    return train_data, val_data, test_data
