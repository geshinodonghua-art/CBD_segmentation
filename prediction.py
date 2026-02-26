# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:53:50 2026

@author: owner
"""
import os
import torch
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from DataLoader import DataLoad
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

model_path = r"C:\DL\best_metric_model.pth"
save_dir = r"C:\DL"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(
    spatial_dims = 3,
    in_channels = 1,
    out_channels = 1,
    channels = (16,32,64,128,256),
    strides = (2,2,2,2),
    num_res_units = 2,
    norm = Norm.BATCH,
    ).to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

_, _, test_loader = DataLoad()

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        images = test_data["image"].to(device)
        masks = test_data["mask"].to(device)
        
        output = sliding_window_inference(
            inputs=images, 
            roi_size=(96, 96, 96), 
            sw_batch_size=4, 
            predictor=model
        )
        pred = torch.argmax(torch.softmax(output, dim = 1), dim = 1, keepdim=True)
        
        slice_idx = 38
        plt.figure(figsize=(12, 12))
        plt.subplot(1, 3, 1); plt.title("Original Image"); plt.imshow(images[0, 0, slice_idx, :, :].cpu(), cmap="gray")
        plt.subplot(1, 3, 2); plt.title("Teacher Mask"); plt.imshow(masks[0, 0, slice_idx, :, :].cpu(), cmap="jet")
        plt.subplot(1, 3, 3); plt.title("AI Prediction"); plt.imshow(pred[0, 0, slice_idx, :, :].cpu(), cmap="jet")
        
        save_path = os.path.join(save_dir, f"result_{i:03d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{test_data['pt_id']}")
        plt.show()
        
        
        if i == 0: break