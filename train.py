# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 11:33:05 2026

@author: kadot
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from DataLoader import DataLoad
from monai.losses import DiceLoss
import torch
from torch.optim import Adam
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
import wandb
wandb.login(key ="")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="bile-duct-segmentation", name="balanced-sampling-run")
    
    train_loader, val_loader, test_loader = DataLoad()
    
    model = Unet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 1,
        channels = (16,32,64,128,256),
        strides = (2,2,2,2),
        num_res_units = 2,
        norm = Norm.BATCH,
        ).to(device)
    
    loss_fnc = DiceLoss(sigmoid=True)
    optimizer = Adam(model.parameters(), lr = 1e-3)
    #0.5以上で１にする（変更可能）
    post_pred = AsDiscrete(threshold=0.4)
    
    max_epoch = 100
    val_interval = 1
    
    best_metric = -1
    best_metric_epoch = -1
    
    epoch_loss_values = list()
    metric_values = list()
    #評価用関数
    dice_metric = DiceMetric(include_background=False, reduction='mean')
    
    for epoch in range(max_epoch):
        print("-" * 20)
        print(f"epoch:{epoch+1} / {max_epoch}")
        model.train()
        step = 0
        epoch_loss = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fnc(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= step
        print(f"Epoch[{epoch + 1} / {max_epoch}] Loss: {epoch_loss:.4f}")
        wandb.log({'train/loss': epoch_loss}, step=epoch+1)
        
        if (epoch+1) % val_interval ==0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device),val_data["mask"].to(device)
                    roi_size = (32,128,128)
                    sw_batch_size = 2
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(torch.sigmoid(i)) for i in val_outputs]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                metric_values.append(metric)
                wandb.log({'ins/metric': metric},step = epoch+1) 
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), r"C:\DL\best_metric_model.pth")
                    print(f"現在の epoch: {epoch + 1} 平均 dice: {metric:.4f}"
                      f" best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
    # ベストスコアをwandbに記録
    wandb.log({"best_metric": best_metric, "best_metric_epoch": best_metric_epoch})

        
        
        
        



