# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:12:37 2026

@author: owner
"""

import nibabel as nib
import imageio
import os
import numpy as np

def nii_to_pngs(nii_path, output_base_path, row_base_path):
    patients = sorted([d for d in os.listdir(nii_path) if os.path.isdir(os.path.join(nii_path,d))])
    for pt in patients:
        pt_row_path = os.path.join(row_base_path,pt)
        pt_nii_path = os.path.join(nii_path,pt)
        pt_output_path = os.path.join(output_base_path,pt)
        
        conditions = sorted([c for c in os.listdir(pt_row_path) if os.path.isdir(os.path.join(pt_row_path,c))])
        os.makedirs(pt_output_path, exist_ok=True)
        
        for cond in conditions:
            cond_output_path = os.path.join(pt_output_path,cond)
            
            # --- ここが修正ポイント ---
            # ループではなく、フォルダ名（cond）と一致するnii.gzを直接指定
            nii_file_path = os.path.join(pt_nii_path, f"{cond}.nii.gz")
            
            if not os.path.exists(nii_file_path):
                print(f"スキップ: {nii_file_path} がありません")
                continue
                
            # 指定した1つのnii.gzだけを読み込む
            img = nib.load(nii_file_path)
            data = img.get_fdata()
            
            os.makedirs(cond_output_path, exist_ok=True)
            
            num_slices = data.shape[2]
            for i in range(num_slices):
                slice_data = data[:, :, i]
                # パターン4（90度回転＋上下反転）
                slice_data = np.flipud(np.rot90(slice_data))
                
                output_path = os.path.join(cond_output_path, f"{i:08d}.png")
                imageio.imwrite(output_path, (slice_data * 255).astype(np.uint8))
                
        print(f"{pt}分終了")
    print("終わった")

nii_to_pngs(r"E:\DL\画像\マスク",
            r"E:\DL\画像\マスク画像\総胆管",
            r"E:\DL\画像\原画像")