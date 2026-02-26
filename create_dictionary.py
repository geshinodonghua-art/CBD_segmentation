
"""
Created on Mon Jan  5 20:03:25 2026

@author: owner
"""

import os 
import numpy as np
import pydicom
import imageio.v3 as iio
from sklearn.model_selection import train_test_split

def create_dictionary(img_root,mask_root,npy_img_base_dir,npy_mask_base_dir):
    os.makedirs(npy_img_base_dir, exist_ok=True)
    os.makedirs(npy_mask_base_dir, exist_ok=True)
    data_dict = []
    patient_list = []
    patients = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root,d))])
    
    for pt in patients:
        pt_img_root = os.path.join(img_root,pt)
        pt_mask_root = os.path.join(mask_root,pt)
        pt_npy_img_dir = os.path.join(npy_img_base_dir,pt)
        pt_npy_mask_dir = os.path.join(npy_mask_base_dir,pt)
        
        os.makedirs(pt_npy_img_dir, exist_ok=True)
        os.makedirs(pt_npy_mask_dir, exist_ok=True)
        
        patient_list.append(pt)
        
        conditions = sorted([c for c in os.listdir(pt_img_root) if os.path.isdir(os.path.join(pt_img_root,c))])
        
        for cond in conditions:
            img_cond_root = os.path.join(pt_img_root,cond)
            mask_cond_root = os.path.join(pt_mask_root,cond)
            npy_img_cond_dir = os.path.join(pt_npy_img_dir,f"{cond}.npy")
            npy_mask_cond_dir = os.path.join(pt_npy_mask_dir,f"{cond}.npy")
            
            if not os.path.exists(mask_cond_root):
                print(f"マスクがないです{pt}_{cond}")
                continue
            
            if not os.path.exists(npy_img_cond_dir):
                dcmfiles = sorted([os.path.join(img_cond_root,f) for f in os.listdir(img_cond_root) if os.path.isfile(os.path.join(img_cond_root, f))])
                img_3d = np.stack([pydicom.dcmread(f).pixel_array for f in dcmfiles], axis = 0).astype(np.float32)
                np.save(npy_img_cond_dir, img_3d)
           
            if not os.path.exists(npy_mask_cond_dir):
                maskfiles = sorted([os.path.join(mask_cond_root, f) for f in os.listdir(mask_cond_root) if f.endswith(".png")])
                mask_3d = np.stack([iio.imread(f) for f in maskfiles], axis = 0).astype(np.uint8)
                np.save(npy_mask_cond_dir, mask_3d)
                
            data_dict.append({
                "image": npy_img_cond_dir,
                "mask": npy_mask_cond_dir,
                "pt_id": pt})
    
    print(f"患者数: {len(patients)}, サンプル数: {len(data_dict)}")
    
    train_pt, temp_pt = train_test_split(patient_list, test_size=0.28, random_state=42)
    val_pt, test_pt = train_test_split(temp_pt, test_size=0.5, random_state=42)
    
    train_file = [d for d in data_dict if d["pt_id"] in train_pt]
    val_file = [d for d in data_dict if d["pt_id"] in val_pt]
    test_file = [d for d in data_dict if d["pt_id"] in test_pt]
    
    return train_file, val_file, test_file, data_dict