# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:11:45 2026

@author: kadot
"""

from create_dictionary import create_dictionary
import numpy as np
import matplotlib.pyplot as plt

def nomal_distribution():
    _, _, _, all_data = create_dictionary(
        r"C:\DL\画像\原画像",
        r"C:\DL\画像\マスク画像\総胆管",
        r"C:\DL\画像\原npy",
        r"C:\DL\画像\マスクnpy")
    
    all_z, all_y, all_x = [], [], []
    
    print("総胆管の分布解析開始")
    
    for item in all_data:
        mask = np.load(item["mask"])
        
        z_idx, y_idx, x_idx = np.where(mask > 0)
        
        all_z.extend(z_idx)
        all_y.extend(y_idx)
        all_x.extend(x_idx)
    
    z_array = np.array(all_z)
    y_array = np.array(all_y)
    x_array = np.array(all_x)
    
    all_array = [("z_axis",z_array,66), ("y_axis", y_array, 512), ("x_axis",x_array,512)]
    
    for name,array,x_range in all_array:
        counts = np.bincount(array, minlength = x_range)
        x = np.arange(len(counts))
        plt.figure()
        plt.bar(x,counts)
        plt.xlabel("座標 / スライス")
        plt.ylabel("頻度")
        plt.title(f"{name}")
        plt.savefig(rf"E:\DL\総胆管分布\{name}.png")
        plt.show()
        plt.close()
    
    stats = {}
    
    for name, array, _ in all_array:
        mu = np.mean(array)
        sigma = np.std(array)
        stats[name] = (mu, sigma)
        print(f"{name} -> 平均(mu): {mu:.2f}, 標準偏差(sigma): {sigma:.2f}")
    
    return stats
    

    
    
        
    