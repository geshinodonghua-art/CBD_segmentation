# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 22:46:01 2026

@author: owner
"""

from transformation import transform
from monai.data import DataLoader
from monai.data import list_data_collate

def DataLoad():
    train_ds, val_ds, test_ds = transform()
    
    train_load = DataLoader(
        train_ds,
        batch_size = 2,
        shuffle = True,
        num_workers = 4,
        collate_fn=list_data_collate
        )
    
    val_load = DataLoader(val_ds, batch_size = 1, shuffle = False)
    test_load = DataLoader(test_ds, batch_size = 1, shuffle = False)
    
    return train_load, val_load, test_load