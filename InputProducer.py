#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:25:01 2021

@author: dacon
"""

from configparser import ConfigParser
import h5py
import numpy as np

parser = ConfigParser()
parser.read('config.ini')

from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transforms import RandomFlip, ToTensor, RandomFlip_3, ToTensor_3

import torch
import random
from sklearn.model_selection import train_test_split

def set_seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def train_val_test_split(seed=42):
    set_seed(random_seed=seed)
    train_idx, test_idx = train_test_split(np.arange(1, 51), test_size=0.2, random_state=42)
    train_idx, valid_idx = train_test_split(train_idx, test_size=0.25, random_state=42)
    # print(train_idx, valid_idx, test_idx)
    return train_idx, valid_idx, test_idx

class IPWrapper:
    def __init__(self, path, is_train, batch_size=32, multiple=False):
        if not multiple:
            ip = IP(path, is_train)
        else:
            ip = IP(path, is_train, multiple=multiple)
        sampler = RandomSampler(ip, replacement=False)
        self.len = ip.__len__()
        self.loader = DataLoader(ip, batch_size=batch_size,
                                 num_workers=4,
                                 # shuffle = False,
                                 sampler=sampler,
                                 drop_last=True)
        self.iterator = iter(self.loader)
        self.idx = 0
    
    def produce(self):
        try:
            batch = next(self.iterator)
        except :
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
            self.idx += 1
        return batch
    
    def reset_iter(self):
        self.iterator = iter(self.loader)

class IP(Dataset):
    def __init__(self,
                 train_case, is_train=True, multiple=False
                 ):
        self.is_train = is_train
        self.multiple = multiple
        dataset_path = parser.get(train_case, 'dataset_name')
        f = h5py.File(dataset_path, 'r')

        train_indices, valid_indices, test_indices = train_val_test_split()

        cancer = f['cancer']
        cancer = np.expand_dims(cancer, axis=-1)
        cancer = np.where(cancer==b'Col', 0, cancer)
        cancer = np.where(cancer==b'Pan', 1, cancer)
        cancer = np.where(cancer==b'Pro', 2, cancer)
        cancer = cancer.astype(int)
        cancer = np.squeeze(cancer)
        cancer_idx = np.asarray(f['cancer_idx']).astype(int)
        mask_train = np.zeros(cancer.shape, dtype=bool)
        mask_valid = np.zeros(cancer.shape, dtype=bool)

        for t_idx in train_indices:
            mask_train = np.where(cancer_idx == t_idx, True, mask_train)
        for v_idx in valid_indices:
            mask_valid = np.where(cancer_idx == v_idx, True, mask_valid)
        # print(np.unique(cancer_idx[mask_train], return_counts=True))
        # print(np.unique(cancer_idx[mask_valid], return_counts=True))

        # for k, key in enumerate(['Col', 'Pan', 'Pro'])
            # print(k, key)
            # print(np.unique(c_flag, return_counts=True))
            # for idx in mask[key]:
            #     ci_flag = np.where(cancer_idx==idx, True, False)
            #     # print(np.unique(ci_flag&c_flag, return_counts=True))
            #     mask_ = (c_flag & ci_flag) | mask_

        
        # if is_train:
        #     mask_ = ~mask_
        if is_train == 'Train':
            mask_ = mask_train
        elif is_train == 'Valid':
            mask_ = mask_valid
        
        self.items = {}
        for key in f.keys():
            temp = f[key][()]
            self.items[key] = temp[mask_]
        # for k in f.keys():
        #     print(k)
        # self.items = [np.asarray(f.get(k))[mask_] for k in f.keys()]


        if is_train == 'Train':
            if not multiple:
                self.transform = Compose([
                    ToTensor(),
                    # RandomRotate(90),
                    RandomFlip(),
                    # RandomCrop(int(crop_size/16))
                    ])
            else:
                self.transform = Compose([
                    ToTensor_3(),
                    RandomFlip_3(),
                ])
        elif is_train == 'Valid':
            if not multiple:
                self.transform = Compose([
                    ToTensor(),
                ])
            else:
                self.transform = Compose([
                    ToTensor_3(),
                ])

        
    def __len__(self):
        # return len(self.items[0])
        return len(self.items['idx'])
    
    def __getitem__(self, idx):
        #original
        # mask = self.items['map'][idx]
        #SM patcher
        infos = "_".join([self.items['cancer'][idx].decode('UTF-8'),
                          str(self.items['cancer_idx'][idx]),
                          str(self.items['idx'][idx]),
                          str(self.items['label'][idx])])

        if not self.multiple:
            img = self.items['img'][idx]
            mask = self.items['map_lv2'][idx]
            img, mask = self.transform((img, mask))
            return img, mask, infos
        else:
            img = self.items['img'][idx]
            mask = self.items['map_lv2'][idx]
            mask_tk = self.items['map_lv2_tk2'][idx]
            img, mask, mask_tk = self.transform((img, mask, mask_tk))
            return img, mask, mask_tk, infos


        

        

if __name__ == '__main__':
    # d_path = parser.get('train14', 'dataset_name')
    # print(d_path)
    ip = IPWrapper(path='train14', is_train='Train')
    # ip = IP('lv0_tk1_512.hdf5', is_train=True)
    # dataloader = DataLoader(ip, batch_size=32, shuffle=True)
    # IP = iter(dataloader)
    for i in range(200):
        img, mask, infos = ip.produce()
        print(infos)
        #print(img.min(), img.max(), mask.min(), mask.max())
        
    
