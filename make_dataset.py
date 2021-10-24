#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 02:41:25 2021

@author: dacon
"""

import glob
import os
import h5py
import numpy as np
import logging

from tqdm import tqdm
from Patcher import *

from configparser import ConfigParser
parser = ConfigParser()
parser.read('config.ini')

svs_path = parser.get('dataset', 'svs_path')
xml_path = parser.get('dataset', 'xml_path')
save_dataset_path = parser.get('dataset', 'save_dataset_path')

svs_fns = sorted(glob.glob(svs_path+'/*.svs'))
xml_fns = sorted(glob.glob(xml_path+'/*.xml'))

def generate(level, thickness, margin, threshold=0):
    os.makedirs(save_dataset_path, exist_ok=True)
    file_name=save_dataset_path+f'/lv{level}_tk{thickness}_{margin}_SM.hdf5'
    
    f = h5py.File(file_name, 'w')
    
    img_list = []
    map_lv0_list = []
    map_lv2_list = []
    map_lv2_tk2_list = []
    label_list = []
    cancer_list = []
    cancer_idx_list = []

    cnt = 0
    for svs_fn, xml_fn in tqdm(zip(svs_fns, xml_fns), total=len(svs_fns)):
        ptc = Patcher(svs_fn, xml_fn)
        ptc.find_region()

        cancer = ptc.cancer
        cancer_idx = ptc.cancer_idx
        # cand_img, cand_map = ptc.Thumnail(level, thickness, margin)
        imgs, maps_lv0, maps_lv2, maps_lv2_tk2 = ptc.Thumnail(level=0,thickness=1, margin_x=margin, margin_y=margin)

        for img, map_lv0, map_lv2, map_lv2_tk2 in zip(imgs, maps_lv0, maps_lv2, maps_lv2_tk2):
            SM = SmallPatcher(img, map_lv0, map_lv2, map_lv2_tk2)
            SM.random_cropping(threshold=threshold)

            cand_img, cand_map_lv0, cand_map_lv2, cand_map_lv2_tk2 = SM.get_candidate()
            cand_length = len(cand_img)
            not_cand_img, not_cand_map_lv0, not_cand_map_lv2, not_cand_map_lv2_tk2 = SM.get_not_candidate()
            ncand_length = len(not_cand_img)
            img_list = img_list + cand_img
            map_lv0_list = map_lv0_list + cand_map_lv0
            map_lv2_list = map_lv2_list + cand_map_lv2
            map_lv2_tk2_list = map_lv2_tk2_list + cand_map_lv2_tk2
            label_list = label_list + [1]*cand_length

            img_list = img_list + not_cand_img
            map_lv0_list = map_lv0_list + not_cand_map_lv0
            map_lv2_list = map_lv2_list + not_cand_map_lv2
            map_lv2_tk2_list = map_lv2_tk2_list + not_cand_map_lv2_tk2
            label_list = label_list + [0]*ncand_length

            cancer_list = cancer_list + [cancer]*(cand_length+ncand_length)
            cancer_idx_list = cancer_idx_list + [cancer_idx]*(cand_length+ncand_length)

            print(f'num of {len(cand_img)+len(not_cand_img)} patches are saved')
            cnt += len(cand_img)+len(not_cand_img)


        print(f'Accumulated {cnt}')
    idx_list = [i for i in range(len(img_list))]

    # print(f'Accumulated {len(img_list)}')
    
    img_list = np.array(img_list).astype(np.uint8)
    map_lv0_list = np.array(map_lv0_list).astype(np.uint8)
    map_lv2_list = np.array(map_lv2_list).astype(np.uint8)
    map_lv2_tk2_list = np.array(map_lv2_tk2_list).astype(np.uint8)
    label_list = np.array(label_list).astype(np.uint8)
    cancer_list = np.array(cancer_list).astype('S3')
    cancer_idx_list = np.array(cancer_idx_list).astype(int)
    idx_list = np.array(idx_list).astype(int)

    # print(img_list)
    f.create_dataset('img', data=img_list)
    f.create_dataset('map_lv0', data=map_lv0_list)
    f.create_dataset('map_lv2', data=map_lv2_list)
    f.create_dataset('map_lv2_tk2', data=map_lv2_tk2_list)
    f.create_dataset('label', data=label_list)
    f.create_dataset('cancer', data=cancer_list)
    f.create_dataset('cancer_idx', data=cancer_idx_list)
    f.create_dataset('idx', data=idx_list)

    print("Successfuly saved")
    

if __name__ == '__main__':
    generate(level=0, thickness=2, margin=512)