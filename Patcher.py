#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:56:30 2021

@author: youngjae
"""

import openslide
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import copy, os
import matplotlib.pyplot as plt

from configparser import ConfigParser
parser = ConfigParser()
parser.read('config.ini')

def max_pooling(arr, level=2):
    arr = np.squeeze(arr)
    print(arr.shape)
    x, y = arr.shape
    new_x, new_y = x//(4**level), y//(4**level)
    arr = arr[:new_x*(4**level), :new_y*(4**level)]
    
    arr = np.max(arr.reshape(new_x, (4**level), new_y, (4**level)), axis=(1,3))
    return arr
    
class Patcher:
    def __init__(self, svs_path, xml_path):
        self.slide = openslide.OpenSlide(svs_path)
        cancer, c_idx = svs_path.split('_PNI2021chall_train_')
        self.cancer = cancer.split('/')[-1]
        self.cancer_idx = int(c_idx.split('.')[0])
        
        self.svs_path = svs_path
        self.xml_path = xml_path
        
        self.w_ratio = []
        self.h_ratio = []
        src_w, src_h = self.slide.level_dimensions[0]
        # print(src_w, src_h)
        for i in [0,1,2]:
            dest_w, dest_h = self.slide.level_dimensions[i]
            self.w_ratio.append(src_w/dest_w)
            self.h_ratio.append(src_h/dest_h)

    def Thumnail(self, level, thickness=1, margin_x=256, margin_y=256, show=False):
        img_list = []
        map_lv0_list = []
        map_lv2_list = []
        map_lv2_tk2_list = []

        for region_idx in range(self.num_regions):
            pts_lv0, pts_lv1, pts_lv2 = self.pts_dict[region_idx]

            min_x_lv0, min_y_lv0 = np.min(pts_lv0, axis=0)
            max_x_lv0, max_y_lv0 = np.max(pts_lv0, axis=0)
            min_x_lv2, min_y_lv2 = np.min(pts_lv2, axis=0)
            max_x_lv2, max_y_lv2 = np.max(pts_lv2, axis=0)

            w_lv0 = max_x_lv0 - min_x_lv0
            h_lv0 = max_y_lv0 - min_y_lv0
            w_lv2 = max_x_lv2 - min_x_lv2
            h_lv2 = max_y_lv2 - min_y_lv2

            pts_lv0 = pts_lv0.reshape((-1,1,2))
            pts_lv2 = pts_lv2.reshape((-1,1,2))

            pts_lv0[:,0,0] -= (min_x_lv0 - margin_x)
            pts_lv0[:,0,1] -= (min_y_lv0 - margin_y)
            pts_lv2[:,0,0] -= (min_x_lv2 - (margin_x//16))
            pts_lv2[:,0,1] -= (min_y_lv2 - (margin_y//16))

            if min_y_lv0 - round(margin_y*self.h_ratio[0]) < 0:
                pts_lv0[:,0,1] -= margin_y
                pts_lv2[:,0,1] -= (margin_y//16)
                img_lv0 = self.slide.read_region((min_x_lv0-round(margin_x*self.w_ratio[0]),
                                                  min_y_lv0), 0, (w_lv0+margin_x*2, h_lv0+margin_y*2))
            else:
                img_lv0 = self.slide.read_region((min_x_lv0-round(margin_x*self.w_ratio[0]),
                                                  min_y_lv0-round(margin_y*self.h_ratio[0])), 0, (w_lv0+margin_x*2, h_lv0+margin_y*2))

            img_lv0 = img_lv0.convert('RGB')
            img_lv0 = np.asarray(img_lv0, dtype=np.uint8)

            img_map_lv0 = copy.deepcopy(img_lv0)
            img_map_lv0 = cv2.polylines(img_map_lv0, [pts_lv0], isClosed=False, color=(255,0,0), thickness=5)
            map_lv0 = np.zeros(img_lv0.shape[:-1], dtype=np.uint8)
            map_lv0 = np.expand_dims(map_lv0, axis=-1)
            cv2.polylines(map_lv0, [pts_lv0], isClosed=False, color=1, thickness=1)

            map_lv2 = np.zeros((w_lv2+(margin_x//16)*2, h_lv2+(margin_y//16)*2), dtype=np.uint8)
            map_lv2 = np.expand_dims(map_lv2, axis=-1)
            cv2.polylines(map_lv2, [pts_lv2], isClosed=False, color=1, thickness=1)

            map_lv2_tk2 = np.zeros((w_lv2+(margin_x//16)*2, h_lv2+(margin_y//16)*2), dtype=np.uint8)
            map_lv2_tk2 = np.expand_dims(map_lv2_tk2, axis=-1)
            cv2.polylines(map_lv2_tk2, [pts_lv2], isClosed=False, color=1, thickness=2)

            img_list.append(img_lv0)
            map_lv0_list.append(map_lv0)
            map_lv2_list.append(map_lv2)
            map_lv2_tk2_list.append(map_lv2_tk2)

            if show:
                print("show image", self.cancer, self.cancer_idx, region_idx)
                fig = plt.figure()
                rows = 2
                cols = 2

                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.imshow(img_lv0)
                ax1.set_title('Raw Image')
                ax1.axis("off")

                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.imshow(img_map_lv0)
                ax2.set_title('Image with Label2')
                # ax2.axis("off")

                ax3 = fig.add_subplot(rows, cols, 3)
                ax3.imshow(map_lv0)
                ax3.set_title('Map LV0')
                # ax3.axis("off")

                ax4 = fig.add_subplot(rows, cols, 4)
                ax4.imshow(map_lv2)
                ax4.set_title('Map LV2')
                # ax4.axis("off")

                plt.show()

        # self.img_list = img_list
        # self.map_lv0_list = map_lv0_list
        # self.map_lv1_list = map_lv2_list

        return img_list, map_lv0_list, map_lv2_list, map_lv2_tk2_list

    def thumnail(self, level, thickness=1, margin=212, show=False):
        img_list = []
        map_list = []
        
        for region_idx in range(self.num_regions):
            pts_lv0 = self.pts_dict[region_idx][0]
            len_pts = int(len(pts_lv0)/2)
            for i in range(2):
                center_lv0 = np.asarray(pts_lv0[i*len_pts]-margin, dtype=np.int)
                img_lv0 = self.slide.read_region(center_lv0, 0, (margin*2, margin*2))
                img_lv0 = np.asarray(img_lv0.convert('RGB'), dtype=np.uint8)
                
                pts = self.pts_dict[region_idx][2]
                center = np.asarray(pts[i*len_pts]-int((1/4)**2*margin), dtype=np.int)
                map_ = np.zeros([int(margin*2/(4**2)), int(margin*2/(4**2)), 1], dtype=np.uint8)
                for j in range(self.num_regions):
                    line = np.reshape(self.pts_dict[j][2]-center, [-1, 1, 2])
                    line = line.astype(np.int)
                    cv2.polylines(map_, [line], isClosed=False, color=1, thickness=thickness)
                
                img_list.append(img_lv0)
                map_list.append(map_)
        
        if show:
            fig = plt.figure(dpi=200)
            
            pts_lv0 = self.pts_dict[0][0]
            center_lv0 = np.asarray(pts_lv0[0*int(len(pts_lv0)/2)]-margin, dtype=np.int)
            img_lv0 = self.slide.read_region(center_lv0, 0, (margin*2, margin*2))
            img_lv0 = np.asarray(img_lv0.convert('RGB'), dtype=np.uint8)
            cv2.polylines(img_lv0, [pts_lv0 - center_lv0], isClosed=False, color=(255,0,0), thickness=8)
            
            map_ = np.zeros([int(margin*2), int(margin*2), 1], dtype=np.uint8)
            cv2.polylines(map_, [pts_lv0 - center_lv0], isClosed=False, color=1, thickness=8)
            
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(img_list[0])
            ax1.set_title('Raw Image')
            
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(img_lv0)
            ax2.set_title('Raw Image with Layer2\n(thickness=8)')
            
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(map_)
            ax3.set_title('Mask Map LV0\n(thickness=8)')
        
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(map_list[0])
            ax4.set_title('Mask Map LV2\n(thickness=1)')
            
            plt.subplots_adjust(hspace=0.7, wspace=-0.3)
        
        return np.asarray(img_list, dtype=np.int), np.asarray(map_list, dtype=np.int) 

    def find_region(self):
        etree = ET.parse(self.xml_path)

        annotations = etree.getroot()
        # label = int(annotations[1].get("Id"))
        regions = annotations[1].findall("Regions")[0]

        self.pts_dict = {}

        for r_idx, region in enumerate(regions.findall("Region")):
            pts_lv0 = []
            pts_lv1 = []
            pts_lv2 = []

            vertices = region.findall("Vertices")[0]
            for vertex in vertices.findall("Vertex"):
                x = float(vertex.get("X"))
                y = float(vertex.get("Y"))
                pts_lv0.append((x,y))
                
                dest_w, dest_h = self.slide.level_dimensions[1]
                x = np.clip(x/self.w_ratio[1], 0, dest_w)
                y = np.clip(y/self.h_ratio[1], 0, dest_h)
                pts_lv1.append((x,y))
                
                x = float(vertex.get("X"))
                y = float(vertex.get("Y"))
                dest_w, dest_h = self.slide.level_dimensions[2]
                x = np.clip(x/self.h_ratio[2], 0, dest_w)
                y = np.clip(y/self.w_ratio[2], 0, dest_h)
                pts_lv2.append((x, y))
                
            self.pts_dict[r_idx] = np.array([pts_lv0,
                                        pts_lv1,
                                        pts_lv2], dtype=np.int32)
        self.num_regions = len(self.pts_dict)

class SmallPatcher:
    def __init__(self, img, map_lv0, map_lv2, map_lv2_tk2, weighted_map=None):
        self.img = img
        self.map_lv0 = map_lv0
        self.map_lv2 = map_lv2
        self.map_lv2_tk2 = map_lv2_tk2

    def get_candidate(self):
        return self.candidate_img, self.candidate_map_lv0, self.candidate_map_lv2, self.candidate_map_lv2_tk2

    def get_not_candidate(self):
        return self.not_candidate_img, self.not_candidate_map_lv0, self.not_candidate_map_lv2, self.not_candidate_map_lv2_tk2

    def random_cropping(self, crop_size=512, threshold=0.005):
        W, H = self.img.shape[:-1]
        window_step = int(crop_size//2)

        total_area = (crop_size//16)**2 # crop_size**2
        wh_pts = []
        wh_pts_not = []

        # print(self.map_lv0.shape, self.map_lv2.shape)

        for w in range(0, W-crop_size+1, window_step):
            for h in range(0, H-crop_size+1, window_step):
                # cropped_map_lv0 = self.map_lv0[w:w+crop_size, h:h+crop_size]
                cropped_map_lv2 = self.map_lv2[(w//16):(w+crop_size)//16, (h//16):(h+crop_size)//16]
                indices, counts = np.unique(cropped_map_lv2, return_counts = True)
                if counts.sum() != 1024:
                    continue
                # print(indices, counts)
                if 1 not in indices:
                    wh_pts_not.append((w,h))
                elif 0 not in indices:
                    wh_pts.append((w,h))
                else:
                    # print(counts, total_area, threshold)
                    if counts[-1] / total_area >= threshold:
                        wh_pts.append((w,h))
                    else:
                        wh_pts_not.append((w,h))
        # print('wh_pts', wh_pts)
        # print('wh_pts_not', wh_pts_not)
        self.candidate_img = [self.img[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts]
        self.candidate_map_lv0 = [self.map_lv0[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts]
        self.candidate_map_lv2 = [self.map_lv2[(w//16):(w+crop_size)//16, (h//16):(h+crop_size)//16, :] for w,h in wh_pts]
        self.candidate_map_lv2_tk2 = [self.map_lv2_tk2[(w//16):(w+crop_size)//16, (h//16):(h+crop_size)//16, :] for w,h in wh_pts]

        self.not_candidate_img = [self.img[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts_not]
        self.not_candidate_map_lv0 = [self.map_lv0[w:w+crop_size, h:h+crop_size, :] for w,h in wh_pts_not]
        self.not_candidate_map_lv2 = [self.map_lv2[(w//16):(w+crop_size)//16, (h//16):(h+crop_size)//16, :] for w,h in wh_pts_not]
        self.not_candidate_map_lv2_tk2 = [self.map_lv2_tk2[(w//16):(w+crop_size)//16, (h//16):(h+crop_size)//16, :] for w,h in wh_pts_not]

        # print(len(self.candidate_img), len(self.candidate_map_lv0), len(self.candidate_map_lv2))
        # print(len(self.not_candidate_img), len(self.not_candidate_map_lv0), len(self.not_candidate_map_lv2))

    def show_candidate(self):
        fig = plt.figure()
        rows = 1
        cols = len(self.candidate_img)

        # for i in range(len(self.candidate_img)):
        #     fig.add_subplot(1,len(self.candidate_img), i+1).imshow(self.candidate_map_lv2[i])
        # for j in range(len(self.not_candidate_img)):
        #     fig.add_subplot(2, len(self.not_candidate_img), len(self.candidate_img)+j+1).imshow(self.not_candidate_map_lv2[j])

        axes = [fig.add_subplot(rows, cols, i+1) for i in range(cols)]
        for idx, ax in enumerate(axes):
            ax.imshow(self.candidate_img[idx]) #*self.candidate_map_lv0[idx])
            # ax.imshow(self.candidate_map_lv2[idx])
            # ax.axis("off")
        plt.show()


if __name__ == '__main__':
    svs_path = parser.get('dataset', 'svs_path')
    xml_path = parser.get('dataset', 'xml_path')
    
    file_name = 'Col_PNI2021chall_train_0001'
    svs_path = os.path.join(svs_path, f'{file_name}.svs')
    xml_path = os.path.join(xml_path, f'{file_name}.xml')
    
    print(svs_path)
    
    img_level = 0
    
    ptc = Patcher(svs_path, xml_path)
    ptc.find_region()

    img_list, mask_lv0_list, mask_lv2_list, mask_lv2_tk2_list = ptc.Thumnail(level=0, margin_x=512, margin_y=512, show=False)
    for img, mask_lv0, mask_lv2, mask_lv2_tk2 in zip(img_list, mask_lv0_list, mask_lv2_list, mask_lv2_tk2_list):
        SM = SmallPatcher(img, mask_lv0, mask_lv2, mask_lv2_tk2)
        SM.random_cropping()
        # SM.show_candidate()

        cand_img, cand_map_lv0, cand_map_lv2, cand_map_lv2_tk2 = SM.get_candidate()
        not_candidates = SM.get_not_candidate()

    # print(img.shape, mask_lv0.shape, mask_lv2.shape)
