import glob, os
from tqdm import tqdm
from configparser import ConfigParser
import xml.etree.ElementTree as ET

import cv2
import openslide
import numpy as np

import matplotlib.pyplot as plt

parser = ConfigParser()
parser.read('../config.ini')

svs_path = parser.get('dataset', 'svs_path')
xml_path = parser.get('dataset', 'xml_path')

svs_fns = sorted(glob.glob(svs_path+'/*.svs'))
xml_fns = sorted(glob.glob(xml_path+'/*.xml'))

def make_GT():
    for svs_fn, xml_fn in tqdm(zip(svs_fns, xml_fns), total=len(svs_fns)):
        etree = ET.parse(xml_fn)
        annotations = etree.getroot()
        regions = annotations[1].findall("Regions")[0]

        slide = openslide.OpenSlide(svs_fn)
        src_w, src_h = slide.level_dimensions[0]
        # print(src_w, src_h)

        w_ratio = []
        h_ratio = []
        for i in [0,1,2]:
            dest_w, dest_h = slide.level_dimensions[i]
            w_ratio.append(src_w/dest_w)
            h_ratio.append(src_h/dest_h)

        pts_dict = {}


        map_lv2 = np.zeros((slide.level_dimensions[2]))
        map_lv2 = np.expand_dims(map_lv2, axis=-1)
        for r_idx, region in enumerate(regions.findall("Region")):
            pts_lv0 = []
            pts_lv1 = []
            pts_lv2 = []

            vertices = region.findall("Vertices")[0]
            for vertex in vertices.findall("Vertex"):
                x = float(vertex.get("X"))
                y = float(vertex.get("Y"))
                pts_lv0.append((x,y))

                dest_w, dest_h = slide.level_dimensions[1]
                x = np.clip(x/w_ratio[1], 0, dest_w)
                y = np.clip(y/h_ratio[1], 0, dest_h)
                pts_lv1.append((x,y))

                x = float(vertex.get("X"))
                y = float(vertex.get("Y"))
                dest_w, dest_h = slide.level_dimensions[2]
                x = np.clip(x/h_ratio[2], 0, dest_w)
                y = np.clip(y/w_ratio[2], 0, dest_h)
                pts_lv2.append((x, y))

            pts_dict[r_idx] = np.array([pts_lv0,
                                        pts_lv1,
                                        pts_lv2], dtype=np.int32)

            cv2.polylines(map_lv2, [np.array(pts_lv2)], isClosed=False, color=1, thickness=1)

        plt.imshow(map_lv2)
        plt.show()
        return

if __name__ == '__main__':
    make_GT()