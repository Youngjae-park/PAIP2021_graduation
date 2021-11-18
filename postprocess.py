from configparser import ConfigParser
import os, glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
parser = ConfigParser()
parser.read('config.ini')

BASE_PATH = os.getcwd()

def postprocess2(pred):
    pred[pred<128] = 0
    thin = cv2.ximgproc.thinning(pred)

def postprocess(pred, line_len_threshold=[10,20], line_act_threshold=0.5, dilate_iters=0):
    # print(np.unique(pred, return_counts=True))
    # print(np.unique(gt, return_counts=True))

    mean_threshold = (pred>10).astype(np.uint8)*255
    # kernel = np.ones([3,3], dtype=np.uint8)
    kernel = np.asarray([[1,1],[1,1]], dtype=np.uint8)
    thin = cv2.ximgproc.thinning(mean_threshold)
    # cv2.imwrite('./results/effb2_L2_UNet_seg(comb)cls2_noweight_max_SMpatcher/lv2_Col_46_thinned.jpg', thin)

    labels, label_map = cv2.connectedComponents(thin, connectivity=8)
    result = np.zeros_like(thin, dtype=np.uint8)
    for i in range(1, labels):
        if np.sum((label_map==i).astype(np.uint8)) > line_len_threshold[0]:
            if np.mean(pred[label_map==i])>int(255*line_act_threshold):
                result[label_map==i] = 255
    # cv2.imwrite('./results/effb2_L2_UNet_seg(comb)cls2_noweight_max_SMpatcher/lv2_Col_46_LT1.jpg', result)

    if dilate_iters>0:
        dilated = cv2.dilate(result, kernel, iterations=dilate_iters)
        # cv2.imwrite('./results/effb2_L2_UNet_seg(comb)cls2_noweight_max_SMpatcher/lv2_Col_46_LT1DT.jpg', dilated)
        thin = cv2.ximgproc.thinning(dilated)
        # cv2.imwrite('./results/effb2_L2_UNet_seg(comb)cls2_noweight_max_SMpatcher/lv2_Col_46_LT1DTTH.jpg', thin)
        labels, label_map = cv2.connectedComponents(thin, connectivity=8)
        result = np.zeros_like(thin, dtype=np.uint8)
        for i in range(1, labels):
            if np.sum((label_map==i).astype(np.uint8))>line_len_threshold[1]:
                result[label_map==i] = 255
        # cv2.imwrite('./results/effb2_L2_UNet_seg(comb)cls2_noweight_max_SMpatcher/lv2_Col_46_LT1DTTHLT2.jpg', result)

    return result

if __name__ == '__main__':
    experiment_idx_list = [11]
    for i in experiment_idx_list:
        experiment_name = parser.get(f'train{i}', 'experiment_name')
        print(experiment_name)
        result_path = os.path.join(BASE_PATH,f'results/{experiment_name}')
        gt_path = os.path.join(BASE_PATH, 'results/GT')
        # th025 = sorted(glob.glob(result_path+'/*_0.25.jpg'))
        # th050 = sorted(glob.glob(result_path+'/*_0.50.jpg'))
        th075 = sorted(glob.glob(result_path+'/*_0.75.jpg'))
        gts = sorted(glob.glob(gt_path+"/*.jpg"), key=lambda x: len(x))[:15]
        # thraw = sorted(glob.glob(result_path+'/*_0.25.jpg'))
        # print(th025)
        # print(th050)
        # print(th075)
        # print(thraw)

        for jpg_fn, gt_fn in zip(th075, gts):
            print(jpg_fn, gt_fn)
            save_fn = jpg_fn.split('.jpg')[0]+'_PP2.jpg'
            pred = cv2.imread(jpg_fn, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_fn)
            pp_img = postprocess(pred, dilate_iters=1)
            cv2.imwrite(save_fn, pp_img)
            break

