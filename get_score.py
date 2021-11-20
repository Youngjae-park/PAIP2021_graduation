import os
import numpy as np
import cv2

from configparser import ConfigParser
from metric import threading_compute_dist_f1_score
from postprocess import postprocess

import multiprocessing

import matplotlib.pyplot as plt
from PIL import Image

parser = ConfigParser()
parser.read('config.ini')

def draw_image(gt_label_map, pred_label_map, tps, pred_tps, fps, fns):
    w, h = gt_label_map.shape
    img = np.zeros((w, h, 3), dtype=np.uint8)

    for tp, pred_tp in zip(tps, pred_tps):
        # Green for both gt and predicted line
        # print(gt_label_map==(tp+1))
        # print(pred_label_map==(pred_tp+1))
        # print(np.logical_or(gt_label_map==(tp+1),pred_label_map==(pred_tp+1)))

        # Yellow for gt line
        img[np.logical_and(gt_label_map==(tp+1),pred_label_map!=(pred_tp+1)), 0] = 255
        img[np.logical_and(gt_label_map==(tp+1),pred_label_map!=(pred_tp+1)), 1] = 255
        # White for predicted line
        img[np.logical_and(gt_label_map!=(tp+1),pred_label_map==(pred_tp+1)), 0] = 255
        img[np.logical_and(gt_label_map!=(tp+1),pred_label_map==(pred_tp+1)), 1] = 255
        img[np.logical_and(gt_label_map!=(tp+1),pred_label_map==(pred_tp+1)), 2] = 255
        # Green for both line
        img[np.logical_and(gt_label_map==(tp+1),pred_label_map==(pred_tp+1)), 0] = 0
        img[np.logical_and(gt_label_map==(tp+1),pred_label_map==(pred_tp+1)), 1] = 255
        img[np.logical_and(gt_label_map==(tp+1),pred_label_map==(pred_tp+1)), 2] = 0
    for fp in fps:
        img[pred_label_map==(fp+1), 0] = 255
    for fn in fns:
        img[gt_label_map==(fn+1), 2] = 255
    return img

def show_result(infos):
    cancers = ['Col', 'Pan', 'Pros']
    indices = ['46', '47', '48', '49', '50']

    cnt = 0
    for cancer in cancers:
        for idx in indices:
            print(cancer+' '+idx+' f1-score: ', infos[cnt])
            cnt += 1

def calc_score(model_name, line_len_threshold=[10, 16], line_act_threshold=0.75, dilate_iters=1, post_name=None):
    print(f'{model_name}{line_len_threshold}{line_act_threshold}{dilate_iters} scoring',
          f'process started!!')
    pred_path = f'/home/dhodwo/young/PAIP2021/results/{model_name}'
    gt_path = '/home/dhodwo/young/PAIP2021/results/GT'
    save_path = f'/home/dhodwo/young/PAIP2021/result_image/{post_name}/{model_name}'
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    cancer_list = ['Col', 'Pan', 'Pros']
    idx_list = [46, 47, 48, 49, 50]
    # cancer_list = ['Col']
    # idx_list = [49]
    f1_list = []

    with open(save_path+'/result.txt', 'w') as file:
        file.write(f'model_name:{model_name}\n'+
                   f'line_len_threshold:{line_len_threshold}\n'+
                   f'line_act_threshold:{line_act_threshold}\n'+
                   f'dilate_iters:{dilate_iters}\n\n')
        for cancer in cancer_list:
            for idx in idx_list:
                print(f'{cancer} {idx}')
                pred = cv2.imread(os.path.join(pred_path, f'lv2_{cancer}_{idx}_raw.jpg'), cv2.IMREAD_GRAYSCALE)
                pred = postprocess(pred, line_len_threshold, line_act_threshold, dilate_iters).astype(np.uint8)

                gt = cv2.imread(os.path.join(gt_path, f'{cancer}_{idx}.jpg'), cv2.IMREAD_GRAYSCALE)
                gt = (gt>128).astype(np.uint8)*2

                dist_f1_score, counts, others, maps = threading_compute_dist_f1_score(pred=pred, gt=gt, num_thread=4)
                gt_count, pred_count = counts
                tp, fp, fn, pred_tp, prec, rec = others
                gt_label_map = maps[0]
                pred_label_map = maps[1]

                # print(gt_count, pred_count)
                # print(tp, fp, fn, prec, rec)

                final_img = draw_image(gt_label_map, pred_label_map, tp, pred_tp, fp, fn)
                # print(final_img.shape)
                # print(np.unique(final_img[:,:,0], return_counts=True))
                # print(np.unique(final_img[:,:,1], return_counts=True))
                # print(np.unique(final_img[:,:,2], return_counts=True))
                final_img = Image.fromarray(final_img, 'RGB')
                final_img.save(f'{save_path}/{cancer}_{idx}.png')
                # print(dist_f1_score)
                f1_list.append(dist_f1_score)

                file.write(f'# {cancer}_{idx} result\n')
                file.write(f'gt_count:{gt_count}, pred_count:{pred_count}\n')
                file.write(f'tps:{tp} => {len(tp)}\n')
                file.write(f'fps:{fp} => {len(fp)}\n')
                file.write(f'fns:{fn} => {len(fn)}\n')
                file.write(f'precision: {prec}\n')
                file.write(f'recall: {rec}\n')
                file.write(f'f1: {dist_f1_score}\n\n')

        file.write(f'\n##TOTAL RESULTS##\naverage f1 score: {np.mean(f1_list)}')
        print('average f1 score: ', np.mean(f1_list))


if __name__ == '__main__':
    # experiment_idx_list = [5, 6, 7, 8, 9, 10, 11]
    experiment_idx_list = [14]
    process = []
    for i in experiment_idx_list:
        experiment_name = parser.get(f'train{i}', 'experiment_name')
        for len_t in [[10,20]]:
            for act_t in [0.7]:
                for d_iter in [0, 1]:
                    post_name = f'L{len_t[0]}A{act_t}L{len_t[1]}D{d_iter}'
                    print(post_name, experiment_name)
                    p = multiprocessing.Process(target=calc_score, args=(experiment_name,len_t,act_t,d_iter,post_name))
                    p.start()
                    process.append(p)
                    continue
                    calc_score(experiment_name,
                               line_len_threshold=len_t,
                               line_act_threshold=act_t,
                               dilate_iters=d_iter,
                               post_name=post_name)
                    continue
                    try:
                        calc_score(experiment_name,
                                   line_len_threshold=len_t,
                                   line_act_threshold=act_t,
                                   dilate_iters=d_iter,
                                   post_name=post_name)
                    except Exception as ex:
                        print(experiment_name, 'has error\n', ex)

    for proc in process:
        proc.join()

    print("Scoring Finished!!")


    exit()
    for i in experiment_idx_list:
        experiment_name = parser.get(f'train{i}', 'experiment_name')

        # calc_score(experiment_name,
        #            line_len_threshold=[10,20],
        #            line_act_threshold=0.5,
        #            dilate_iters=1)
        try:
            calc_score(experiment_name,
                       line_len_threshold=[10,20],
                       line_act_threshold=0.5,
                       dilate_iters=1)
        except Exception as ex:
            print(experiment_name, 'has error\n', ex)

        # for len_t in [[5, 15], [5, 20], [5, 25], [10, 15], [10, 20], [10, 25], [15, 15], [15, 20], [15, 25]]:
        #     for act_t in [0.4, 0.5, 0.6]:
        #         for iters in [0, 1, 2]:
        #             print(f'len_threshold: {len_t}, act_threshold: {act_t}, iters: {iters}')
        #             calc_score(experiment_name,line_len_threshold=len_t, line_act_threshold=act_t, dilate_iters=iters)
