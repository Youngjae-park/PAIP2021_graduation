#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 17:07:55 2021

@author: dacon
"""
from configparser import ConfigParser
import os
import time

import tqdm
import torch.optim as optim
import torch.nn as nn

import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

from loss import *
from InputProducer import IPWrapper
from models import L2_UNET

parser = ConfigParser()
parser.read('config.ini')


def train(train_case, model_type, encoder_name, pooling_type, loss_type, lr, batch_size, experiment_name, multiple, patch_size=512):
    ip = IPWrapper(path=train_case, is_train=True, batch_size=int(batch_size), multiple=multiple)
    valid_ip = IPWrapper(path=train_case, is_train=False, batch_size=int(batch_size), multiple=multiple)

    print(f'train_len : {ip.len}')
    print(f'validation_len : {valid_ip.len}')
    print("Data Loaded!!")

    if model_type == 'L2_UNET':
        model = L2_UNET(1, out_size=patch_size, encoder_name=encoder_name)

    if pooling_type == 'max':
        pool = nn.AdaptiveMaxPool2d(1)

    model = nn.DataParallel(model).cuda()
    
    dice_loss = smp.utils.losses.DiceLoss()
    ce = nn.BCELoss()

    writer = SummaryWriter(log_dir=os.path.join('runs', experiment_name))
    
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    
    best_iou = 0.
    for step in tqdm.tqdm(range(1, 10000)):
        if multiple:
            img, mask, mask_tk, info = ip.produce()
            img = img.cuda()
            mask = mask.cuda()
            mask_tk = mask_tk.cuda()
        else:
            img, mask, info = ip.produce()
            img = img.cuda()
            mask = mask.cuda()

        optimizer.zero_grad()
        outputs = model(img)
        
        loss = 0.
        # Setting losses
        if loss_type == 'seg':
            if multiple:
                seg_loss = get_L2_sloss(outputs, mask_tk, dice_loss)
                loss = seg_loss
            else:
                seg_loss = get_L2_sloss(outputs, mask, dice_loss)
                loss = seg_loss
        elif loss_type == 'segcls':
            if multiple:
                seg_loss = get_L2_sloss(outputs, mask_tk, dice_loss)
                cls_loss = get_L2_closs(outputs, mask_tk, ce, pool)
                loss = seg_loss + cls_loss
            else:
                seg_loss = get_L2_sloss(outputs, mask, dice_loss)
                cls_loss = get_L2_closs(outputs, mask, ce, pool)
                loss = seg_loss + cls_loss
        elif loss_type == 'seg(comb)':
            if multiple:
                seg1_loss = get_L2_sloss(outputs, mask, dice_loss)
                seg2_loss = get_L2_sloss(outputs, mask_tk, dice_loss)
                loss = seg1_loss+seg2_loss
            else:
                print("impossible_traincase")
                return False
        elif loss_type == 'seg(comb)cls':
            if multiple:
                seg1_loss = get_L2_sloss(outputs, mask, dice_loss)
                seg2_loss = get_L2_sloss(outputs, mask_tk, dice_loss)
                cls_loss = get_L2_closs(outputs, mask, ce, pool)
                loss = seg1_loss + seg2_loss + cls_loss
            else:
                print("impossible_traincase")
                return False
        # print(loss)

        if step%10 == 0:
            if loss_type == 'seg':
                writer.add_scalar('train/seg_loss', seg_loss, step)
                writer.add_scalar('train/total_loss', loss, step)
            elif loss_type == 'segcls':
                writer.add_scalar('train/seg_loss', seg_loss, step)
                writer.add_scalar('train/cls_loss', cls_loss, step)
                writer.add_scalar('train/total_loss', loss, step)
            elif loss_type == 'seg(comb)':
                writer.add_scalar('train/seg1_loss', seg1_loss, step)
                writer.add_scalar('train/seg2_loss', seg2_loss, step)
                writer.add_scalar('train/total_loss', loss, step)
            elif loss_type == 'seg(comb)cls':
                writer.add_scalar('train/seg1_loss', seg1_loss, step)
                writer.add_scalar('train/seg2_loss', seg2_loss, step)
                writer.add_scalar('train/cls_loss', cls_loss, step)
                writer.add_scalar('train/total_loss', loss, step)
                
            if multiple and step%100==0:
                writer.add_images('train/img', img[:4], step)
                writer.add_images('train/mask', mask[:4], step)
                writer.add_images('train/mask_tk', mask_tk[:4], step)
                writer.add_images('train/outputs', outputs[:4], step)
            elif not multiple and step%100==0:
                writer.add_images('train/img', img[:4], step)
                writer.add_images('train/mask', mask[:4], step)
                writer.add_images('train/outputs', outputs[:4], step)

        loss.backward()
        optimizer.step()
        
        if step%200 == 0:
            with torch.no_grad():
                model.eval()
                for batch in valid_ip.iterator:
                    if multiple:
                        valid_img, valid_mask, valid_mask_tk, valid_info = batch
                        valid_img = img.cuda()
                        valid_mask = valid_mask.cuda()
                        valid_mask_tk = valid_mask_tk.cuda()
                    else:
                        valid_img, valid_mask, valid_info = batch
                        valid_img = img.cuda()
                        valid_mask = valid_mask.cuda()
                    valid_outputs = model(valid_img)

                    if multiple:
                        seg_loss = get_L2_sloss(valid_outputs, valid_mask_tk, dice_loss)
                        cls_loss = get_L2_closs(valid_outputs, valid_mask_tk, ce, pool)
                        total_loss = seg_loss + cls_loss
                    else:
                        seg_loss = get_L2_sloss(valid_outputs, valid_mask, dice_loss)
                        cls_loss = get_L2_closs(valid_outputs, valid_mask, ce, pool)
                        total_loss = seg_loss + cls_loss

                    if loss_type == 'seg':
                        writer.add_scalar('valid/seg_loss', seg_loss, step)
                    elif loss_type == 'segcls':
                        writer.add_scalar('valid/seg_loss', seg_loss, step)
                        writer.add_scalar('valid/cls_loss', cls_loss, step)
                        writer.add_scalar('valid/total_loss', total_loss, step)

                    if multiple:
                        writer.add_images('valid/img', valid_img[:4], step)
                        writer.add_images('valid/mask', valid_mask[:4], step)
                        writer.add_images('valid/mask_tk', valid_mask_tk[:4], step)
                        writer.add_images('valid/outputs', valid_outputs[:4], step)
                    else:
                        writer.add_images('valid/img', valid_img[:4], step)
                        writer.add_images('valid/mask', valid_mask[:4], step)
                        writer.add_images('valid/outputs', valid_outputs[:4], step)

                valid_ip.reset_iter()
    return True

def experiment_info(section):
    print(f"##### Experiment: {section} #####")
    for key in parser.options(section):
        data = parser.get(section, key)
        print(f'{key} => {data}')
    print('###############################\n')

if __name__ == '__main__':
    experiment_idx_list = [9, 10]
    for i in experiment_idx_list:
        experiment_info(f'train{i}')

        learning_rate = parser.get(f'train{i}', 'learning_rate')
        batch_size = parser.get(f'train{i}', 'batch_size')
        model_type = parser.get(f'train{i}', 'model_type')
        encoder_name = parser.get(f'train{i}', 'encoder_name')
        loss_type = parser.get(f'train{i}', 'loss_type')
        pooling_type = parser.get(f'train{i}', 'pooling_type')
        experiment_name = parser.get(f'train{i}', 'experiment_name')
        multiple = bool(parser.get(f'train{i}', 'multiple'))

        try:
            flag = train(f'train{i}', model_type, encoder_name, pooling_type, loss_type, learning_rate, batch_size, experiment_name, multiple)
            if flag == True:
                time.sleep(3)
            else:
                print("Exception occured!!\nExperiment Failed!!")
        except Exception as ex:
            print("Error: ",ex)
            print("Skipped the experiment!!")