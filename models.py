#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 01:55:15 2021

@author: dacon
"""

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

from loss import *

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   

class L2_UNET(nn.Module):
    def __init__(self, n_class, out_size, encoder_name='efficientnet-b2'):
        super().__init__()
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            decoder_channels=[1,1,1,1,1],
            activation='sigmoid',
            in_channels=3,
            classes=n_class,
        ).cuda()
        
        self.encoder = model.encoder
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if encoder_name == 'efficientnet-b2':
            self.dconv_up = double_conv(472, 256)
            self.last_conv = nn.Conv2d(256, n_class, 1)
        elif encoder_name == 'efficientnet-b4':
            self.dconv_up = double_conv(608, 312)
            self.last_conv = nn.Conv2d(312, n_class, 1)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.encoder(x)
        # print(x[-1][0].shape)
        up = self.upsample(x[-1])
        # print(up[0].shape)
        x = torch.cat([x[-2], up], dim=1)
        x = self.dconv_up(x)
        x = self.last_conv(x)
        x = self.act(x)
        return x

if __name__ == '__main__':
    input_tensor = torch.zeros(1,3,512,512).cuda()
    gt_tensor = torch.zeros(1,32,32,1).cuda()
    model = L2_UNET(n_class=1, out_size=512//16)
    model = nn.DataParallel(model).cuda()
    outputs = model(input_tensor)
    pool = nn.AdaptiveMaxPool2d(1)
    # print(gt_tensor.shape, input_tensor.shape, outputs.shape)
    # print(get_L2_closs(outputs, gt_tensor,
    #                    smp.utils.losses.BCELoss(), pool))
    
