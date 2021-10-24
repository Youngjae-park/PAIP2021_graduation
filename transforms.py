#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:30:14 2021

@author: dacon
"""
import torch
import numpy as np
from torchvision.transforms.functional import rotate, vflip, hflip

class RandomCrop(object):
    def __init__(self, output_size, scale=16):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.scale = scale

    def __call__(self, x):
        image, mask = x
        h, w = mask.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top*self.scale: (top + new_h)*self.scale, left*self.scale: (left + new_w)*self.scale]
        mask = mask[:, top: top + new_h, left: left + new_w]
        return image, mask


class CenterCrop(object):
    def __init__(self, output_size, scale=16):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.scale = scale

    def __call__(self, x):
        image, mask = x
        h, w = mask.shape[1:]
        new_h, new_w = self.output_size

        top = int((h - new_h)/2)
        left = int((w - new_w)/2)

        image = image[:, top*self.scale: (top + new_h)*self.scale, left*self.scale: (left + new_w)*self.scale]
        mask = mask[:, top: top + new_h, left: left + new_w]
        return image, mask


class RandomRotate(object):
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, x):
        image, mask = x
        angle = np.random.randint(0, self.max_angle)
        image = rotate(image, angle)
        mask = rotate(mask, angle)
        return image, mask


class RandomFlip(object):
    def __call__(self, x):
        image, mask = x
        if np.random.random(1)>0.5:
            image = hflip(image)
            mask = hflip(mask)
        if np.random.random(1)>0.5:
            image = vflip(image)
            mask = vflip(mask)

        return image, mask

class RandomFlip_3(object):
    def __call__(self, x):
        image, mask, mask_tk = x
        if np.random.random(1)>0.5:
            image = hflip(image)
            mask = hflip(mask)
            mask_tk = hflip(mask_tk)
        if np.random.random(1)>0.5:
            image = vflip(image)
            mask = vflip(mask)
            mask_tk = vflip(mask_tk)

        return image, mask, mask_tk


class ToTensor(object):
    def __call__(self, x):
        image, mask = x
        image = np.transpose(image, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])
        image = torch.tensor(image)/255
        mask = torch.tensor(mask)/1.
        
        return image, mask

class ToTensor_3(object):
    def __call__(self, x):
        image, mask, mask_tk = x

        image = np.transpose(image, [2, 0, 1])
        mask = np.transpose(mask, [2, 0, 1])
        mask_tk = np.transpose(mask_tk, [2, 0, 1])

        image = torch.tensor(image)/255.
        mask = torch.tensor(mask)/1.
        mask_tk = torch.tensor(mask_tk)/1.
        return image, mask, mask_tk
