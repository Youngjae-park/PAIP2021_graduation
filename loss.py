#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 02:07:46 2021

@author: dacon
"""

import torch

def get_L2_closs(out, target, criterion, pool=None):
    # target = target.permute(0,3,1,2)
    total_loss = criterion(pool(out), pool(target))
    return total_loss

def get_L2_sloss(out, target, criterion):
    # target = target.permute(0,3,1,2)
    total_loss = criterion(out, target)
    return total_loss
