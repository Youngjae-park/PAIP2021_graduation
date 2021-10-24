from configparser import ConfigParser
import os

import torch
import tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np

import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

from loss import *
from InputProducer import IPWrapper
from models import L2_UNET
import matplotlib.pyplot as plt

parser = ConfigParser()
parser.read('config.ini')

if __name__ == '__main__':
    i=5

    learning_rate = parser.get(f'train{i}', 'learning_rate')
    batch_size = parser.get(f'train{i}', 'batch_size')
    model_type = parser.get(f'train{i}', 'model_type')
    encoder_name = parser.get(f'train{i}', 'encoder_name')
    loss_type = parser.get(f'train{i}', 'loss_type')
    pooling_type = parser.get(f'train{i}', 'pooling_type')
    experiment_name = parser.get(f'train{i}', 'experiment_name')
    multiple = bool(parser.get(f'train{i}', 'multiple'))

    ip = IPWrapper(path=f'train{i}', is_train=False, batch_size=int(batch_size), multiple=multiple)

    for step in range(10):
        img, mask, mask_tk, info = ip.produce()

        fig = plt.figure()
        pool = nn.AdaptiveMaxPool2d(1)

        ax1 = fig.add_subplot(3, 2, 1)
        print(info[0])
        temp_img = img[0].permute(1,2,0)
        ax1.imshow(temp_img)
        ax1.set_title(f'Img #{step}')

        ax2 = fig.add_subplot(3, 2, 3)
        temp_mask = mask[0].permute(1,2,0)
        ax2.imshow(temp_mask)
        ax2.set_title(f'Mask #{step}')

        ax3 = fig.add_subplot(3, 2, 4)
        temp_mask_tk = mask_tk[0].permute(1,2,0)
        ax3.imshow(temp_mask_tk)
        ax3.set_title(f'Mask_tk #{step}')

        fig.suptitle(info[0])

        # print(mask)
        print(pool(mask)[0], pool(mask_tk)[0])

        # ax4 = fig.add_subplot(3, 2, 5)
        # ax4.imshow(pool(mask)[0])
        # ax4.set_title(f'Pool Mask #{step}')
        #
        # ax5 = fig.add_subplot(3, 2, 6)
        # ax5.imshow(pool(mask_tk)[0])
        # ax5.set_title(f'Pool Mask #{step}')

        plt.show()