"""
Author: Kaiqing Lin
Date: 2024/6/23
File: Programming.py
"""
import os
import os.path as osp
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
import imageio.v2 as imageio
import cv2
from termcolor import cprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from termcolor import cprint


class Trainable_Resize(nn.Module):
    def __init__(self, output_size=(3, 256, 256)):
        super(Trainable_Resize, self).__init__()
        self.height_out = output_size[1]
        self.width_out = output_size[2]
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.l_pad = 0.0  # left padding
        self.r_pad = 0.0  # right padding
        self.u_pad = 0.0  # upper padding
        self.d_pad = 0.0  # lower padding

    def forward(self, image):  # image.shape [batch_sz, channel, H, W]
        self.l_pad = int((self.width_out - image.shape[3] + 1) / 2)
        self.r_pad = int((self.width_out - image.shape[3]) / 2)
        self.u_pad = int((self.height_out - image.shape[2] + 1) / 2)
        self.d_pad = int((self.height_out - image.shape[2]) / 2)

        x = torch.nn.functional.pad(
            image, (self.l_pad, self.r_pad, self.u_pad, self.d_pad), value=0)
        x = kornia.geometry.transform.scale(x, self.scale)

        return x, min(int(image.shape[2] * self.scale), self.height_out), min(int(image.shape[3] * self.scale),
                                                                              self.width_out)


# Original
class InputPadding(nn.Module):
    def __init__(self, img_size=(3, 32, 32), output_size=(3, 256, 256), normalization=None, input_aware=False,
                 padding_size=None, model_name=None, device=None, zero_padding=False):
        super(InputPadding, self).__init__()
        self.img_size = img_size
        self.channel = img_size[0]
        self.img_h = img_size[1]
        self.img_w = img_size[2]

        self.output_size = output_size
        self.out_h = output_size[1]
        self.out_w = output_size[2]

        self.u_pad = int((output_size[1] - img_size[1] + 1) / 2)
        self.d_pad = int((output_size[1] - img_size[1]) / 2)
        self.l_pad = int((output_size[2] - img_size[2] + 1) / 2)
        self.r_pad = int((output_size[2] - img_size[2]) / 2)

        self.normalization = normalization

        self.padding_size = padding_size

        self.model_name = model_name

        self.device = device

        self.input_aware = input_aware

        self.mask = None
        self.init_mask()

        self.delta = torch.nn.Parameter(
            data=torch.zeros(3, output_size[1], output_size[2]))
        self.zero_padding = zero_padding

    def init_mask(self):
        self.mask = torch.ones(self.output_size).to(self.device)
        # upper triangle and the diagonal are set to True, others are False
        up_tri = np.invert(np.tri(N=self.img_h, M=self.img_w, k=0, dtype=bool))
        up_tri = torch.from_numpy(up_tri)
        if self.padding_size is None or self.padding_size < int((self.out_h - self.img_h) // 2):
            self.mask[:, int((self.out_h - self.img_h) // 2):int((self.out_h + self.img_h) // 2), int(
                    (self.out_w - self.img_w) // 2):int(
                    (self.out_w + self.img_w) // 2)] = 0  # the location of img is set to zero
        else:
            self.mask[:, self.padding_size:self.out_h - self.padding_size,
            self.padding_size:self.out_w - self.padding_size] = 0

    def redefind_mask(self, img, img_h, img_w):
        self.channel = img.size()[1]
        self.img_h = img.size()[2]
        self.img_w = img.size()[3]

        # replace with real size from Trainable_Resize
        if img_h != -1:
            self.img_h = img_h
        if img_w != -1:
            self.img_w = img_w

        self.mask = torch.ones(self.output_size).to(self.device)
        # upper triangle and the diagonal are set to True, others are False
        up_tri = np.invert(np.tri(N=self.img_h, M=self.img_w, k=0, dtype=bool))
        up_tri = torch.from_numpy(up_tri).to(self.device)
        if self.padding_size is None or self.padding_size < int((self.out_h - self.img_h) // 2):
            self.mask[:, int((self.out_h - self.img_h) // 2):int((self.out_h + self.img_h) // 2), int(
                    (self.out_w - self.img_w) // 2):int(
                    (self.out_w + self.img_w) // 2)] = 0  # the location of img is set to zero
        else:
            self.mask[:, self.padding_size:self.out_h - self.padding_size,
            self.padding_size:self.out_w - self.padding_size] = 0

        self.u_pad = int((self.out_h - img.size()[2] + 1) / 2)
        self.d_pad = int((self.out_h - img.size()[2]) / 2)
        self.l_pad = int((self.out_w - img.size()[3] + 1) / 2)
        self.r_pad = int((self.out_w - img.size()[3]) / 2)

    def forward(self, image, img_h, img_w, zero_padding=False):
        self.redefind_mask(image, img_h, img_w)
        image = image.repeat(1, 3 - self.channel + 1, 1, 1)

        x = torch.nn.functional.pad(
            image, (self.l_pad, self.r_pad, self.u_pad, self.d_pad), value=0)

        if self.input_aware is True:
            print("Not Implement!")
            return

        delta = self.delta
        masked_delta = delta * self.mask

        if self.zero_padding or zero_padding:
            masked_delta = masked_delta * 0

        x_adv = x + masked_delta
        if self.normalization is not None:
            x_adv = self.normalization(x_adv)
        return x_adv


if __name__ == '__main__':
    pass
