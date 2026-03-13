#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ipdb as ipdb
import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np
import random
import skimage.io as skio
import collections


# def transform_image(image, flip_num, rotate_num0, rotate_num):
#     image = image.astype(np.float32)

#     # 数据增强：水平翻转
#     if flip_num == 1:
#         image = image[:, :, ::-1]
#     # 数据增强：旋转
#     C, H, W = image.shape
#     if rotate_num0 == 1:
#         # -90旋转角度
#         if rotate_num == 2:
#             image = image.transpose(0, 2, 1)[::-1, :]
#         # 90
#         elif rotate_num == 1:
#             image = image.transpose(0, 2, 1)[:, ::-1]
#         # 180
#         else:
#             image = image.reshape(C, H * W)[:, ::-1].reshape(C, H, W)

#     image = torch.from_numpy(image.copy())
#     return image

def transform(x):
    channel, _, _ = x.shape
    newdata = np.zeros(x.shape)
    for i in range(channel):
        data = x[i,:,:]
        data[data < 0] = 0
        # data[data > 10000] = 10000  # 将大于10000的值设为10000
        data = data.astype(np.float32)
      #  out = data * 0.0001
        newdata[i, :, :] = data
    return newdata

class Dataset(data.Dataset):
    def __init__(self, args, split, isTrain=True):
        super(Dataset, self).__init__()

        self.args = args

        self.isTrain = isTrain
        self.s2_channels = 4   # 可变
        self.l8_channels = 4

        # self.root = '../traindata'
        self.root = r'./traindata/net3_fusion'
        self.split = split
        self.files = collections.defaultdict(list)

        for split in ["train", "val"]:
            file_list = tuple(open(self.root + '/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __getitem__(self, index):

        img_name = self.files[self.split][index]

        L8_real = self.root + '/L8_real/' + img_name
        S2_30m = self.root + '/S2_30m/' + img_name

        L8_real = skio.imread(L8_real).transpose(2,0,1)
        L8_real = transform(L8_real)

        S2_30m = skio.imread(S2_30m).transpose(2,0,1)
        S2_30m = transform(S2_30m)

        # # 随机选择翻转和旋转操作
        # flip_num = np.random.choice(2)
        # rotate_num0 = np.random.choice(2)
        # rotate_num = np.random.choice(3)
        # L8_real = transform_image(L8_real, flip_num, rotate_num0, rotate_num)
        # S2_30m = transform_image(S2_30m, flip_num, rotate_num0, rotate_num)

        return {"S2_30m":S2_30m,
                'L8_real':L8_real,
                "name":img_name
                }

    def __len__(self):
        return len(self.files[self.split])       #len(self.img_S2)


