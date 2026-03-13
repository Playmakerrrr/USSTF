#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The framework of traing process
~~~~~
Before python train.py, please ensure running the "python -m visdom.server -port=xxx" where xxx is the port assign in options
"""

import imgvision as iv
from model.net3 import *
# from data.net3dataset import *
from osgeo import gdal
import scipy.misc as m
import argparse
# import options.options as option
import numpy as np
import options.options as option
from skimage.metrics import structural_similarity as compare_ssim  # (realimage(H x W x N), generate(H x W x N))
from skimage.metrics import peak_signal_noise_ratio


#计算CC相关系数 mask:融合图像 label:理想参考图像
def compute_cc(mask,label):

    dim = mask.ndim
    if dim == 2:
        mask = mask - np.mean(mask)
        label = label - np.mean(label)
        cov = np.sum(mask * label)
        d1 = np.sum(mask * mask)
        d2 = np.sum(label * label)
        cc = cov / (np.sqrt(d1) * np.sqrt(d2))       #+ 1e-8)
        return cc
    else:
        # shape:(N,W,H)
        c = []
        for i in range(np.shape(mask)[0]):
            fake = mask[i]
            real = label[i]
            fake = fake - np.mean(fake)
            real = real - np.mean(real)
            cov = np.sum(fake * real)
            d1 = np.sum(fake * fake)
            d2 = np.sum(real * real)
            cc = cov / (np.sqrt(d1) * np.sqrt(d2))  #分波段计算
            c.append(cc)
        num = np.shape(mask)[0]
        cc = np.sum(c) / num
        return cc


def imsave(img, path, Dtype):
    if len(img.shape) == 3:
        (n, h, w) = img.shape
    else:
        (h, w) = img.shape
        n = 1
    driver = gdal.GetDriverByName("GTiff")

    if Dtype == 'uint8':
        datatype = gdal.GDT_Byte
    elif Dtype == 'uint16':
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    dataset = driver.Create(path, w, h, n, datatype)
    if len(img.shape) == 3:
        for t in range(n):
            dataset.GetRasterBand(t + 1).WriteArray(img[t])
    else:
        dataset.GetRasterBand(1).WriteArray(img)

    del dataset


def transform(x):
    # x = x.transpose(2, 0, 1)
    channel, _, _ = x.shape
    newdata = np.zeros(x.shape)
    for i in range(channel):
        data = x[i,:,:]
        # data[data < 0] = 0
        # data[data > 10000] = 10000  # 将大于10000的值设为10000
        data = data.astype(np.float32)
      #  out = data * 0.0001
        newdata[i, :, :] = data
    return newdata

if __name__ == "__main__":

    # train_opt = TrainOptions().parse()
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='./options/test/net3_fusion.json', help='Path to option JSON file.')
    # parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    testopt = option.dict_to_nonedict(opt)
    PATCH_SIZE = 32 # 256 # 64 # 16 #32 # 64  #训练时大小

    test_model = fusion(testopt)
    test_model.load_network(opt)

    l8path1 = "Dezhou_data/S2_20180922_10m_cut.tif"
    s2path1 = "Dezhou_data/S2_20180922_10m_cut.tif"

    # 读取L8
    imagel8 = gdal.Open(l8path1)
    l8image_width = imagel8.RasterXSize
    l8image_height = imagel8.RasterYSize
    l8_channels = imagel8.RasterCount
    raw_L8 = imagel8.ReadAsArray(0, 0, l8image_width, l8image_height)

    # raw_L8 = transform(raw_L8)

    ch_L8, h_L8, w_L8 = raw_L8.shape
    image_size = h_L8
    # ch_S2, h_S2, w_S2 = raw_S2_1.shape

    images2 = gdal.Open(s2path1)
    s2image_width = images2.RasterXSize
    s2image_height = images2.RasterYSize
    s2_channels = images2.RasterCount
    raw_S2 = images2.ReadAsArray(0, 0, s2image_width, s2image_height)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs2 = raw_S2

    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (image_size - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (image_size - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
    if (image_size - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(image_size - PATCH_SIZE)
    if (image_size - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(image_size - PATCH_SIZE)

    output_image = np.zeros(inputs2.shape)
    input_lr = inputs2
    # target_hr = raw_L8
    # 遍历所有的 Patch
    for i in range(len(h_index_list)):
        for j in range(len(w_index_list)):
            h_start = h_index_list[i]
            w_start = w_index_list[j]

            input_patch = input_lr[:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
            target_patch = raw_L8[:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]

            # 预处理图像
            input_patch = transform(input_patch)
            target_patch = transform(target_patch)

            input_patch = torch.from_numpy(input_patch).float().unsqueeze(0).cuda()

            # 预测输出
            # output = model(input_patch)
            output = test_model.test(input_patch)
            output = output.squeeze()

            # 将预测结果拼接回完整图像
            h_end = h_start + PATCH_SIZE
            w_end = w_start + PATCH_SIZE
            cur_h_start, cur_h_end = 0, PATCH_SIZE
            cur_w_start, cur_w_end = 0, PATCH_SIZE

            if i != 0:
                h_start += PATCH_SIZE // 4
                cur_h_start = PATCH_SIZE // 4
            if i != len(h_index_list) - 1:
                h_end -= PATCH_SIZE // 4
                cur_h_end -= PATCH_SIZE // 4
            if j != 0:
                w_start += PATCH_SIZE // 4
                cur_w_start = PATCH_SIZE // 4
            if j != len(w_index_list) - 1:
                w_end -= PATCH_SIZE // 4
                cur_w_end -= PATCH_SIZE // 4

            output_image[:, h_start: h_end, w_start: w_end] = \
                output[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()

    # target = test_model.test(opt, inputs2)
    target = output_image

    save_path = "./result/" + 'Swinresult222' + ".tif"
    imsave(target, save_path, 'float32')



