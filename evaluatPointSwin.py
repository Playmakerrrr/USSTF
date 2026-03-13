# -*- coding: UTF-8 -*-

import argparse
import os
from pathlib import Path
import numpy as np
import torch
from osgeo import gdal_array
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

from skimage.metrics import structural_similarity as compare_ssim

from sewar import rmse, ssim, sam, psnr
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from Config import ConfigForEvaluation, ConfigForEvaluationForSwin


# def uiqi(im1, im2, block_size=64, return_map=False):
#     if im1.shape[0] == 6:  # 调整成标准的[长，宽，通道]
#         im1 = im1.transpose(1, 2, 0)
#         im2 = im2.transpose(1, 2, 0)
#     if len(im1.shape) == 3:
#         return np.array(
#             [uiqi(im1[:, :, i], im2[:, :, i], block_size, return_map=return_map) for i in range(im1.shape[2])])
#     delta_x = np.std(im1, ddof=1)
#     delta_y = np.std(im2, ddof=1)
#     delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
#     mu_x = np.mean(im1)
#     mu_y = np.mean(im2)
#     q1 = delta_xy / (delta_x * delta_y)
#     q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
#     q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
#     q = q1 * q2 * q3
#     return q


def calculate_ergas(real_images, predicted_images):
    """
    计算增强的灰度相似性指数（ERGAS）。

    参数:
    real_images -- 真实图像的列表，每个元素是一个通道。
    predicted_images -- 预测图像的列表，每个元素是一个通道。

    返回:
    ergas -- ERGAS指标的值。
    """
    ergas_sum = 0.0
    num_channels = len(real_images)
    # 遍历所有通道
    for real_img, pred_img in zip(real_images, predicted_images):
        # 计算RMSE
        channel_rmse = rmse(real_img, pred_img)

        # 计算图像平均亮度
        mean_brightness = np.mean(real_img)

        # 避免除以零
        mean_brightness_squared = mean_brightness ** 2+1e-100

        # 计算ERGAS值
        channel_ergas = (channel_rmse ** 2) / mean_brightness_squared

        # 累加ERGAS值
        ergas_sum += channel_ergas

    # 计算平均ERGAS值
    average_ergas = ergas_sum / num_channels

    # 缩放ERGAS值
    scaled_ergas = np.sqrt(average_ergas) * 6

    return scaled_ergas


def evaluate(y_true, y_pred, func):
    assert y_true.shape == y_pred.shape
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, :]
        y_pred = y_pred[np.newaxis, :]
    metrics = []
    for i in range(y_true.shape[0]):
        metrics.append(func(y_true[i], y_pred[i]))
    return metrics


def rmse_loss(y_true, y_pred):
    return evaluate(y_true, y_pred,
                    lambda x, y: sqrt(mean_squared_error(x.ravel(), y.ravel())))


# def ssim(y_true, y_pred, data_range=1):
#     return evaluate(y_true, y_pred,
#                     lambda x, y: compare_ssim(x, y, data_range=data_range))

def getMean(data):
    return sum(data) / len(data)


def cc(real_image, predicted_image):
    """
    计算两个图像的相关系数。

    参数:
    real_image -- 真实图像，形状为 (channels, height, width)
    predicted_image -- 预测图像，形状应与 real_image 相同

    返回:
    cc_array -- 一个数组，包含每个通道的相关系数
    """
    # 确保输入图像的形状相同
    if real_image.shape != predicted_image.shape:
        raise ValueError("The shapes of real_image and predicted_image must be the same.")

    # 计算每个通道的相关系数
    cc_array = []
    for i in range(real_image.shape[0]):  # 遍历所有通道
        real_channel = real_image[i]
        pred_channel = predicted_image[i]

        # 计算均值
        mu_x = np.mean(real_channel)
        mu_y = np.mean(pred_channel)

        # 计算协方差和标准差
        cov_xy = np.sum((real_channel - mu_x) * (pred_channel - mu_y))
        var_x = np.sum(np.square(real_channel - mu_x))
        var_y = np.sum(np.square(pred_channel - mu_y))

        # 计算相关系数
        cc = cov_xy / (np.sqrt(var_x * var_y) + 1e-100)  # 添加一个小的常数以避免除以零

        cc_array.append(cc)

    return np.array(cc_array)


def trans_sam(real_image, predicted_image):
    return sam(real_image.transpose(1, 2, 0), predicted_image.transpose(1, 2, 0)) * 180 / np.pi

def difference_map(real_image, predicted_image):
    if real_image.shape[0] == 6:  # 调整成标准的[长，宽，通道]
        real_image = real_image.transpose(1, 2, 0)
        predicted_image = predicted_image.transpose(1, 2, 0)
    difference_map=np.zeros((real_image.shape[0],real_image.shape[1]))
    difference_map_band = np.zeros((real_image.shape[0], real_image.shape[1],6))
    for i in range(real_image.shape[2]):  # 遍历所有通道
        difference_img=abs(predicted_image[:,:,i]-real_image[:,:,i])
        difference_map=difference_img+difference_map
        difference_map_band[:,:,i]=abs(predicted_image[:,:,i]-real_image[:,:,i])
    return difference_map,difference_map_band
import matplotlib.pyplot as plt

if __name__ == '__main__':


    ground_truth_dir = r"./image/EWZ/2019-529-520/L8_20190529_p.tif"

    predict_dir = r"./result/Swinresult_20190529.tif"

    ix = gdal_array.LoadFile(predict_dir)#.astype(np.int16)
    iy = gdal_array.LoadFile(ground_truth_dir).astype(np.int16)
    print(ix.shape)
    print(iy.shape)
    # if config.choice == 'CIA':
    #     ix[iy == 0] = 0
    scale_factor = 0.0001

    # xx = ix * 1 # scale_factor # 1
    xx = ix #* scale_factor # 1
    yy = iy * scale_factor
    print('RMSE', rmse_loss(yy, xx))
    print('RMSE', getMean(rmse_loss(yy, xx)))
    ssimc=[]
    for i in range(4):
        ssimc.append(ssim(yy[i], xx[i], MAX=1.0)[0])
    print('SSIM', getMean(ssimc))

    # print('UIQI', uiqi(xx, yy))
    # print('UIQI', getMean(uiqi(xx, yy)))

    print('CC', cc(yy, xx))
    print('CC', getMean(cc(yy, xx)))

    print('SAM', trans_sam(iy, ix))  # 在原论文中，只有sam是真实数据比的，其他指标都是放缩后再比的
    print('ERGAS', calculate_ergas(yy, xx))
    print('PSNR', psnr(yy, xx, MAX=1.0))


    import pandas as pd

    data = pd.DataFrame(columns=['Band', 'RMSE', 'SSIM', 'CC', 'SAM', 'ERGAS','PSNR'])

    data['Band'] = ['Blue','Green','Red','NIR','Mean']
    data['RMSE'] = rmse_loss(yy, xx)+[getMean(rmse_loss(yy, xx))]
    print(rmse_loss(yy, xx)+[getMean(rmse_loss(yy, xx))])
    data['SSIM'] = ssimc+[getMean(ssimc)]
    # print(len(uiqi(xx, yy).append(getMean(uiqi(xx, yy)))))
    # print(np.append(uiqi(xx, yy),getMean(uiqi(xx, yy))))
    # data['UIQI'] = np.append(uiqi(xx, yy),getMean(uiqi(xx, yy)))
    data['CC'] = np.append(cc(yy, xx),getMean(cc(yy, xx)))
    data['SAM'] = trans_sam(iy, ix)
    data['ERGAS'] = calculate_ergas(yy, xx)
    data['PSNR'] = psnr(yy, xx, MAX=1.0)

    print(data)
    # Save to CSV
    # data.to_csv('metrics_results.csv', index=False)
    data.to_csv('2_Swimtrs_selfblock.csv', mode='a',  index=False)
