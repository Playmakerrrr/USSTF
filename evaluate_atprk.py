import tifffile
from skimage.metrics import structural_similarity as compare_ssim  # (realimage(H x W x N), generate(H x W x N))
from skimage.metrics import peak_signal_noise_ratio
import scipy.io as scio
import gdal
import numpy as np
import imgvision as iv


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


# def datanormal(x):
#     # x = x.transpose(2, 0, 1)
#     channel, _, _ = x.shape
#     newdata = np.zeros(x.shape)
#     for i in range(channel):
#         data = x[i,:,:]
#         max = np.max(data)
#         min = np.min(data)
#         newdata[i, :, :] = (data - min)/(max - min)  # * 255
#     return newdata.astype(np.float32)

def transform(x):
    # x = x.transpose(2, 0, 1)
    channel, _, _ = x.shape
    newdata = np.zeros(x.shape)
    for i in range(channel):
        data = x[i,:,:]
        data[data < 0] = 0
        # data[data > 10000] = 10000  # 将大于10000的值设为10000
        data = data.astype(np.float32)
        out = data * 0.0001
        newdata[i, :, :] = out
    return newdata

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


# atprkpath = r'F:\projects\atprk\result\atprk_L8_20190224.mat'
atprkpath = r'G:\projects\atprk\EWZ20190529\atprkresult_20190529_p.mat'
atprk = scio.loadmat(atprkpath)['Z'].astype(np.float32).transpose(2,0,1)
# realimagepath = r'F:\projects\HyperFusion-main\HyperFusion-main_band4\hailardatab4\L8\L8_20190224.tif'
realimagepath = r'G:/projects/threenet_orign_10000/image/EWZ/2019-529-520/L8_20190529_p.tif'

# 读取L8
imagel8 = gdal.Open(realimagepath)
l8image_width = imagel8.RasterXSize
l8image_height = imagel8.RasterYSize
l8_channels = imagel8.RasterCount
raw_L8 = imagel8.ReadAsArray(0, 0, l8image_width, l8image_height)
ch_x, h_L8, w_L8 = raw_L8.shape
# raw_L8 = datanormal(raw_L8)
# atprk = datanormal(atprk)
raw_L8 = transform(raw_L8)
atprk = transform(atprk)

CCz = compute_cc(atprk, raw_L8)
SSIM = compare_ssim(raw_L8.transpose(1, 2, 0), atprk.transpose(1, 2, 0), multichannel=True)  # (realimage, generate)

# 图像范围 0~1 时， 默认l=1，不需填入参数
Metric = iv.spectra_metric(raw_L8, atprk)
# 评价SAM：
SAM = Metric.SAM()
# 评价PSNR：
# PSNR = Metric.PSNR()
PSNR = peak_signal_noise_ratio(raw_L8, atprk)
# 评价SSIM：
# SSIM = Metric.SSIM()
# 评价ERGAS:
ERGAS = Metric.ERGAS()
# 评价MSE:
MSE = Metric.MSE()
# MSE1 = compare_mse1(l8_label, target)

RMSE = np.sqrt(MSE)
# 评价PSNR, SAM, ERGAS, SSIM, RMSE
# PSNR, SAM, ERGAS, SSIM, RMSE = Metric.get_Evaluation()
# print(f'whole: SAM: {SAM},PSNR:{PSNR},SSIM:{SSIM},ERGAS:{ERGAS},RMSE:{RMSE},CC:{CCz}')
print(f'whole: PSNR:{PSNR},SSIM:{SSIM},RMSE:{RMSE},CC:{CCz}')

for i in range(ch_x):
    # databand = reconstruct[i,:,:]
    databand = atprk[i, :, :]
    CC = compute_cc(databand, raw_L8[i, :, :])
    Metric1 = iv.spectra_metric(raw_L8[i, :, :], databand)
    # 评价PSNR：
    # PSNR_band = Metric1.PSNR()
    PSNR_band = peak_signal_noise_ratio(raw_L8[i, :, :], databand)
    # 评价SSIM：
    # SSIM_band = Metric1.SSIM()
    testSSIM = compare_ssim(raw_L8[i, :, :], databand)
    # 评价ERGAS:
    ERGAS_band = Metric1.ERGAS()
    # 评价MSE:
    MSE_band = Metric1.MSE()
    RMSE_band = np.sqrt(MSE_band)
    # print(f'band{i+1}: SAM: {SAM_band},PSNR:{PSNR_band},SSIM:{SSIM_band},ERGAS:{ERGAS_band},MSE:{MSE_band}')
    print(f'band{i + 2}: PSNR:{PSNR_band},SSIM:{testSSIM},RMSE:{RMSE_band},CC:{CC}')

# 获取数据维度
num_bands, num_rows, num_cols = atprk.shape
save_path = r'G:\projects\atprk\EWZ20190529\EWZrs_20190529.tif'
imsave(atprk, save_path, 'float32')    # 保存为TIFF文件



