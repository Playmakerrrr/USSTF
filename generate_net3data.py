import skimage.io as io
import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from osgeo import gdal

from osgeo import gdal, gdalconst

# def datanormal(x):
#     # x = x.transpose(2, 0, 1)
#     channel, _, _ = x.shape
#     newdata = np.zeros(x.shape)
#     for i in range(channel):
#         data = x[i,:,:]
#         max = np.max(data)
#         min = np.min(data)
#         newdata[i, :, :] = (data - min)/(max - min)  # * 255
#     return newdata


def crop(l8_real, s2_respl, crop_size_h,crop_size_w,prefix,save_dir,crop_label=False):

    ch_L8, h_L8, w_L8 = l8_real.shape
    ch_S2, h_S2, w_S2 = s2_respl.shape

    index = 0

    x2,y2 = 0,0
    x0,y0 = 0,0

    # stride_h = crop_size_h  #32
    # stride_w = crop_size_w
    stride_h = crop_size_h // 4  # 设置为裁剪块的四分之一，75% 重叠
    stride_w = crop_size_w // 4  # 设置为裁剪块的四分之一，75% 重叠

    while(y2<h_L8 and y2<h_S2):
        while(x2<w_L8 and x2<w_S2):
            x1 = x0
            x2 = x1 + crop_size_w
            y1 = y0
            y2 = y1 +crop_size_h

            print(x1,y1,x2,y2)

            if(x2>w_L8 or y2>h_L8):
                break
            elif(x2>w_S2 or y2>h_S2):
                break

            else:
                #每次取一块进行操作
                patch_L8_real = l8_real[:, y1:y2,x1:x2]  # 30m
                patch_S2_30m = s2_respl[:, y1:y2,x1:x2]

            x0 = x1 + stride_w

            io.imsave(os.path.join(save_dir,'L8_real',prefix+"_%d.tif"%(index)),patch_L8_real)  # 90

            io.imsave(os.path.join(save_dir,'S2_30m',prefix+"_%d.tif"%(index)),patch_S2_30m)  # 90m
            # io.imsave(os.path.join(save_dir,'S2_1',prefix+"_%d.tif"%(index)),patch_S2)  # 30m
            # io.imsave(os.path.join(save_dir,'S2_1_label',prefix+"_%d.tif"%(index)),patch_S2_label)  # reals2

            index = index + 1

        x0,x1,x2 = 0,0,0
        y0 = y1 + stride_h
    print("test")

def generate_trainval_list(pathdir):  #../dataset/train_data_S2L8_1
    labels_img_paths = os.listdir(os.path.join(pathdir,'L8_real'))
    labels_count_list=dict()
    for labels_img_path in tqdm(labels_img_paths):
        label = io.imread(os.path.join(pathdir,'L8_real',labels_img_path))
        most_count_label= np.argmax(np.bincount(label.flatten().tolist()))  #先将label拉成一维列表，然后找出数组中出现次数最多的元素
        labels_count_list[labels_img_path] = most_count_label
    values= labels_count_list.values()
    count_dict= Counter(values)
    print(count_dict)


def write_train_list(pathdir):   #"../dataset/train_data_S2L8_1"
    labels_img_paths = os.listdir(os.path.join(pathdir,'L8_real'))
    num_sets = len(labels_img_paths)
    indexs = list(range(num_sets))
    np.random.shuffle(indexs)
    train_set_num = 0.8 * num_sets   # 0.9 -> 0.8
    train_f = open(os.path.join(pathdir,'train.txt'),'w')
    val_f = open(os.path.join(pathdir,'val.txt'),'w')
    # trainval_f = open(os.path.join(pathdir,'trainval.txt'),'w')
    for index in range(num_sets):
        if(index<train_set_num):
            # print >>train_f,labels_img_paths[indexs[index]]
            # print(): file 参数的默认值为 sys.stdout，该默认值代表了系统标准输出，也就是屏幕
            print(labels_img_paths[indexs[index]], file=train_f)
        else:
            # print >>val_f,labels_img_paths[indexs[index]]
            print(labels_img_paths[indexs[index]], file=val_f)
    train_f.close()
    val_f.close()
    # trainval_f.close()

if __name__ == "__main__":

    # dataset_dir = r'F:\projects\HyperFusion-main\HyperFusion-main\HyperFusion\temp'
    dataset_dir = '../traindata/net3_fusion'
    L8_90m_path = os.path.join(dataset_dir, 'L8_real')
    S2_90m_path = os.path.join(dataset_dir, 'S2_30m')

    if (not os.path.exists(L8_90m_path)):
        os.mkdir(L8_90m_path)

    if (not os.path.exists(S2_90m_path)):
        os.mkdir(S2_90m_path)

    # l8path1 = "../image/AHB/L8_20210425.tif"
    # s2path1 = "../image/AHB/S2_20210414_30m.tif"
    # l8path1 = "../image/EWZ/2019-529-520/L8_20190529_p.tif"
    # s2path1 = "../image/EWZ/2019-529-520/S2_20190520_30m.tif"
    # l8path1 = "../image/EWZ/2019-3-26&3-11/L8_20190326_p.tif"
    # s2path1 = "../image/EWZ/2019-3-26&3-11/S2_20190311_30m.tif"
    # l8path1 = "../image/EWZ/2019-918/L8_20190918_p.tif"
    # s2path1 = "../image/EWZ/2019-918/S2_20190902_30m.tif"
    # l8path1 = "../image/EWZ/2019-1105-1027/L8_20191105_p.tif"
    # s2path1 = "../image/EWZ/2019-1105-1027/S2_20191027_30m.tif"
    # l8path1 = "../image/EWZ/20200101/L8_20200101_p.tif"
    # s2path1 = "../image/EWZ/20200101/S2_20191211_30m.tif"
    # l8path1 = "../image/EWZ/2020-328-310/L8_20200328_p.tif"
    # s2path1 = "../image/EWZ/2020-328-310/S2_20200310_30m.tif"
    # l8path1 = "../image/EWZ/2020-04-/L8_20200413_p.tif"
    # s2path1 = "../image/EWZ/2020-04-/S2_20200404_30m.tif"
    # l8path1 = "../image/EWZ/20200429/L8_20200429_p.tif"
    # s2path1 = "../image/EWZ/20200429/S2_20200424_30m.tif"
    # l8path1 = "../image/EWZ/20200524/L8_20200524_p.tif"
    # s2path1 = "../image/EWZ/20200524/S2_20200519_30m.tif"
    # l8path1 = "../image/EWZ/20200904/L8_20200904_p.tif"
    # s2path1 = "../image/EWZ/20200904/S2_20200901_30m.tif"
    # l8path1 = "../image/EWZ/20200920/L8_20200920_p.tif"
    # s2path1 = "../image/EWZ/20200920/S2_20200916_30m.tif"
    # l8path1 = "../image/EWZ/20201218/L8_20201218_p.tif"
    # s2path1 = "../image/EWZ/20201218/S2_20201210_30m.tif"
    # l8path1 = "../image/EWZ/20201225/L8_20201225_p.tif"
    # s2path1 = "../image/EWZ/20201225/S2_20201220_30m.tif"
    #-----------AHB--------------
    # l8path1 = "../image/AHB/L8_20190130_p.tif"
    # s2path1 = "../image/AHB/S2_20190120_cut_30m.tif"
    # l8path1 = "../image/AHB/L8_20190623_p.tif"
    # s2path1 = "../image/AHB/S2_20190619_cut_30m.tif"
    # l8path1 = "../image/AHB/L8_20200524_p.tif"
    # s2path1 = "../image/AHB/S2_20200519_30m.tif"
    # l8path1 = "../image/AHB/L8_20201031_p.tif"
    # s2path1 = "../image/AHB/S2_20201016_30m.tif"

    # Dezhou
    # l8path1 = "../Dezhou_data/L8_20180111_10m_cut.tif"
    # s2path1 = "../Dezhou_data/S2_20180105_10m_cut.tif"
    # l8path1 = "../Dezhou_data/L8_20180212_10m_cut.tif"
    # s2path1 = "../Dezhou_data/S2_20180204_10m_cut.tif"
    # l8path1 = "../Dezhou_data/L8_20180908_10m_cut.tif"
    # s2path1 = "../Dezhou_data/S2_20180907_10m_cut.tif"
    # l8path1 = "../Dezhou_data/L8_20180924_10m_cut.tif"
    # s2path1 = "../Dezhou_data/S2_20180922_10m_cut.tif"
    # l8path1 = "../Dezhou_data/L8_20181010_10m_cut.tif"
    # s2path1 = "../Dezhou_data/S2_20181002_10m_cut.tif"
    l8path1 = "../Dezhou_data/L8_20181026_10m_cut.tif"
    s2path1 = "../Dezhou_data/S2_20181017_10m_cut.tif"


    imagel8 = gdal.Open(l8path1)
    l8image_width = imagel8.RasterXSize
    l8image_height = imagel8.RasterYSize
    l8_channels = imagel8.RasterCount
    raw_L8 = imagel8.ReadAsArray(0, 0, l8image_width, l8image_height)
    # raw_L8 = datanormal(raw_L8)
    # raw_L8 = raw_L8 / 10000.0

    images2 = gdal.Open(s2path1)
    s2image_width = images2.RasterXSize
    s2image_height = images2.RasterYSize
    s2_channels = images2.RasterCount
    raw_S2 = images2.ReadAsArray(0, 0, s2image_width, s2image_height)
    # raw_S2 = datanormal(raw_S2)
    # raw_S2 = raw_S2 / 10000.0

    cropsize = 32 # 256 # 64 #16 # 32 # 64 # 240   # 32  16

    crop(raw_L8, raw_S2, cropsize, cropsize, prefix='net3', save_dir=dataset_dir)

    generate_trainval_list(dataset_dir)
    write_train_list(dataset_dir)

