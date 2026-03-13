import skimage.io as io
import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from skimage import transform
import scipy.misc as m
from osgeo import gdal
import glob

def crop(l8, S2_root1, S2_root2, crop_size_h,crop_size_w,prefix,save_dir,crop_label=False):
    # crop(L8_20170615_root,S2_20170620_root,crop_size,crop_size,prefix='L8S2_a1',save_dir=dataset_dir,crop_label=True)
    # 读取L8
    imagel8 = gdal.Open(l8)
    image_width = imagel8.RasterXSize
    image_height = imagel8.RasterYSize
    l8_channels = imagel8.RasterCount
    raw_L8 = imagel8.ReadAsArray(0, 0, image_width, image_height).transpose(1,2,0)
    # self.img_list.append(io.loadmat(self.imgpath_list[i])['img'])  # scipy.io.loadmat()加载高光谱数据集.mat;读取.mat格式的数据
    # self.img_L8.append(data)
    # raw_L8 = img_L8   #沿维度2堆叠
    # raw_S2 = img_S2   #沿维度2堆叠

    # 读取S2
    imageS2_root1 = gdal.Open(S2_root1)
    image_width = imageS2_root1.RasterXSize
    image_height = imageS2_root1.RasterYSize
    s2_channels = imageS2_root1.RasterCount
    raw_S2_root1 = imageS2_root1.ReadAsArray(0, 0, image_width, image_height).transpose(1,2,0)

    imageS2_root2 = gdal.Open(S2_root2)
    image_width = imageS2_root2.RasterXSize
    image_height = imageS2_root2.RasterYSize
    s2_channels = imageS2_root2.RasterCount
    raw_S2_root2 = imageS2_root2.ReadAsArray(0, 0, image_width, image_height).transpose(1,2,0)

    # raw_S2 = [raw_S2_root1, raw_S2_root2]
    raw_S2 = np.dstack((raw_S2_root1, raw_S2_root2))   #按深度拼接（shape[]方向）


    h_L8,w_L8,ch_L8 = raw_L8.shape
    h_S2,w_S2,ch_S2 = raw_S2.shape

    index = 0

    x2,y2 = 0,0
    x0,y0 = 0,0

    stride_h = crop_size_h  #32
    stride_w = crop_size_w

    while(y2<h_L8 and y2*3<h_S2):
        while(x2<w_L8 and x2*3<w_S2):
            x1 = x0
            x2 = x1 + crop_size_w
            y1 = y0
            y2 = y1 +crop_size_h

            print(x1,y1,x2,y2)

            if(x2>w_L8 or y2>h_L8):
                break
            elif(x2*3>w_S2 or y2*3>h_S2):
                break

            else:
                #每次取一块进行操作
                patch_L8_label = raw_L8[y1:y2,x1:x2]  # 30m
                patch_S2_label = raw_S2[y1*3:y2*3,x1*3:x2*3]

                patch_L8 = np.zeros((crop_size_h // 3, crop_size_w // 3, ch_L8), dtype=np.uint8)    # 90m
                patch_L8_up = np.zeros((crop_size_h,crop_size_w, ch_L8),dtype=np.uint8)   # 30m
                patch_S2 = np.zeros((crop_size_h,crop_size_w,ch_S2),dtype=np.uint8)  # 30m
                for i in range(ch_L8):
                    patch_L8[:,:,i] = m.imresize(patch_L8_label[:,:,i], (crop_size_h//3,crop_size_w//3), 'bicubic')  #crop_size_h=32  30->90
                    # patch_L8[i, :,:] = m.imresize(patch_L8_label[i, :,:], (crop_size_h//3,crop_size_w//3), 'bicubic')  #crop_size_h=32  30->90

                for i in range(ch_L8):
                    patch_L8_up[:, :, i] = m.imresize(patch_L8[:, :, i], (crop_size_h, crop_size_w), 'bicubic')#90->30
                    # patch_L8_up[i, :,:] = m.imresize(patch_L8[i, :,:], (crop_size_h,crop_size_w), 'bicubic')   #90->30

                for i in range(ch_S2):
                    patch_S2[:, :, i] = m.imresize(patch_S2_label[:, :, i], (crop_size_h, crop_size_w), 'bicubic')  # 10m->30m
                    # patch_S2[i, :,:] = m.imresize(patch_S2_label[i, :,:], (crop_size_h,crop_size_w), 'bicubic')  # 10m->30m

                #patch_S2 = np.uint8(patch_S2)

                # patch_L8_vis = patch_L8[:,:,1:4][:,:,::-1]
                # patch_L8_label_vis = patch_L8_label[:,:,1:4][:,:,::-1]
                #
                # patch_S2_vis = patch_S2[:,:,:3][:,:,::-1]
                # patch_S2_label_vis = patch_S2_label[:,:,:3][:,:,::-1]
                #
                # io.imsave(os.path.join(save_dir,'L8_vis',prefix+"_%d.tif"%(index)),patch_L8_vis)
                # io.imsave(os.path.join(save_dir,'L8_label_vis',prefix+"_%d.tif"%(index)),patch_L8_label_vis)
                #
                # io.imsave(os.path.join(save_dir,'S2_1_vis',prefix+"_%d.tif"%(index)),patch_S2_vis)
                # io.imsave(os.path.join(save_dir,'S2_1_label_vis',prefix+"_%d.tif"%(index)),patch_S2_label_vis)

            x0 = x1 + stride_w

            io.imsave(os.path.join(save_dir,'L8',prefix+"_%d.tif"%(index)),patch_L8)  # 90
            io.imsave(os.path.join(save_dir,'L8_up',prefix+"_%d.tif"%(index)),patch_L8_up)  # 90->30
            io.imsave(os.path.join(save_dir,'L8_label',prefix+"_%d.tif"%(index)),patch_L8_label)   # reall8

            io.imsave(os.path.join(save_dir,'S2_1',prefix+"_%d.tif"%(index)),patch_S2)   # 10->30
            io.imsave(os.path.join(save_dir,'S2_1_label',prefix+"_%d.tif"%(index)),patch_S2_label)

            index = index + 1

        x0,x1,x2 = 0,0,0
        y0 = y1 + stride_h
    print("test")

def generate_trainval_list(pathdir):  #../dataset/train_data_S2L8_1
    labels_img_paths = os.listdir(os.path.join(pathdir,'L8_label'))
    labels_count_list=dict()
    for labels_img_path in tqdm(labels_img_paths):
        label = io.imread(os.path.join(pathdir,'L8_label',labels_img_path))
        most_count_label= np.argmax(np.bincount(label.flatten().tolist()))  #先将label拉成一维列表，然后找出数组中出现次数最多的元素
        labels_count_list[labels_img_path] = most_count_label
    values= labels_count_list.values()
    count_dict= Counter(values)
    print(count_dict)


def write_train_list(pathdir):   #"../dataset/train_data_S2L8_1"
    labels_img_paths = os.listdir(os.path.join(pathdir,'L8_label'))
    num_sets = len(labels_img_paths)
    indexs = list(range(num_sets))
    np.random.shuffle(indexs)
    train_set_num = 0.9 * num_sets
    train_f = open(os.path.join(pathdir,'train.txt'),'w')
    val_f = open(os.path.join(pathdir,'val.txt'),'w')
    trainval_f = open(os.path.join(pathdir,'trainval.txt'),'w')
    for index in range(num_sets):
        if(index<train_set_num):
            # print >>train_f,labels_img_paths[indexs[index]]
            #print(): file 参数的默认值为 sys.stdout，该默认值代表了系统标准输出，也就是屏幕
            print(labels_img_paths[indexs[index]], file=train_f)
        else:
            # print >>val_f,labels_img_paths[indexs[index]]
            print(labels_img_paths[indexs[index]], file=val_f)
        # print >>trainval_f,labels_img_paths[indexs[index]]
        print(labels_img_paths[indexs[index]], trainval_f)
    train_f.close()
    val_f.close()
    trainval_f.close()

if __name__ == "__main__":

    dataset_dir = r'F:\projects\HyperFusion-main\HyperFusion-main\HyperFusion\temp_one2two'
    L8_path = os.path.join(dataset_dir, 'L8')
    L8_vis_path = os.path.join(dataset_dir, 'L8_vis')
    L8_up_path = os.path.join(dataset_dir, 'L8_up')
    L8_label_path = os.path.join(dataset_dir, 'L8_label')
    L8_label_vis_path = os.path.join(dataset_dir, 'L8_label_vis')
    S2_1_path = os.path.join(dataset_dir, 'S2_1')
    S2_1_vis_path = os.path.join(dataset_dir, 'S2_1_vis')
    S2_1_label_path = os.path.join(dataset_dir, 'S2_1_label')
    S2_1_label_vis_path = os.path.join(dataset_dir, 'S2_1_label_vis')

    if (not os.path.exists(L8_path)):
        os.mkdir(L8_path)
    if (not os.path.exists(L8_vis_path)):
        os.mkdir(L8_vis_path)
    if (not os.path.exists(L8_up_path)):
        os.mkdir(L8_up_path)
    if (not os.path.exists(L8_label_path)):
        os.mkdir(L8_label_path)
    if (not os.path.exists(L8_label_vis_path)):
        os.mkdir(L8_label_vis_path)
    if (not os.path.exists(S2_1_path)):
        os.mkdir(S2_1_path)
    if (not os.path.exists(S2_1_vis_path)):
        os.mkdir(S2_1_vis_path)
    if (not os.path.exists(S2_1_label_path)):
        os.mkdir(S2_1_label_path)
    if (not os.path.exists(S2_1_label_vis_path)):
        os.mkdir(S2_1_label_vis_path)

    # s2path = "../../S2data/S2/"
    # l8path = "../../S2data/L8"

    l8path1 = "../../S2data/L8/L8_20180111.tif"
    l8path2 = "../../S2data/L8/L8_20180212.tif"
    s2path1 = "../../S2data/S2/S2_20180105_10m.tif"
    s2path2 = "../../S2data/S2/S2_20180204_10m.tif"
    s2path3 = "../../S2data/S2/S2_20180316_10m.tif"
    # default_datapath_S2 = s2path  # S2path
    # default_datapath_L8 = l8path  # S2path
    #
    # # data_folder = os.path.join(default_datapath, args.data_name)
    # if os.path.exists(default_datapath_S2):
    #     data_path_S2 = os.path.join(default_datapath_S2, "*.tif")
    #     data_path_L8 = os.path.join(default_datapath_L8, "*.tif")


    # imgpath_list_S2 = sorted(glob.glob(data_path_S2))  # 对应数据集所有文件
    # imgpath_list_L8 = sorted(glob.glob(data_path_L8))  # 对应数据集所有文件
    # self.img_list = []
    # self.img_S2 = []
    # for i in range(len(imgpath_list_S2)):
    crop(l8path1, s2path1, s2path2, 32, 32, prefix='L8S2_a1', save_dir=dataset_dir)
    crop(l8path1, s2path2, s2path3, 32, 32, prefix='L8S2_a2', save_dir=dataset_dir)
    crop(l8path1, s2path1, s2path3, 32, 32, prefix='L8S2_a3', save_dir=dataset_dir)
    crop(l8path2, s2path1, s2path2, 32, 32, prefix='L8S2_b1', save_dir=dataset_dir)
    crop(l8path2, s2path2, s2path3, 32, 32, prefix='L8S2_b2', save_dir=dataset_dir)
    crop(l8path2, s2path1, s2path3, 32, 32, prefix='L8S2_b4', save_dir=dataset_dir)

    generate_trainval_list(dataset_dir)
    write_train_list(dataset_dir)

