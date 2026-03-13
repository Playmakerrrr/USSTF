import matplotlib.pyplot as plt
import numpy as np
import re

path_log = r"./checkpoints/net1/l8tos2/loss.txt"
# path_log = r"./checkpoints/net2/s2tol8/loss.txt"
# path_log = r"./checkpoints/net3/fusion/loss.txt"

with open(path_log, "r") as fr:
    lines = fr.readlines()

def get_index(list_tmp, val):
    for index, value in enumerate(list_tmp):
        if val == value:
            return index
    return -1


lossname = 'loss'

list_iteration = []
list_iteration_val = []
list_loss_val = []
list_loss = []
for line in lines:
    # line = line.rstrip()  # 此函数只会删除头和尾的字符，中间的不会删除；如果strip()的参数为空，那么会默认删除字符串头和尾的空白字符(包括\n，\r，\t这些)
    # line_split = line.split(":")

    if ("loss:" in line) and ("validation>" not in line):   #("epoch: " in line) and  #筛选有用的行
        line = line.rstrip()  #此函数只会删除头和尾的字符，中间的不会删除；如果strip()的参数为空，那么会默认删除字符串头和尾的空白字符(包括\n，\r，\t这些)
        #line=str(line)
        # line.replace('/',':')
        b = re.sub('/', ':', line)
        # b1 = re.sub(':', ',', b)
        # b2 = re.sub(' ', '', b1)  #对对应行进行处理，便于后续分离各个
        # c = b2.split(',')

        #print(line)
        # line_split = b2.split(",")
        line_split = b.split(":")
        index_iteration = get_index(line_split, "epoch")   #Iteration #定位到需要的数据
        # index_loss = get_index(line_split, "loss")  #loss
        index_loss = get_index(line_split, lossname)  #loss
        if -1 == index_iteration or -1 == index_loss:
            continue

        iteration_ = line_split[index_iteration + 1]  #将<epoch对应的后面的数值取出来
        loss_ = line_split[index_loss + 1]     #2  #将l_pix对应的后面的数值取出来
        list_iteration.append(int(iteration_))
        list_loss.append(float(loss_))
    else:
        line = line.rstrip()  # 此函数只会删除头和尾的字符，中间的不会删除；如果strip()的参数为空，那么会默认删除字符串头和尾的空白字符(包括\n，\r，\t这些)
        # line=str(line)
        # line.replace('/',':')
        b_val = re.sub(',', ':', line)
        b1_val = re.sub('>', ':', b_val)
        # b2 = re.sub('', '', b1)  #对对应行进行处理，便于后续分离各个
        # c = b2.split(',')

        line_split_val = b1_val.split(":")
        index_iteration_val = get_index(line_split_val, "epoch")  # Iteration #定位到需要的数据
        # index_loss_val = get_index(line_split_val, "loss")  # loss
        index_loss_val = get_index(line_split_val, lossname)  # loss
        if -1 == index_iteration or -1 == index_loss:
            continue

        iteration_val = line_split_val[index_iteration_val + 1]  # 将<epoch对应的后面的数值取出来
        loss_val = line_split_val[index_loss_val + 1]  # 2  #将l_pix对应的后面的数值取出来
        list_iteration_val.append(int(iteration_val))
        list_loss_val.append(float(loss_val))

plt.figure(figsize=(8, 6))  # 定义图的大小
plt.xlabel("epoch")  # X轴标签
plt.ylabel("trainloss")  # Y轴坐标标签
plt.title("trainloss")  # 曲线图的标题

plt.plot(list_iteration, list_loss, color="red", linewidth=2)
plt.savefig(r"./lossimage/2-EWZ_L8_20200413_net3-train_" + lossname + "-1.jpg")
plt.show()

plt.figure(figsize=(8, 6))  # 定义图的大小
plt.xlabel("valepoch")  # X轴标签
plt.ylabel("valloss")  # Y轴坐标标签
plt.title("valloss")  # 曲线图的标题

plt.plot(list_iteration_val, list_loss_val, color="red", linewidth=2)
plt.savefig(r"./lossimage/2-EWZ_L8_20200413_net3-val_" + lossname + "-1.jpg")
plt.show()
