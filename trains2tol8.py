#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The framework of traing process
~~~~~
Before python train.py, please ensure running the "python -m visdom.server -port=xxx" where xxx is the port assign in options
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from random import random
import time
# import numpy as np
import hues
# import os
import argparse
# from data import get_dataloader
# from model import create_model
# from options.train_options import TrainOptions
from model.net2 import *
# from torch.utils.tensorboard import SummaryWriter
# from utils.visualizer import Visualizer
from data.net2dataset import *
import options.options as option
# from tensorboardX import SummaryWriter

# writer = SummaryWriter("./mylog")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    # a = torch.cuda.get_device_name(0)  # 返回GPU名字
    # print("a is ",a)
    # b = torch.cuda.get_device_name(1)
    # print("b is ", b)


    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train/net2_s2tol8.json', help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    train_opt = option.dict_to_nonedict(opt)

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    hues.info('Random seed: {}'.format(seed))
    set_random_seed(seed)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-opt', type=str, default='options/train/train_ESRCNN_S2L8_2.json', help='Path to option JSON file.')
    # train_dataloader = get_dataloader(train_opt, split='train', isTrain=True)
    trainloader = Dataset(opt, split='train')   # ['datasets']['train']['dataroot']
    train_dataloader = torch.utils.data.DataLoader(trainloader, batch_size=opt['datasets']['train']['batch_size'], num_workers=0, shuffle=True)
    dataset_size = len(train_dataloader)
    valloader = Dataset(opt['datasets']['train']['dataroot'], split='val')
    val_dataloader = torch.utils.data.DataLoader(valloader, batch_size=opt['val']['batch_size'], num_workers=0, shuffle=True)

    model = s2tol8(train_opt)  #原代码
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 强制使用 GPU 1
    # model = s2tol8(train_opt).to(device)  # 把模型放到 GPU 1

    total_epo = train_opt['train']['epoch_count']
    message = ''

    for epoch in range(1, total_epo + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        current_step = 0
        loss = 0.0
        for i, data in enumerate(train_dataloader):
            
            iter_start_time = time.time()
            # total_steps += train_opt.batchsize
            # epoch_iter += train_opt.batchsize
            current_step += 1
            # data = data.to(device)  #把数据放到GPU2
            model.set_input(data, True)
            loss += model.optimize_parameters()
            model.update_learning_rate()

        lossepo = loss / current_step
        lr_now = model.get_current_learning_rate()
        # writer.add_scalar("loss", lossepo, epoch)
        message += f'epoch:{epoch}/{total_epo}:lr:{lr_now}:loss:{lossepo}\n'
        hues.info("epoch: {:d}/{} :lr:{} : loss:{}".format(epoch, total_epo, lr_now, lossepo))
        # val:
        if epoch % train_opt['train']['print_freq'] == 0:
            loss_val = 0.0
            idx = 0
            with torch.no_grad():
                for i_val, val_data in enumerate(val_dataloader):
                    idx += 1
                    # img_name = val_data[3][0].split('.')[0]
                    # train_model.eval()
                    model.set_input(val_data, True)
                    loss_val += model.val() # 前向
                    # train_model.train()

            loss_val_epo = loss_val / idx
            # writer.add_scalar("loss_val", loss_val_epo, epoch)

            message += f'epoch: {epoch}, validation>loss:{loss_val_epo}\n'
            # hues.info("epoch: {:d}/{} : loss:{}".format(epoch, train_opt.epoch_count, lossepo))
            hues.info(f'<epoch:{epoch}/{total_epo}, validation> loss:{loss_val_epo}\n')
            model.save(epoch)

        with open('./checkpoints/net2' + '/s2tol8' + '/loss.txt', "w") as file:
            file.write(str(message))

    hues.info('Saving the final model.')
    model.save('latest')
    hues.info('End of training')
    # model.save_networks(train_opt.epoch_count)   # #save
    # train_model.saveAbundance()
