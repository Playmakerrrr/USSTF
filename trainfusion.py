#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The framework of traing process
~~~~~
Before python train.py, please ensure running the "python -m visdom.server -port=xxx" where xxx is the port assign in options
"""
from random import random
import time
import hues
import argparse
from model.net3 import *
# from torch.utils.tensorboard import SummaryWriter
# from utils.visualizer import Visualizer
from data.net3dataset import *
import options.options as option
from datetime import datetime

# from tensorboardX import SummaryWriter

# writer = SummaryWriter("./mylog")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    count = torch.cuda.device_count()
    print(f"gpu count:{count}")
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train/net3_fusion.json', help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    train_opt = option.dict_to_nonedict(opt)

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    hues.info('Random seed: {}'.format(seed))
    set_random_seed(seed)
    trainloader = Dataset(opt, split='train')   # ['datasets']['train']['dataroot']
    train_dataloader = torch.utils.data.DataLoader(trainloader, batch_size=opt['datasets']['train']['batch_size'], num_workers=0, shuffle=True)
    dataset_size = len(train_dataloader)

    valloader = Dataset(opt, split='val')  # ['datasets']['train']['dataroot']
    val_dataloader = torch.utils.data.DataLoader(valloader, batch_size=opt['val']['batch_size'], num_workers=0, shuffle=True)
    # test_dataloader = get_dataloader(train_opt, split='val',isTrain=False)

    # # 检查gpu是否可以用
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fusion(train_opt)
    # model = model.to(device)
    # if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型
    # device = torch.device("cuda:0")
    # model.to(device)  # 把并行的模型移动到GPU上
    # 记录开始时间
    start_time = datetime.now()

    total_epo = train_opt['train']['epoch_count']
    message = ''
    for epoch in range(1, total_epo + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        current_step = 0
        loss1_train = 0.0
        loss2_train = 0.0
        lossz_train = 0.0
        loss_train = 0.0
        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            current_step += 1
            model.set_input(data, True)
            # model.module.set_input(data, True)
            loss1, loss2, lossz = model.optimize_parameters()
            # loss1, loss2, lossz = model.module.optimize_parameters()
            # model.update_learning_rate()
            loss1_train += loss1
            loss2_train += loss2
            loss_train += lossz
        loss1epo = loss1_train / current_step
        loss2epo = loss2_train / current_step
        lossepo = loss_train / current_step
        lr1_now, lr2_now = model.get_lr()
        # writer.add_scalar("loss", lossepo, epoch)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message += f'{current_time} - epoch: {epoch}/{total_epo}:lr1:{lr1_now}:lr2:{lr2_now}:loss1:{loss1epo}:loss2:{loss2epo}:loss:{lossepo}\n'
        # message += f'epoch: {epoch}/{total_epo}:lr1:{lr1_now}:lr2:{lr2_now}:loss1:{loss1epo}:loss2:{loss2epo}:loss:{lossepo}\n'
        hues.info("epoch: {:d}/{} :lr1:{}:lr2:{}:loss1:{}:loss2:{} : loss:{}".format(epoch, total_epo, lr1_now, lr2_now, loss1epo, loss2epo, lossepo))
        # val:
        if epoch % train_opt['train']['print_freq'] == 0:
            loss1_val = 0.0
            loss2_val = 0.0
            loss_val = 0.0
            idx = 0
            with torch.no_grad():
                for i_val, val_data in enumerate(val_dataloader):
                    idx += 1
                    # img_name = val_data[3][0].split('.')[0]
                    # train_model.eval()
                    model.set_input(val_data, True)
                    loss11, loss22, loss33 = model.val() # 前向
                    loss1_val += loss11
                    loss2_val += loss22
                    loss_val += loss33
                    # train_model.train()
            loss1_valepo = loss1_val / idx
            loss2_valepo = loss2_val / idx
            loss_val_epo = loss_val / idx
            # writer.add_scalar("loss_val", loss_val_epo, epoch)

            message += f'epoch: {epoch}, validation>loss1:{loss1_valepo}:loss2:{loss2_valepo}:loss:{loss_val_epo}\n'
            # hues.info("epoch: {:d}/{} : loss:{}".format(epoch, train_opt.epoch_count, lossepo))
            hues.info(f'<epoch:{epoch}/{total_epo}, validation>loss1:{loss1_valepo},loss2:{loss2_valepo},loss:{loss_val_epo}\n')
            model.save(epoch)

            # print("test")
        # model.update_learning_rate()   # 更新学习率

        with open('./checkpoints/net3'  +'/fusion' + '/loss.txt', "w") as file:   #  + '/s2tol8'
            file.write(str(message))

    # 记录开始时间
    end_time = datetime.now()
    total_time = end_time - start_time
    hues.info('Saving the final model.')
    model.save('latest')
    hues.info(f'End of training, times{total_time}')
    # model.save_networks(train_opt.epoch_count)   # #save
    # train_model.saveAbundance()
