#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The framework of traing process
~~~~~
Before python train.py, please ensure running the "python -m visdom.server -port=xxx" where xxx is the port assign in options
"""
from random import random

import time
import argparse
from model.net1 import *
from data.msi2dataset import *
import options.options as option
from tensorboardX import SummaryWriter
from datetime import datetime
import hues
# writer = SummaryWriter("./mylog")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train/net1_l82s2.json', help='Path to option JSON file.')
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

    valloader = Dataset(opt['datasets']['train']['dataroot'], split='val')
    val_dataloader = torch.utils.data.DataLoader(valloader, batch_size=opt['val']['batch_size'], num_workers=0, shuffle=True)

    model = l8tos2(train_opt)

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
            # visualizer.reset()
            model.update_learning_rate()
            model.set_input(data, True)
            loss += model.optimize_parameters()
        lossepo = loss / current_step
        lr_now = model.get_current_learning_rate()
        # writer.add_scalar("loss", lossepo, epoch)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message += f'{current_time} -epoch: {epoch}/{total_epo}:lr:{lr_now}:loss:{lossepo}\n'
        hues.info("epoch: {:d}/{} :lr:{} : loss:{}".format(epoch, total_epo, lr_now, lossepo))
        # val:
        if epoch % train_opt['train']['print_freq'] == 0:
            loss_val = 0.0
            idx = 0
            with torch.no_grad():
                for i_val, val_data in enumerate(val_dataloader):
                    idx += 1
                    model.set_input(val_data, True)
                    loss_val += model.val() # 前向
                    # train_model.train()

            loss_val_epo = loss_val / idx
            # writer.add_scalar("loss_val", loss_val_epo, epoch)

            message += f'epoch: {epoch}, validation>loss:{loss_val_epo}\n'
            # hues.info("epoch: {:d}/{} : loss:{}".format(epoch, train_opt.epoch_count, lossepo))
            hues.info(f'<epoch:{epoch}/{total_epo}, validation>loss:{loss_val_epo}\n')
            model.save(epoch)

        with open('./checkpoints/net1' + '/l8tos2' + '/loss.txt', "w") as file:
            file.write(str(message))

    hues.info('Saving the final model.')
    model.save('latest')
    hues.info('End of training')
    # model.save_networks(train_opt.epoch_count)   # #save
    # train_model.saveAbundance()
