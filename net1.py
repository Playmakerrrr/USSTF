#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn
from torch.autograd import Variable
from loss import GeneratorLoss

from . import network
from .base_model import BaseModel
# import skimage.measure as ski_measure

# writer = SummaryWriter("mylog")

# class l8tos2(torch.nn.Module):
class l8tos2(BaseModel):
    def __init__(self, opt):
        super(l8tos2, self).__init__(opt)
        train_opt = opt['train']
        s2_channels = 4
        l8_channels = 4
        self.net = network.define_netl82s2(input_ch=l8_channels, output_ch=s2_channels, init_type='kaiming')   # gpu_ids=self.gpu_ids,
        # if self.is_train:
        if opt['train']['is_train']:
            self.net.train()
        # load() mx
        if opt['train']['is_train']:
            self.cri_pix = torch.nn.MSELoss().cuda(0)
            # self.cri_pix = GeneratorLoss().cuda(0)
            self.optimizer_l8tos2 = torch.optim.Adam(self.net.parameters(), lr=train_opt['lr'], betas=(0.9, 0.999))
            self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(self.optimizer_l8tos2, train_opt['lr_steps'], train_opt['lr_gamma']))

    def set_input(self, input, isTrain):
        if isTrain:
            self.L8_90m = Variable(input['L8_90m'].float(), requires_grad=True).cuda(0)  # .to(self.device)
            self.S2_90m = Variable(input['S2_90m'].float(), requires_grad=True).cuda(0)
            # self.S2_real = Variable(input['S2_real'].float(), requires_grad=True).cuda(0)  # .to(self.device)   # 10m
        else:
            with torch.no_grad():
                self.L8_90m = Variable(input['L8_90m'].float(), requires_grad=True).cuda(0)  # .to(self.device)
                self.S2_90m = Variable(input['S2_90m'].float(), requires_grad=True).cuda(0)
                # self.S2_real = Variable(input['S2_real'].float(), requires_grad=True).cuda(0)  # .to(self.device)   # 10m
        # self.image_name = self.opt.name
        self.image_name = self.opt['name']

    # def optimize_parameters(self):
    #     self.optimizer_l8tos2.zero_grad()
    #     self.results = self.net(self.L8_90m)
    #     loss = self.cri_pix(self.results, self.S2_90m)
    #     loss.backward()
    #     self.optimizer_l8tos2.step()
    #     return loss.item()

    def optimize_parameters(self):
        self.optimizer_l8tos2.zero_grad()
        self.results = self.net(self.L8_90m)
        loss = self.cri_pix(self.results, self.S2_90m)
        # 不使用掩码
        # loss = self.cri_pix(self.results, self.S2_90m, is_ds=False)
        loss.backward()
        self.optimizer_l8tos2.step()
        return loss.item()

    def val(self):
        self.net.eval()
        self.results = self.net(self.L8_90m)
        loss_val = self.cri_pix(self.results, self.S2_90m)
        # loss_val = self.cri_pix(self.results, self.S2_90m, is_ds=False)
        self.net.train()
        return loss_val

    def save(self, iter_step):
        self.save_network(self.net, 'l8tos2', iter_step)

    def get_lr(self):
        lr1 = self.optimizer_l8tos2.param_groups[0]['lr']
        # lr2 = self.optimizer_l8tos2.param_groups[0]['lr']
        return lr1


