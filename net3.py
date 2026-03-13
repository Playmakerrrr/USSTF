#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn
from torch.autograd import Variable
import itertools
from . import finalnetwork
from .base_model import BaseModel
from loss import GeneratorLoss
import torch.nn.functional as F
# import hues
# import os
# import numpy as np
# import skimage.measure as ski_measure

# writer = SummaryWriter("mylog")

# class l8tos2(torch.nn.Module):
class fusion(BaseModel):
    def __init__(self, opt):
        super(fusion, self).__init__(opt)
        train_opt = opt['train']
        s2_channels = 4
        l8_channels = 4
        # self.net = finalnetwork.define_nets2tol8(input_ch=s2_channels, output_ch=l8_channels)   # gpu_ids=self.gpu_ids,

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nets2tol8 = finalnetwork.define_nets2tol8(input_ch=s2_channels, output_ch=l8_channels)
        self.netl8tos2 = finalnetwork.define_netl82s2(input_ch=l8_channels, output_ch=s2_channels)

        # if self.is_train:
        if opt['train']['is_train']:
            self.nets2tol8.train()
            self.netl8tos2.train()
        # load() mx
        if opt['train']['is_train']:
            self.cri_pix = torch.nn.MSELoss().to(self.device)
            # self.cri_pix = GeneratorLoss().cuda(0)
            # self.cri_pix = torch.nn.MSELoss().cuda(0)
            # self.optimizer_s2tol8 = torch.optim.Adam(self.net.parameters(), lr=train_opt['lr'], betas=(0.9, 0.999))
            self.optimizer_s2tol8 = torch.optim.Adam(self.nets2tol8.parameters(), lr=train_opt['lr1'], betas=(0.9, 0.999)) # 优化器
            self.schedulers_s2tol8 = (torch.optim.lr_scheduler.MultiStepLR(self.optimizer_s2tol8, milestones=train_opt['lr_steps1'], gamma=train_opt['lr_gamma1']))

            self.optimizer_l8tos2 = torch.optim.Adam(self.netl8tos2.parameters(), lr=train_opt['lr2'], betas=(0.9, 0.999))
            self.schedulers_l8tos2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_l8tos2, milestones=train_opt['lr_steps2'], gamma=train_opt['lr_gamma2'])

    def set_input(self, input, isTrain):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isTrain:
            self.L8_real = Variable(input['L8_real'].float(), requires_grad=True).to(self.device)  # .to(self.device)
            self.S2_30m = Variable(input['S2_30m'].float(), requires_grad=True).to(self.device)
            # self.S2_real = Variable(input['S2_real'].float(), requires_grad=True).cuda(0)  # .to(self.device)   # 10m
        else:
            with torch.no_grad():
                self.L8_real = Variable(input['L8_real'].float(), requires_grad=True).to(self.device)     #.cuda(0)  # .to(self.device)
                self.S2_30m = Variable(input['S2_30m'].float(), requires_grad=True).to(self.device)       # .cuda(0)
                # self.S2_real = Variable(input['S2_real'].float(), requires_grad=True).cuda(0)  # .to(self.device)   # 10m
        # self.image_name = self.opt.name
        self.image_name = self.opt['name']

    def optimize_parameters(self):
        self.optimizer_s2tol8.zero_grad()
        self.optimizer_l8tos2.zero_grad()
        self.zjresult = self.nets2tol8(self.S2_30m)   # 中间结果：10mL8(30m)
        self.result = self.netl8tos2(self.zjresult)    # 最后结果：10mS2(30m)
        self.newzjresult = F.interpolate(self.zjresult, scale_factor=1 / 3, mode="bilinear", align_corners=False)
        self.newl8real = F.interpolate(self.L8_real, scale_factor=1 / 3, mode="bilinear", align_corners=False)
        loss1 = self.cri_pix(self.newzjresult, self.newl8real)
  
        loss2 = self.cri_pix(self.result, self.S2_30m)


        lossz = loss1 + loss2
        lossz.backward()
        self.optimizer_s2tol8.step()
        self.optimizer_l8tos2.step()  # y+

        self.schedulers_s2tol8.step()  # y+
        self.schedulers_l8tos2.step()
        return loss1.item(), loss2.item(), lossz.item()

    def get_lr(self):
        lr1 = self.optimizer_s2tol8.param_groups[0]['lr']
        lr2 = self.optimizer_l8tos2.param_groups[0]['lr']
        return lr1, lr2

    def val(self):
        self.nets2tol8.eval()
        self.netl8tos2.eval()

        self.zjresult = self.nets2tol8(self.S2_30m)   # 中间结果：10mL8(30m)
        self.result = self.netl8tos2(self.zjresult)    # 最后结果：10mS2(30m)
        self.zjresult_3 = F.interpolate(self.zjresult, scale_factor=1 / 3, mode="bilinear", align_corners=False)
        self.L8_real_3 = F.interpolate(self.L8_real, scale_factor=1 / 3, mode="bilinear", align_corners=False)
        loss1_val = self.cri_pix(self.zjresult_3, self.L8_real_3)
        loss2_val = self.cri_pix(self.result, self.S2_30m)

        lossz_val = loss1_val + loss2_val
        self.nets2tol8.train()
        self.netl8tos2.train()
        return loss1_val.item(), loss2_val.item(), lossz_val.item()

    def save(self, iter_step):
        self.save_network(self.nets2tol8, 's2tol8', iter_step)
        self.save_network(self.netl8tos2, 'l8tos2', iter_step)

    def load_network(self, opt):
        firstnet = torch.load(opt['path']['s2tol8_root'])
        twonet = torch.load(opt['path']['l8tos2_root'])
        self.nets2tol8.load_state_dict(firstnet)
        self.netl8tos2.load_state_dict(twonet)

    def test(self, s2_30m):
        self.nets2tol8.eval()
        self.netl8tos2.eval()
        target = self.nets2tol8(s2_30m)
        # sem_s2 = self.netl8tos2(target)
        # return target, sem_s2
        return target

# class fusion(BaseModel):
#     def initialize(self, opt, s2_channels, l8_channels):
#         self.opt = opt
#         self.nets2tol8 = finalnetwork.define_nets2tol8(input_ch=s2_channels, output_ch=l8_channels)
#         self.netl8tos2 = finalnetwork.define_netl82s2(input_ch=l8_channels, output_ch=s2_channels)
#     def optimize_parameters(self):
#         self.optimizer_s2tol8 = torch.optim.Adam(self.nets2tol8.parameters(), lr=self.opt['train']['lr'], betas=(0.9, 0.999))
#         self.optimizer_l8tos2 = torch.optim.Adam(self.netl8tos2.parameters(), lr=self.opt['train']['lr'], betas=(0.9, 0.999))





