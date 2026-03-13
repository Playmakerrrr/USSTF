import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import numpy as np
from model.swinbasic import BasicLayer


def get_scheduler(optimizer, opt):  #使用pytorch提供的接口调整学习率
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)  # 1- max(epo+1+1-10)/11
            lr_l = 1.0 - max(0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)  # 1- max(epo+1+1-10)/11
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):  # 内置初始化
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal': #正态分布 - N(mean, std)
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier': # xavier_uniform 初始化
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)   # apply对net里的所有layer调用init_func这个函数（遍历一遍）
    return net

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


class ResBlock(nn.Module):
    def __init__(self, input_ch):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_ch, input_ch, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_ch, input_ch, 1, 1, 0)
        )
    def forward(self, x):
        out = self.net(x)
        return out + x



class SwinFusionNet(nn.Module):
    # def __init__(self, input_c, output_c, dim=128, num_heads=4, window_size=8):  # window_size=8
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(SwinFusionNet, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(dim, 64, 1, 1, 0, bias=True),
        #     nn.LeakyReLU())
        self.layer1 = nn.Conv2d(dim, 64, 1, 1, 0, bias=True)
        self.swinblock1 = nn.Sequential(
            BasicLayer(dim=64, input_resolution=input_resolution, depth=depth, num_heads=num_heads, window_size=window_size, mlp_ratio=1),
            nn.Tanh())

        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3, 1, 1, bias=True),
        #     nn.LeakyReLU())
        self.layer2 = nn.Conv2d(64, 256, 1, 1, 0, bias=True)

        self.swinblock2 = nn.Sequential(
            BasicLayer(dim=256, input_resolution=input_resolution, depth=4, num_heads=num_heads, window_size=window_size, mlp_ratio=1),
            nn.Tanh())
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, 1, 1, bias=True),
        #     nn.LeakyReLU())
        self.layer3 = nn.Conv2d(256, 64, 1, 1, 0, bias=True)
        
        self.swinblock3 = nn.Sequential(
            BasicLayer(dim=64, input_resolution=input_resolution, depth=depth, num_heads=num_heads, window_size=window_size, mlp_ratio=1),
            nn.Tanh())
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(64, dim, 3, 1, 1, bias=True),
        #     nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, dim, 1, 1, 0, bias=True),
            nn.ReLU())

    def forward(self, x):

        x1 = self.layer1(x)
        # x = self.relu1(x)
        x2 = self.swinblock1(x1)
        x3 = self.layer2(x2)
        x4 = self.swinblock2(x3)
        x5 = self.layer3(x4)
        x6 = self.swinblock3(x5)
        x7 = self.layer4(x6)
        return x7


# one：
def define_netl82s2(input_ch, output_ch, init_type='kaiming', init_gain=0.02):#
    # net = netl82s2(input_c=input_ch, output_c=output_ch)
    # net = SwinFusionNet(input_c=input_ch, output_c=output_ch)
    net = SwinFusionNet(dim=4, input_resolution=(32, 32), depth=2, num_heads=4, window_size=8)  # input_resolution：（H,W）
    return init_weights(net, init_type, gain=init_gain).cuda(0)

# class netl82s2(nn.Module):
#     def __init__(self, input_c, output_c):
#         super(netl82s2, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(input_c, 64, 3, 1, 1, bias=True),
#             # nn.LeakyReLU(0.2, True),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 3, 1, 1, bias=True),
#             # nn.LeakyReLU(0.2, True),
#             nn.ReLU(),
#             nn.Conv2d(128, 64, 3, 1, 1, bias=True),
#             # nn.LeakyReLU(0.2, True),
#             nn.ReLU(),
#             nn.Conv2d(64, output_c, 3, 1, 1, bias=True),
#             # nn.Tanh()

#             # nn.LeakyReLU(0.2, True),
#             # nn.Conv2d(ngf * 8, output_c, 1, 1, 0),
#             # nn.ReLU()
#         )
#     def forward(self, inputs):
#         outputs = self.net(inputs)
#         return outputs

# two：
# def define_nets2tol8(input_ch, output_ch, init_type='kaiming', init_gain=0.02):
#     net = nets2tol8(input_c=input_ch, output_c=output_ch)
#     return init_weights(net, init_type, gain=init_gain).cuda(0)
def define_nets2tol8(input_ch, output_ch, init_type='kaiming', init_gain=0.02):
    # net = netl82s2(input_c=input_ch, output_c=output_ch)
    # net = SwinFusionNet(input_c=input_ch, output_c=output_ch)
    net = SwinFusionNet(dim=4, input_resolution=(32, 32), depth=2, num_heads=4, window_size=8)  # input_resolution：（H,W）
    return init_weights(net, init_type, gain=init_gain).cuda(0)

# class nets2tol8(nn.Module):
#     def __init__(self, input_c, output_c):
#         super(nets2tol8, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(input_c, 64, 3, 1, 1, bias=True),
#             # nn.LeakyReLU(0.2, True),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 3, 1, 1, bias=True),
#             # nn.LeakyReLU(0.2, True),
#             nn.ReLU(),
#             nn.Conv2d(128, 64, 3, 1, 1, bias=True),
#             # nn.LeakyReLU(0.2, True),
#             nn.ReLU(),
#             nn.Conv2d(64, output_c, 3, 1, 1, bias=True),
#             # nn.Tanh()
#
#             # nn.LeakyReLU(0.2, True),
#             # nn.Conv2d(ngf * 8, output_c, 1, 1, 0),
#             # nn.ReLU()
#         )
#     def forward(self, inputs):
#         outputs = self.net(inputs)
#         return outputs
