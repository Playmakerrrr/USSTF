#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import torch
import torch.nn
from torch.autograd import Variable
from loss import GeneratorLoss
import itertools
from . import network
from .base_model import BaseModel

import numpy as np
# import skimage.measure as ski_measure

# writer = SummaryWriter("mylog")

# class l8tos2(torch.nn.Module):
class s2tol8(BaseModel):
    def __init__(self, opt):
        super(s2tol8, self).__init__(opt)
        train_opt = opt['train']
        s2_channels = 4
        l8_channels = 4
        self.net = network.define_nets2tol8(input_ch=s2_channels, output_ch=l8_channels, init_type='kaiming')   # gpu_ids=self.gpu_ids,
        # if self.is_train:
        if opt['train']['is_train']:
            self.net.train()
        # load() mx
        if opt['train']['is_train']:
            self.cri_pix = torch.nn.MSELoss().cuda(0)
            # self.cri_pix = GeneratorLoss().cuda(0)
            self.optimizer_s2tol8 = torch.optim.Adam(self.net.parameters(), lr=train_opt['lr'], betas=(0.9, 0.999))
            self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(self.optimizer_s2tol8, train_opt['lr_steps'], train_opt['lr_gamma']))

    def to(self, device):
        """ 让 self.network 支持 .to(device) """
        self.net.to(device)
        return self

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

    def optimize_parameters(self):
        self.optimizer_s2tol8.zero_grad()
        self.results = self.net(self.S2_90m)
        loss = self.cri_pix(self.results, self.L8_90m)
        # loss = self.cri_pix(self.results, self.S2_90m, is_ds=False)

        loss.backward()
        self.optimizer_s2tol8.step()
        return loss.item()

    def val(self):
        self.net.eval()
        self.results = self.net(self.S2_90m)
        loss_val = self.cri_pix(self.results, self.L8_90m)
        # loss_val = self.cri_pix(self.results, self.S2_90m, is_ds=False)
        self.net.train()
        return loss_val

    def save(self, iter_step):
        self.save_network(self.net, 's2tol8', iter_step)








# class Fusion(BaseModel):
#     def name(self):
#         return 'FusionGan'
#
#     @staticmethod
#     def modify_commandline_options(parser, isTrain=True):
#
#         parser.set_defaults(no_dropout=True)
#         if isTrain:
#             parser.add_argument('--num_theta', type=int, default=128)
#             parser.add_argument('--n_res', type=int, default=3)
#             parser.add_argument('--avg_crite', action="store_true")
#             parser.add_argument('--useGan', action="store_true")
#             parser.add_argument('--isCalSP', action="store_true")
#             parser.add_argument("--useSoftmax", action='store_false')
#         return parser
#
#
#     # def initialize(self, opt, hsi_channels, msi_channels, sp_matrix, sp_range):
#     def initialize(self, opt, hsi_channels, msi_channels):
#
#         BaseModel.initialize(self, opt)
#         self.opt = opt
#         num_s = 70  # test   #128
#         l8_channels = hsi_channels
#         s2_channels = msi_channels
#
#         # net generate abundance (encoder for msi) 下面？？
#         # Y->A
#         self.net_MSI2S = network.define_msi2s(input_ch=s2_channels, output_ch=num_s, gpu_ids=self.gpu_ids, n_res=opt.n_res,
#                                                 useSoftmax=opt.useSoftmax)
#         # shared endmember (also represents decoder)   lunwen:E
#         # A->Z~
#         self.net_s2img = network.define_s2img(input_ch=num_s, output_ch=l8_channels, gpu_ids=self.gpu_ids)   # A->Z~
#         # encoder for hsi  上面？？
#         # Z->Aah
#         self.net_LR2s = network.define_lr2s(input_ch=l8_channels, output_ch=num_s, gpu_ids=self.gpu_ids, n_res=opt.n_res,
#                                                 useSoftmax=opt.useSoftmax)   # Z -> Aah
#         # define psf function
#         self.net_PSF = network.define_psf(scale=opt.scale_factor,gpu_ids=self.gpu_ids)
#         # self.net_SRF = network.define_srf(gpu_ids=self.gpu_ids)
#         self.net_SRF = network.define_srf(input_ch=l8_channels, output_ch=s2_channels, gpu_ids=self.gpu_ids)
#
#         # LOSS
#         if self.opt.avg_crite == False:
#             self.criterionL1Loss = torch.nn.L1Loss(size_average=False).to(self.device)
#         else:
#             self.criterionL1Loss = torch.nn.L1Loss(size_average=True).to(self.device)
#         self.criterionPixelwise = self.criterionL1Loss
#         self.cri_pix = torch.nn.MSELoss()  # 损失函数：均方差损失   y+
#         self.criterionSumToOne = network.SumToOneLoss().to(self.device)
#         self.criterionSparse = network.SparseKLloss().to(self.device)
#
#         # self.model_names = ['MSI2S', 's2img', 'LR2s', 'PSF', 'G_HR2MSI']  #HrMSI前半部； E; LrHSI左半边； PSF； SRF
#         self.model_names = ['MSI2S', 's2img', 'LR2s', 'PSF', 'SRF']  #HrMSI前半部； E; LrHSI左半边； PSF； SRF
#
#         self.setup_optimizers()
#         self.visual_corresponding_name = {}
#
#     def setup_optimizers(self, lr=None):
#         if lr == None:
#             lr = self.opt.lr
#         else:
#             isinstance(lr, float)
#             lr = lr
#         self.optimizers = []
#         # 0.5
#         self.optimizer_G_MSI2S = torch.optim.Adam(itertools.chain(self.net_MSI2S.parameters()),
#                                             lr=lr*0.5,betas=(0.9, 0.999))  # lr=lr*0.5  # lr=0.00005    # Y->A
#         self.optimizers.append(self.optimizer_G_MSI2S)
#         self.optimizer_G_s2img = torch.optim.Adam(itertools.chain(self.net_s2img.parameters()),
#                                             lr=lr,betas=(0.9, 0.999))   # lr=lr  #lr=0.0001  # E
#         self.optimizers.append(self.optimizer_G_s2img)
#         self.optimizer_G_LR2s = torch.optim.Adam(itertools.chain(self.net_LR2s.parameters()),
#                                             lr=lr,betas=(0.9, 0.999))  # lr=lr   # lr=0.0001  # Z ->Aha
#         self.optimizers.append(self.optimizer_G_LR2s)
#         # 0.2
#         self.optimizer_G_PSF = torch.optim.Adam(itertools.chain(self.net_PSF.parameters()),
#                                             lr=lr*0.2,betas=(0.9, 0.999))   # lr=lr*0.2  # lr=0.00002  # psf
#         self.optimizers.append(self.optimizer_G_PSF)
#         # +SRF
#         self.optimizer_G_SRF = torch.optim.Adam(itertools.chain(self.net_SRF.parameters()),
#                                             lr=lr*0.2,betas=(0.9, 0.999))    #  lr=lr*0.2  # Z->Ylr
#         self.optimizers.append(self.optimizer_G_SRF)
#
#         # if self.opt.isCalSP == True:
#         #     # 0.2
#         #     self.optimizer_G_HR2MSI = torch.optim.Adam(itertools.chain(self.net_G_HR2MSI.parameters()),
#         #                                                lr=lr*0.2,betas=(0.9, 0.999))
#         #     self.optimizers.append(self.optimizer_G_HR2MSI)
#
#
#     def set_input(self, input, isTrain=True):
#         if isTrain:
#
#             self.real_l8 = Variable(input['L8_real'].float(), requires_grad=True).cuda(0)   # .to(self.device)
#             self.S2_30 =  Variable(input['S2_30m'].float(), requires_grad=True).cuda(0)
#             self.S2_real = Variable(input['S2_real'].float(), requires_grad=True).cuda(0)   # .to(self.device)   # 10m
#
#         else:
#             with torch.no_grad():
#                 self.real_l8 = Variable(input['L8_real'].float(), requires_grad=True).cuda(0)  # .to(self.device)
#                 self.S2_30 = Variable(input['S2_30m'].float(), requires_grad=True).cuda(0)
#                 self.S2_real = Variable(input['S2_real'].float(), requires_grad=True).cuda(0)  # .to(self.device)   # 10m
#
#         # self.image_name = input['name']
#         self.image_name = self.opt.name
#
#         # self.real_input = input
#
#     def forward(self):  #论文：Fig 3
#
#         # self.real_lhsi = self.real_l8
#         # self.real_hmsi = self.real_s2
#         self.real_lhsi = self.rec90_l8
#         self.real_hmsi = self.rec30_s2
#         # first lr process
#         self.rec_lr_s = self.net_LR2s(self.real_lhsi)  # LrHSI左半边（Z）Aah  # 10
#         self.rec_lr_lr = self.net_s2img(self.rec_lr_s)  #  E -> ~Z^a
#         #second msi process
#         self.rec_msi_s = self.net_MSI2S(self.real_hmsi)  # HrMSI前半部   (Y)->A   # 32
#         self.rec_msi_hr = self.net_s2img(self.rec_msi_s)  # E : A->~X
#         # self.rec_msi_msi = self.net_G_HR2MSI(self.rec_msi_hr)  # SRF : ~X->~Y
#         self.rec_msi_msi = self.net_SRF(self.rec_msi_hr)  # SRF : ~X->~Y  # 32
#
#         # third msi s lr
#         self.rec_msi_lrs = self.net_PSF(self.rec_msi_s)   # PSF : A -> A^bh   # 10
#         self.rec_msi_lrs_lr = self.net_s2img(self.rec_msi_lrs)  # E  A^bh-> ~Z^b   # 10
#         # four hr-msi-->psf-->lr-msi == lr-hsi-->sp-->lr-msi
#         # self.rec_lrhsi_lrmsi = self.net_G_HR2MSI(self.real_lhsi)  # SRF : Z-> ~Ylr
#
#         self.rec_lrhsi_lrmsi = self.net_SRF(self.real_lhsi)  # SRF : Z-> ~Ylr   # 10
#         self.rec_hrmsi_lrmsi = self.net_PSF(self.real_hmsi)  # PSF:  Y->~Ylr   # 10
#
#         # self.visual_corresponding_name['real_lhsi'] = 'rec_lr_lr'   # ~Z^a
#         # self.visual_corresponding_name['real_hmsi'] = 'rec_msi_msi'  # ~Y
#         # # if self.opt.isRealFusion == 'No':
#         # self.visual_corresponding_name['real_hhsi'] = 'rec_msi_hr'  # ~X
#         self.visual_corresponding_name['rec90_l8'] = 'rec_lr_lr'   # ~Z^a
#         self.visual_corresponding_name['rec30_s2'] = 'rec_msi_msi'  # ~Y
#         # if self.opt.isRealFusion == 'No':
#         self.visual_corresponding_name['real_l8'] = 'rec_msi_hr'  # ~X
#
#
#     def backward_joint(self):
#         # loss
#         # # new:
#         self.za_loss = self.cri_pix(self.rec_lr_lr, self.real_lhsi) * self.opt.lambda_A   # self.opt.lambda_A = 10  # (Z,~Z^a)loss
#         self.loss_lr_s_sumtoone = self.criterionSumToOne(self.rec_lr_s) * self.opt.lambda_D  # E
#         self.loss_lr_sparse = self.criterionSparse(self.rec_lr_s) * self.opt.lambda_E
#         self.loss_lr = self.za_loss + self.loss_lr_s_sumtoone + self.loss_lr_sparse
#         # self.za_loss.backward(retain_graph=True)
#
#         # msi
#         self.YY_loss = self.cri_pix(self.rec_msi_msi, self.real_hmsi) * self.opt.lambda_B   # (Y, ~Y)loss
#         # self.YY_loss = self.criterionPixelwise(self.real_hmsi, self.rec_msi_msi) * self.opt.lambda_B  # (Y, ~Y)
#         self.loss_msi_s_sumtoone = self.criterionSumToOne(self.rec_msi_s) * self.opt.lambda_D  # A
#         self.loss_msi_sparse = self.criterionSparse(self.rec_msi_s) * self.opt.lambda_E   # A
#         self.loss_msi = self.YY_loss + self.loss_msi_s_sumtoone + self.loss_msi_sparse  #
#
#         # self.YY_loss.backward(retain_graph=True)
#         # PSF
#         # self.loss_msi_ss_lr =  self.criterionPixelwise(self.real_lhsi, self.rec_msi_lrs_lr) * self.opt.lambda_C  # （Z, ~Z^b）
#         self.Zb_loss = self.cri_pix(self.rec_msi_lrs_lr, self.real_lhsi)  * self.opt.lambda_C # （Z, ~Z^b）loss
#         # self.Zb_loss.backward(retain_graph=True)
#         # lrmsi
#         # self.loss_lrmsi_pixelwise = self.criterionPixelwise(self.rec_hrmsi_lrmsi, self.rec_lrhsi_lrmsi) * self.opt.lambda_F  # (Z-> ~Ylr, Y->~Ylr)
#         self.Ylr2_loss = self.cri_pix(self.rec_hrmsi_lrmsi, self.rec_lrhsi_lrmsi) * self.opt.lambda_F   # (Z-> ~Ylr, Y->~Ylr)loss
#         # self.Ylr2_loss = self.criterionPixelwise(self.rec_lrhsi_lrmsi, self.rec_hrmsi_lrmsi) * self.opt.lambda_F   # (Z-> ~Ylr, Y->~Ylr)loss
#         # self.Ylr2_loss.backward(retain_graph=True)
#         # self.loss_joint.backward(retain_graph=True)
#
#         self.loss_joint = self.loss_lr  + self.loss_msi  + self.Zb_loss + self.Ylr2_loss
#         self.loss_joint.backward(retain_graph=True)
#         # print("za_loss:{}, YY_loss:{}, Zb_loss:{}, Ylr2_loss:{}".format(self.za_loss.item(), self.YY_loss.item(), self.Zb_loss.item(), self.Ylr2_loss.item()))
#         return self.za_loss.item(), self.YY_loss.item(), self.Zb_loss.item(), self.Ylr2_loss.item(), self.loss_joint.item()
#         # return self.loss_joint.item()
#         # self.loss_joint.backward(retain_graph=True)
#
#     def optimize_joint_parameters(self):
#         self.loss_names = ["loss_joint","loss_lr","loss_msi","loss_msi_ss_lr","loss_lrmsi_pixelwise"]
#
#         # self.visual_names = ['rec90_l8', 'rec_lr_lr', 'rec30_s2','rec_msi_msi','real_l8','rec_msi_hr']  # Z,Z~,Y,Y~,X,X~
#         self.forward()
#         self.optimizer_G_LR2s.zero_grad()
#         self.optimizer_G_s2img.zero_grad()
#         self.optimizer_G_MSI2S.zero_grad()
#         self.optimizer_G_PSF.zero_grad()
#         # if self.opt.isCalSP == 'Yes':
#         #     self.optimizer_G_HR2MSI.zero_grad()
#         self.optimizer_G_SRF.zero_grad()
#
#         # zaloss, yyloss, zbloss, Ylr2loss = self.backward_joint(epoch)   # loss
#         zaloss, yyloss, zbloss, Ylr2loss, loss_joint = self.backward_joint()   # loss
#         # loss_joint = self.backward_joint(epoch)   # loss
#
#         self.optimizer_G_LR2s.step()
#         self.optimizer_G_s2img.step()
#         self.optimizer_G_MSI2S.step()
#         self.optimizer_G_PSF.step()
#         # if self.opt.isCalSP == 'Yes':
#         #     self.optimizer_G_HR2MSI.step()
#         self.optimizer_G_SRF.step()
#
#         lr = self.optimizers[0].param_groups[0]['lr']
#         # clipper_nonzero = network.NonZeroClipper()
#         # self.net_G_s2img.apply(clipper_nonzero)
#         cliper_zeroone = network.ZeroOneClipper()  # （0,1）
#         self.net_PSF.apply(cliper_zeroone)      # ????
#         self.net_s2img.apply(cliper_zeroone)    # ????
#
#         # if self.opt.isCalSP == 'Yes':
#         #     cliper_sumtoone = network.SumToOneClipper()
#         #     self.net_G_HR2MSI.apply(cliper_sumtoone)
#         # self.net_SRF.apply(cliper_zeroone)  # 1train
#         cliper_sumtoone = network.SumToOneClipper()
#         self.net_SRF.apply(cliper_sumtoone)  # new
#
#         return zaloss, yyloss, zbloss, Ylr2loss, lr,loss_joint
#         # return lr,loss_joint
#
#     def val(self):
#         # self.forward()
#
#         # self.real_lhsi = self.real_l8
#         # self.real_hmsi = self.real_s2
#         self.real_lhsi = self.rec90_l8
#         self.real_hmsi = self.rec30_s2
#         # first lr process
#         self.rec_lr_s = self.net_LR2s(self.real_lhsi)  # LrHSI左半边（Z）Aah  # 10
#         self.rec_lr_lr = self.net_s2img(self.rec_lr_s)  # E -> ~Z^a
#         # second msi process
#         self.rec_msi_s = self.net_MSI2S(self.real_hmsi)  # HrMSI前半部   (Y)->A   # 32
#         self.rec_msi_hr = self.net_s2img(self.rec_msi_s)  # E : A->~X
#         # self.rec_msi_msi = self.net_G_HR2MSI(self.rec_msi_hr)  # SRF : ~X->~Y
#         self.rec_msi_msi = self.net_SRF(self.rec_msi_hr)  # SRF : ~X->~Y  # 32
#         # third msi s lr
#         self.rec_msi_lrs = self.net_PSF(self.rec_msi_s)  # PSF : A -> A^bh   # 10
#         self.rec_msi_lrs_lr = self.net_s2img(self.rec_msi_lrs)  # E  A^bh-> ~Z^b   # 10
#         # four hr-msi-->psf-->lr-msi == lr-hsi-->sp-->lr-msi
#         # self.rec_lrhsi_lrmsi = self.net_G_HR2MSI(self.real_lhsi)  # SRF : Z-> ~Ylr
#         self.rec_lrhsi_lrmsi = self.net_SRF(self.real_lhsi)  # SRF : Z-> ~Ylr   # 10
#         self.rec_hrmsi_lrmsi = self.net_PSF(self.real_hmsi)  # PSF:  Y->~Ylr   # 10
#         return self.rec_msi_hr
#
#     def feeddata(self,real_l8, rec90_l8, rec30_s2):
#         # self.real_l8 = torch.from_numpy(real_l8)
#         # self.rec90_l8 = torch.from_numpy(rec90_l8)
#         # self.rec30_s2 = torch.from_numpy(rec30_s2)
#         self.real_l8 = Variable(torch.from_numpy(real_l8).float().unsqueeze(0).cuda(0))  # +cuda
#         self.rec90_l8 = Variable(torch.from_numpy(rec90_l8).float().unsqueeze(0).cuda(0))
#         self.rec30_s2 = Variable(torch.from_numpy(rec30_s2).float().unsqueeze(0).cuda(0))
#
#     def savePSFweight(self):
#         # save_np = self.net_PSF.module.net.weight.data.cpu().numpy().reshape(self.opt.scale_factor,self.opt.scale_factor)
#         save_np = self.net_PSF.module.net.weight.data.cpu().numpy()   # .reshape(self.opt.scale_factor,self.opt.scale_factor)
#         # save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'rec_psf_weight.npy')
#         # np.save(save_path, save_np)
#         save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'rec_psf_weight.mat')
#         io.savemat(save_path,{'psf_weight':save_np})
#
#     # def imsave(self, img, path, Dtype):
#     #     if len(img.shape) == 3:
#     #         (n, h, w) = img.shape
#     #     else:
#     #         (h, w) = img.shape
#     #         n = 1
#     #     driver = gdal.GetDriverByName("GTiff")
#     #
#     #     if Dtype == 'uint8':
#     #         datatype = gdal.GDT_Byte
#     #     elif Dtype == 'uint16':
#     #         datatype = gdal.GDT_UInt16
#     #     else:
#     #         datatype = gdal.GDT_Float32
#     #     dataset = driver.Create(path, w, h, n, datatype)
#     #     if len(img.shape) == 3:
#     #         for t in range(n):
#     #             dataset.GetRasterBand(t + 1).WriteArray(img[t])
#     #     else:
#     #         dataset.GetRasterBand(1).WriteArray(img)
#     #
#     #     del dataset
#
#     # def saveresult(self, num):
#     #     X = self.rec_msi_hr.data.cpu().numpy().squeeze()
#     #     ch_x, h_x, w_x = X.shape
#     #     # save_path = "./result/t1.tif"
#     #     save_path = "./result/" + str(num +1) + ".tif"
#     #
#     #     # if (not os.path.exists(save_path)):
#     #     #     print("目录文件不存在")
#     #     #     os.makedirs(save_path)
#     #     self.imsave(X, save_path, 'float32')
#
#
#         # ----------gdal---------------
#         # in_ds = gdal.Open(data_root_L8 + '/B02.tif')
#         # datatype = X.dtype
#         # # datatype = in_ds.GetRasterBand(1).DataType
#         # gtif = gdal.GetDriverByName("GTiff")
#         # # out_ds = gtif.Create(save_path + '/SRCNN_S2L8_1_D3.tif', h_x, w_x, ch_x, datatype)  # , datatype=np.uint8)
#         #
#         # for i in range(ch_x):
#         #     out_ds = gtif.Create(save_path + '/t' + str(i+1) +'.tif', h_x, w_x, ch_x, gdal.GDT_Float32)  # , datatype=np.uint8)  datatype
#         #     out_ds.GetRasterBand(i+1).WriteArray(X[i, :, :])
#         # out_ds.FlushCache()
#         # print(f"di {i+1} FlushCache succeed")
#         # del out_ds
#         # print("test")
#
#     def saveAbundance(self):
#         self.forward()
#
#         LHSI_A_a = self.rec_lr_s.data.cpu().numpy()
#         # save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'LHSI_A_a.npy')
#         # np.save(save_path, LHSI_A_a)
#         save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'Abundance_lhsi_a.mat')
#         io.savemat(save_path,{'abundance_lhsi_a':LHSI_A_a})
#
#         HMSI_A = self.rec_msi_s.data.cpu().numpy()   # A
#         # save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'HMSI_A.npy')
#         # np.save(save_path, HMSI_A)
#         save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'Abundance_hmsi.mat')
#         io.savemat(save_path,{'abundance_hmsi':HMSI_A})
#
#         LHSI_A_b = self.rec_msi_lrs.data.cpu().numpy()
#         # save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'LHSI_A_b.npy')
#         # np.save(save_path, LHSI_A_b)
#         save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'Abundance_lhsi_b.mat')
#         io.savemat(save_path,{'abundance_lhsi_b':LHSI_A_b})
#
#         X = self.rec_msi_hr.data.cpu().numpy() # X
#         save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'Abundance_X.mat')
#         io.savemat(save_path, {'abundance_X': X})
#
#     def get_visual_corresponding_name(self):
#         return self.visual_corresponding_name
#
#     def cal_psnr(self):
#         # real_hsi = self.real_hhsi.data.cpu().float().numpy()[0]
#         # rec_hsi = self.rec_msi_hr.data.cpu().float().numpy()[0]
#         # return self.compute_psnr(real_hsi, rec_hsi)
#         # real_l8 = self.real_l8.data.cpu().float().numpy()
#         # real_l8 = self.l8rep.data.cpu().float().numpy()
#         real_l8 = self.real_l8.data.cpu().float().numpy()
#         rec30_l8 = self.rec_msi_hr.data.cpu().float().numpy()
#
#         import imgvision as iv
#         # 创建评价器
#         # Metric = iv.spectra_metric(Hyperspectral_Image, Reconstruction)
#         Metric = iv.spectra_metric(real_l8, rec30_l8)
#         # 评价PSNR：
#         PSNR = Metric.PSNR()
#         SAM = Metric.SAM()  # 图像像素范围的最大值为1时
#         MSE = Metric.MSE()
#
#         za_loss = self.cri_pix(self.rec_lr_lr, self.real_lhsi) * self.opt.lambda_A   # self.opt.lambda_A = 10  # (Z,~Z^a)loss
#         loss_lr_s_sumtoone = self.criterionSumToOne(self.rec_lr_s) * self.opt.lambda_D  # E
#         loss_lr_sparse = self.criterionSparse(self.rec_lr_s) * self.opt.lambda_E
#         loss_lr = za_loss + loss_lr_s_sumtoone + loss_lr_sparse
#         # self.za_loss.backward(retain_graph=True)
#
#         # msi
#         YY_loss = self.cri_pix(self.rec_msi_msi, self.real_hmsi) * self.opt.lambda_B   # (Y, ~Y)loss
#         # self.YY_loss = self.criterionPixelwise(self.real_hmsi, self.rec_msi_msi) * self.opt.lambda_B  # (Y, ~Y)
#         loss_msi_s_sumtoone = self.criterionSumToOne(self.rec_msi_s) * self.opt.lambda_D  # A
#         loss_msi_sparse = self.criterionSparse(self.rec_msi_s) * self.opt.lambda_E   # A
#         loss_msi = YY_loss + loss_msi_s_sumtoone + loss_msi_sparse  #
#
#         # self.YY_loss.backward(retain_graph=True)
#         # PSF
#         # self.loss_msi_ss_lr =  self.criterionPixelwise(self.real_lhsi, self.rec_msi_lrs_lr) * self.opt.lambda_C  # （Z, ~Z^b）
#         Zb_loss = self.cri_pix(self.rec_msi_lrs_lr, self.real_lhsi)  * self.opt.lambda_C # （Z, ~Z^b）loss
#         # self.Zb_loss.backward(retain_graph=True)
#         # lrmsi
#         # self.loss_lrmsi_pixelwise = self.criterionPixelwise(self.rec_hrmsi_lrmsi, self.rec_lrhsi_lrmsi) * self.opt.lambda_F  # (Z-> ~Ylr, Y->~Ylr)
#         Ylr2_loss = self.cri_pix(self.rec_hrmsi_lrmsi, self.rec_lrhsi_lrmsi) * self.opt.lambda_F   # (Z-> ~Ylr, Y->~Ylr)loss
#
#         loss_joint = loss_lr  + loss_msi  + Zb_loss + Ylr2_loss
#         # return compute_psnr(real_l8, rec30_l8)
#         return PSNR, SAM, MSE, za_loss.item(), YY_loss.item(), Zb_loss.item(), Ylr2_loss.item(), loss_joint.item()

def compute_psnr(img1, img2):
    import imgvision as iv
    # 创建评价器
    # Metric = iv.spectra_metric(Hyperspectral_Image, Reconstruction)
    Metric = iv.spectra_metric(img1, img2)
    # 评价PSNR：
    PSNR = Metric.PSNR()
    SAM = Metric.SAM()  # 图像像素范围的最大值为1时
    MSE = Metric.MSE()
    return PSNR, SAM, MSE


    # def compute_psnr(self, img1, img2):
    #     assert img1.ndim == 3 and img2.ndim ==3
    #
    #     img_c, img_w, img_h = img1.shape
    #     ref = img1.reshape(img_c, -1)
    #     tar = img2.reshape(img_c, -1)
    #     msr = np.mean((ref - tar)**2, 1)  # 计算每一行的均值
    #     max2 = np.max(ref,1)**2
    #     psnrall = 10*np.log10(max2/msr)
    #     out_mean = np.mean(psnrall)
    #     return out_mean

