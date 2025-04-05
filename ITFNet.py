#!/usr/bin/env python
# coding=utf-8
'''
https://github.com/MrCPlusPlus/SRPPNN/blob/master/SRPPNN_model.py
Author: wjm
Date: 2020-11-11 20:37:09
LastEditTime: 2020-12-09 23:12:50
Description: Super-Resolution-Guided Progressive Pansharpening Based on a Deep Convolutional Neural Network
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
from TFB import CrossSwinTransformer_1
from ScConv import ScConv
from INN import *
from Extendedconvolution import RMRS
from Transformer_block import *
# from CDDformer import *

class ITFNet(nn.Module):
    def __init__(self, num_channels):
        super(ITFNet, self).__init__()

        out_channels = 3
        n_resblocks = 4



        self.ms_feature_extract = ConvBlock(3, 32, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.pan_feature_extract = ConvBlock(1, 32, 3, 1, 1, activation='prelu', norm=None, bias=True)
        # self.cross_attn_1 = CrossSwinTransformer_1(n_feats=32, n_heads=4, head_dim=8, win_size=4,
        #                                        cross_module=['ms', 'pan'], cat_feat=['ms', 'pan'])

        # 前方卷积
        self.conv_SL = ConvBlock(64, 32, 5, 1, 2, activation='prelu')

        # ScConv
        self.scconv = ScConv(32)

        # 可逆神经网络
        self.interact = nn.Sequential(DetailFeatureExtraction())

        # Transformer
        self.baseFeature =BaseFeatureExtraction1(dim=32, num_heads=8, ffn_expansion_factor=2)

        # self.extendconv = RMRS(32)

        self.conv_mid = ConvBlock(64, 90, 3, 1, 1, activation='prelu', norm=None, bias=True)
        # self.conv_mid = ConvBlock(32, 90, 3, 1, 1, activation='prelu', norm=None, bias=True)

        self.conv_end1 = ConvBlock(90, 64, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.conv_end2 = ConvBlock(64, 32, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.conv_end3 = ConvBlock(32, 3, 3, 1, 1, activation='prelu', norm=None, bias=True)

        # # unet
        # self.unet1 = unet(32,32,3)
        # self.unet2 = unet(64,32,3)
        # self.unet3 = unet(160,64,3)

        # self.conv_final1 = ConvBlock(64, 32, 3, 1, 1, activation='prelu', norm=None, bias=True)
        # self.conv_final2 = ConvBlock(32, 3, 3, 1, 1, activation='prelu', norm=None, bias=True)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_pan, l_ms):
        hp_pan_4 = x_pan - F.interpolate(
            F.interpolate(x_pan, scale_factor=1 / 4, mode='bicubic', align_corners=False, recompute_scale_factor=True),
            scale_factor=4, mode='bicubic', align_corners=False, recompute_scale_factor=True)  # 4倍high-pass
        r_ms = F.interpolate(l_ms, scale_factor=4, mode='bicubic', align_corners=False,
                             recompute_scale_factor=True)  # 四倍上采样MS
        # 前方特征提取
        ms = self.ms_feature_extract(r_ms)  # 32
        pan = self.pan_feature_extract(x_pan) # 32
        # cross_att = self.cross_attn_1(ms,pan)
        # fstart = self.conv_SL(cross_att)
        ms_sc = self.scconv(ms) # 32
        pan_sc = self.scconv(pan) # 32
        first_feature = torch.cat([ms_sc, pan_sc], 1)  # 64
        fstart = self.conv_SL(first_feature) # 32

        transformer = self.baseFeature(fstart,ms_sc,pan_sc) # 64
        # transformer = self.baseFeature(fstart)  # 64
        fmid = self.conv_mid(transformer) # 90
        # fmid = self.conv_mid(first_feature)  # 90

        fmid = self.interact(fmid) # 90
        fend = self.conv_end1(fmid)
        fend = self.conv_end2(fend)
        hrms = self.conv_end3(fend) + hp_pan_4 + r_ms
        # # 可逆神经网络
        # fmid = self.interact(fstart)
        # # 下方扩展卷积
        # fbelow = self.extendconv(fstart)
        #
        # # 后方Unet
        # unet_up = self.unet1(fup)
        # unet_mid = self.unet1(fmid)
        # unet_below = self.unet1(fbelow)
        #
        # unet_up_mid = self.unet2(torch.cat([unet_up, unet_mid], 1))
        # unet_below_mid = self.unet2(torch.cat([unet_mid, unet_below], 1))
        #
        # unet_final = self.unet3(torch.cat([unet_up_mid, unet_up, unet_mid, unet_below, unet_below_mid], 1))
        # # 三合一
        # hrms = self.conv_final1(unet_final)
        # hrms = self.conv_final2(hrms) + hp_pan_4 + r_ms
        return hrms


if __name__ == '__main__':
    # model = SRPPNN_Net(num_channels = WV-2_1)
    model = ITFNet(num_channels=4)
    print("===> Parameter numbers : %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.train()
    input_ms, input_pan = torch.rand(1, 3, 64, 64), torch.rand(1, 1, 256, 256)
    sr = model(input_pan, input_ms)
    print('sr输出', sr.size())
