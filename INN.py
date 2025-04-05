import torch
import torch.nn as nn
from dynamic_conv import Dynamic_conv2d
from torch.nn import functional as F
import math
import numpy as np
from torchvision import transforms as T
from torchvision.utils import save_image
import time
import numbers
from einops import rearrange
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup):
        super(InvertedResidualBlock, self).__init__()
        self.conv2_1 = nn.Conv2d(in_channels=inp, out_channels=oup // 3, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=inp, out_channels=oup // 3, kernel_size=3, stride=1,
                                 padding=2, dilation=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=inp, out_channels=oup // 3, kernel_size=3, stride=1,
                                 padding=3, dilation=3, bias=True)
        self.dyconv = Dynamic_conv2d(in_planes= inp, out_planes= oup, kernel_size=3, ratio=0.25, padding=1)
        self.act = torch.nn.PReLU(init=0.5)
        # hidden_dim = int(inp * expand_ratio)
        # self.bottleneckBlock = nn.Sequential(
        #     # pw
        #     nn.Conv2d(inp, hidden_dim, 1, bias=False),
        #     # nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU6(inplace=True),
        #     # dw
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
        #     # nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU6(inplace=True),
        #     # pw-linear
        #     nn.Conv2d(hidden_dim, oup, 1, bias=False),
        #     # nn.BatchNorm2d(oup),
        # )

    def forward(self, x):
        out1 = x
        out21 = self.conv2_1(out1)
        out22 = self.conv2_2(out1)
        out23 = self.conv2_3(out1)
        # out24 = self.conv2_4(out1)
        out2 = torch.cat([out21, out22, out23], 1)
        out2 = self.act(out2)
        out3 = self.dyconv(out2)
        out3 = self.act(out3)

        return x + out3

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=45, oup=45)
        self.theta_rho = InvertedResidualBlock(inp=45, oup=45)
        self.theta_eta = InvertedResidualBlock(inp=45, oup=45)
        self.shffleconv = nn.Conv2d(90, 90, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        x = torch.cat((z1, z2), dim=1)
        return x




