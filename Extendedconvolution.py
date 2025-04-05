# AWFLN
from torch import nn
import torch
class RMRS(nn.Module):
    def __init__(self, out_channels):
        super(RMRS, self).__init__()

        self.conv2_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=2, dilation=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=3, dilation=3, bias=True)
        # self.conv2_4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 4, kernel_size=3, stride=1,
        #                          padding=4, dilation=4, bias=True)


    def forward(self, x):

        out1 = x
        out21 = self.conv2_1(out1)
        out22 = self.conv2_2(out1)
        out23 = self.conv2_3(out1)
        # out24 = self.conv2_4(out1)
        out2 = torch.cat([out21, out22, out23], 1)

        return out2




