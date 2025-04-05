import torch
import math
import torch.nn as nn


######################################
#            common model
######################################



class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, activation='prelu', bias=True):
        super(Upsampler, self).__init__()
        modules = []
        if scale == 3:
            modules.append(ConvBlock(n_feat, 9 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(3))  # 亚像素卷积
            if bn:
                modules.append(torch.nn.BatchNorm2d(n_feat))
        # else:
        #     for _ in range(int(math.log(scale, 2))):
        #         modules.append(ConvBlock(n_feat, 4 * n_feat, WV-2, 1, 1, bias, activation=None, norm=None))
        #         modules.append(torch.nn.PixelShuffle(2))
        #         if bn:
        #             modules.append(torch.nn.BatchNorm2d(n_feat))
        else:
            for _ in range(int(math.log(scale, 2))):
                modules.append(ConvBlock(n_feat, 64, 3, 1, 1, bias, activation='prelu', norm=None))
                modules.append(ConvBlock(64, 32, 3, 1, 1, bias, activation='prelu', norm=None))
                modules.append(ConvBlock(32, 4 * n_feat, 3, 1, 1, bias, activation='prelu', norm=None))
                modules.append(torch.nn.PixelShuffle(2))
                if bn:
                    modules.append(torch.nn.BatchNorm2d(n_feat))

        self.up = torch.nn.Sequential(*modules)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class MeanShift(nn.Conv2d):  # 均值漂移
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ConvBlock(torch.nn.Module):  # 卷积
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)  # 样本维度进行归一化(一个批次内不同样本的相同特征计算均值和方差)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(
                self.output_size)  # 一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。

        self.activation = activation  # 激活函数
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        if self.pad_model == None:
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding,
                                        bias=self.bias)
        elif self.pad_model == 'reflection':  # 对于越界的位置在网格中采用关于边界的对称值进行填充
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,
                                        bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class TransConvBlock(ConvBlock):  # 反卷积 放大图片作用
    def __init__(self, *args, **kwargs):
        super(ConvBlock, self).__init__()

        if self.pad_model == None:
            self.conv = torch.nn.ConvTranspose2d(self.input_size, self.output_size, self.kernel_size, self.stride,
                                                 self.padding, bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.ConvTranspose2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,
                                                 bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


######################################
#           resnet_block
######################################
class ResnetBlock(torch.nn.Module):  # ResModules
    # (layers): Sequential(
    # (0): Conv2d(32, 32, kernel_size=(WV-2, WV-2), stride=(1, 1), padding=(1, 1))
    # (1): PReLU(num_parameters=1)
    # (2): Conv2d(32, 32, kernel_size=(WV-2, WV-2), stride=(1, 1), padding=(1, 1))
    # (WV-2): PReLU(num_parameters=1)
    # )
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu',
                 norm='batch', pad_model=None):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale

        if self.norm == 'batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None,
                        [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer,
                         self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out


class ResnetBlock_triple(ResnetBlock):  # 精确残差
    def __init__(self, *args, middle_size, output_size, **kwargs):
        ResnetBlock.__init__(self, *args, **kwargs)

        if self.norm == 'batch':
            self.normlayer1 = torch.nn.BatchNorm2d(middle_size)
            self.normlayer2 = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.normlayer1 = torch.nn.InstanceNorm2d(middle_size)
            self.normlayer2 = torch.nn.BatchNorm2d(output_size)
        else:
            self.normlayer1 = None
            self.normlayer2 = None

        if self.pad_model == None:
            self.conv1 = torch.nn.Conv2d(self.input_size, middle_size, self.kernel_size, self.stride, self.padding,
                                         bias=self.bias)
            self.conv2 = torch.nn.Conv2d(middle_size, output_size, self.kernel_size, self.stride, self.padding,
                                         bias=self.bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv1 = torch.nn.Conv2d(self.input_size, middle_size, self.kernel_size, self.stride, 0, bias=self.bias)
            self.conv2 = torch.nn.Conv2d(middle_size, output_size, self.kernel_size, self.stride, 0, bias=self.bias)

        layers = filter(lambda x: x is not None,
                        [self.pad, self.conv1, self.normlayer1, self.act, self.pad, self.conv2, self.normlayer2,
                         self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out


class unet(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3):
        super(unet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            ConvBlock(self.input_size, self.output_size, self.kernel_size, padding=1, stride=1),
            nn.BatchNorm2d(self.output_size),
            nn.PReLU(init=0.5),
            # ResnetBlock(self.output_size),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


