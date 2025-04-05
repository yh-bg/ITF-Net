import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torchvision import transforms as T
from torchvision.utils import save_image
import time
import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True).mean(-2, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False).mean(-2, keepdim=True)
#
#         return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(
#             -1).unsqueeze(-1)
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, dim):
#         super(LayerNorm, self).__init__()
#
#         self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        # self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        # self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.qkv1 = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, bias=qkv_bias, groups=dim*5)
        self.proj = nn.Conv2d(64, 64, kernel_size=1, bias=qkv_bias)
        # self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        # self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,stride=1, padding=1, bias=qkv_bias, groups=dim*3)
        # self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    # def forward(self, x):
    #     # [batch_size, num_patches + 1, total_embed_dim]
    #     b, c, h, w = x.shape
    #     qkv = self.qkv2(self.qkv1(x))
    #     q, k, v = qkv.chunk(3, dim=1)
    #     q = rearrange(q, 'b (head c) h w -> b head c (h w)',
    #                   head=self.num_heads)
    #     k = rearrange(k, 'b (head c) h w -> b head c (h w)',
    #                   head=self.num_heads)
    #     v = rearrange(v, 'b (head c) h w -> b head c (h w)',
    #                   head=self.num_heads)
    #     q = torch.nn.functional.normalize(q, dim=-1)
    #     k = torch.nn.functional.normalize(k, dim=-1)
    #     # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
    #     # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
    #     attn = (q @ k.transpose(-2, -1)) * self.scale
    #     attn = attn.softmax(dim=-1)
    #
    #     out = (attn @ v)
    #
    #     out = rearrange(out, 'b head c (h w) -> b (head c) h w',
    #                     head=self.num_heads, h=h, w=w)
    #
    #     out = self.proj(out)
    #     return out

    def forward(self, x, y, z):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q1, q2, k1, k2, v = qkv.chunk(5, dim=1)
        # 逐元素相加
        q1 = q1 + y
        k1 = k1 + y
        q2 = q2 + z
        k2 = k2 + z

        # 增加维度
        # q1 = torch.cat([q1, y], 1)
        # k1 = torch.cat([k1, y], 1)
        # q2 = torch.cat([q2, z], 1)
        # k2 = torch.cat([k2, z], 1)
        # v = torch.cat([v, x], 1)
        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)',
                       head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)',
                       head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)

        out1 = (attn1 @ v)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)

        out2 = (attn2 @ v)

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = torch.cat([out1,out2],1)
        out = self.proj(out)
        return out



class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeatureExtraction1(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False,):
        super(BaseFeatureExtraction1, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(64)
        self.mlp = Mlp(in_features=64, ffn_expansion_factor=2,)
        self.relu = nn.ReLU(inplace=True)
        #self.conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    def forward(self, x, y ,z):

        x = self.attn(self.norm1(x),self.norm1(y),self.norm1(z))
        #x2 = self.conv(x)
        x = torch.cat([y,z],1) + x
        # x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # x = self.relu(x)
        return x

# class BaseFeatureExtraction1(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  ffn_expansion_factor=1.,
#                  qkv_bias=False,):
#         super(BaseFeatureExtraction1, self).__init__()
#         self.norm1 = LayerNorm(dim)
#         self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
#         self.norm2 = LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, ffn_expansion_factor=2,)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         # x = self.relu(x)
#         return x


