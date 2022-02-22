import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# non-local模块的定义
class NONLocalBlock2D(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N * C * H * W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N * C * H * W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None):
        super(NONLocalBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        # 特征图的输出通道数
        if out_channels == None:
            self.out_channels = in_channels

        # K            
        self.f_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),)
        
        # Q              
        self.f_query = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),)
       
        # V
        self.f_value = nn.Conv2d(self.in_channels, self.value_channels, kernel_size=1, bias=False)

        # 特征图通道重建
        self.W = nn.Sequential(
            nn.Conv2d(self.value_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),)
                    

    def forward(self, x, y):
        # 特征图维度信息　读取
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        # Q
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        # K
        key = self.f_key(y).view(batch_size, self.key_channels, -1)

        # V
        value = self.f_value(y).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)


        # 矩阵相乘
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        # softmax函数
        sim_map = F.softmax(sim_map, dim=-1)

        # 矩阵相乘
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        # 通道数扩展为原始值
        context = self.W(context)

        # 残差连接
        out = x + context

        return out


class Nonlocal_Block(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels):
        super(Nonlocal_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = key_channels

        # non-local模型部分
        self.non_local = NONLocalBlock2D(in_channels=self.in_channels, key_channels=self.inter_channels, 
                                             value_channels=self.inter_channels, out_channels=self.out_channels)    

    def forward(self, x, y):
        n, c, h, w = x.size()   # [16, 512, 14, 14]

        # non-local模型部分 　
        out = self.non_local(x, y)
        
        return out














