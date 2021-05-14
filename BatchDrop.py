# encoding: utf-8
import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, resnet50
from torchvision.transforms import functional


# BatchDrop
class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio  # 高
        self.w_ratio = w_ratio  # 宽
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)   # 返回浮点数x的四舍五入值
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)   # random.randint(12, 20)  #生成的随机数n: 12 <= n <= 20 
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())    # 返回一个与size大小相同的用1填充的张量
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0  # 将mask的区域置为0
            x = x * mask   # mask操作
        return x 
