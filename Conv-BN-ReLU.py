import torch
import torchvision
import torch.nn as nn
import torch.tensor as tensor
from torch.nn import functional as F

self.feature = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 通道维度减少
                  nn.BatchNorm2d(out_channels),  # BN操作
                  nn.ReLU(inplace=True) )  # ReLU操作







