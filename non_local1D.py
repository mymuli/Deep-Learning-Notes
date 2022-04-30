import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

class Rs_GCN(nn.Module):
    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # 1D卷积
        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        # 是否使用BN层
        if bn_layer:
            self.W = nn.Sequential(
                # 1D卷积升维度
                nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(self.in_channels), )            
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # 1D卷积
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        # 1D卷积
        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)



    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)

        # 1D卷积降维
        g_v = self.g(v).view(batch_size, self.inter_channels, -1)   # [16, 512, 80]
        g_v = g_v.permute(0, 2, 1)      # [16, 80, 512]

        # 1D卷积降维
        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)   # [16, 512, 80]
        theta_v = theta_v.permute(0, 2, 1)      # [16, 80, 512]

        # 1D卷积降维
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)   # [16, 512, 80]

        # 矩阵相乘
        R = torch.matmul(theta_v, phi_v)    # [16, 80, 512]　* [16, 512, 80] ==> [16, 80, 80]
        N = R.size(-1)
        R_div_C = R / N

        # 矩阵相乘            
        y = torch.matmul(R_div_C, g_v)  # [16, 80, 80] * [16, 80, 512] ==> [16, 80, 512]
        y = y.permute(0, 2, 1).contiguous() # [16, 512, 80]

        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        # 1D卷积升维度
        W_y = self.W(y)
        # 残差连接
        v_star = W_y + v

        return v_star, R_div_C
