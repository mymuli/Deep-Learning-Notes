import torch
from torch import nn
from torch.nn import functional as F


# 图卷积网络
class Rs_GCN(nn.Module):
    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # 1x1　卷积
        conv_nd = nn.Conv1d
        # z最大池化
        max_pool = nn.MaxPool1d
        # BN正则化
        bn = nn.BatchNorm1d

        # 通道数降低
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            # 通道数升高
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels))   # 加入BN层
            
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            # 通道数升高
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        # 参数1
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        # 参数2
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)




    def forward(self, v):
        '''
        :param v: (B, D, N)  例如: [16, 2048, 80]
        :return:
        '''
        batch_size = v.size(0)

        # 通道数降低
        g_v = self.g(v).view(batch_size, self.inter_channels, -1)   # [16, 512, 80]
        g_v = g_v.permute(0, 2, 1)  # [16, 80, 512]

        # 参数1 
        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)  # [16, 80, 512]

        # 参数2
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1) # [16, 512, 80]
        # 矩阵相乘
        R = torch.matmul(theta_v, phi_v)    # [16, 80, 512] x [16, 512, 80] ==> [16, 80, 80]
        N = R.size(-1)
        R_div_C = R / N # 正则化?

        # 矩阵相乘
        y = torch.matmul(R_div_C, g_v)  # [16, 80, 80] x [16, 80, 512] ==> [16, 80, 512]
        y = y.permute(0, 2, 1).contiguous()     # [16, 512, 80]
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])

        # 通道数升高
        W_y = self.W(y)     # [16, 2048, 80]

        # 残差连接
        v_star = W_y + v

        return v_star








