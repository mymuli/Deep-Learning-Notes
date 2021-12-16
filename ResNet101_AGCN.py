import copy
import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU, Parameter
import torchvision
import torchvision.models as models
from util import *


class Ada_GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=2048):
        super(Ada_GCNResnet, self).__init__()
        # 加载 Resnet 模型中需要的结构
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4, )    # Conv_5

        # 图像正则化
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        # 定义类别数
        self.num_classes = num_classes


        # 两层GCN结构的定义
        self.gc_1 = GraphConvolution(1024, 1024)
        self.gc_2 = GraphConvolution(1024, 1024)


        # 降维 [ 14*14*2048 -> 14*14*1024 ]
        self.conv1x1_down = nn.Conv2d(in_channels=in_channel,
                                      out_channels=1024, 
                                      kernel_size=1, stride=1, padding=0)

        # 升维 [ 14*14*1024 -> 14*14*2048 ]
        self.conv1x1_up = nn.Sequential(
                            nn.Conv2d(in_channels=1024, 
                                      out_channels=in_channel, 
                                      kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(in_channel))


        # 最大池化
        self.pooling = nn.MaxPool2d(14, 14)

        self.fc1 = nn.Linear(2048, self.num_classes) 
        nn.init.normal_(self.fc1.weight, std=0.01)



    def forward(self, feature, inp=None):
        # ResNet卷积层的特征
        x_f = self.features(feature)       # conv_5　[16, 2048, 14, 14]

        # 降维
        x_down = self.conv1x1_down(x_f)         # b, 1024, 14, 14

        # 维度变换
        feat = x_down.permute(0, 2, 3, 1).reshape(-1, x_down.shape[1]) # (N*H*W,C)   
        # 矩阵乘法＋正则化操作得到　邻接矩阵A 
        adj = torch.mm(feat, torch.t(feat)) # (N*H*W, N*H*W)
        adj = row_normalize(adj)

        # 两层图卷积网络操作
        gc=F.relu(self.gc_1(feat, adj))
        gc=F.relu(self.gc_2(gc, adj))   # (16*14*14, 1024)

        # 得到输出
        gc = gc.reshape(x_down.shape[0], x_down.shape[2], x_down.shape[3], 1024)
        gc_out = gc.permute(0, 3, 1, 2)       # (B, 1024, 14, 14) 

        # 升维
        x_up = self.conv1x1_up(gc_out)     # b, 2048, 14, 14

        # 残差连接
        x_redisual = x_f + x_up    # b, 2048, 14, 14

        # 最大池化层处理
        x_pool = self.pooling(x_redisual)
        # 特征图形变
        x_resized = x_pool.view(x_pool.size(0), -1)

        # 全连接层
        out = self.fc1(x_resized) 

        return out


    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())

        # 学习率设置
        return [ {'params': self.features.parameters(), 'lr': lr * lrp},
                 {'params': large_lr_layers, 'lr': lr}, ]



def row_normalize(mx):
    """Row-normalize sparse matrix"""
    # 行标准化稀疏矩阵
    rowsum = torch.sum(mx, dim=1)
    r_inv = 1 / (rowsum + 1e-10)
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


# 图卷积神经网络
################################
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    # 定义输入维度和输出维度
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
               

# 模型参数
# num_classes=num_classes:  数据集类别数
def Ada_GCN_ResNet(num_classes, in_channel=2048):
    # 加载 预训练模型
    print("\n ==> 加载官方的resnet101预训练文件... \n")
    model = torchvision.models.resnet101(pretrained=True) 
    return Ada_GCNResnet(model, num_classes, in_channel=in_channel)


