# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.nn import functional as F

import pdb

# ===================
#     RGA Module
# RGA空间注意力模块(薄片与通道特征图做矩阵乘法)
# ===================

class RGA_Module(nn.Module):
	def __init__(self, in_channel=2048, in_spatial=196, use_spatial=True, use_channel=True, \
					cha_ratio=8, spa_ratio=8, down_ratio=8):
		super(RGA_Module, self).__init__()

		self.in_channel = in_channel
		self.in_spatial = in_spatial
		
		self.use_spatial = use_spatial
		self.use_channel = use_channel

		print ('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

		self.inter_channel = in_channel // cha_ratio
		self.inter_spatial = in_spatial // spa_ratio
		
		# Embedding functions for original features
		if self.use_spatial:
			self.gx_spatial = nn.Sequential(
				# 1x1卷积  ==> 通道数降维
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		if self.use_channel:
			self.gx_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		
		# Embedding functions for relation features
		if self.use_spatial:
			self.gg_spatial = nn.Sequential(
				# 1x1卷积 ==> 通道数降低
				nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		if self.use_channel:
			self.gg_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel*2, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		
		# Networks for learning attention weights
		if self.use_spatial:
			# 对拼接后的空间注意力特征图进行操作
			num_channel_s = 1 + self.inter_spatial
			self.W_spatial = nn.Sequential(
				# 第1次通道数降低
				nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_s//down_ratio),
				nn.ReLU(),
				# 第2次通道数降低
				nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)
		if self.use_channel:	
			num_channel_c = 1 + self.inter_channel
			self.W_channel = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_c//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_c//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)

		# Embedding functions for modeling relations
		# 嵌入函数 ==> 建模关系
		if self.use_spatial:
			self.theta_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
			self.phi_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)

		if self.use_channel:
			self.theta_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
			self.phi_channel = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
				
	def forward(self, x):
		b, c, h, w = x.size()

		# 使用空间注意力机制
		if self.use_spatial:
			# spatial attention
			theta_xs = self.theta_spatial(x) # 通道数降维 [B, C, H, W]
			phi_xs = self.phi_spatial(x) # 通道数降维
			theta_xs = theta_xs.view(b, self.inter_channel, -1) #[B, C, HxW]
			theta_xs = theta_xs.permute(0, 2, 1)  # [B, HxW, C]
			phi_xs = phi_xs.view(b, self.inter_channel, -1) # [B, C, HxW]
			# 矩阵乘法
			Gs = torch.matmul(theta_xs, phi_xs) # [B, HxW, HxW]
			# 取行
			Gs_in = Gs.permute(0, 2, 1).view(b, h*w, h, w)   # 维度有变换
			# 取列
			Gs_out = Gs.view(b, h*w, h, w)
			# 按照通道维度进行拼接
			Gs_joint = torch.cat((Gs_in, Gs_out), 1)
			# 1x1卷积 ==> 通道数降低
			Gs_joint = self.gg_spatial(Gs_joint)

			# 1x1卷积 == > 通道数降维
			g_xs = self.gx_spatial(x)
			# 取均值
			g_xs = torch.mean(g_xs, dim=1, keepdim=True)

			# 拼接 ==> 两部分拼接
			ys = torch.cat((g_xs, Gs_joint), 1)

			# 对拼接后的空间注意力特征图进行操作
			W_ys = self.W_spatial(ys)  # 通道数为1
			if not self.use_channel:
				# 注意：W_ys是一个薄面，尺度为HxWx1
				# 将其扩展为 HxWxC ==> 与特征图x 的每个通道相乘
				out = F.sigmoid(W_ys.expand_as(x)) * x
				return out
			else:
				x = F.sigmoid(W_ys.expand_as(x)) * x

		if self.use_channel:
			# channel attention
			xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
			theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
			phi_xc = self.phi_channel(xc).squeeze(-1)
			Gc = torch.matmul(theta_xc, phi_xc)
			Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
			Gc_out = Gc.unsqueeze(-1)
			Gc_joint = torch.cat((Gc_in, Gc_out), 1)
			Gc_joint = self.gg_channel(Gc_joint)

			g_xc = self.gx_channel(xc)
			g_xc = torch.mean(g_xc, dim=1, keepdim=True)
			yc = torch.cat((g_xc, Gc_joint), 1)

			W_yc = self.W_channel(yc).transpose(1, 2)
			out = F.sigmoid(W_yc) * x

			return out
