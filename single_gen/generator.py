import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import math
import copy

class Generator(nn.Module):
	def __init__(self, z_dim, generator_channels):
		super(Generator, self).__init__()
		ngf = generator_channels
		nc = 3
		self.init_G_net = nn.Sequential(
            # # input is Z, going into a convolution
            # nn.ConvTranspose2d(     z_dim + 2 * generator_channels, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # # state size. (ngf*8) x 4 x 4
            # nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            # # state size. (ngf*4) x 8 x 8
            # nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True))
			# # state size. (ngf*2) x 32 x 32

			# 1
			self.conv_unit(z_dim + 2 * generator_channels, ngf * 32, kernel_size = 1, stride = 1, padding = 0, activation = 'relu', batch_norm = True, max_pool = False, upsample = False),
			#  1
			self.conv_unit(ngf * 32, ngf * 16, kernel_size = 5, stride = 1, padding = 2, activation = 'relu', batch_norm = True, max_pool = False, upsample = True),
			# 2
			self.conv_unit(ngf * 16, ngf * 8, kernel_size = 5, stride = 1, padding = 2, activation = 'relu', batch_norm = True, max_pool = False, upsample = True),
			# 4
			self.conv_unit(ngf * 8, ngf * 4, kernel_size = 5, stride = 1, padding = 2, activation = 'relu', batch_norm = True, max_pool = False, upsample = True),
			# 8
			self.conv_unit(ngf * 4, ngf * 2, kernel_size = 5, stride = 1, padding = 2, activation = 'relu', batch_norm = True, max_pool = False, upsample = True),
			# 16
			self.conv_unit(ngf * 2, ngf, kernel_size = 5, stride = 1, padding = 2, activation = 'relu', batch_norm = True, max_pool = False, upsample = True),
			# 32
		)

		self.stage1_G_net = nn.Sequential(
            # nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True))
			# state size. (ngf*2) x 64 x 64

			self.conv_unit(ngf, ngf, kernel_size = 5, stride = 1, padding = 2, activation = 'relu', batch_norm = True, max_pool = False, upsample = True),
			# 64
		)


		self.stage2_G_net = nn.Sequential(
            # nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True))

			self.conv_unit(ngf, ngf, kernel_size = 5, stride = 1, padding = 2, activation = 'relu', batch_norm = True, max_pool = False, upsample = True),
			# 128
		)
		self.stage3_G_net = nn.Sequential(
            # # state size. (ngf) x 128 x 128
            # nn.ConvTranspose2d(ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            # # state size. (nc) x 256 x 256
			self.conv_unit(ngf, ngf, kernel_size = 5, stride = 1, padding = 2, activation = 'relu', batch_norm = True, max_pool = False, upsample = True),
			self.conv_unit(ngf, nc, kernel_size = 1, stride = 1, padding = 0, activation = 'tanh', batch_norm = False, max_pool = False, upsample = False),
			# 256
        )

	def conv_unit(self, in_ch, out_ch, kernel_size, stride = 1, padding = 0, activation = 'relu', batch_norm = True, max_pool = False, upsample = True):
		seq_list = []
		seq_list.append(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding))

		if batch_norm:
			seq_list.append(nn.BatchNorm2d(num_features = out_ch))
		
		if activation == 'relu':
			seq_list.append(nn.ReLU())
		elif activation == 'sigmoid':
			seq_list.append(nn.Sigmoid())
		elif activation == 'tanh':
			seq_list.append(nn.Tanh())
		
		if max_pool:
			seq_list.append(nn.MaxPool2d(2))
		

		if upsample:
			seq_list.append(nn.Upsample(scale_factor = 2))

		return nn.Sequential(*seq_list)


	def attention(self, x, w):
		# w is text (batch, num_words, transformer dim)
		# x is image (batch, gen_channels, img width, img height)
		
		batch = x.size(0)
		width = x.size(3)
		height = x.size(2)
		num_pixels = x.size(2) * x.size(3)
		num_words = w.size(1)
		internal_dim = x.size(1)

		w_transpose = torch.transpose(w, 1, 2)
		x = x.view(x.size(0), -1, num_pixels)
		x_transpose = torch.transpose(x, 1, 2).contiguous()

		attn = torch.bmm(x_transpose, w_transpose)
		# attn -> (batch, num_pixels, num_words)
		attn = attn.view(batch * num_pixels, num_words)
		attn = nn.Softmax()(attn)
		attn = attn.view(batch, num_pixels, num_words)

		attn = attn.transpose(1, 2).contiguous()
		# attn -> (batch, num_words, num_pixels)
		weighted_context = torch.bmm(w_transpose, attn)
		weighted_context = weighted_context.view(batch, internal_dim, height, width)

		# # s -> (batch, num_words, num_pixels)
		# s = torch.bmm(w, x)
		# s = s.view(s.size(0) * num_pixels, -1)
		# s = nn.Softmax()(s)
		# s = s.view(batch, num_pixels, num_words)

		# word_context = torch.bmm(s, w)
		# word_context = word_context.view(batch, internal_dim, width, height)

		return weighted_context

	def forward(self, z, w, sent):

		z = torch.cat((z, sent), dim = 1)
		# print('z', z.size())
		x = self.init_G_net(z)
		# print('init', x.size())
		x = self.attention(x = x, w = w)
		x = self.stage1_G_net(x)
		# print('1', x.size())
		# x = self.attention(x = x, w = w)
		x = self.stage2_G_net(x)
		# print('2', x.size())
		# x = self.attention(x = x, w = w)

		return self.stage3_G_net(x)
		

# G = Generator(z_dim = 256, generator_channels = 128).to('cuda:0')
# from torchsummary import summary
# summary(G, (512, 1, 1))