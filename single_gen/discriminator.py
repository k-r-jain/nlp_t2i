import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import math
import copy

class Discriminator(nn.Module):
	def __init__(self, current_size = 256, start_channels = 64, final_conv_size = 4):
		super(Discriminator, self).__init__()
		self.start_channels = start_channels
		self.final_conv_size = final_conv_size
		self.current_size = current_size
		# self.D_net = self.create_network(current_size = self.current_size, current_channels = 3)

		ndf = 16
		self.D_net = nn.Sequential(

			# 256
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
            # 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.7),
            # 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout(0.7),
			# nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout(0.7),
			# nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # 4
			# self.conv_unit(3, ndf, kernel_size = 4, stride = 2, padding = 1, activation = 'relu', batch_norm = True),
			# self.conv_unit(ndf, ndf * 2, kernel_size = 4, stride = 2, padding = 1, activation = 'relu', batch_norm = True),
			# self.conv_unit(ndf * 2, ndf * 4, kernel_size = 4, stride = 2, padding = 1, activation = 'relu', batch_norm = True),
			# self.conv_unit(ndf * 4, ndf * 8, kernel_size = 4, stride = 2, padding = 1, activation = 'relu', batch_norm = True),
			# self.conv_unit(ndf * 8, ndf * 16, kernel_size = 4, stride = 2, padding = 1, activation = 'relu', batch_norm = True),
			# self.conv_unit(ndf * 16, ndf * 32, kernel_size = 4, stride = 2, padding = 1, activation = 'relu', batch_norm = True)

		)

		self.D_net_continue = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
			# self.conv_unit(ndf * 32, 1, kernel_size = 4, stride = 1, padding = 0, activation = 'sigmoid', batch_norm = False)
        )

	def conv_unit(self, in_ch, out_ch, kernel_size, stride = 1, padding = 0, activation = 'relu', batch_norm = True, max_pool = False):
		seq_list = []
		seq_list.append(nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size, stride = stride, padding = padding))

		if batch_norm:
			seq_list.append(nn.BatchNorm2d(num_features = out_ch))
		
		if activation == 'relu':
			seq_list.append(nn.ReLU())
		elif activation == 'sigmoid':
			seq_list.append(nn.Sigmoid())
		
		if max_pool:
			seq_list.append(nn.MaxPool2d(2))
		return nn.Sequential(*seq_list)



	def create_network(self, current_size, current_channels):
		modules = []
		modules.append(self.downblock(in_channels = current_channels, out_channels = self.start_channels))
		current_channels = self.start_channels
		current_size = int(current_size / 2)
		while(current_size > self.final_conv_size):

			next_stage_channels = current_channels * 2
			modules.append(self.downblock(in_channels = current_channels, out_channels = next_stage_channels))

			current_channels = next_stage_channels
			current_size = int(current_size / 2)
			
		modules.append(nn.Conv2d(in_channels = current_channels, out_channels = 1, kernel_size = 4, padding = 0, bias = False))
		modules.append(nn.Sigmoid())
		return nn.Sequential(*modules)

	def downblock(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, activation = 'leakyrelu', batchnorm = True):
		modules = []
		modules.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False))
		if batchnorm:
			modules.append(nn.BatchNorm2d(num_features = out_channels))
		if activation == 'leakyrelu':
			modules.append(nn.LeakyReLU(negative_slope = 0.2))
		return nn.Sequential(*modules)
	
	def forward(self, im):
		# batch = im.size(0)
		features = self.D_net(im)
		pred = self.D_net_continue(features)
		return pred, features

# D = Discriminator().to('cuda:0')
# summary(D, (3, 256, 256))