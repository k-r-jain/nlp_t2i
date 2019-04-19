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

class Generator(nn.Module):
	def __init__(self, z_dim, start_channels = 512, start_size = 4, final_conv_size = 64):
		super(Generator, self).__init__()
		self.start_channels = start_channels
		self.start_size = start_size

		self.init_linear = nn.Sequential(
			nn.Linear(z_dim, start_channels * start_size * start_size),
			nn.LeakyReLU(negative_slope = 0.01),
			nn.BatchNorm1d(start_channels * start_size * start_size)
		)

		modules = []
		current_size = start_size
		current_channels = start_channels
		while(current_size < final_conv_size):

			modules.append(self.upblock(in_channels = current_channels, out_channels = int(current_channels / 2)))

			current_channels = int(current_channels / 2)
			current_size *= 2
		
		self.init_G = nn.Sequential(*modules)


	def upblock(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, activation = 'leakyrelu', batchnorm = True, scale_factor = 2):
		modules = []
		modules.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding))

		if activation == 'leakyrelu':
			modules.append(nn.LeakyReLU(negative_slope = 0.01))

		if batchnorm:
			modules.append(nn.BatchNorm2d(num_features = out_channels))
		
		modules.append(nn.Upsample(scale_factor = scale_factor, mode = 'nearest'))

		return nn.Sequential(*modules)

	def forward(self, z):
		z = z.view(-1, z.size(2)) # To remove the fake dimention - single channel
		z = self.init_linear(z)
		z = z.view(-1, self.start_channels, self.start_size, self.start_size)
		x = self.init_G(z)
		return x

Z_DIM = 256
g = Generator(z_dim = Z_DIM).to('cuda:0')

summary(g, ((1, Z_DIM)))