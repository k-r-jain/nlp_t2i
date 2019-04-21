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
	def __init__(self, discriminator_channels, current_size, start_channels = 16, final_conv_size = 1):
		super(Discriminator, self).__init__()
		self.discriminator_channels = discriminator_channels
		self.start_channels = start_channels
		self.final_conv_size = final_conv_size
		self.current_size = current_size

		if self.current_size == 64:
			self.stage1_D = self.create_network(current_size = 64, current_channels = 3)
		elif self.current_size == 128:
			self.stage2_D = self.create_network(current_size = 128, current_channels = 3)
		elif self.current_size == 256:
			self.stage3_D = self.create_network(current_size = 256, current_channels = 3)

	def create_network(self, current_size, current_channels):
		modules = []
		modules.append(self.downblock(in_channels = current_channels, out_channels = self.start_channels))
		current_channels = self.start_channels
		current_size = int(current_size / 2)
		# print(current_size, current_channels)
		while(current_size > self.final_conv_size):

			if current_channels >= self.discriminator_channels:
				next_stage_channels = self.discriminator_channels
			else:
				next_stage_channels = current_channels * 2
			modules.append(self.downblock(in_channels = current_channels, out_channels = next_stage_channels))

			current_channels = next_stage_channels
			current_size = int(current_size / 2)
			
		modules.append(nn.Conv2d(in_channels = current_channels, out_channels = 1, kernel_size = 1, padding = 0))
		modules.append(nn.Sigmoid())
		return nn.Sequential(*modules)

	def downblock(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, activation = 'leakyrelu', batchnorm = True, scale_factor = 2):
		modules = []
		modules.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding))

		if activation == 'leakyrelu':
			modules.append(nn.LeakyReLU(negative_slope = 0.01))

		if batchnorm:
			modules.append(nn.BatchNorm2d(num_features = out_channels))
		
		modules.append(nn.MaxPool2d(kernel_size = scale_factor))
		return nn.Sequential(*modules)
	
	def forward(self, im):
		batch = im.size(0)
		if self.current_size == 64:
			pred = self.stage1_D(im).view(batch)
		elif self.current_size == 128:
			pred = self.stage2_D(im).view(batch)
		elif self.current_size == 256:
			pred = self.stage3_D(im).view(batch)

		# pred2 = self.stage2_D(im_128).view(batch)
		# pred3 = self.stage3_D(im_256).view(batch)
		return pred