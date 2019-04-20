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
	def __init__(self, z_dim, generator_channels, start_channels = 512, start_size = 4, final_conv_size = 32):
		super(Generator, self).__init__()
		self.start_channels = start_channels
		self.start_size = start_size
		self.generator_channels = generator_channels

		self.init_linear = nn.Sequential(
			nn.Linear(z_dim, start_channels * start_size * start_size),
			nn.LeakyReLU(negative_slope = 0.01),
			nn.BatchNorm1d(start_channels * start_size * start_size)
		)

		modules = []
		current_size = start_size
		current_channels = start_channels
		while(current_size < final_conv_size):

			if current_channels <= generator_channels:
				next_stage_channels = generator_channels
			else:
				next_stage_channels = int(current_channels / 2)
			modules.append(self.upblock(in_channels = current_channels, out_channels = next_stage_channels))

			current_channels = next_stage_channels
			current_size *= 2
		
		self.init_G = nn.Sequential(*modules)

		self.stage1_G = self.upblock(in_channels = self.generator_channels, out_channels = generator_channels)
		self.stage2_G = self.upblock(in_channels = self.generator_channels, out_channels = generator_channels)
		self.stage3_G = self.upblock(in_channels = self.generator_channels, out_channels = generator_channels)

		self.image_gen_64x64 = self.to_3_channels(in_channels = self.generator_channels)
		self.image_gen_128x128 = self.to_3_channels(in_channels = self.generator_channels)
		self.image_gen_256x256 = self.to_3_channels(in_channels = self.generator_channels)




	def upblock(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, activation = 'leakyrelu', batchnorm = True, scale_factor = 2):
		modules = []
		modules.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding))

		if activation == 'leakyrelu':
			modules.append(nn.LeakyReLU(negative_slope = 0.01))

		if batchnorm:
			modules.append(nn.BatchNorm2d(num_features = out_channels))
		
		modules.append(nn.Upsample(scale_factor = scale_factor, mode = 'nearest'))

		return nn.Sequential(*modules)

	def to_3_channels(self, in_channels, kernel_size = 3, stride = 1, padding = 1):
		return nn.Conv2d(in_channels = in_channels, out_channels = 3, kernel_size = kernel_size, stride = stride, padding = padding)

	def attention(self, x, w):
		# w is text (batch, num_words, transformer dim)
		# x is image (batch, gen_channels, img width, img height)
		batch = x.size(0)
		width = x.size(2)
		height = x.size(3)
		num_pixels = x.size(2) * x.size(3)
		num_words = w.size(1)
		internal_dim = x.size(1)
		x = x.view(x.size(0), x.size(1), -1)

		# s -> (batch, num_words, num_pixels)
		s = torch.bmm(w, x)
		s = s.view(s.size(0) * num_pixels, -1)
		s = nn.Softmax()(s)
		s = s.view(batch, num_pixels, num_words)

		word_context = torch.bmm(s, w)
		word_context = word_context.view(batch, internal_dim, width, height)
		return word_context



	def forward(self, z, w):
		z = z.view(-1, z.size(2)) # To remove the fake dimension - single channel
		z = self.init_linear(z)
		z = z.view(-1, self.start_channels, self.start_size, self.start_size)
		x = self.init_G(z)
		
		word_context = self.attention(x = x, w = w)
		x = self.stage1_G(word_context)
		image_64x64 = self.image_gen_64x64(x)
		image_64x64 = nn.Tanh()(image_64x64)

		word_context = self.attention(x = x, w = w)
		x = self.stage2_G(word_context)
		image_128x128 = self.image_gen_128x128(x)
		image_128x128 = nn.Tanh()(image_128x128)

		word_context = self.attention(x = x, w = w)
		x = self.stage3_G(word_context)
		image_256x256 = self.image_gen_256x256(x)
		image_256x256 = nn.Tanh()(image_256x256)
		
		return image_64x64, image_128x128, image_256x256
