import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torchsummary import summary
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
# from pycocotools.coco import COCO
from cocoloader import COCODataLoader
from selfattn import Embedder, PositionalEncoder, Encoder
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt


import PIL
import torch as th
import os

from torch.utils.data import Dataset, DataLoader
from t2floader import Face2TextDataset
from t2ftextextractor import load_pickle
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from inception_score import inception_score








import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy










device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

LFW_DIR = '/home/kartik/data/face2text/lfw'
PROC_PICKLE_PATH = '/home/kartik/data/face2text/processed_text.pkl'
CAPTION_LEN = 10
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR_G = 1e-3
LR_D = 1e-4
# EMBEDDING_DIM = 512

TRANSFORMER_DIM = 50
GENERATOR_CHANNELS = TRANSFORMER_DIM
DISCRIMINATOR_CHANNELS = 512
TRANSFORMER_HEADS = 2
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.3
Z_DIM = 256
FINAL_IMAGE_RES = 256

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

transformation = transforms.Compose([
			transforms.Resize(FINAL_IMAGE_RES),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])

face2text_dataset = Face2TextDataset(pro_pick_file = PROC_PICKLE_PATH, img_dir = LFW_DIR, img_transform = transformation, captions_len = CAPTION_LEN)


print("INCEPTION", inception_score(face2text_dataset, cuda = True, batch_size = 4, resize = True, splits = 2))


rev_vocab = face2text_dataset.rev_vocab
vocab = face2text_dataset.vocab
# print(rev_vocab)
# print('vocab', vocab)
VOCAB_SIZE = len(vocab)
print(VOCAB_SIZE)
# print(vocab)
data_loader = DataLoader(face2text_dataset, batch_size = BATCH_SIZE, num_workers = 8, shuffle = True)



encoder = Encoder(vocab_size = VOCAB_SIZE, d_model = TRANSFORMER_DIM, N = TRANSFORMER_LAYERS, heads = TRANSFORMER_HEADS, dropout = TRANSFORMER_DROPOUT, vocab = (vocab, rev_vocab)).to(device)
generator = Generator(z_dim = Z_DIM, generator_channels = GENERATOR_CHANNELS).to(device)
generator.apply(weights_init)
# summary(generator, ((1, Z_DIM)))
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
# summary(discriminator, (3, 64, 64))
# summary(discriminator, (3, 128, 128))
# summary(discriminator, (3, 256, 256))


cost_fn = nn.BCELoss().to(device)
mse_loss = nn.MSELoss().to(device)

optimizer_g = torch.optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr = LR_G, betas = (0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = LR_D, betas = (0.5, 0.999))

NUM_BATCHES = len(data_loader)
print(NUM_BATCHES)

g_plot = []
d_plot = []
for epoch in range(NUM_EPOCHS):
	print('Epoch', epoch, '-' * 30)

	stacked_gen_images = None
	for i, sample in enumerate(data_loader):


		# if i > 200:
		#   break

		images_256 = sample[0].to(device)
		captions = sample[1].to(device)
		batch, channels, im_w, im_h = images_256.size()


		real = torch.ones((batch,), requires_grad = False).to(device)
		fake = torch.zeros((batch,), requires_grad = False).to(device)

		# masks from the index values
		captions = captions.transpose(0, 1)
		captions_mask = (captions != rev_vocab['<pad>']).unsqueeze(1)
		# print(captions.size(), captions_mask.size(), images_256.size())
		

		########################### Discriminator ##############################

		optimizer_d.zero_grad()
		real_pred, _ = discriminator(images_256)
		d_loss_real = cost_fn(real_pred, real)
		d_loss_real.backward()
		D_x = real_pred.mean().item()



		representations, sent_embeddings = encoder(captions, captions_mask) 
		# Batch first
		representations = representations.transpose(0, 1)
		sent_embeddings = sent_embeddings.transpose(0, 1)
		sent_embeddings = sent_embeddings[:, sent_embeddings.size(1) - 1, :]
		sent_embeddings = sent_embeddings.view(sent_embeddings.size(0), 2 * TRANSFORMER_DIM, 1, 1)

		z = torch.randn((sent_embeddings.size(0), Z_DIM, 1, 1)).to(device)

		gen_images_256 = generator(z, representations, sent_embeddings)
		fake_pred, _ = discriminator(gen_images_256.detach())
		d_loss_fake = cost_fn(fake_pred, fake)
		d_loss_fake.backward()
		D_G_z1 = fake_pred.mean().item()

		for p in discriminator.parameters():
			if p.grad is not None:
				p.grad.data = p.grad.data.clamp(-0.01, 0.01)
		d_loss = d_loss_real + d_loss_fake
		optimizer_d.step()
		
		########################### Discriminator  end ##############################


		optimizer_g.zero_grad()
		adv_pred, adv_pred_features = discriminator(gen_images_256)
		_, real_pred_features = discriminator(images_256)
		# g_loss = mse_loss(adv_pred_features, real_pred_features)
		# g_loss = mse_loss(gen_images_256, images_256)
		# g_loss += cost_fn(adv_pred, real)
		g_loss = cost_fn(adv_pred, real)
		g_loss.backward()
		# D_G_z2 = adv_pred.mean().item()
		D_G_z2 = g_loss.item()
		optimizer_g.step()

		print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
			  % (epoch, NUM_EPOCHS, i, NUM_BATCHES,
				 d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
		
		g_plot.append(g_loss.item())
		d_plot.append(d_loss.item())
		if i % 50 == 0:
			import matplotlib.pyplot as plt
			print(face2text_dataset.get_english_caption(captions.transpose(0, 1)[0].cpu().detach()))
			# plt.imshow(np.transpose(images_256[0].cpu().detach(), (1, 2, 0)))
			# plt.show()
			# plt.imshow(np.transpose(gen_images_256[0].cpu().detach(), (1, 2, 0)))
			# plt.show()
	
		# if stacked_gen_images is None:
		# 	stacked_gen_images = gen_images_256.cpu().detach()
		# else:
		# 	stacked_gen_images = torch.cat((stacked_gen_images, gen_images_256.cpu().detach()), dim = 0)
		# print(stacked_gen_images.size())
		# if i % 10 == 0:
		# 	# print(inception_score(stacked_gen_images.cpu().detach(), cuda = True, batch_size = 4, resize = True, splits = 2, is_tensor = True))
		# 	# stacked_gen_images = None
		# 	print(inception_score(gen_images_256.cpu().detach(), cuda = True, batch_size = 4, resize = True, splits = 2, is_tensor = True))


	
		# print(images_256.cpu().numpy().shape)
plt.clf()
plt.plot(g_plot, label = 'G_loss')
plt.plot(d_plot, label = 'D_loss')
plt.legend(loc = 'best')
plt.show()

# for i, sample in enumerate(dataloader):

#     captions = sample[0].to('cuda:0')
#     images = sample[1].to('cuda:0')

#     # print(captions.size(), images.size())
#     print(captions)
#     img = np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0))
#     plt.imshow(img)
#     plt.pause(1)