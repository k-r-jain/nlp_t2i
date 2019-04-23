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
from cocoloader import COCODataLoader
from selfattn import Embedder, PositionalEncoder, Encoder
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
VOCAB_FILE = '/home/kartik/data/coco/vocab.pkl'
with open(VOCAB_FILE, 'rb') as f:
		vocab = pickle.load(f)
VOCAB_SIZE = len(vocab)
# print(vocab('<pad>'))
# print(vocab('<start>'))
# print(vocab('<end>'))
# print(vocab('<unk>'))
# print(vocab('a'))
# print(vocab('the'))
# print(vocab('blue'))
# print(vocab('gzorpazorp'))

COCO_DIR = '/home/kartik/data/coco/train2014'
ANNOTATION_FILE = '/home/kartik/data/coco/captions_train2014.json'
BATCH_SIZE = 80
NUM_EPOCHS = 25
LR_G = 1e-3
LR_D = 1e-3
EMBEDDING_DIM = 512

TRANSFORMER_DIM = 128
GENERATOR_CHANNELS = TRANSFORMER_DIM
DISCRIMINATOR_CHANNELS = 512
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1
Z_DIM = 256
FINAL_IMAGE_RES = 256
fraction = 0.01

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)


transform = transforms.Compose([
	transforms.Resize(FINAL_IMAGE_RES),
	transforms.RandomCrop(FINAL_IMAGE_RES),
	# transforms.RandomHorizontalFlip(), 
	transforms.ToTensor(), 
	transforms.Normalize((0.5, 0.5, 0.5), 
						(0.5, 0.5, 0.5))])
# transforms.Normalize((0.485, 0.456, 0.406), 
					# (0.229, 0.224, 0.225))])

data_loader = COCODataLoader(COCO_DIR, ANNOTATION_FILE, vocab, 
							transform, BATCH_SIZE,
							shuffle=True, num_workers=8, fraction = fraction) 


# embedder = Embedder(vocab_size = VOCAB_SIZE, d_model = EMBEDDING_DIM).to(device)
encoder = Encoder(vocab_size = VOCAB_SIZE, d_model = TRANSFORMER_DIM, N = TRANSFORMER_LAYERS, heads = TRANSFORMER_HEADS, dropout = TRANSFORMER_DROPOUT).to(device)

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
for epoch in range(NUM_EPOCHS):
	print('Epoch', epoch, '-' * 30)
	
	for i, sample in enumerate(data_loader):

		# if i > 100:
		# 	break

		images_256 = sample[0].to(device)
		captions = sample[1].to(device)
		batch, channels, im_w, im_h = images_256.size()


		real = torch.ones((batch,), requires_grad = False).to(device)
		fake = torch.zeros((batch,), requires_grad = False).to(device)

		# masks from the index values
		captions = captions.transpose(0, 1)
		captions_mask = (captions != vocab('<pad>')).unsqueeze(1)
		# print(captions.size(), captions_mask.size())
		

		########################### Discriminator ##############################

		discriminator.zero_grad()
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
				p.grad.data = p.grad.data.clamp(0.0, 1.0)
		d_loss = d_loss_real + d_loss_fake
		optimizer_d.step()
		
		########################### Discriminator  end ##############################


		generator.zero_grad()
		# _, adv_pred_features = discriminator(gen_images_256)
		# _, real_pred_features = discriminator(images_256)
		# g_loss = mse_loss(adv_pred_features, real_pred_features)
		g_loss = mse_loss(gen_images_256, images_256)
		# g_loss = cost_fn(adv_pred, real)
		g_loss.backward()
		# D_G_z2 = adv_pred.mean().item()
		D_G_z2 = g_loss.item()
		optimizer_g.step()

		print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
			  % (epoch, NUM_EPOCHS, i, NUM_BATCHES,
				 d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
			
		if i % 100 == 0:
			import matplotlib.pyplot as plt
			plt.imshow(np.transpose(images_256[0].cpu().detach(), (1, 2, 0)))
			plt.pause(1)
			plt.imshow(np.transpose(gen_images_256[0].cpu().detach(), (1, 2, 0)))
			plt.pause(1)
			plt.close()
