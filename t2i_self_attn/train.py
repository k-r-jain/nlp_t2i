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
BATCH_SIZE = 32
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

transform = transforms.Compose([
	transforms.Resize(FINAL_IMAGE_RES),
	transforms.RandomCrop(FINAL_IMAGE_RES),
	# transforms.RandomHorizontalFlip(), 
	transforms.ToTensor(), 
	# transforms.Normalize((0.5, 0.5, 0.5), 
						# (0.5, 0.5, 0.5))])
transforms.Normalize((0.485, 0.456, 0.406), 
					(0.229, 0.224, 0.225))])

data_loader = COCODataLoader(COCO_DIR, ANNOTATION_FILE, vocab, 
							transform, BATCH_SIZE,
							shuffle=True, num_workers=8) 


# embedder = Embedder(vocab_size = VOCAB_SIZE, d_model = EMBEDDING_DIM).to(device)
encoder = Encoder(vocab_size = VOCAB_SIZE, d_model = TRANSFORMER_DIM, N = TRANSFORMER_LAYERS, heads = TRANSFORMER_HEADS, dropout = TRANSFORMER_DROPOUT).to(device)

generator = Generator(z_dim = Z_DIM, generator_channels = GENERATOR_CHANNELS, start_channels = Z_DIM + 2 * TRANSFORMER_DIM).to(device)
# summary(generator, ((1, Z_DIM)))

discriminator1 = Discriminator(discriminator_channels = DISCRIMINATOR_CHANNELS, current_size = 64).to(device)
discriminator2 = Discriminator(discriminator_channels = DISCRIMINATOR_CHANNELS, current_size = 128).to(device)
discriminator3 = Discriminator(discriminator_channels = DISCRIMINATOR_CHANNELS, current_size = 256).to(device)
# summary(discriminator, (3, 64, 64))
# summary(discriminator, (3, 128, 128))
# summary(discriminator, (3, 256, 256))


cost_fn = nn.BCELoss().to(device)

optimizer_g = torch.optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr = LR_G)
optimizer_d1 = torch.optim.Adam(discriminator1.parameters(), lr = LR_D)
optimizer_d2 = torch.optim.Adam(discriminator2.parameters(), lr = LR_D)
optimizer_d3 = torch.optim.Adam(discriminator3.parameters(), lr = LR_D)

NUM_BATCHES = len(data_loader)
print(NUM_BATCHES)
for epoch in range(NUM_EPOCHS):
	print('Epoch', epoch, '-' * 30)
	avg_d_loss = 0.0
	avg_g_loss = 0.0
	for i, sample in enumerate(data_loader):

		if i > 200:
			break


		optimizer_d1.zero_grad()
		optimizer_d2.zero_grad()
		optimizer_d3.zero_grad()
		optimizer_g.zero_grad()

		# discriminator1.zero_grad()
		# discriminator2.zero_grad()
		# discriminator3.zero_grad()
		# generator.zero_grad()
		# encoder.zero_grad()

		images_256 = sample[0].to(device)
		captions = sample[1].to(device)

		batch, channels, im_w, im_h = images_256.size()

		images_128 = nn.functional.interpolate(images_256, (128, 128), mode = 'nearest')
		images_64 = nn.functional.interpolate(images_256, (64, 64), mode = 'nearest')

		real = torch.ones((batch), requires_grad = False).to(device)
		# real_128 = torch.ones((batch), dtype = torch.long).to(device)
		# real_256 = torch.ones((batch), dtype = torch.long).to(device)

		fake = torch.zeros((batch), requires_grad = False).to(device)
		# fake_128 = torch.zeros((batch), dtype = torch.long).to(device)
		# fake_256 = torch.zeros((batch), dtype = torch.long).to(device)


		# randperm = torch.randperm(BATCH_SIZE * 2)
		# labels = torch.cat((real, fake), dim = 0)
		# labels = labels[randperm]

		# current_batch_max_seq_len = captions.size(1)

		# embeddings = embedder(captions)
		# # Creating positional embedding object here since no learning and to save max seqeunce length space
		# posenc = PositionalEncoder(d_model = EMBEDDING_DIM, max_seq_len = current_batch_max_seq_len, dropout = 0.0).to(device)
		# encodings = posenc(embeddings)
		# print(embeddings.size(), encodings.size())

		# masks from the index values
		captions = captions.transpose(0, 1)
		captions_mask = (captions != vocab('<pad>')).unsqueeze(1)
		# print(captions.size(), captions_mask.size())
		
		representations, sent_embeddings = encoder(captions, captions_mask) 
		# Batch first
		representations = representations.transpose(0, 1)
		sent_embeddings = sent_embeddings.transpose(0, 1)
		# print(representations.size())

		z = torch.randn((BATCH_SIZE, 1, Z_DIM)).to(device)
		gen_images_64, gen_images_128, gen_images_256 = generator(z, representations, sent_embeddings)

		# predr_1, predr_2, predr_3 = discriminator(images_64, images_128, images_256)
		# predf_1, predf_2, predf_3 = discriminator(gen_images_64, gen_images_128, gen_images_256)

		# images_64 = torch.cat((images_64, gen_images_64.detach()), dim = 0)
		# images_128 = torch.cat((images_128, gen_images_128.detach()), dim = 0)
		# images_256 = torch.cat((images_256, gen_images_256.detach()), dim = 0)
		
		# images_64 = images_64[randperm]
		# images_128 = images_128[randperm]
		# images_256 = images_256[randperm]

		labels = real
		pred_64_1 = discriminator1(images_64)
		pred_128_1 = discriminator2(images_128)
		pred_256_1 = discriminator3(images_256)

		d_loss = 0
		d_loss_real = 0
		d_loss_fake = 0

		loss1 = cost_fn(pred_64_1, labels)
		d_loss += loss1.item()
		d_loss_real += loss1.item()
		loss1.backward()

		loss2 = cost_fn(pred_128_1, labels)
		d_loss += loss2.item()
		d_loss_real += loss1.item()
		loss2.backward()

		loss3 = cost_fn(pred_256_1, labels)
		d_loss += loss3.item()
		d_loss_real += loss1.item()
		loss3.backward()

		# optimizer_d1.step()
		# optimizer_d2.step()
		# optimizer_d3.step()

		labels = fake
		pred_64_2 = discriminator1(gen_images_64.detach())
		pred_128_2 = discriminator2(gen_images_128.detach())
		pred_256_2 = discriminator3(gen_images_256.detach())

		d_loss = 0
		loss4 = cost_fn(pred_64_2, labels)
		d_loss += loss4.item()
		d_loss_fake += loss4.item()
		loss4.backward()

		loss5 = cost_fn(pred_128_2, labels)
		d_loss += loss5.item()
		d_loss_fake += loss4.item()
		loss5.backward()

		loss6 = cost_fn(pred_256_2, labels)
		d_loss += loss6.item()
		d_loss_fake += loss4.item()
		loss6.backward()

		# optimizer_d1.step()
		# optimizer_d2.step()
		# optimizer_d3.step()

		# G trying to fool D by comparing with real
		labels = real
		g_loss = 0

		# representations, sent_embeddings = encoder(captions, captions_mask) 
		# # Batch first
		# representations = representations.transpose(0, 1)
		# sent_embeddings = sent_embeddings.transpose(0, 1)

		# gen_images_64, gen_images_128, gen_images_256 = generator(z, representations, sent_embeddings)
		pred_64_3 = discriminator1(gen_images_64)
		loss7 = cost_fn(pred_64_3, labels)
		g_loss += loss7.item()
		# loss7.backward(retain_graph = True)
		# optimizer_g.step()
		
		# optimizer_g.zero_grad()
		# representations, sent_embeddings = encoder(captions, captions_mask) 
		# # Batch first
		# representations = representations.transpose(0, 1)
		# sent_embeddings = sent_embeddings.transpose(0, 1)

		# gen_images_64, gen_images_128, gen_images_256 = generator(z, representations, sent_embeddings)
		pred_128_3 = discriminator2(gen_images_128)
		loss8 = cost_fn(pred_128_3, labels)
		g_loss += loss8.item()
		# loss8.backward(retain_graph = True)
		# optimizer_g.step()

		# optimizer_g.zero_grad()
		# representations, sent_embeddings = encoder(captions, captions_mask) 
		# # Batch first
		# representations = representations.transpose(0, 1)
		# sent_embeddings = sent_embeddings.transpose(0, 1)

		# gen_images_64, gen_images_128, gen_images_256 = generator(z, representations, sent_embeddings)
		pred_256_3 = discriminator3(gen_images_256)
		loss9 = cost_fn(pred_256_3, labels)
		g_loss += loss9.item()
		# loss9.backward()
		final_g_loss = loss7 + loss8 + loss9
		final_g_loss.backward()


		if d_loss > g_loss:
			optimizer_d1.step()
			optimizer_d2.step()
			optimizer_d3.step()
			del pred_64_3, pred_128_3, pred_256_3
		else:
			optimizer_g.step()
			del pred_64_1, pred_64_2, pred_128_1, pred_128_2, pred_256_1, pred_256_2
		
		print('\r iter', i , 'loss:', d_loss_fake, d_loss_real, g_loss, end = '')
		avg_d_loss += d_loss
		avg_g_loss += g_loss
	
		# print(word_context.size())
		# print(image.size(), target.size())
		# print(target)
		# print(sample)
		if i % 100 == 0:
			import matplotlib.pyplot as plt
			plt.imshow(np.transpose(images_256[0].cpu().detach(), (1, 2, 0)))
			plt.pause(1)
			# plt.show()
			plt.imshow(np.transpose(gen_images_64[0].cpu().detach(), (1, 2, 0)))
			plt.pause(1)
			# plt.show()
			plt.imshow(np.transpose(gen_images_128[0].cpu().detach(), (1, 2, 0)))
			plt.pause(1)
			# plt.show()
			plt.imshow(np.transpose(gen_images_256[0].cpu().detach(), (1, 2, 0)))
			plt.pause(1)
			# plt.show()
			# transforms.ToPILImage()(images_64[0].cpu().detach()).show()
			# transforms.ToPILImage()(images_128[0].cpu().detach()).show()
			# transforms.ToPILImage()(images_256[0].cpu().detach()).show()
			# transforms.ToPILImage()(gen_images_64[0].cpu().detach()).show()
			# transforms.ToPILImage()(gen_images_128[0].cpu().detach()).show()
			# transforms.ToPILImage()(gen_images_256[0].cpu().detach()).show()
			print('AVG D and G loss:', avg_d_loss, avg_g_loss)
			avg_d_loss, avg_g_loss = 0.0, 0.0
	# rows = 3
	# cols = 3
	# fig = plt.figure(figsize = (8, 8))

	# fig.add_subplot(rows, cols, 1)
	# plt.imshow(gen_images_64[0].cpu().detach().numpy())
	# fig.add_subplot(rows, cols, 2)
	# plt.imshow(gen_images_64[1].cpu().detach().numpy())
	# fig.add_subplot(rows, cols, 3)
	# plt.imshow(gen_images_64[2].cpu().detach().numpy())

	# fig.add_subplot(rows, cols, 4)
	# plt.imshow(gen_images_128[0].cpu().detach().numpy())
	# fig.add_subplot(rows, cols, 5)
	# plt.imshow(gen_images_128[1].cpu().detach().numpy())
	# fig.add_subplot(rows, cols, 6)
	# plt.imshow(gen_images_128[2].cpu().detach().numpy())

	# fig.add_subplot(rows, cols, 7)
	# plt.imshow(gen_images_256[0].cpu().detach().numpy())
	# fig.add_subplot(rows, cols, 8)
	# plt.imshow(gen_images_256[1].cpu().detach().numpy())
	# fig.add_subplot(rows, cols, 9)
	# plt.imshow(gen_images_256[2].cpu().detach().numpy())

	
	# plt.imshow(np.transpose(gen_images_64[0].cpu().detach().transpose().numpy(), (1, 2, 0)))
	# plt.show()

	# plt.imshow(gen_images_128[0].cpu().detach().numpy())
	# plt.show()

	# plt.imshow(gen_images_256[0].cpu().detach().numpy())
	# plt.show()