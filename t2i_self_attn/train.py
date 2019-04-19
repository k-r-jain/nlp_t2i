import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
# from pycocotools.coco import COCO
from cocoloader import COCODataLoader
from selfattn import Embedder, PositionalEncoder, Encoder


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
BATCH_SIZE = 64
EMBEDDING_DIM = 512
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 4
TRANSFORMER_DROPOUT = 0.1


transform = transforms.Compose([ 
	transforms.RandomCrop(224),
	# transforms.RandomHorizontalFlip(), 
	transforms.ToTensor(), 
	transforms.Normalize((0.485, 0.456, 0.406), 
						(0.229, 0.224, 0.225))])

data_loader = COCODataLoader(COCO_DIR, ANNOTATION_FILE, vocab, 
							transform, BATCH_SIZE,
							shuffle=True, num_workers=8) 


# embedder = Embedder(vocab_size = VOCAB_SIZE, d_model = EMBEDDING_DIM).to(device)
encoder = Encoder(vocab_size = VOCAB_SIZE, d_model = TRANSFORMER_DIM, N = TRANSFORMER_LAYERS, heads = TRANSFORMER_HEADS, dropout = TRANSFORMER_DROPOUT).to(device)
# print(len(data_loader))
for i, sample in enumerate(data_loader):
	image = sample[0].to(device)
	captions = sample[1].to(device)
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
	
	representations = encoder(captions, captions_mask).transpose(0, 1) # Batch first
	print(representations.size())


	# print(image.size(), target.size())
	# print(target)
	# print(sample)