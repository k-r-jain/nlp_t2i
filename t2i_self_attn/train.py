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


VOCAB_FILE = '/home/kartik/data/coco/vocab.pkl'
COCO_DIR = '/home/kartik/data/coco/train2014'
ANNOTATION_FILE = '/home/kartik/data/coco/captions_train2014.json'
BATCH_SIZE = 64
transform = transforms.Compose([ 
	transforms.RandomCrop(224),
	transforms.RandomHorizontalFlip(), 
	transforms.ToTensor(), 
	transforms.Normalize((0.485, 0.456, 0.406), 
						(0.229, 0.224, 0.225))])


with open(VOCAB_FILE, 'rb') as f:
		vocab = pickle.load(f)
	

data_loader = COCODataLoader(COCO_DIR, ANNOTATION_FILE, vocab, 
							transform, BATCH_SIZE,
							shuffle=True, num_workers=8) 

print(len(data_loader))
for i, sample in enumerate(data_loader):
	image = sample[0]
	target = sample[1]
	# print(image.size(), target.size())
	# print(target)
	# print(sample)