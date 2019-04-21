import PIL
import torch as th
import os

from torch.utils.data import Dataset, DataLoader
from t2ftextextractor import load_pickle
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class Face2TextDataset(Dataset):
    """ PyTorch Dataset wrapper around the Face2Text dataset """

    def __load_data(self):
        """
        private helper for loading the data
        :return: data => dict of data objs
        """
        data = load_pickle(self.pickle_file_path)

        return data

    def __init__(self, pro_pick_file, img_dir, img_transform=None, captions_len=100):
        """
        constructor of the class
        :param pro_pick_file: processed pickle file
        :param img_dir: path to the images directory
        :param img_transform: torch_vision transform to apply
        :param captions_len: maximum length of the generated captions
        """

        # create state:
        self.base_path = img_dir
        self.pickle_file_path = pro_pick_file
        self.transform = img_transform
        self.max_caption_len = captions_len

        data_obj = self.__load_data()

        # extract all the data
        self.text_data = data_obj['data']
        self.rev_vocab = data_obj['rev_vocab']
        self.vocab = data_obj['vocab']
        self.images = data_obj['images']
        self.vocab_size = len(self.vocab)



        print(self.vocab_size, len(self.rev_vocab), self.rev_vocab['<pad>'], self.rev_vocab['a'])

    def __len__(self):
        """
        obtain the length of the data-items
        :return: len => length
        """
        return len(self.images)

    def __getitem__(self, ix):
        """
        code to obtain a specific item at the given index
        :param ix: index for element query
        :return: (caption, img) => caption and the image
        """

        # read the image at the given index
        img_file_path = os.path.join(self.base_path, self.images[ix])
        img = PIL.Image.open(img_file_path)

        # transform the image if required
        if self.transform is not None:
            img = self.transform(img)

        # get the encoded caption:
        caption = self.text_data[ix]

        # pad or truncate the caption length:
        if len(caption) < self.max_caption_len:
            while len(caption) != self.max_caption_len:
                caption.append(self.rev_vocab["<pad>"])

        elif len(caption) > self.max_caption_len:
            caption = caption[: self.max_caption_len]

        caption = th.tensor(caption, dtype=th.long)

        # return the data element
        return img, caption

    def get_english_caption(self, sent):
        """
        obtain the english words list for the given numeric sentence
        :param sent: numeric id sentence
        :return: sent => list[String]
        """
        return list(map(lambda x: self.vocab[x], sent.numpy()))
