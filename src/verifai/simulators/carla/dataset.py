import os
import collections
import json

import numpy as np
from PIL import Image
import collections
import torch
#import torchvision

from torch.utils import data
import random 

from scipy.io import loadmat


class roadDataset(data.Dataset):
    def __init__(self,root='', split='train', img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.Imglist = np.load(f'{self.root}/{split}.npy')
        self.Imglist = np.transpose(self.Imglist, (0, 3, 2, 1))
        self.GT = np.load(f'{self.root}/{split}_label.npy')
        
        assert self.GT.shape[0] == self.Imglist.shape[0]
    def __len__(self):
        return self.Imglist.shape[0]

    def __getitem__(self, index):
        img = self.Imglist[index]
        lbl = self.GT[index]

        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self.img_transform is not None:
            img_o = self.img_transform(img)
            imgs = img_o
        else:
            imgs = img
        random.seed(seed)
        if self.label_transform is not None:
            label_o = self.label_transform(lbl)
            lbls = label_o
        else:
            lbls = lbl

        return imgs, lbls
