# encoding=utf-8
import pdb

import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms
import torchvision.transforms as transforms

import random
import numpy as np
import cv2

from dataloader.transforms.custom_transform import read_image


class TrainValDataset(dataset.Dataset):
    """ImageDataset for training.

    Args:
        datadir(str): dataset root path, default input and label dirs are 'input' and 'gt'
        aug(bool): data argument (×8)
        norm(bool): normalization

    Example:
        train_dataset = ImageDataset('train.txt', aug=False)
        for i, data in enumerate(train_dataset):
            input, label = data['input']. data['label']

    """

    def __init__(self, file_list, transforms, max_size=None):
        self.im_names = []
        self.labels = []
        with open(file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                img, label = line.split(' ')
                label = int(label)
                self.im_names.append(img)
                self.labels.append(label)

        self.transforms = transforms
        self.max_size = max_size

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            {'input': input,
             'label': label,
             'path': path
            }

        """

        input = read_image(self.im_names[index])
        label = self.labels[index]

        sample = self.transforms(**{
            'image': input,
        })

        sample = {
            'input': sample['image'],
            'label': label,
            'path': self.im_names[index],
        }

        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)


class TestDataset(dataset.Dataset):
    """ImageDataset for test.

    Args:
        datadir(str): dataset path'
        norm(bool): normalization

    Example:
        test_dataset = ImageDataset('test', crop=256)
        for i, data in enumerate(test_dataset):
            input, file_name = data

    """

    def __init__(self, file_list, transforms, max_size=None):
        self.im_names = []
        with open(file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                img = line
                self.im_names.append(img)

        self.transforms = transforms
        self.max_size = max_size


    def __getitem__(self, index):

        input = read_image(self.im_names[index])

        sample = self.transforms(**{
            'image': input,
        })

        sample = {
            'input': sample['image'],
            'path': self.im_names[index],
        }

        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)

