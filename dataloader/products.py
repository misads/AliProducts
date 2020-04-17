import pdb

import torchvision.transforms.functional as F
import os
from PIL import Image
import torch.utils.data.dataset as dataset
from torchvision import transforms
import random
import numpy as np
import cv2


from torch_template import torch_utils

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

    def __init__(self, file_list, crop=None, aug=True, norm=False, max_size=None):
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

        self.trans_dict = {0: Image.FLIP_LEFT_RIGHT, 1: Image.FLIP_TOP_BOTTOM, 2: Image.ROTATE_90, 3: Image.ROTATE_180,
                           4: Image.ROTATE_270, 5: Image.TRANSPOSE, 6: Image.TRANSVERSE}

        if type(crop) == int:
            crop = (crop, crop)

        self.crop = crop
        self.aug = aug
        self.norm = norm
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

        input = Image.open(self.im_names[index]).convert("RGB")
        label = self.labels[index]

        r = random.randint(0, 7)
        if self.aug and r != 7:
            input = input.transpose(self.trans_dict[r])

        if self.norm:
            input = F.normalize(F.to_tensor(input), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            input = F.to_tensor(input)

        return {'input': input, 'label': label, 'path': self.im_names[index]}

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
    def __init__(self, file_list, norm=False, max_size=None):
        self.im_names = []
        with open(file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                img = line
                self.im_names.append(img)

        self.norm = norm
        self.max_size = max_size

    def __getitem__(self, index):

        input = Image.open(self.im_names[index]).convert("RGB")
        if self.norm:
            input = F.normalize(F.to_tensor(input), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            input = F.to_tensor(input)

        return {'input': input, 'path':self.im_names[index]}

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)


def preview_dataset(dataset, path='path'):
    for i, data in enumerate(dataset):
        if i == min(10, len(dataset) - 1):
            break

        c = 'input'
        img = data[c]
        img = np.array(img)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if path in data:
            cv2.putText(img, data[path][-11:], (1, 35), 0, 1, (255, 255, 255), 2)
        if 'label' in data:
            cv2.putText(img, 'label: %05d' % data['label'], (1, 70), 0, 1, (255, 255, 255), 2)
            cv2.imshow('train', img)

        else:
            cv2.imshow('test', img)

        cv2.waitKey(0)


if __name__ == '__main__':

    dataset = TrainValDataset('../train.txt', aug=False)
    preview_dataset(dataset)

    dataset = TestDataset('../test.txt')
    preview_dataset(dataset)

    # for i, data in enumerate(train_dataset):
    #     input, label, file_name = data
    #     torch_utils.write_image(writer, 'train', '0_input', input, i)
    #     torch_utils.write_image(writer, 'train', '2_label', label, i)
    #     print(i, file_name)

    # for i, data in enumerate(test_dataset):
    #     input, file_name = data
    #     torch_utils.write_image(writer, 'train', file_name, input, i)



