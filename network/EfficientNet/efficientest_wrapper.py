import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.EfficientNet.efficientnet import EfficientNet

classes = 50030

archs = {
    'EfficientNet-B7',
    'EfficientNet-B5',
}


class Classifier(nn.Module):
    def __init__(self, arch):
        super(Classifier, self).__init__()

        # self.network = EfficientNet.from_name(arch.lower())  # not pre-trained
        self.network = EfficientNet.from_pretrained(arch.lower())

        num_ftrs = self.network._fc.in_features
        self.network._fc = nn.Linear(num_ftrs, classes)

    def forward(self, input):
        x = input
        y = self.network(x)
        return y


# a = Classifier('EfficientNet-B7')
# img = torch.randn([1, 3, 256, 256])
# a(img)
