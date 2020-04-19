import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnest import resnest200

classes = 50030


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.network = resnest200(pretrained=False)
        # pdb.set_trace()
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, classes)

    def forward(self, input):
        x = input
        y = self.network(x)
        return y


# a = Classifier()
# img = torch.randn([1, 3, 256, 256])
# a(img)
