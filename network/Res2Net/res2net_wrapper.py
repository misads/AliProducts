import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .res2net_v1b import res2net101_v1b

classes = 50030


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.network = res2net101_v1b(pretrained=True)
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
