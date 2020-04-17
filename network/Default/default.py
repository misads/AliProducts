import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet101

classes = 50030


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.network = resnet101()
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, classes)

    def forward(self, input):
        x = input
        return self.network(x)


# a = classifier()
# img = torch.randn([1, 3, 256, 256])

