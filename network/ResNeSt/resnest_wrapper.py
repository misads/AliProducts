import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnest import resnest50, resnest101, resnest200
from options import opt

classes = opt.num_classes

arch_dict = {
    'ResNeSt50': resnest50,
    'ResNeSt101': resnest101,
    'ResNeSt200': resnest200

}


class Classifier(nn.Module):
    def __init__(self, arch):
        super(Classifier, self).__init__()
        self.network = arch_dict[arch](pretrained=True)
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
