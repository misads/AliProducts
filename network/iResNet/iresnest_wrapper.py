import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .iresnet.iresnet import iresnet101, iresnet152, iresnet200, iresnet1001

classes = 50030

arch_dict = {
    'iResNet101': iresnet101,
    'iResNet152': iresnet152,
    'iResNet200': iresnet200,
    'iResNet1001': iresnet1001

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


# a = Classifier('iResNet101')
# img = torch.randn([1, 3, 256, 256])
# a(img)
