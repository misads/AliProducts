import pdb

import numpy as np
import torch
import torch.nn as nn
from loss import DiscCentroidsLoss
from .MetaEmbeddingClassifier import MetaEmbedding_Classifier
import torch.nn.functional as F

from .resnest import resnest101, resnest200

classes = 50030

arch_dict = {
    'ResNeSt101': resnest101,
    'ResNeSt200': resnest200

}


class Classifier(nn.Module):
    def __init__(self, arch):
        super(Classifier, self).__init__()
        self.network = arch_dict[arch](pretrained=True)
        # pdb.set_trace()
        self.disc_loss = DiscCentroidsLoss(num_classes=classes, feat_dim=self.network.fc.in_features)

        num_ftrs = self.network.fc.in_features
        # self.networks['classifier'](self.features, self.centroids)
        self.clf = MetaEmbedding_Classifier(feat_dim=num_ftrs, num_classes=classes)
        self.centroids = self.disc_loss.centroids.data if self.training else None

        self.network._modules.pop('fc')
        """
        self.clf.fc_hallucinator = init_weights(model=self.clf.fc_hallucinator,
                                           weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset,
                                           classifier=True)
        """

        # self.network.fc = self.clf.fc_hallucinator

    def forward(self, input, centroids):
        x = input


        y = self.network(x)
        return y


# a = Classifier()
# img = torch.randn([1, 3, 256, 256])
# a(img)
