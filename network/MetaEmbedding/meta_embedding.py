# encoding=utf-8
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


class NoModule(nn.Module):
    def __init__(self):
        super(NoModule, self).__init__()

    def forward(self, x):
        return x


class DirectFeature(nn.Module):
    def __init__(self, arch):
        super(DirectFeature, self).__init__()
        self.network = arch_dict[arch](pretrained=True)
        # pdb.set_trace()
        self.num_ftrs = self.network.fc.in_features
        self.network.fc = NoModule()  # Direct Feature没有fc了

    def forward(self, input):
        x = input
        y = self.network(x)
        return y

    def get_feature_num(self):
        return self.num_ftrs


class Classifier(nn.Module):
    def __init__(self, num_features):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(num_features, classes)

    def forward(self, input):
        return self.fc(input)




# class Classifier(nn.Module):
#     def __init__(self, arch):
#         super(Classifier, self).__init__()
#         self.network = arch_dict[arch](pretrained=True)
#         # pdb.set_trace()
#         self.disc_loss = DiscCentroidsLoss(num_classes=classes, feat_dim=self.network.fc.in_features)
#
#         num_ftrs = self.network.fc.in_features
#         # self.networks['classifier'](self.features, self.centroids)
#         self.clf = MetaEmbedding_Classifier(feat_dim=num_ftrs, num_classes=classes)
#         self.centroids = self.disc_loss.centroids.data if self.training else None
#
#         self.network._modules.pop('fc')
#         """
#         self.clf.fc_hallucinator = init_weights(model=self.clf.fc_hallucinator,
#                                            weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset,
#                                            classifier=True)
#         """
#
#         # self.network.fc = self.clf.fc_hallucinator
#
#     def forward(self, input, centroids):
#         x = self.network(input)
#
#         y = self.clf(x, centroids)
#
#         return y  # logits, [direct_feature, infused_feature]




# a = Classifier()
# img = torch.randn([1, 3, 256, 256])
# a(img)
