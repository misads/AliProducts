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
    'MetaEmbedding': resnest101,
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


class MetaEmbedding(nn.Module):

    def __init__(self, feat_dim=2048, num_classes=1000):
        super(MetaEmbedding, self).__init__()
        self.num_classes = num_classes
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.classifier = Classifier(feat_dim)

    def forward(self, x, centroids, *args):
        # storing direct feature
        direct_feature = x.clone()

        batch_size = x.size(0)
        feat_size = x.size(1)

        # set up visual memory
        x_expand = x.clone().unsqueeze(1).expand(-1, self.num_classes, -1)
        centroids_expand = centroids.clone().unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids.clone()

        # computing reachability
        dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
        values_nn, labels_nn = torch.sort(dist_cur, 1)
        scale = 10.0
        reachability = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        # computing memory feature by querying and associating visual memory
        values_memory = self.fc_hallucinator(x.clone())
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x.clone())
        concept_selector = concept_selector.tanh()
        x = reachability * (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature

        logits = self.classifier(x)

        return logits, [direct_feature, infused_feature]



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
