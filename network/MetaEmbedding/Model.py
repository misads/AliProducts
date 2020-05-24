# encoding=utf-8
import pdb

from collections import OrderedDict
import numpy as np
import pickle
import torch
import os

from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from itertools import chain
from torch_template.utils.torch_utils import ExponentialMovingAverage, print_network

from scheduler import get_scheduler
from options import opt
from dataloader import train_dataloader_plain, train_dataset

from .meta_embedding import DirectFeature, MetaEmbedding
import misc_utils as utils

#  criterionCE = nn.CrossEntropyLoss()

"""
训练步骤：
    ① 先用普通的dataloader训第一阶段(direct_feature+classifier)
    ② 把第一阶段的classifier load到meta_embedding的hallucinator里
    ③ 用普通的dataloader计算类别中心centroids(feature相加除以该类别的个数)
    ④ 用带sampler的dataloader训练第二阶段
"""


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.direct_feature = DirectFeature(opt.model)
        self.feature_nums = self.direct_feature.get_feature_num()
        self.meta_embedding = MetaEmbedding(self.feature_nums, 50030)

        print_network(self.direct_feature)
        print_network(self.meta_embedding)

        # TODO: 这里学习率是不是可以调成 direct_feature 0.01 meta_embedding 0.1
        self.optimizer = optim.SGD(chain(self.direct_feature.parameters(), self.meta_embedding.parameters()),
                                   lr=0.01, momentum=0.9, weight_decay=0.0005)

        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

        # different weight for different classes
        self.criterionCE = nn.CrossEntropyLoss()

    @staticmethod
    def class_count(dataset):
        labels = np.array(dataset.labels)
        class_data_num = []
        for l in np.unique(labels):
            class_data_num.append(len(labels[labels == l]))
        return class_data_num

    def centroids_cal(self, dataloader):
        #在embedding模式下生成mem,建议在train里的if opt.load 和 model.train()之间添加model.centroids_cal(train_dataloader)
        centroids = torch.zeros(50030, self.feature_nums).to(opt.device)

        # print('Calculating centroids.')

        self.eval()

        with torch.set_grad_enabled(False):

            for i, data in enumerate(dataloader):
                utils.progress_bar(i, len(dataloader), 'Calculating centroids...')
                inputs, labels = data['input'], data['label']
                inputs = inputs.to(opt.device)
                direct_features = self.direct_feature(inputs)
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += direct_features[i]

        # Average summed features with class count
        centroids /= torch.tensor(self.class_count(train_dataset)).float().unsqueeze(1).to(opt.device)  #class count为每一类的样本数，需要单独写。
        self.mem = centroids
        pdb.set_trace()

    def update(self, input, label):

        predicted = self.forward(input)
        # TODO：loss加上DiscCentroidsLoss
        loss = self.criterionCE(predicted, label)

        self.avg_meters.update({'Cross Entropy': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'predicted': predicted}

    def forward(self, x):
        direct_feature = self.direct_feature(x)
        # meta embedding里带有了classifier
        logits, _ = self.meta_embedding(direct_feature, self.mem)
        return logits

    def load(self, ckpt_path):
        load_dict = torch.load(ckpt_path, map_location=opt.device)
        if 'direct_feature' not in load_dict:  # 旧的checkpoint
            direct_feature = load_dict['classifier']
            classifier_dict = OrderedDict()
            classifier_dict['weight'] = direct_feature.pop('network.fc.weight')
            classifier_dict['bias'] = direct_feature.pop('network.fc.bias')
            self.direct_feature.load_state_dict(direct_feature)
            self.meta_embedding.fc_hallucinator.load_state_dict(classifier_dict)
            # 如果是从stage1 load的，计算centroids
            self.centroids_cal(train_dataloader_plain)

        else:  # 新的checkpoint
            self.direct_feature.load_state_dict(load_dict['direct_feature'])
            self.meta_embedding.load_state_dict(load_dict['meta_embedding'])
            self.mem.load_state_dict(load_dict['centroids'])

        if opt.resume:
            self.optimizer.load_state_dict(load_dict['optimizer'])
            self.scheduler.load_state_dict(load_dict['scheduler'])
            epoch = load_dict['epoch']
            utils.color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)
        else:
            epoch = load_dict['epoch']
            utils.color_print('Load checkpoint from %s.' % ckpt_path, 3)

        return epoch

    def save(self, which_epoch):
        save_filename = f'{which_epoch}_{opt.model}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = OrderedDict()
        save_dict['direct_feature'] = self.direct_feature.state_dict()
        save_dict['meta_embedding'] = self.meta_embedding.state_dict()
        save_dict['centroids'] = self.mem.state_dict()

        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['scheduler'] = self.scheduler.state_dict()
        save_dict['epoch'] = which_epoch
        torch.save(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)

        # self.save_network(self.discriminitor, 'D', which_epoch)
