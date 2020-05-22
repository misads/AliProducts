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
from torch_template.utils.torch_utils import ExponentialMovingAverage, print_network
from optimizer import get_optimizer
from scheduler import get_scheduler
from options import opt

from .meta_embedding import DirectFeature, Classifier
import misc_utils as utils

#  criterionCE = nn.CrossEntropyLoss()


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.direct_feature = DirectFeature(opt.model)
        feature_nums = self.direct_feature.get_feature_num()
        self.classifier = Classifier(feature_nums)  #.cuda(device=opt.device)

        print_network(self.direct_feature)
        print_network(self.classifier)

        self.optimizer = get_optimizer(opt, self.classifier)
        self.scheduler = get_scheduler(opt, self.optimizer)

        # # load networks
        # if opt.load:
        #     pretrained_path = opt.load
        #     save_filename = '%s_net_%s.pt' % (opt.which_epoch, 'G')
        #     save_path = os.path.join(pretrained_path, save_filename)
        #     state_dict = torch.load(save_path, map_location=opt.device)
        #     fc_state = OrderedDict()
        #     fc_state['network.fc.weight'] = state_dict['network.fc.weight']
        #     fc_state['network.fc.bias'] = state_dict['network.fc.bias']
        #     state_dict.pop('network.fc.weight')
        #     state_dict.pop('network.fc.bias')
        #     model_dict = self.classifier.state_dict()
        #     for k, v in state_dict:
        #         if v.size() == model_dict[k].size():
        #             model_dict[k] = v
        #
        #     fc_dict = self.classifier.clf.fc_hallucinator.state_dict()
        #     for k, v in fc_state:
        #         if v.size() == model_dict[k].size():
        #             fc_dict[k] = v
        #
        #     self.load_network(self.classifier, 'G', opt.which_epoch, pretrained_path)
        #
        #     # if self.training:
        #     #     self.load_network(self.discriminitor, 'D', opt.which_epoch, pretrained_path)

        # self.init_criterions()
        # if self.memory['init_centroids']:
        #     self.criterions['FeatureLoss'].centroids.data = \
        #         self.centroids_cal(self.data['train_plain'])

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

        # different weight for different classes
        self.criterionCE = nn.CrossEntropyLoss()

    def update(self, input, label):

        predicted = self.forward(input)
        loss = self.criterionCE(predicted, label)

        self.avg_meters.update({'Cross Entropy': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'predicted': predicted}

    def forward(self, x):
        direct_feature = self.direct_feature(x)
        y = direct_feature(direct_feature)
        return y

    def load(self, ckpt_path):
        load_dict = torch.load(ckpt_path, map_location=opt.device)
        if 'direct_feature' not in load_dict:  # 旧的checkpoint
            direct_feature = load_dict['classifier']
            classifier_dict = OrderedDict()
            classifier_dict['fc.weight'] = direct_feature.pop('network.fc.weight')
            classifier_dict['fc.bias'] = direct_feature.pop('network.fc.bias')

        else:  # 新的checkpoint
            self.direct_feature.load_state_dict(load_dict['direct_feature'])
            self.classifier.load_state_dict(load_dict['classifier'])

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
        # self.save_network(self.classifier, 'G', which_epoch)
        save_filename = f'{which_epoch}_{opt.model}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = OrderedDict()
        save_dict['direct_feature'] = self.direct_feature.state_dict()
        save_dict['classifier'] = self.classifier.state_dict()
        # save_dict['discriminitor'] = self.discriminitor.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['scheduler'] = self.scheduler.state_dict()
        save_dict['epoch'] = which_epoch
        torch.save(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)

        # self.save_network(self.discriminitor, 'D', which_epoch)
