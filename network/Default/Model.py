import pdb

import numpy as np
import torch
import os


from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network

from options import opt
import misc_utils as utils
from scheduler import get_scheduler
from loss import criterionCE

from .resnet import Classifier


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.classifier = Classifier()
        #####################
        #    Init weights
        #####################
        # self.classifier.apply(weights_init)

        print_network(self.classifier)

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=opt.lr, betas=(0.95, 0.999))
        self.scheduler = get_scheduler(opt, self.optimizer)

        # load networks
        # if opt.load:
        #     pretrained_path = opt.load
        #     self.load_network(self.classifier, 'G', opt.which_epoch, pretrained_path)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update(self, input, label):

        predicted = self.classifier(input)
        loss = criterionCE(predicted, label)

        self.avg_meters.update({'Cross Entropy': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'predicted': predicted}

    def forward(self, x):
        return self.classifier(x)

    def load(self, ckpt_path):
        load_dict = torch.load(ckpt_path, map_location=opt.device)
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
        save_dict['classifier'] = self.classifier.state_dict()
        # save_dict['discriminitor'] = self.discriminitor.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['scheduler'] = self.scheduler.state_dict()
        save_dict['epoch'] = which_epoch
        torch.save(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)

        # self.save_network(self.discriminitor, 'D', which_epoch)
