import pdb

import numpy as np
import pickle
import torch
import os

from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from collections import OrderedDict
from mscv import ExponentialMovingAverage, print_network
from optimizer import get_optimizer
from scheduler import get_scheduler
from options import opt
import misc_utils as utils

from .resnest_wrapper import Classifier
from loss import label_smooth_loss
from mscv.image import tensor2im

#  criterionCE = nn.CrossEntropyLoss()

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.classifier = Classifier(opt.model)  #.cuda(device=opt.device)
        #####################
        #    Init weights
        #####################
        # self.classifier.apply(weights_init)

        print_network(self.classifier)

        self.optimizer = get_optimizer(opt, self.classifier)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

        self.criterionCE = nn.CrossEntropyLoss()

    def update(self, input, label):

        predicted = self.classifier(input)
        # smooth_loss = label_smooth_loss(predicted, label)
        ce_loss = criterionCE(predicted, label)

        loss = ce_loss

        self.avg_meters.update({'CE loss': ce_loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'predicted': predicted}

    def forward(self, x):
        return self.classifier(x)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)