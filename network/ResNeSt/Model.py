import pdb

import numpy as np
import pickle
import torch
import os

from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from network.base_model import BaseModel
from torch_template.utils.torch_utils import ExponentialMovingAverage, print_network
from optimizer import RAdam, Ranger, Lookahead
from options import opt

from .resnest_wrapper import Classifier


#  criterionCE = nn.CrossEntropyLoss()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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

        if opt.optimizer == 'adam':
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=opt.lr, betas=(0.95, 0.999))
        elif opt.optimizer == 'sgd':  # 从头训练 lr=0.1 fine_tune lr=0.01
            self.optimizer = optim.SGD(self.classifier.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
        elif opt.optimizer == 'radam':
            self.optimizer = RAdam(self.classifier.parameters(), lr=opt.lr, betas=(0.95, 0.999))
        elif opt.optimizer == 'lookahead':
            self.optimizer = Lookahead(self.classifier.parameters())
        elif opt.optimizer == 'ranger':
            self.optimizer = Ranger(self.classifier.parameters(), lr=opt.lr)
        else:
            raise NotImplementedError

        # load networks
        if opt.load:
            pretrained_path = opt.load
            self.load_network(self.classifier, 'G', opt.which_epoch, pretrained_path)
            # if self.training:
            #     self.load_network(self.discriminitor, 'D', opt.which_epoch, pretrained_path)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

        with open('datasets/class_weight.pkl', 'rb') as f:
            class_weight = pickle.load(f, encoding='bytes')
            class_weight = np.array(class_weight, dtype=np.float32)
            class_weight = torch.from_numpy(class_weight).to(opt.device)
            self.criterionCE = nn.CrossEntropyLoss(weight=class_weight)

    def update(self, input, label):

        predicted = self.classifier(input)
        loss = self.criterionCE(predicted, label)

        self.avg_meters.update({'Cross Entropy': loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'predicted': predicted}

    def forward(self, x):
        return self.classifier(x)

    def save(self, which_epoch):
        self.save_network(self.classifier, 'G', which_epoch)
        # self.save_network(self.discriminitor, 'D', which_epoch)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        # for param_group in self.d_optimizer.param_groups:
        #     param_group['lr'] = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
