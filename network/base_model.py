import os
from abc import abstractmethod

import torch
import sys

import misc_utils as utils

from mscv import ExponentialMovingAverage
from options import opt

from collections import OrderedDict

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def forward(self, x):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def load(self, ckpt_path):
        load_dict = torch.load(ckpt_path, map_location=opt.device)
        self.classifier.load_state_dict(load_dict['classifier'])
        if opt.resume:
            self.optimizer.load_state_dict(load_dict['optimizer'])
            self.scheduler.load_state_dict(load_dict['scheduler'])
            self.scheduler.step()
            
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
        save_dict['classifier'] = self.classifier.state_dict()

        save_dict['optimizer'] = self.optimizer.state_dict()
        save_dict['scheduler'] = self.scheduler.state_dict()
        save_dict['epoch'] = which_epoch
        torch.save(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)

