# python 3.5, pytorch 1.14

import os, sys
import ipdb
from collections import defaultdict

import dataloader as dl
from options import opt
from mscv.summary import write_loss

import torch
import torchvision
import numpy as np
import subprocess
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from collections.abc import Iterable
from PIL import Image
from utils import *

import misc_utils as utils


def evaluate(model, dataloader, epochs, writer, logger, data_name='val'):

    save_root = os.path.join(opt.result_dir, opt.tag, str(epochs), data_name)

    utils.try_make_dir(save_root)

    correct = 0
    ct_num = 0
    counts = defaultdict(int)
    corrects = defaultdict(int)

    # print('Start testing ' + tag + '...')
    for i, data in enumerate(dataloader):
        if data_name == 'val':
            input, label, path = data['input'], data['label'], data['path']
            utils.progress_bar(i, len(dataloader), 'Eva... ')
            # ct_num += 1
            with torch.no_grad():
                img_var = Variable(input, requires_grad=False).to(device=opt.device)
                label_var = Variable(label, requires_grad=False).to(device=opt.device)
                predicted = model(img_var)
                _, predicted = torch.max(predicted, 1)
                ct_num += label.size(0)
                correct += (predicted == label_var).sum().item()

            for idx, l in enumerate(label):
                l = l.item()
                counts[l] += 1
                p = predicted[idx].item()
                if p == l:
                    corrects[l] += 1

        elif data_name == 'test':
            pass

        else:
            raise Exception('Unknown dataset name: %s.' % data_name)

    if data_name == 'val':
        # write_loss(writer, 'val/%s' % data_name, 'psnr', ave_psnr / float(ct_num), epochs)
        logger.info('Eva(%s) epoch %d ,' % (data_name, epochs) + 'Top-1 Acc: ' + str(correct / float(ct_num)) + '.')

        return str(round(correct / float(ct_num), 3))
    else:
        return ''


if __name__ == '__main__':
    from options import opt
    from network import get_model
    import misc_utils as utils
    from mscv.summary import create_summary_writer

    if not opt.load:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    Model = get_model(opt.model)
    model = Model(opt)
    model = model.to(device=opt.device)

    load_epoch = model.load(opt.load)
    if load_epoch is not None:
        opt.which_epoch = load_epoch

    model.eval()

    log_root = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))
    utils.try_make_dir(log_root)

    writer = create_summary_writer(log_root)

    logger = init_log(training=False)
    evaluate(model, dl.val_dataloader, opt.which_epoch, writer, logger, 'val')

