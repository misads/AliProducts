# encoding = utf-8
"""
    基于Torch-Template的一个baseline。

    如何添加新的模型：

    ① 复制network目录下的Default文件夹，改成另外一个名字(比如MyNet)。

    ② 在network/__init__.py中import你的Model并且在models = {}中添加它。
        from MyNet.Model import Model as MyNet
        models = {
            'default': Default,
            'MyNet': MyNet,
        }

    ③ 尝试 python train.py --model MyNet 看能否成功运行


    File Structure:
    AliProducts
        ├── train.py                :Train and evaluation loop, errors and outputs visualization (Powered by TensorBoard)
        ├── eval.py                 :Evaluation and test (with visualization)
        ├── test.py                 :Test
        │
        ├── clear.py                :Clear cache, be CAREFUL to use it
        │
        ├── run_log.txt             :Record your command logs (except --tag cache)
        │
        ├── network
        │     ├── __init__.py       :Declare all models here so that `--model` can work properly
        │     ├── Default
        │     │      ├── Model.py   :Define default model, losses and parameter updating procedure
        │     │      └── res101.py
        │     └── MyNet
        │            ├── Model.py   :Define your model, losses and parameter updating procedure
        │            └── mynet.py
        │
        ├── options
        │     └── options.py        :Define options
        │
        │
        ├── dataloader/             :Define Dataloaders
        │     ├── __init__.py       :imports all dataloaders in dataloaders.py
        │     ├── dataloaders.py    :Define all dataloaders here
        │     └── products.py       :Custom Dataset
        │
        ├── checkpoints/<tag>       :Trained checkpoints
        ├── logs/<tag>              :Logs and TensorBoard event files
        └── results/<tag>           :Test results


    Datasets:

        datasets
           ├── train
           │     ├── 00001
           │     ├── 00002
           │     └── .....
           ├──  val
           │     ├── 00001
           │     ├── 00002
           │     └── .....
           ├── train.json
           ├── val.json
           └── product_tree.json

    Usage:

    #### Train

        python train.py --tag train_1 --epochs 500 -b 8 --gpu 1

    #### Resume or Fine Tune

        python train.py --load checkpoints/train_1 --which-epoch 500

    #### Evaluation

        python eval.py --tag eval_1 --model MyNet --load checkpoints/MyNet --which-epoch 499

    #### Test

        python test.py --tag test_1

    #### Clear

        python clear.py [--tag cache]  # (DO NOT use this command unless you know what you are doing.)


    License: MIT

"""

import os
import pdb
import time
import numpy as np
from collections.abc import Iterable

import torch
from torch import optim
from torch.autograd import Variable

import dataloader as dl
from options import opt

from network import get_model
from eval import evaluate

from utils import *
# from torch_template.utils.torch_utils import create_summary_writer, write_meters_loss, LR_Scheduler
from utils.torch_utils import create_summary_writer, write_meters_loss
from utils.send_sms import send_notification

import misc_utils as utils

######################
#       Paths
######################
save_root = os.path.join(opt.checkpoint_dir, opt.tag)
log_root = os.path.join(opt.log_dir, opt.tag)

utils.try_make_dir(save_root)
utils.try_make_dir(log_root)

train_dataloader = dl.train_dataloader
val_dataloader = dl.val_dataloader
# init log
logger = init_log(training=True)

######################
#     Init model
######################
Model = get_model(opt.model)
model = Model(opt)

# if len(opt.gpu_ids):
#     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model = model.to(device=opt.device)

if opt.which_epoch and not opt.reset:
    start_epoch = opt.which_epoch + 1
else:
    start_epoch = 1

model.train()

# Start training
print('Start training...')
start_step = start_epoch * len(train_dataloader)
global_step = start_step
total_steps = opt.epochs * len(train_dataloader)
start = time.time()

#####################
#   定义scheduler
#####################

optimizer = model.optimizer
if opt.scheduler == 'cos':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 0.1)
elif opt.scheduler == 'step':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.epochs // 3, gamma=0.1)
elif opt.scheduler == 'exp':
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
elif opt.scheduler == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=opt.lr, max_lr=0.1 * opt.lr, step_size_up=opt.epochs // 10,
                                                  step_size_down=opt.epochs // 10)
elif opt.scheduler == 'lambda':
    def lambda_decay(epoch) -> float:
        warm_epoch = int(0.1 * opt.epochs)
        if epoch <= warm_epoch:
            return epoch * 0.1 / warm_epoch
        else:
            return 0.1 * 0.99 ** (epoch - warm_epoch)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)

######################
#    Summary_writer
######################
writer = create_summary_writer(log_root)

start_time = time.time()
######################
#     Train loop
######################
try:
    eval_result = ''

    for epoch in range(start_epoch, opt.epochs + 1):
        for iteration, data in enumerate(train_dataloader):
            global_step += 1
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            img, label = data['input'], data['label']  # ['label'], data['image']  #

            img_var = Variable(img, requires_grad=False).to(device=opt.device)
            label_var = Variable(label, requires_grad=False).to(device=opt.device)

            ##############################
            #       Update parameters
            ##############################
            update = model.update(img_var, label_var)
            predicted = update.get('predicted')

            pre_msg = 'Epoch:%d' % epoch

            msg = '(loss) %s ETA: %s' % (str(model.avg_meters), utils.format_time(remaining))
            utils.progress_bar(iteration, len(train_dataloader), pre_msg, msg)
            # print(pre_msg, msg)

            if global_step % 1000 == 999:
                write_meters_loss(writer, 'train', model.avg_meters, global_step)

        logger.info(f'Train epoch: {epoch}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % opt.save_freq == opt.save_freq - 1 or epoch == opt.epochs - 1:  # 每隔10次save checkpoint
            model.save(epoch)

        ####################
        #     Validation
        ####################
        if epoch % opt.eval_freq == (opt.eval_freq - 1):

            model.eval()
            eval_result = evaluate(model, val_dataloader, epoch, writer, logger)
            model.train()

        if opt.scheduler is not None:
            scheduler.step()

    # send_notification([opt.tag[:12], '', '', eval_result])

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Accuracy:' + eval_result + '\n')

except Exception as e:

    # if not opt.debug:  # debug模式不会发短信 12是短信模板字数限制
    #     send_notification([opt.tag[:12], str(e)[:12]], template='error')

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Error: ' + str(e)[:120] + '\n')

    # print(e)
    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的trace back信息
