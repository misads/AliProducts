from network import get_model
import misc_utils as utils
import argparse
import torch
import os
from dataloader import test_dataloader as dataloader
from torch.autograd import Variable
import csv
import ipdb


def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default='cache',
                        help='folder name to save the outputs')
    parser.add_argument('--gpu_ids', '--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    parser.add_argument('--model', type=str, default='default', help='which model to use')

    # to load correctly
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'radam', 'lookahead', 'ranger'], default='ranger')
    parser.add_argument('--scheduler', choices=['cos', 'step', 'exp', 'cyclic', 'lambda', 'None'], default='cos')
    parser.add_argument('--epochs', '--max_epoch', type=int, default=10, help='epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')

    # batch size
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='input batch size')
    # data argumentation
    parser.add_argument('--norm-input', action='store_true')

    # scale
    parser.add_argument('--scale', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop', type=int, default=None, help='then crop to this size')

    parser.add_argument('--load', type=str, default=None, help='load checkpoint')
    parser.add_argument('--which-epoch', type=int, default=None, help='which epoch to resume')

    return parser.parse_args()


opt = parse_args()

if not opt.load:
    print('Usage: submit.py --model your_model --load LOAD --gpu 0')
    utils.color_print('Exception: submit.py: the following arguments are required: --load', 1)
    exit(1)

opt.device = 'cuda:' + opt.gpu_ids if torch.cuda.is_available() and opt.gpu_ids != '-1' else 'cpu'

Model = get_model(opt.model)
model = Model(opt)
model = model.to(device=opt.device)

opt.which_epoch = model.load(opt.load)

model.eval()

with open('submission.csv', 'w') as f:  # 如果在windows下打开csv出现空行的情况,加一个newline=''参数
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'predicted'])  # 写一行
    for i, data in enumerate(dataloader):

        input, path = data['input'], data['path']

        if 'label' in data:
            label = data['label']

        utils.progress_bar(i, len(dataloader), 'Testing... ')
        # ct_num += 1
        with torch.no_grad():
            img_var = Variable(input, requires_grad=False).to(device=opt.device)
            predicted = model(img_var)
            _, predicted = torch.max(predicted, 1)
            # ct_num += label.size(0)
            # correct += (predicted == label_var).sum().item()
        # ipdb.set_trace()
        for idx in range(len(path)):  # batch
            filename = os.path.basename(path[idx])
            line = [filename, predicted[idx].item()]
            csv_writer.writerow(line)  # 写一行




