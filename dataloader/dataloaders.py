# encoding=utf-8
from dataloader.products import *
from torch.utils.data import DataLoader
from options import opt
import pdb

###################

TEST_DATASET_HAS_OPEN = True  # 有没有开放测试集

###################

train_list = "./datasets/train.txt"
val_list = "./datasets/val.txt"

max_size = 128 if opt.debug else None  # debug模式时dataset的最大大小

train_dataset = TrainValDataset(train_list, scale=opt.scale, aug=opt.aug, max_size=max_size, norm=opt.norm_input)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)

val_dataset = TrainValDataset(val_list, scale=opt.scale, aug=False, max_size=max_size, norm=opt.norm_input)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)


if TEST_DATASET_HAS_OPEN:
    test_list = "./datasets/test.txt"  # 测试集

    test_dataset = TestDataset(test_list, scale=opt.scale, max_size=max_size, norm=opt.norm_input)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

else:
    test_dataloader = None
