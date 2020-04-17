# encoding=utf-8
from dataloader.products import *
from torch.utils.data import DataLoader
from options import opt

###################

TEST_DATASET_HAS_OPEN = False  # 有没有开放测试集

###################

train_list = "./datasets/train.txt"
val_list = "./datasets/val.txt"

max_size = 10 if opt.debug else None

train_dataset = TrainValDataset(train_list, scale=opt.scale, aug=False, max_size=max_size)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

val_dataset = TrainValDataset(val_list, scale=opt.scale, aug=False, max_size=None)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)


if TEST_DATASET_HAS_OPEN:
    test_list = "./datasets/test.txt"  # 还没有

    test_dataset = TestDataset(test_list, scale=opt.scale, max_size=max_size)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

else:
    test_dataloader = None
