# encoding=utf-8
from dataloader.products import *
from dataloader.transforms import get_transform
from torch.utils.data import DataLoader
from options import opt
import pdb

###################

TEST_DATASET_HAS_OPEN = False  # 有没有开放测试集

###################

train_list = "./datasets/train.txt"
val_list = "./datasets/val.txt"

max_size = 128 if opt.debug else None  # debug模式时dataset的最大大小

# transforms
transform = get_transform(opt.transform)
train_transform = transform.train_transform
val_transform = transform.val_transform

# datasets和dataloaders
train_dataset = TrainValDataset(train_list, transforms=train_transform, max_size=max_size)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)

val_dataset = TrainValDataset(val_list, transforms=val_transform, max_size=max_size)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers//2)

if TEST_DATASET_HAS_OPEN:
    test_list = "./datasets/test.txt"  # 测试集

    test_dataset = TestDataset(test_list, transforms=val_transform, max_size=max_size)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

else:
    test_dataloader = None
