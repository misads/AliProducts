# encoding=utf-8
import numpy as np
import torch
import torch.nn as nn


class OHEM(object):
    def __init__(self, hard_sample_per_batch=2):
        self.hard_sample_pool = []
        self.criterionCE = nn.CrossEntropyLoss()
        self.hard_sample_per_batch = hard_sample_per_batch

    def collect_batch(self, minibatch, input, label):
        batch_size = minibatch.size(0)
        losses = []
        for i in range(batch_size):
            losses.append((self.criterionCE(minibatch[i:i + 1], label[i:i + 1]).item(), i))

        losses.sort(reverse=True)
        hard_losses = losses[:min(batch_size, self.hard_sample_per_batch)]
        hard_ids = set([idx for _, idx in hard_losses])
        for idx in hard_ids:
            hi = input[idx:idx+1]
            hl = label[idx:idx+1]
            if not self.hard_sample_pool:
                self.hard_sample_pool = (hi, hl)
            else:
                pi, pl = self.hard_sample_pool
                self.hard_sample_pool = (torch.cat([pi, hi], dim=0), torch.cat([pl, hl], dim=0))

            # self.hard_sample_pool.append((minibatch, label))

        # print(hard_losses)
        # print(hard_ids)
        # print(self.hard_sample_pool)

    def get_pool_size(self):
        return self.hard_sample_pool[0].size(0)

    def get_hard_batch(self):
        """
            >>> if ohem.get_pool_size() >= opt.batch_size:
            >>>     ohem.get_hard_batch()
        """
        print('get_hard_batch')
        hard_batch = self.hard_sample_pool
        self.hard_sample_pool = []
        return hard_batch


if __name__ == '__main__':
    ohem = OHEM()
    for i in range(12):
        input = torch.rand([24, 3, 256, 246])
        output = torch.rand([24, 10])
        label = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3]).long()
        ohem.collect_batch(output, input, label)

        if ohem.get_pool_size() >= 24:
            input, label = ohem.get_hard_batch()
            print(input.shape)
            print(label.shape)