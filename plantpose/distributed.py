import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler


class RankLocalGraphLoss(nn.Module):
    """Compute the criterion per local batch; DDP averages parameter gradients."""
    def __init__(self, model, criterion):
        super().__init__()
        self.model, self.criterion = model, criterion

    def forward(self, images, target):
        h, outputs = self.model(images)
        return self.criterion(h, outputs, target)


class GlobalBatchSampler(Sampler):
    """One seeded global ordering, no padding or duplicate samples across ranks."""
    def __init__(self, size, batch_size, rank, world, seed, shuffle):
        if batch_size % world or batch_size < world:
            raise ValueError('global batch must be divisible by world size')
        self.size, self.batch_size = size, batch_size
        self.rank, self.world, self.seed, self.shuffle = rank, world, seed, shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def global_indices(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        values = torch.randperm(self.size, generator=generator).tolist() if self.shuffle else list(range(self.size))
        return values[:self.size // self.batch_size * self.batch_size]

    def __iter__(self):
        return iter(self.global_indices()[self.rank::self.world])

    def __len__(self):
        return self.size // self.batch_size * self.batch_size // self.world


def rank_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
