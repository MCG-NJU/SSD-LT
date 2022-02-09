# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
from PIL import ImageFilter
import random
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, k_transform):
        self.base_transform = base_transform
        self.k_transform = k_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.k_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
##################################

class RandomCycleIter:
    
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list) 
        return self.data_list[self.i]
    
def class_aware_sample_generator (cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0
    j = 0
    while i < n:
        
#         yield next(data_iter_list[next(cls_iter)])
        
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1


class DistributedClassAwareSampler(Sampler):

    def __init__(self, dataset, num_samples_cls=1, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_samples_cls = num_samples_cls

        self.num_classes = len(np.unique(dataset.targets))

        print('==> Using Class Aware Sampling')
        print('num_samples_cls: {}'.format(num_samples_cls))

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        self.epoch += 1
        cls_data_list = [list() for _ in range(self.num_classes)]
        for idx in indices:
            cls_data_list[self.dataset.targets[idx]].append(idx)
        data_iter_list = []
        for x in cls_data_list:
            if len(x) > 0:
                data_iter_list.append(RandomCycleIter(x))
        class_iter = RandomCycleIter(range(len(data_iter_list)))
        return class_aware_sample_generator(class_iter, data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch