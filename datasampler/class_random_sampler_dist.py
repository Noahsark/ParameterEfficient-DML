import numpy as np
import torch.distributed as dist
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import math

from datasampler.class_random_sampler import Sampler


"""======================================================"""
REQUIRES_STORAGE = False


class Sampler(Sampler):
    def __init__(self, opt, image_dict, image_list, rank=0, **kwargs):
        super().__init__(opt, image_dict, image_list, **kwargs)

        self.num_replicas = opt.world_size
        
        self.rank = rank

        self.epoch = 0

    def __iter__(self):
        
        batches = list(super(Sampler, self).__iter__())
        
        # Determine how many samples to add to make it evenly divisible
        length = len(batches)
        length_per_replica = int(math.ceil(length / self.num_replicas))
        total_length = length_per_replica * self.num_replicas

        # add extra samples by duplicating some batches to make it evenly divisible
        batches += batches[:(total_length - length)]
        assert len(batches) == total_length

        # subsample
        batches = batches[self.rank:total_length:self.num_replicas]
        assert len(batches) == length_per_replica

        return iter(batches)
    
    def __len__(self):
        batches = list(super(Sampler, self).__iter__())
        return int(math.ceil(len(batches) / float(self.num_replicas)))
    
    def set_epoch(self, epoch):
        self.epoch = epoch