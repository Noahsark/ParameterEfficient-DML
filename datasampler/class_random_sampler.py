import numpy as np
import torch.distributed as dist
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import math



"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, image_list, **kwargs):

        #####
        self.image_dict         = image_dict
        self.image_list         = image_list

        #####
        self.internal_split = opt.internal_split
        self.use_meta_split = self.internal_split!=1
        self.classes        = list(self.image_dict.keys())
        self.tv_split       = int(len(self.classes)*self.internal_split)
        self.train_classes  = self.classes[:self.tv_split]
        self.val_classes    = self.classes[self.tv_split:]

        ####
        self.batch_size         = opt.batch_size
        self.samples_per_class  = opt.samples_per_class
        self.sampler_length     = len(image_list)//opt.batch_size  # Number of batches per epoch
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'
        self.name             = 'class_random_sampler'
        self.requires_storage = False


    def _smart_sample(self, S, N):
        M = len(S)
        if N <= M:
            return random.sample(S, N)
        else:
            multiple_samples = N // M
            remaining_samples = N % M
            result = []
            for _ in range(multiple_samples):
                result.extend(S)
            result.extend(random.sample(S, remaining_samples))
            random.shuffle(result)
            return result
    
    def _sample(self, subset, draws, classes):
        # Randomly draw classes
        class_keys = self._smart_sample(classes, draws)
        for cls in class_keys:
            sample = self._smart_sample(self.image_dict[cls], self.samples_per_class)
            subset.extend([s[-1] for s in sample])
        return subset
        
    def __iter__(self):
        
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            if self.use_meta_split:
                train_draws = int((self.batch_size//self.samples_per_class)*self.internal_split)
                val_draws   = self.batch_size//self.samples_per_class-train_draws
            else:
                train_draws = self.batch_size//self.samples_per_class
                val_draws   = 0
            
            # Randomly select classes
            if train_draws > 0:
                subset = self._sample(subset, train_draws, self.train_classes)
            if val_draws > 0:
                subset = self._sample(subset, val_draws, self.val_classes)
            
            
            yield subset

    def __len__(self):
        return self.sampler_length



class DistributedBatchSampler(Sampler):
    def __init__(self, opt, image_dict, image_list, rank=None, **kwargs):
        super().__init__(opt, image_dict, image_list, **kwargs)

        self.num_replicas = self.opt.world_size
        
        self.rank = rank

        self.epoch = 0

    def __iter__(self):
        
        batches = list(super().__iter__())

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
        return int(math.ceil(len(super()) / float(self.num_replicas)))
    
    def set_epoch(self, epoch):
        self.epoch = epoch