import torch
import torch.nn.functional as F
import numpy as np
import math

DIS_ALPHA = 2.0
DIS_BETA = 2.0


ALLOWED_MINING_OPS = None
REQUIRES_BATCHMINER = False
#REQUIRES_OPTIM = True
REQUIRES_OPTIM = False

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Module ProxyAnchor/NCA loss for ViT
        """
        super(Criterion, self).__init__()

        self.opt = opt
        self.num_proxies = opt.n_classes
        self.embed_dim = opt.embed_dim
        self.mode = opt.loss_oproxy_mode # ProxyAnchor
        self.name = 'proxynca' if self.mode == 'nca' else 'proxyanchor'
        self.proxy_dim = 1 if self.mode == 'nca' else 0 # anchor: 0, nca: 1
        self.class_idxs = torch.arange(self.num_proxies)
        self.new_epoch = True
        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM
    
    def _masked_logsumexp(self, sims, dim=0, mask=None):
        # select features by mask for proxy DML
        if mask is not None:
            sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
        dims = list(sims.shape)
        dims[dim] = 1 # select between nca and anchor loss
        zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
        sims = torch.cat([sims, zeros], dim=dim)
        logsumexp_sims = torch.logsumexp(sims, dim=dim, keepdim=True)
        if mask is not None:
            logsumexp_sims = logsumexp_sims.masked_fill(
                ~torch.any(mask, dim=dim, keepdim=True), 0)
        return logsumexp_sims


    def forward(self, batch, proxies, labels, **kwargs):    
        # assume input proxies: n_class x dim
        
        proxies, batch_proxies = proxies
        batch_proxies = F.normalize(batch_proxies, dim=1) # BS x dim
        if not proxies is None:
            proxies = F.normalize(proxies, dim=1) # n_class x dim
        
        batch = F.normalize(batch, dim=1) # BS x dim
        
        self.labels = labels.unsqueeze(1) # BS x 1
        self.f_labels = self.labels.view(-1) # BS x 1  flat labels
        self.u_labels = torch.unique_consecutive(self.f_labels) # unique labels, BS//k x 1
        
        if self.opt.world_size > 1:
            self.class_idxs = self.class_idxs.to(batch.device)
        
        # in_batch: only optimize proxies inside batch; \
        # both_all: optimize all in batch, positive and negative in all proxies.'
        if self.opt.proxy_data_range == 'in_batch':
            pos_sims = batch.mm(batch_proxies.T) # BS x BS//k
            neg_sims = pos_sims
            self.same_labels = (self.labels.T == self.u_labels.view(-1, 1)).to(
                batch.device).T # BS x BS//k  
            self.diff_labels = (self.u_labels.unsqueeze(1) != self.labels.T).to(
                batch.device).T  # BS x BS//k  
        elif self.opt.proxy_data_range == 'both_all':
            pos_sims = batch.mm(proxies.T) # BS x n_class
            neg_sims = batch.mm(proxies.T) # BS x n_class
            self.same_labels = (self.class_idxs.unsqueeze(1) == self.labels.T).to(
                batch.device).T # BS x NClass  positive mask: one-hot labels
            self.diff_labels = (self.class_idxs.unsqueeze(1) != self.labels.T).to(
                batch.device).T # BS x NClass  negative mask: invert of one-hot labels
        else:
            raise NotImplementedError
        
        w_pos_sims = -self.opt.loss_oproxy_pos_alpha * (pos_sims - self.opt.loss_oproxy_pos_delta)
        w_neg_sims = self.opt.loss_oproxy_neg_alpha * (neg_sims - self.opt.loss_oproxy_neg_delta)
        
        pos_s = self._masked_logsumexp(w_pos_sims,
                                      mask=self.same_labels.type(torch.bool),
                                      dim=self.proxy_dim)
        neg_s = self._masked_logsumexp(w_neg_sims,
                                      mask=self.diff_labels.type(torch.bool),
                                      dim=self.proxy_dim)
        return pos_s.mean() + neg_s.mean()

          
 


