import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pdb
# load vision transformers


class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        
        self.pars = opt
        self.name = opt.pretrained_model
        
        self.model = timm.create_model(opt.pretrained_model, pretrained=True)
        
        # load pretrained configuration
        self.pretrained_cfg = self.model.pretrained_cfg
        # trainable head
        self.model.head = nn.Linear(self.model.embed_dim, opt.embed_dim)
        self.model.layer_norm = nn.LayerNorm(self.model.embed_dim)
        
        # setting freezing
        if self.pars.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        if self.pars.vit_head:
            for param in self.model.head.parameters():
                param.requires_grad = True
            for param in self.model.layer_norm.parameters():
                param.requires_grad = True
                
        self.model.cuda().train()
    
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    # updated for deit-3
    def _pos_embed(self, x):
        if self.model.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.model.pos_embed
            if hasattr(self.model, 'dist_token') and  self.model.dist_token is not None:
                x = torch.cat((self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            if hasattr(self.model, 'cls_token') and self.model.cls_token is not None:
                x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if hasattr(self.model, 'dist_token') and  self.model.dist_token is not None:
                x = torch.cat((self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            if hasattr(self.model, 'cls_token') and  self.model.cls_token is not None:
                x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.model.pos_embed
        return self.model.pos_drop(x)
    
    def forward(self, x, warmup=False, freeze=False, **kwargs):
        
        x = self.model.patch_embed(x)
        x = self._pos_embed(x) 
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
            
        if self.pars.vit_global_pooling == 'avg':
            x = x[:, 1:].mean(dim=1)  # avg pooling
        else:
            x = x[:, 0]  # use cls_token
        x = self.model.layer_norm(x)
        
        x = self.model.head(x)
        x = F.normalize(x, dim=-1)
        
        return {
            'embeds': x
        }
    
