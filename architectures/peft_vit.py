import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint

import timm
from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import Attention, LayerScale
import math
import numpy as np

class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        
        self.pars = opt
        self.name = opt.pretrained_model
        
        self.model = timm.create_model(opt.pretrained_model, pretrained=True)
        self.embed_dim = self.model.embed_dim 
        self.block_depth = len(self.model.blocks)
        self.input_size = self.model.default_cfg['input_size'] # 224
        self.p_channels = self._cal_channels()
        self.n_classes = opt.n_classes # 
        self.distill = ('distilled' in opt.pretrained_model)
        # n_classes_batch is the number of classes in a batch
        self.n_classes_batch = opt.batch_size // opt.samples_per_class
        
        model_params = []
        proxy_net_params = []
        self.optimizer = None
        
        # pretrained configuration
        self.pretrained_cfg = self.model.pretrained_cfg
        
        # setting freezing
        if not self.pars.full_fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            model_params += list(self.model.parameters())
        
        ###################### All trainable setting must be below this line ######################
        
        # setting head
        if self.pars.vit_head:
            if self.pars.vit_global_pooling == 'combined':
                self.head, self.fc_norm, self.dist_head, self.dist_norm = self._setup_head(in_dim=self.embed_dim*2)
            else:
                self.head, self.fc_norm, self.dist_head, self.dist_norm = self._setup_head()
            model_params += list(self.head.parameters())
            model_params += list(self.fc_norm.parameters())
            if self.distill:
                model_params += list(self.dist_head.parameters())
                model_params += list(self.dist_norm.parameters())
            
        self.dropout = nn.Dropout(self.pars.feature_dropout)
        
        # proxy bias and container
        self.proxies_bias = nn.Parameter(nn.init.xavier_uniform_(
            torch.zeros(self.n_classes, self.pars.embed_dim)
           ) / 8.0)
        
        self.proxies_container = nn.init.xavier_uniform_(
                        torch.zeros(self.n_classes, 
                        self.pars.embed_dim).to(self.pars.device) / 8.0)
        
        self.proxies_container = F.normalize(self.proxies_container, dim=1)
        
        if "gru" in self.pars.semantic_mix_type and self.pars.vit_proxy_net:
            self.rnn_update = Simple_GRUCell(self.pars.embed_dim, self.pars.embed_dim, 
                                             drop=self.pars.rnn_dropout, type=self.pars.semantic_mix_type)
            proxy_net_params += list(self.rnn_update.parameters())
            
            
        if self.pars.adapter:
            model_params = self._setup_adapter(model_params)
        
        if self.pars.vit_proxy_net:
            if self.pars.vit_global_pooling == 'combined':
                self.p_head, self.p_fc_norm, self.p_dist_head, self.p_dist_norm = self._setup_head(in_dim=self.embed_dim*2)
            else:
                self.p_head, self.p_fc_norm, self.p_dist_head, self.p_dist_norm = self._setup_head()
            
            proxy_net_params += list(self.p_head.parameters())
            proxy_net_params += list(self.p_fc_norm.parameters())
            if self.distill:
                proxy_net_params += list(self.p_dist_head.parameters())
                proxy_net_params += list(self.p_dist_norm.parameters())
        
        # setting bitfit
        if self.pars.vit_bitfit:
            for n, param in self.model.named_parameters():
                if self._check_params(n, ['bias'], all_match=False):
                    param.requires_grad = True
                    model_params += [param]
        
        memory_bank = []
        # setting prompt
        if self.pars.vit_prompt:
            self.prompts = self._setup_prompt()
            model_params += list(self.prompts.parameters())
            if self.pars.vit_proxy_net:
                self.p_prompts = self._setup_prompt(self.n_classes, 
                                                self.pars.p_prompt_depth, 
                                                self.pars.num_p_prompt,
                                                requires_grad=not self.pars.prompt_memory_bank)
            
            # depth x n_class_batch x num_p_prompt x embed_dim
            
            if self.pars.prompt_memory_bank:
                self.p_prompts_buffer = self._setup_prompt(self.n_classes_batch, 
                                                self.pars.p_prompt_depth, 
                                                self.pars.num_p_prompt,
                                                init=False)
                
                ##############################
                # parameters of client will be udpated automatically, but we have to update 
                # their optimizing states manually
                # this only works for adamw or sgd, other optimizer may have different states
                ##############################
                
                self.memory_state_bank = {}
                if self.pars.optim == 'adamw':
                    self.state_key_list = ['step', 'exp_avg', 'exp_avg_sq']
                    for key in self.state_key_list:
                        self.memory_state_bank[key] = []
                        if key == 'step':
                            for i in range(self.pars.p_prompt_depth):
                                self.memory_state_bank[key].append(torch.zeros(1))
                        else:
                            for i in range(self.pars.p_prompt_depth):
                                self.memory_state_bank[key].append(torch.zeros(self.n_classes, 
                                                            self.pars.num_p_prompt,
                                                            self.embed_dim))
                elif self.pars.optim == 'sgd':
                    self.state_key_list = ['momentum_buffer']
                    for key in self.state_key_list:
                        self.memory_state_bank[key] = []
                        if key == 'step':
                            for i in range(self.pars.p_prompt_depth):
                                self.memory_state_bank[key].append(torch.zeros(1))
                        else:
                            for i in range(self.pars.p_prompt_depth):
                                self.memory_state_bank[key].append(torch.zeros(self.n_classes, 
                                                            self.pars.num_p_prompt,
                                                            self.embed_dim))
                else:
                    raise NotImplementedError
                ##############################
                
                # save buffer to optmizer
                memory_bank += list(self.p_prompts_buffer.parameters())
                
            elif self.pars.vit_proxy_net:
                proxy_net_params += list(self.p_prompts.parameters())
            
        # setup optim_dict_list
        self.optim_bank_idx = 1
        self.optim_dict_list = [
            {
                'params': nn.ParameterList(proxy_net_params),
                'lr': opt.lr,
                'weight_decay': opt.decay
            },
                 #optim_bank_idx = 1
            {
                'params': nn.ParameterList(memory_bank),
                'lr': opt.lr,
                'weight_decay': opt.decay
            },
            {
                'params': nn.ParameterList(model_params),
                'lr': opt.lr,
                'weight_decay': opt.decay
            }]
        
        self.p_optim_dict_list = [{
                'params': nn.ParameterList([self.proxies_bias]),
                'lr': opt.lr_proxy,
                'weight_decay': opt.decay
            }]
        self.model.cuda().train()

    # save states to bank
    def bank_optim_states_save(self):
        with torch.no_grad():
            for param_idx, param in enumerate(
                    self.optimizer.param_groups[self.optim_bank_idx]['params']
                ):
                state = self.optimizer.state[param]
                
                if len(state.keys()) == 0:
                    continue
                
                for k in self.state_key_list:
                    # n_classes_batch x num_p_prompt x embed_dim should be same to the buffer size
                    if k == 'step':
                        self.memory_state_bank[k][param_idx] = state[k].cpu()
                    else:
                        self.memory_state_bank[k][param_idx][self.u_labels_local] = state[k].cpu()
    
    # save model to bank
    def bank_prompt_p_save(self):
        with torch.no_grad():
            for idx, prompt in enumerate(self.p_prompts_buffer):
                prompt_local = prompt.detach().cpu()
                prompt_local.requires_grad = False
                self.p_prompts[idx][self.u_labels_local] = prompt_local
    
    def _update_optim_parameters(self, p_depth_idx, update_values, update_states):
        with torch.no_grad():
            param = self.optimizer.param_groups[self.optim_bank_idx]['params']
            
            # Update parameter values
            param[p_depth_idx].copy_(update_values)
            
            # Update optimizer state (e.g., momentum)
            if len(self.optimizer.state.keys()) > 0 and \
                len(self.optimizer.state[param[p_depth_idx]]) > 0:

                for k in self.state_key_list:
                    self.optimizer.state[param[p_depth_idx]][k].copy_(update_states[k])
    
    # load model and state from bank
    def bank_prompt_p_widthdraw(self):
        with torch.no_grad():
            for idx, prompt in enumerate(self.p_prompts):
                _prompt_local = prompt[self.u_labels_local].clone().to(self.pars.device) # BS x num x 384
                _prompt_local.requires_grad = True
                update_state = {}
                for key in self.state_key_list:
                    if key == 'step':
                        update_state[key] = self.memory_state_bank[key][idx].clone().to(self.pars.device)
                    else:
                        update_state[key] = self.memory_state_bank[key][idx][self.u_labels_local].clone().to(self.pars.device)
                # update parameters and states to optimizer
                self._update_optim_parameters(idx, _prompt_local, update_state)
    
    
    def _recover_indices(self, input_tensor, target_tensor):
        with torch.no_grad():
            idx = torch.nonzero(torch.eq(input_tensor.view(-1, 1), target_tensor).T, as_tuple=True)[1]
        return idx
    
    # check params name
    def _check_params(self, module_name, safe_list, all_match=True):
        check = [partial_name in module_name for partial_name in safe_list]
        return all(check) if all_match else any(check)
    
    def _cal_channels(self):
        conv = self.model.patch_embed.proj
        in_height = self.input_size[1]
        in_width = self.input_size[2]
        out_height = (in_height + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        out_width = (in_width + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1
        return out_height * out_width + 1
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        else:
            raise NotImplementedError("Not init implemented for {}".format(module))
    
    def _setup_head(self, in_dim=None, out_dim=None):
        if in_dim is None:
            in_dim = self.model.embed_dim
        if out_dim is None:
            out_dim = self.pars.embed_dim
        
        head = nn.Linear(in_dim, out_dim).apply(self._init_weights)
        norm = nn.LayerNorm(in_dim).apply(self._init_weights)
        for _, param in head.named_parameters():
            param.requires_grad = True
        for _, param in norm.named_parameters():
            param.requires_grad = True
        
        if self.distill:
            dist_head = nn.Linear(in_dim, out_dim).apply(self._init_weights)
            dist_norm = nn.LayerNorm(in_dim).apply(self._init_weights)  
            for _, param in dist_head.named_parameters():
                param.requires_grad = True
            for _, param in dist_norm.named_parameters():
                param.requires_grad = True
            return head, norm, dist_head, dist_norm
        
        return head, norm, None, None
    
    def _copy_parameters(self, A: nn.Module, B: nn.Module):
        dict_A = A.state_dict()
        dict_B = B.state_dict()

        new_dict_B = dict_B.copy()
        for name_A, param_A in dict_A.items():
            if name_A in dict_B:
                if dict_B[name_A].shape == param_A.shape:
                    new_dict_B[name_A] = param_A
        B.load_state_dict(new_dict_B)
    
    def _setup_adapter(self, model_params):
        depth = self.block_depth
        adapter_num_blocks = max(min(self.pars.adapter_num_blocks, depth), 0)
        for i in range(depth-1, depth - adapter_num_blocks-1, -1):
            adapter = AdaptBlock(self.pars, 
                                dim=self.model.embed_dim,
                                drop_path=self.pars.feature_dropout)
            
            self._copy_parameters(self.model.blocks[i], adapter)
            self.model.blocks[i] = adapter
            
            for _, param in self.model.blocks[i].named_parameters():
                param.requires_grad = False
            for _, param in self.model.blocks[i].adapter.named_parameters():
                param.requires_grad = True
                model_params += [param]
        return model_params
    
    def _setup_prompt(self, length=None, depth=None, num_prompt=None, requires_grad=True, init=True):
        if depth is None:
            depth = max(min(self.pars.prompt_depth, self.block_depth), 0)
            self.pars.prompt_depth = depth
        if num_prompt is None:
            num_prompt = self.pars.num_prompt
        if length is None:
            length = 1
        # Setup VPTs
        # depth x n_class x num_prompt x embed_dim
        if init:
            prompts = nn.ParameterList(
                [nn.Parameter(nn.init.xavier_normal_(
                    torch.zeros(length, num_prompt, self.embed_dim))) 
                 for _ in range(depth)])
        else:
            prompts = nn.ParameterList(
                [nn.Parameter(
                    torch.zeros(length, num_prompt, self.embed_dim))
                 for _ in range(depth)])
        
        if requires_grad:
            for _, param in prompts.named_parameters():
                param.requires_grad = True
        else:
            for _, param in prompts.named_parameters():
                param.requires_grad = False
        return prompts
    
    def _setup_proxies(self):
        '''setup proxy prompts'''
        proxies = nn.Parameter(nn.init.normal_(
            torch.empty(self.n_classes, self.p_channels, self.embed_dim),
                        mean=0.5, std=0.5)
                                )
        proxies.requires_grad = True
        return proxies
    
    
    def _print_trainable_params(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n + ' ' + str(p.shape) + '\n')
                
    def _split_tensor(self, A, k, stack=True):
        # return k x (n//k) x dim
        assert (A.shape[0] % k == 0) & (k > 0)
        split_tensors = [A[i::k] for i in range(k)] # list of k x (BS//k) x dim
        if stack:
            return torch.stack(split_tensors)
        else:
            return split_tensors
        
    def forward_head(self, x, **kwargs):
        # new head for dml
        if self.pars.vit_global_pooling == 'avg':
            x = x[:, 1:].mean(dim=1)  # avg pooling
        else:
            x_dist = x[:, 1]
            x = x[:, 0]  # use cls_token
        
        # train head or run zero-shot
        if self.pars.vit_head:
            x = self.fc_norm(x)
            x = self.head(x)
            if self.distill:
                x_dist = self.dist_norm(x_dist)
                x_dist = self.dist_head(x_dist)
                x = (x + x_dist) / 2
        
        x = F.normalize(x, dim=-1)
        return x
    
    
    def forward_p_head(self, x, **kwargs):
        if self.pars.vit_global_pooling == 'avg':
            x = x[:, 1:].mean(dim=1)  # avg pooling
        elif self.pars.vit_global_pooling == 'combined':
            x = torch.cat((x[:, 0], x[:, 1:].mean(dim=1)), dim=-1)  # cls+avg 2 x embed_dim
        else:
            x_dist = x[:, 1]
            x = x[:, 0]  # use cls_token
        
        # train head or run zero-shot
        if self.pars.vit_head:
            x = self.p_fc_norm(x)
            x = self.p_head(x)
            if self.distill:
                x_dist = self.p_dist_norm(x_dist)
                x_dist = self.p_dist_head(x_dist)
                x = (x + x_dist) / 2
            
        x = F.normalize(x, dim=-1)
        return x
    
    
    def forward_proxy_mix_bias(self, x, labels, **kwargs):
        
        f_labels = labels.view(-1)
        # unique labels in each batch: BS//k

        self.f_labels = labels.view(-1)
        self.u_labels = torch.unique_consecutive(self.f_labels) # BS//k, in order
        self.BS = self.f_labels.shape[0]
        self.k = self.pars.samples_per_class
        self.u_labels_local = self.u_labels.cpu() # BS//k, in order
        
        assert x.shape[0] == f_labels.shape[0]
        
        batch_p_prompts = []
        
        if self.pars.prompt_memory_bank:
            self.batch_memory_bank = []
            self.bank_prompt_p_widthdraw()

        ## ================== memory bank ==================
        for idx in range(self.pars.p_prompt_depth):
            # assign prompt frmo bank to buffer

            if not self.pars.prompt_memory_bank:
                prompt = self.p_prompts[idx][f_labels] # BS x num x 384
            else:
                r_idx = self._recover_indices(self.u_labels, f_labels)
                prompt = self.p_prompts_buffer[idx][r_idx]
            
            batch_p_prompts.append(prompt)
        
        
        ## ================== pre_embedding ==================
        if not self.distill:
            x = self.model.patch_embed(x) # B x 384 x 14 x 14
            x = self.model._pos_embed(x) # B x 197 x 384
            x = self.model.norm_pre(x) # B x 197 x 384
        else:
            x = self._dist_pre_embed(x)
        
        p = x
        
        ## ================== proxy net ==================
        for idx, blk in enumerate(self.model.blocks):
            # transfer i to block index
            if self.pars.vit_prompt:
                # add vpt with the running blocks
                num_p_vpt = 0
                num_vpt = 0
                vpt = torch.tensor([]).to(x.device)
                if idx < self.pars.p_prompt_depth:
                    num_p_vpt = max((self.pars.num_p_prompt - idx * self.pars.prompt_p_block_step), 0)
                    if num_p_vpt > 0:
                        p_prompts = batch_p_prompts[idx][:, :num_p_vpt, :]
                        vpt = torch.cat([p_prompts, vpt], dim=1)
                
                p = self._forward_vpt_block(p, blk, vpt, num_vpt + num_p_vpt)
            else:
                p = self.model.blocks[idx](p)
        p = self.model.norm(p) # B x 197 x 384
        
        return self._proxies_generator(p)
    
    # fuse the proxy with container
    def _semantic_mix_with_container(self, p):
        pc = self.proxies_container[self.u_labels] # BS//k x 384
        if self.pars.semantic_mix_type == 'mix':
            p = self._split_tensor(p, self.k).mean(dim=0) # BS//k x 384
            pc = self.pars.proxy_mix_keep_ratio * pc + (1 - self.pars.proxy_mix_keep_ratio) * p
        elif 'gru' in self.pars.semantic_mix_type:
            p = p.view(self.BS//self.k, self.k, -1)
            h = self.proxies_container[self.u_labels]
            for i in range(self.k):
                h = self.rnn_update(p[:, i, :], h)
            pc = h
        else:
            raise NotImplementedError
        self.proxies_container[self.u_labels] = pc
        self.proxies_container = F.normalize(self.proxies_container, dim=-1)
        return pc
    
    def _update_proxy_bias(self, p):
        with torch.no_grad():
            self.proxies_bias[self.u_labels].copy_(p)
    
    
    def _proxies_generator(self, p):
        
        if self.pars.vit_global_pooling == 'avg':
            p = p[:, 1:].mean(dim=1)  # avg pooling
        else:
            p_dist = p[:, 1]
            p = p[:, 0] # B x 384

        # train head, probing or run zero-shot
        if self.pars.vit_head:
            p = self.p_fc_norm(p)
            p = self.p_head(p)
            
            if self.distill:
                p_dist = self.p_dist_norm(p_dist)
                p_dist = self.p_dist_head(p_dist)
                p = (p + p_dist) / 2
            
        p = F.normalize(p, dim=-1)
        
        p = self._semantic_mix_with_container(p)
        
        # for return
        p = self.proxies_bias[self.u_labels] * self.pars.proxy_bias_ratio + \
            p * (1 - self.pars.proxy_bias_ratio)
        
        p_all = self.proxies_bias * self.pars.proxy_bias_ratio + \
            self.proxies_container * (1 - self.pars.proxy_bias_ratio)
        p = F.normalize(p, dim=-1)
        
        # detach proxies_container to avoid multi updates on tensor graph
        # in next batch forward
        self.proxies_container = self.proxies_container.detach()
        
        return p_all, p
    
    
    def forward_simple_proxies(self, x, labels, **kwargs):
        
        self.f_labels = labels.view(-1)
        self.u_labels = torch.unique_consecutive(self.f_labels) # BS//k, in order
        self.BS = self.f_labels.shape[0]
        self.k = self.pars.samples_per_class
        
        if not self.distill:
            x = self.model.patch_embed(x) # B x 384 x 14 x 14
            x = self.model._pos_embed(x) # B x 197 x 384
            x = self.model.norm_pre(x) # B x 197 x 384
        else:
            x = self._dist_pre_embed(x)
        
        if not self.pars.vit_prompt:
            x = self.model.blocks(x)
        else:
            for idx, blk in enumerate(self.model.blocks):
                if idx < self.pars.prompt_depth and self.pars.simple_proxies_vpt:
                    num_vpt = max((self.pars.num_prompt - idx * self.pars.prompt_block_step), 0)
                    prompts = self.prompts[idx][:, :num_vpt, :]
                    vpt = prompts.repeat(x.shape[0], 1, 1) # 1 x N x 384 -> B x N x 384
                    x = self._forward_vpt_block(x, blk, vpt, num_vpt)
                else:
                    x = blk(x)            
        p = self.model.norm(x)
        
        return self._proxies_generator(p)
    
    
    def _forward_vpt_block(self, x, unit, vpt, num_vpt):
        s = x.shape[1]
        if num_vpt > 0:
            x = torch.cat([vpt, x], dim=1)
        if not self.pars.checkpoint:
            x = unit(x)
        else:
            x = checkpoint(unit, x)
        x = x[:, num_vpt:, :]
        assert x.shape[1] == s
        return x
    
    def _dist_pre_embed(self, x):
        assert self.distill
        x = self.model.patch_embed(x)
        x = torch.cat((
            self.model.cls_token.expand(x.shape[0], -1, -1),
            self.model.dist_token.expand(x.shape[0], -1, -1),
            x),
            dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        return x
    
    def forward_feature(self, x, **kwargs):
        
        if not self.distill:
            x = self.model.patch_embed(x) # B x 384 x 14 x 14
            x = self.model._pos_embed(x) # B x 197 x 384
            x = self.model.norm_pre(x) # B x 197 x 384
        else:
            x = self._dist_pre_embed(x)
        
        if not self.pars.vit_prompt:
            if not self.pars.checkpoint:
                x = self.model.blocks(x)
            else:
                for blk in self.model.blocks:
                    x = checkpoint(blk, x)
        else:
            for idx, blk in enumerate(self.model.blocks):
                if idx < self.pars.prompt_depth:
                    num_vpt = max((self.pars.num_prompt - idx * self.pars.prompt_block_step), 0)
                    prompts = self.prompts[idx][:, :num_vpt, :]
                    vpt = prompts.repeat(x.shape[0], 1, 1) # 1 x N x 384 -> B x N x 384
                    x = self._forward_vpt_block(x, blk, vpt, num_vpt)
                else:
                    if not self.pars.checkpoint:
                        x = blk(x)       
                    else:
                        x = checkpoint(blk, x)     
        x = self.model.norm(x)
        return x
    
    def forward(self, x, labels=None, anchor=None, **kwargs):
        x_res = x
        x = self.forward_feature(x, **kwargs)
        x = self.forward_head(x, **kwargs)
        x = self.dropout(x)
        
        # train proxy net
        if self.pars.vit_proxy_net and self.training:
            if self.pars.proxy_mix_bias:
                pc = self.forward_proxy_mix_bias(x_res, labels, **kwargs)
            elif self.pars.simple_proxies:
                pc = self.forward_simple_proxies(x_res, labels, **kwargs)
            else:
                raise NotImplementedError ("Not implemented proxy net")
        else:
            if labels is not None:
                self.u_labels = torch.unique_consecutive(labels)
                pc = (self.proxies_bias, self.proxies_bias[self.u_labels])
            else:
                pc = None
        return {
            'embeds': x,
            'proxies': pc,
            'labels': labels
        }

def init_bert_weights(module):
    """Initialize the weights for BERT."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
        

class Simple_GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, drop=0.0, type='gru'):
        super(Simple_GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_drop = nn.Dropout(drop)
        self.type = type
        self.W_ir = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))

        self.W_iz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_iz = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_in = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_in = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_ia = nn.Parameter(torch.Tensor(input_size, 1))
        
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
    
    def forward(self, x, h_prev):
        
        z = torch.sigmoid(x @ self.W_iz + h_prev @ self.W_hz + self.b_iz)
        r = torch.sigmoid(x @ self.W_ir + h_prev @ self.W_hr + self.b_ir)
        if self.type == 'gru':
            n = torch.tanh(x @ self.W_in + r * (h_prev @ self.W_hn) + self.b_in)
            h_next = (1 - z) * n + z * h_prev
        elif self.type == 'gru_relu':
            n = F.relu(x @ self.W_in + r * (h_prev @ self.W_hn) + self.b_in)
            h_next = (1 - z) * n + z * h_prev
        else:
            raise NotImplementedError
        
        h_next = self.rnn_drop(h_next)
        return h_next




# https://github.com/ShoufaChen/AdaptFormer/blob/main/models/adapter.py
class AdapterLayer(nn.Module):
    def __init__(self, opt, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.down_size = opt.adapter_bottleneck_dim
        self.ln_position = opt.adapter_ln_position
        self.adapter_ln = nn.LayerNorm(self.n_embd)
        self.scale = float(opt.adapter_scalar)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = opt.feature_dropout
        
        # IMPROVE ME
        if opt.adapter_init_option == "bert":
            self.apply(init_bert_weights)
        elif opt.adapter_init_option == "lora":
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
        else:
            raise NotImplementedError("Adapter init option not implemented!")
        
    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        
        if self.ln_position == 'pre':
            x = self.adapter_ln(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        if self.ln_position == 'post':
            up = self.adapter_ln(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output

# modified from timm/models/vision_transformer.py
class AdaptBlock(nn.Module):

    def __init__(
            self, 
            opt, 
            dim,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.opt = opt
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adapter = AdapterLayer(self.opt, n_embd=dim)
    
    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        if self.opt.adapter_option == 'parallel':
            adapt_x = self.adapter(x, add_residual=False)
        residual = x
        x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        if self.opt.adapter_option == 'sequential':
            x = self.adapter(x, add_residual=True)
        elif self.opt.adapter_option == 'parallel':
            x = x + adapt_x
        else:
            raise NotImplementedError
        x = residual + x
        return x