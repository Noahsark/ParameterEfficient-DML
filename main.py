import argparse
import os
import time
import warnings
import yaml
import contextlib

import numpy as np
import termcolor
from tqdm import tqdm
import params as par
import utilities.misc as misc

### ---------------------------------------------------------------
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import timm
import architectures as archs
import datasampler as dsamplers
import datasets as dataset_library
import criteria as criteria
import metrics as metrics
import batchminer as bmine
from utilities import misc

from metrics import query_gallery_metrics

warnings.filterwarnings("ignore")

# define datasets with query and gallery
QUERY_GALLERY_DATASETS = ['inshop']
QUERY_GALLERY_RECALL_KS = [1, 10, 20, 30, 40, 50]

# define datasets need memory bank
LARGE_DATASETS = ['inshop', 'sop']


def create_optimizer(opt, to_optim):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(to_optim)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(to_optim)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(to_optim,
                                momentum=opt.sdg_momentum)
    else:
        raise Exception('Optimizer <{}> not available!'.format(opt.optim))
    return optimizer

def create_scheduler(opt, scheduler, optimizer):
    if scheduler == 'multi':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=opt.lr_reduce_multi_steps,
                                                 gamma=opt.lr_reduce_rate)
    elif scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=opt.lr_reduce_step,
                                            gamma=opt.lr_reduce_rate)
    elif scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                            start_factor=opt.lr_linear_start,
                                            end_factor=opt.lr_linear_end,
                                            total_iters=opt.lr_linear_length)
    elif scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                            T_max=opt.lr_cosine_length,
                                            eta_min=opt.lr_cosine_min)
    else:
        raise Exception('Scheduler <{}> not available!'.format(opt.scheduler))
    return scheduler



def train():
    
    ### INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser = par.basic_training_parameters(parser)
    parser = par.scale_optimizing_parameters(parser)
    parser = par.batch_creation_parameters(parser)
    parser = par.batchmining_specific_parameters(parser)
    parser = par.loss_specific_parameters(parser)
    
    ### Additional, non-default parameters.
    opt = parser.parse_args()
    
    # update setting from config
    with open(opt.config) as file:
        if file is not None:
            _config = yaml.load(file, Loader=yaml.FullLoader)
            for key, value in _config.items():
                setattr(opt, key, value['value'])
    assert opt.world_size == 1, 'Distributed training not yet supported for turn.py!'
    
    # setup auto memory bank
    if opt.batch_size >= 256 and opt.dataset in LARGE_DATASETS:
        opt.prompt_memory_bank = True
        opt.checkpoint = True
    
    if opt.batch_size >= 256 and opt.embed_dim >= 512:
        opt.prompt_memory_bank = True
        opt.checkpoint = True
    
    opt.source_path += '/' + opt.dataset
    opt.save_path += '/' + opt.dataset

    # Assert that the construction of the batch makes sense, i.e. the division into
    # class-subclusters.
    assert_text = 'Batchsize needs to fit number of samples per class for distance '
    assert_text += 'sampling and margin/triplet loss!'
    assert not opt.batch_size % opt.samples_per_class, assert_text
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])
    misc.set_seed(opt.seed)

    ### ---------------------------------------------------------------
    ### Embedding Network model (resnet, inception).
    
    # load default model to get cfg info
    _model_for_cfg = timm.create_model(opt.pretrained_model, pretrained=False)
    
    opt.pretrained_cfg = None
    if hasattr(_model_for_cfg, 'pretrained_cfg'):
        opt.pretrained_cfg = _model_for_cfg.pretrained_cfg
    
    del _model_for_cfg
    
    ####### Datasetse & Dataloaders.
    
    datasets = dataset_library.select(opt)
    opt.n_classes = datasets['training'].n_classes
    
    opt.device = torch.device('cuda')
    model = archs.select(opt.arch, opt)
    
    if hasattr(model, 'optim_dict_list') and len(model.optim_dict_list):
        to_optim = model.optim_dict_list
    else:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        to_optim = [{'params': model_parameters, 'lr': opt.lr, 'weight_decay': opt.decay}]
    
    ####### Build Criterion.
    batchminer = bmine.select(opt.batch_mining, opt)
    criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
    _ = criterion.to(opt.device)
    
    if opt.show_trainable_pars:
        model._print_trainable_params()
    
    _ = model.to(opt.device)

    dataloaders = {}
    if not opt.dataset in QUERY_GALLERY_DATASETS:
        dataloaders['evaluation'] = torch.utils.data.DataLoader(
                datasets['evaluation'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                shuffle=False)
        dataloaders['testing'] = torch.utils.data.DataLoader(
                datasets['testing'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                shuffle=False)
    else:
        dataloaders['query'] = torch.utils.data.DataLoader(
                datasets['query'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                shuffle=False)
        dataloaders['gallery'] = torch.utils.data.DataLoader(
                datasets['gallery'],
                num_workers=opt.kernels,
                batch_size=opt.batch_size,
                shuffle=False)
    
    train_data_sampler = dsamplers.select(opt.data_sampler, opt,
                                      datasets['training'].image_dict,
                                      datasets['training'].image_list)
    
    if train_data_sampler.requires_storage:
        train_data_sampler.create_storage(dataloaders['evaluation'], model,
                                      opt.device)
    
    dataloaders['training'] = torch.utils.data.DataLoader(
        datasets['training'],
        num_workers=opt.kernels,
        batch_sampler=train_data_sampler)
    
    if not opt.dataset in QUERY_GALLERY_DATASETS:
        opt.n_test_classes = len(dataloaders['testing'].dataset.avail_classes)
    
        metric_evaluation_keys = ['testing', 'evaluation']
    
    
    ### ---------------------------------------------------------------
    if opt.prompt_memory_bank:
        model.p_prompts.to('cpu')
    
    ### Optimizer ####
    model.optimizer = create_optimizer(opt, to_optim)
    model.p_optimizer = create_optimizer(opt, model.p_optim_dict_list)
    scheduler = create_scheduler(opt, opt.scheduler, model.optimizer)

    p_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.p_optimizer,
                                            T_max=opt.lr_cosine_length,
                                            eta_min=opt.lr_cosine_min)
    
    ### Metric Computer.
    metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)
    
    ### ---------------------------------------------------------------
    ### Summary.
    trainable_par = misc.gimme_params(model)
    extra_par = misc.gimme_params(criterion)
    summary = 'Dataset:\t {} \n'.format(opt.dataset)
    summary += 'Objective:\t {} \n'.format(opt.loss)
    summary += 'Backbone:\t {} \n'.format(opt.arch)
    summary += 'Pretrained:\t {} \n'.format(opt.pretrained_model)
    summary += 'Optimizer:\t {} \n'.format(opt.optim)
    summary += 'Trainable Pars:\t {:.2f}M ({}) \n'.format(trainable_par/1e6, trainable_par)
    summary += 'Extra Pars:\t {:.2f}M ({}) \n'.format(extra_par/1e6, extra_par)
    print(summary)
    ### ----------------------------------------------------------------
    
    #### Main training. #####
    print('\n' + termcolor.colored('===============================================', 'red', 
        attrs=['bold']) + '\n')
    
    iter_count = 0
    loss_args = {
        'batch': None,
        'labels': None,
        'batch_features': None,
        'f_embed': None
    }

    opt.epoch = 0
    epochs = range(opt.epoch, opt.n_epochs)
    
    scaler = torch.cuda.amp.GradScaler()
    context = torch.cuda.amp.autocast(
        ) if opt.mix_precision else contextlib.nullcontext()
    
    torch.cuda.empty_cache()


    ################# Training Loop #################
    max_r1 = 0.0
    max_map = 0.0
    
    for epoch in epochs:
        opt.epoch = epoch
        
        misc.set_seed(opt.seed + epoch)
        
        epoch_start_time = time.time()
        
        # Train one epoch
        data_iterator = tqdm(dataloaders['training'],
                bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt}, '
                           '{elapsed}/{remaining}{postfix}]',
                ncols=96, ascii=True, desc='[GPU:%d Train Ep:%d]: ' % (opt.gpu[0], epoch))
        
        _ = model.train()
        criterion.new_epoch = True
        
        total_time = 0.0
        avg_time = 0.0
        # for each minibatch
        for i, out in enumerate(data_iterator):

            if opt.time:
                torch.cuda.synchronize()
                start = time.time()
            
            class_labels, input_dict, sample_indices = out

            # load data to generator
            input = input_dict['image']
            
            model_args = {
                'x': input.to(opt.device),
                'labels': class_labels.to(opt.device),
                'anchor': None
            }
            
            model.optimizer.zero_grad(set_to_none=True)
            model.p_optimizer.zero_grad(set_to_none=True)
            
            with context:
                out_dict = model(**model_args)
                
                loss_args['batch'] = out_dict['embeds']
                loss_args['labels'] = class_labels
                loss_args['proxies'] = out_dict['proxies']
                
                loss = criterion(**loss_args)
            
            data_iterator.set_postfix_str('Loss: {0:.4f}'.format(loss.item()))
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.step(model.p_optimizer)
            scaler.update()

            if opt.prompt_memory_bank:
                model.bank_optim_states_save()
                model.bank_prompt_p_save()   
            
            if opt.time:
                end = time.time()
                total_time = total_time + (end - start)
                avg_time = total_time / (i+1)
                if i >= 100:
                    print('Average time per batch: {:.8f}s'.format(avg_time))
                    opt.time = False
        
        # update learning rate  
        if opt.scheduler != 'none':
            scheduler.step()
            p_scheduler.step()
        # clean memory
        torch.cuda.empty_cache()
        
        if epoch < opt.eval_start:
            continue
        
        # Evaluate Metric for Training & Test (& Validation)
        _ = model.eval()
        
        # clean memory
        torch.cuda.empty_cache()
        
        # run test metrics
        print('\n' + termcolor.colored(
        'Computing Testing Metrics...', 'green', attrs=['bold']))
        
        if not opt.dataset in QUERY_GALLERY_DATASETS:
            # run test metrics
            with context:
                computed_metrics, extra_infos = metric_computer.compute_standard(
                        opt, model, dataloaders['testing'], opt.evaltypes, opt.device, mode='Val')
            # print eval
            numeric_metrics = {}
            for main_key in computed_metrics.keys():
                for name, value in computed_metrics[main_key].items():
                    if main_key not in numeric_metrics:
                        numeric_metrics[main_key] = {}
                    numeric_metrics[main_key][name] = value
            
            # run training metrics
            if not opt.no_train_metrics:
                computed_metrics, extra_infos = metric_computer.compute_standard(
                        opt, model, dataloaders['evaluation'], opt.evaltypes, opt.device, mode='Train')
                if 'e_recall@1' in computed_metrics['embeds'].keys():
                    numeric_metrics['embeds']['train_recall@1'] = computed_metrics['embeds']['e_recall@1']
                if 'mAP_R' in computed_metrics['embeds'].keys():
                    numeric_metrics['embeds']['train_mAP_R'] = computed_metrics['embeds']['mAP_R']
        else:
            
            # query and gallery
            with context:
                recalls, map = query_gallery_metrics.evaluate_query_gallery(model, dataloaders['query'], 
                                                             dataloaders['gallery'], QUERY_GALLERY_RECALL_KS)
            numeric_metrics = {}
            numeric_metrics['embed'] = {}
            for k, recall in zip(QUERY_GALLERY_RECALL_KS, recalls):
                key = "e_recall@{}".format(k)
                numeric_metrics['embed'][key] = recall
            numeric_metrics['embed']['MAP'] = map
        
        full_result_str = ''
        for evaltype in numeric_metrics.keys():
            full_result_str += 'Embed-Type: {}:\n'.format(evaltype)
            for i,(metricname, metricval) in enumerate(numeric_metrics[evaltype].items()):
                full_result_str += '{0}{1}: {2:4.4f}'.format('\n' if i>0 else '',metricname, metricval)
            
            # save maximum recall@1
            if numeric_metrics[evaltype]['e_recall@1'] > max_r1:
                max_r1 = numeric_metrics[evaltype]['e_recall@1']
            numeric_metrics[evaltype]['max_r@1'] = max_r1
            
            # save maximum mAP
            if numeric_metrics[evaltype].keys().__contains__('MAP'):
                if numeric_metrics[evaltype]['MAP'] > max_map:
                    max_map = numeric_metrics[evaltype]['MAP']
                numeric_metrics[evaltype]['max_map'] = max_map
            
            full_result_str += '\n'
        
        print(full_result_str)
        print('\nTotal Epoch Runtime: {0:4.2f}s'.format(time.time() -
                                                    epoch_start_time))
        print('\n' + termcolor.colored('===============================================', 'red', 
                attrs=['bold']) + '\n')
        
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    del model
    del criterion
    return 0
    
    

if __name__ == '__main__':
    train()



