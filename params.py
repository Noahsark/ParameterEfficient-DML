
def basic_training_parameters(parser):
    ### General Training Parameters
    parser.add_argument('--name', default='default', type=str,
                        help='Name of the experiment.')
    parser.add_argument('--dataset', default='cub200', type=str,
                        help='Dataset to use: cub200, cars196, sop, inshop')
    
    parser.add_argument('--no_train_metrics', action='store_true',
                        help='Flag. If set, no training metrics are computed and logged.')
    parser.add_argument('--no_val_metrics', action='store_true',
                        help='Flag. If set, no val metrics are computed and logged.')
    parser.add_argument('--no_test_metrics', action='store_true',
                        help='Flag. If set, no test metrics are computed and logged.')

    parser.add_argument('--evaluation_metrics', nargs='+', default=['e_recall@1', 'e_recall@2', 'e_recall@4', 'e_recall@8', 'nmi', 'f1', 'mAP_R'], type=str,
                        help='Metrics to evaluate performance by.')
                        
    parser.add_argument('--evaltypes', nargs='+', default=['embeds'], type=str)
    parser.add_argument('--storage_metrics', nargs='+', default=['e_recall@1'], type=str,
                        help='Improvement in these metrics will trigger checkpointing.')
    parser.add_argument('--store_improvements', action='store_true',
                        help='If set, will store checkpoints whenever the storage metric improves.')
    parser.add_argument('--save_step', default = 0, type=int, help='save model for x epochs')
    parser.add_argument('--save_results', action='store_true', help='')
    
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        help='gpus')
    parser.add_argument('--savename', default='group_plus_seed',   type=str,
                        help='Appendix to save folder name if any special information is to be included.')
    parser.add_argument('--source_path', default='../dml_data',   type=str,
                        help='Path to training data.')
    parser.add_argument('--save_path', default='../Results', type=str,
                        help='Where to save everything.')
    parser.add_argument('--config', default='', type=str)
    
    ### General Optimization Parameters
    parser.add_argument('--n_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--kernels', default=14, type=int,
                        help='Number of workers for pytorch dataloader.')
    parser.add_argument('--batch_size', default=64 , type=int,
                        help='Mini-Batchsize to use.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--sdg_momentum', default=0.9, type=float)
    
    ### lr scheduler settings
    parser.add_argument('--lr',  default=0.001, type=float,
                        help='Learning Rate for network parameters.')
    parser.add_argument('--lr_proxy', default=0.5, type=float,
                        help='Learning Rate for proxies.')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        help='Type of learning rate scheduling. Currently: step & multi, linear, cosine.')     
    parser.add_argument('--lr_cosine_length', default=50, type=int)
    parser.add_argument('--lr_cosine_min', default=1e-6, type=float)
    parser.add_argument('--lr_reduce_rate', default=0.5, type=float,
                        help='Learning rate reduction after --lr_reduce_step epochs.')
    parser.add_argument('--lr_reduce_step', default=5, type=int,
                        help='step before reducing learning rate.')
    parser.add_argument('--lr_reduce_multi_steps', default=[20, 45, 70], nargs='+', type=int, 
                        help='milestones (epoch) before reducing learning rate.')
    parser.add_argument('--lr_linear_start', default=1.0, type=float)
    parser.add_argument('--lr_linear_end', default=0.01, type=float)
    parser.add_argument('--lr_linear_length', default=20, type=int)
    parser.add_argument('--decay', default=0.0, type=float,
                        help='Weight decay for optimizer.')
                        
    parser.add_argument('--augmentation', default='base', type=str,
                        help='image augmentation: base, big, adv, v2, auto')
    parser.add_argument('--warmup', default=-1, type=int,
                        help='warmup stage: freeze model and train the last layer and others only.')

    parser.add_argument('--optim', default='sgd', type=str,
                        help='Optimizer to use. adam, adamw, sgd')
    parser.add_argument('--loss', default='dml_proxy_peft', type=str,
                        help='Trainin objective to use. See folder <criteria> for available methods.')
    parser.add_argument('--batch_mining', default='distance', type=str,
                        help='Batchmining method to use. See folder <batchminer> for available methods.')
    
    # representation settings
    parser.add_argument('--embed_dim', default=384, type=int)
    # Vit settings
    parser.add_argument('--arch', default='peft_vit',  type=str)
    parser.add_argument('--pretrained_model', default='vit_small_patch16_224_in21k', type=str)
    parser.add_argument('--vit_global_pooling', default='token', help='avg, token or combined')
    
    # Vit Fine turn setting
    parser.add_argument('--full_fine_tune', action='store_true')
    parser.add_argument('--vit_head', action='store_true', help='fine turn the head parameters')
    parser.add_argument('--vit_prompt', action='store_true', help='active vit prompt')
    parser.add_argument('--vit_proxy_net' , action='store_true', help='active vit proxy net')
    parser.add_argument('--vit_bitfit', action='store_true', help='active vit bitfit')
    
    # adapter settings
    parser.add_argument('--adapter', action='store_true', help='active adapter')
    parser.add_argument('--adapter_option', default='parallel', type=str, help='parallel or sequential')
    parser.add_argument('--adapter_num_blocks', default=12, type=int, help='number blocks applied to the adapter')
    parser.add_argument('--adapter_bottleneck_dim', default=64, type=int)
    parser.add_argument('--adapter_scalar', default=1.0, type=float)
    parser.add_argument('--adapter_ln_position', default='post', type=str, help='pre or post')
    parser.add_argument('--adapter_init_option', default='lora', type=str, help='bert or lora')
    
    # prompt settings
    parser.add_argument('--num_prompt', default=10, type=int)
    parser.add_argument('--prompt_depth', default=12, type=int)
    
    # proxy net settings
    parser.add_argument('--proxy_net_depth', default=12, type=int,
                        help='depth of the proxy net')
    
    parser.add_argument('--num_p_prompt', default=5, type=int)
    parser.add_argument('--p_prompt_depth', default=3, type=int)
    
    parser.add_argument('--proxy_net_head', default='isolated', type=str,
                        help='shared or isolated')
    parser.add_argument('--proxy_mix_bias', action='store_true', help='mix original proxies into vpts proxies')
    parser.add_argument('--proxy_mix_keep_ratio', default=0.8, type=float)
    parser.add_argument('--proxy_bias_ratio', default=0.5, type=float)
    parser.add_argument('--prompt_p_block_step', default=0, type=int)
    parser.add_argument('--prompt_block_step', default=0, type=int)
    parser.add_argument('--proxy_data_range', default="in_batch", type=str,
                        help='  in_batch: only optimize proxies inside batch; \
                                both_all: optimize all in batch, positive and negative in all proxies.')
    parser.add_argument('--simple_proxies', action='store_true', help='active simple proxies')
    parser.add_argument('--simple_proxies_vpt', action='store_true', help='active simple proxies vpt')
    parser.add_argument('--semantic_mix_type', default='gru_relu', type=str, help='mix, gru ...')
    parser.add_argument('--rnn_dropout', default=0.0, type=float)
    parser.add_argument('--feature_dropout', default=0.0, type=float)
    
    return parser


def scale_optimizing_parameters(parser):
    
    parser.add_argument('--mix_precision', action='store_true', help='active mix precision')
    parser.add_argument('--world_size', default=1, type=int, help='number of gpus')
    parser.add_argument('--prompt_memory_bank', action='store_true', help='active prompt memory bank')
    parser.add_argument('--normal', action='store_true', help='normalize in testing')
    parser.add_argument('--checkpoint', action='store_true', help='active gpu checkpoint')
    parser.add_argument('--show_trainable_pars', action='store_true', help='show trainable parameters')
    parser.add_argument('--eval_start', default=0, type=int, help='start epoch for evaluation')
    parser.add_argument('--time', action='store_true', help='show running time')
    
    return parser


def loss_specific_parameters(parser):
 
    ### oproxy.
    parser.add_argument('--loss_oproxy_mode', default='anchor', type=str,
                        help='Proxy-method: anchor = ProxyAnchor, nca = ProxyNCA.')
    parser.add_argument('--loss_oproxy_lrmulti', default=200, type=float,
                        help='Learning rate multiplier for proxies.')
    parser.add_argument('--loss_oproxy_pos_alpha', default=32, type=float,
                        help='Inverted temperature/scaling for positive sample-proxy similarities.')
    parser.add_argument('--loss_oproxy_neg_alpha', default=32, type=float,
                        help='Inverted temperature/scaling for negative sample-proxy similarities.')
    parser.add_argument('--loss_oproxy_pos_delta', default=0.1, type=float,
                        help='Threshold for positive sample-proxy similarities')
    parser.add_argument('--loss_oproxy_neg_delta', default=-0.1, type=float,
                        help='Threshold for negative sample-proxy similarities')
    return parser


def batchmining_specific_parameters(parser):
    """
    Hyperparameters for various batchmining methods.
    """
    ### Distance-based_Sampling.
    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float,
                        help='Cutoff distance value below which pairs are ignored.')
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float,
                        help='Cutoff distance value above which pairs are ignored.')
    ### Spectrum-Regularized Miner.
    parser.add_argument('--miner_rho_distance_lower_cutoff', default=0.5, type=float,
                        help='Same behaviour as with standard distance-based mining.')
    parser.add_argument('--miner_rho_distance_upper_cutoff', default=1.4, type=float,
                        help='Same behaviour as with standard distance-based mining.')
    parser.add_argument('--miner_rho_distance_cp', default=0.2, type=float,
                        help='Probability with which label assignments are flipped.')
    ### Semihard Batchmining.
    parser.add_argument('--miner_semihard_margin', default=0.2, type=float,
                        help='Margin value for semihard mining.')
    parser.add_argument('--internal_split', action='store_true')
    return parser


def batch_creation_parameters(parser):
    """
    Parameters for batch sampling methods.
    """
    parser.add_argument('--data_sampler', default='class_random', type=str,
                        help='Batch-creation method. Default <class_random> ensures that for each class, \
                        at least --samples_per_class samples per class are available in each minibatch.')
    parser.add_argument('--data_ssl_set', action='store_true',
                        help='Obsolete. Only relevant for SSL-based extensions.')
    parser.add_argument('--samples_per_class', default=2, type=int,
                        help='Number of samples in one class drawn before choosing the next class. \
                        Set to >1 for losses other than ProxyNCA.')
    return parser
