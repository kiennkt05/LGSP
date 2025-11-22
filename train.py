import argparse
import importlib
from utils import *
import torch

MODEL_DIR=None
DATA_DIR = './local_datasets/'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-out', type=str, default=None)

    parser.add_argument('-epochs_base', type=int, default=50)
    parser.add_argument('-epochs_new', type=int, default=20)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=2e-4)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=80)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=64)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=128)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos', 'ft_comb', 'ft_euc']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=50)
    parser.add_argument('-episode_shot', type=int, default=1)
    parser.add_argument('-episode_way', type=int, default=15)
    parser.add_argument('-episode_query', type=int, default=15)

    parser.add_argument('-start_session', type=int, default=1)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=-1)
    # parser.add_argument('-seeds', nargs='+', help='<Required> Set flag', required=True, default=1)
    parser.add_argument('-fraction_to_keep', type=float, default=0.1)
    
    parser.add_argument('-ED_hp', type=float, default=0.1)

    parser.add_argument('-Prompt_Token_num', type=int, default=5) 
    parser.add_argument('-lr_PromptTokens_base', type=float, default=2e-4) 
    parser.add_argument('-lr_PromptTokens_novel', type=float, default=2e-4)

    parser.add_argument("--first_kernel_size", type=int, default=3)
    parser.add_argument("--second_kernel_size", type=int, default=5)
    parser.add_argument("--prompt_hid_dim", type=int, default=3)
    parser.add_argument('-lr_local', type=float, default=2e-4)  

    parser.add_argument('-Dropout_Prompt', type=float, default=0.1) 

    parser.add_argument('-pool_size', type=int, default=24)

    parser.add_argument('-pixel_prompt', type=str, default='YES', choices=['NO', 'YES'])
    parser.add_argument('-Frequency_mask', type=bool, default=True)
    parser.add_argument('-lr_Frequency_mask', type=float, default=0.03) 
    parser.add_argument('-num_r', type=int, default=100)
    parser.add_argument('-adaptive_weighting', default=False, type=bool,)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100', 'FGVCAircraft', 'iNF200'])
    
    # Method selection
    parser.add_argument('-method', type=str, default='base',
                        choices=['base', 'rainbow'], help='Training method: base or rainbow')
    
    # Rainbow-specific arguments
    parser.add_argument('-rainbow_prompt_length', type=int, default=5,
                        help='Length of Rainbow prompts')
    parser.add_argument('-rainbow_proj_dim', type=int, default=None,
                        help='Projection dimension for Rainbow (default: embed_dim // 8)')
    parser.add_argument('-rainbow_align_hidden_dim', type=int, default=None,
                        help='Hidden dimension for alignment (default: embed_dim // 8)')
    parser.add_argument('-rainbow_gate_tau_start', type=float, default=1.0,
                        help='Starting temperature for Rainbow gate')
    parser.add_argument('-rainbow_gate_tau_end', type=float, default=0.3,
                        help='Ending temperature for Rainbow gate')
    parser.add_argument('-rainbow_gate_harden_at', type=float, default=0.6,
                        help='Epoch ratio to harden Rainbow gate')
    parser.add_argument('-rainbow_lambda_sparse', type=float, default=0.0,
                        help='Sparsity regularization weight for Rainbow')
    parser.add_argument('-rainbow_save_dir', type=str, default='./checkpoint/rainbow_prompts',
                        help='Directory to save Rainbow prompts')
    parser.add_argument('-rainbow_use_paper_evolution', action='store_true',
                        help='Use paper evolution mode for Rainbow')

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    
    # Set default Rainbow dimensions if not specified
    if args.method == 'rainbow':
        if args.rainbow_proj_dim is None:
            # Default to embed_dim // 8 (embed_dim is typically 768 for ViT-Base)
            args.rainbow_proj_dim = 768 // 8
        if args.rainbow_align_hidden_dim is None:
            args.rainbow_align_hidden_dim = 768 // 8
    
    if args.method == 'rainbow':
        from models.rainbow.ViT_fscil_trainer import ViT_RainbowTrainer
        trainer = ViT_RainbowTrainer(args)
    else:
        from models.base.ViT_fscil_trainer import ViT_FSCILTrainer
        trainer = ViT_FSCILTrainer(args)
    
    trainer.train()