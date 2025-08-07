import argparse
import torch
from train import train_model_camelyon, train_model_tcga
import random
import numpy as np
import torch.multiprocessing as mp


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train MMMIL Model')

    # Model and data paths
    parser.add_argument('--weight_attention_checkpoint', type=str,
                        default='F:/MM_First/used_checkpoint/new_checkpoint.pth')
    parser.add_argument('--h5_path', type=str, default='E:/h5_files/')
    parser.add_argument('--pt_path', type=str, default='H:/TCGA-BRCA R50/pt_files/')
    parser.add_argument('--t5model_path', type=str, default='F:/MM_First/used_checkpoint/T5/')

    # Hyperparameters
    parser.add_argument('--task_target', type=str, default='tcga', help='choose task (camelyon or tcga)')
    parser.add_argument('--subtyping_task', type=str, default='brca', help='choose brca or nsclc')
    parser.add_argument('--batch_size_slide', type=int, default=1)
    parser.add_argument('--batch_size_patient', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--num_positive', type=int, default=5)
    parser.add_argument('--log_path', type=str, default='result')
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)

    return parser.parse_args()


def resolve_paths(args):
    if args.task_target == 'camelyon':
        args.csv_path = 'data_splits_stage_folds'
        args.text_data_path = 'text/camelyon17.json'
        args.train_model = train_model_camelyon

    elif args.task_target == 'tcga' and args.subtyping_task == 'nsclc':
        args.csv_path = 'data_splits_tcga_nsclc_folds'
        args.text_data_path = 'text/tcga.json'
        args.train_model = train_model_tcga

    elif args.task_target == 'tcga' and args.subtyping_task == 'brca':
        args.csv_path = 'data_splits_tcga_brca_folds'
        args.text_data_path = 'text/tcga.json'
        args.train_model = train_model_tcga
    else:
        raise ValueError("Unknown task_target/subtyping_task combination")

    return args


def main():
    args = parse_args()
    args = resolve_paths(args)
    set_seed(2025)
    world_size = torch.cuda.device_count()
    mp.spawn(args.train_model, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
