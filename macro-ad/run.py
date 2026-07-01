"""
MacroAD — Entry point for training and evaluation.

Usage:
    python run.py --config configs/default.json --data MSL --mode train
    python run.py --config configs/default.json --data MSL --mode test
    python run.py --config configs/default.json --data MSL --mode train_test
"""
import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MacroAD
from exp.trainer import Trainer, Configs


def get_device(gpu_type='auto'):
    if gpu_type == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda:0')
    elif gpu_type == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    elif gpu_type == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')


def get_data_loaders(data_name, root_path, seq_len, batch_size):
    """Load data using the parent project's data_provider."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from data_provider.data_provider import data_provider

    train_set, train_loader = data_provider(
        root_path=root_path, datasets=data_name,
        batch_size=batch_size, win_size=seq_len, step=1, flag='train'
    )
    _, vali_loader = data_provider(
        root_path=root_path, datasets=data_name,
        batch_size=batch_size, win_size=seq_len, step=seq_len, flag='val'
    )
    _, test_loader = data_provider(
        root_path=root_path, datasets=data_name,
        batch_size=batch_size, win_size=seq_len, step=seq_len, flag='test'
    )
    return train_loader, vali_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='MacroAD: Multi-scale Anomaly Detection')
    parser.add_argument('--config', type=str, default='configs/default.json', help='Path to config JSON')
    parser.add_argument('--data', type=str, default='MSL', help='Dataset name')
    parser.add_argument('--root_path', type=str, default='../dataset', help='Dataset root directory')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'train_test'])
    parser.add_argument('--gpu', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    args = parser.parse_args()

    # Load configs
    configs = Configs(args.config)
    device = get_device(args.gpu)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.data}")

    # Build model
    model = MacroAD(configs)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Setup trainer
    save_path = os.path.join(args.save_path, args.data)
    trainer = Trainer(model, device, configs, save_path)

    # Load data
    train_loader, vali_loader, test_loader = get_data_loaders(
        args.data, args.root_path, configs.seq_len, configs.batch_size
    )

    if args.mode in ['train', 'train_test']:
        print("\n--- Training ---")
        trainer.train(train_loader, vali_loader)

    if args.mode in ['test', 'train_test']:
        print("\n--- Testing ---")
        # Load best checkpoint
        ckpt_path = os.path.join(save_path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded checkpoint: {ckpt_path}")

        scores = trainer.test(test_loader)
        print(f"Anomaly scores shape: {scores.shape}")
        print(f"Score range: [{scores.min():.6f}, {scores.max():.6f}]")

        # Save scores
        os.makedirs(f'./results/{args.data}', exist_ok=True)
        import numpy as np
        np.save(f'./results/{args.data}/scores.npy', scores)
        print(f"Scores saved to ./results/{args.data}/scores.npy")


if __name__ == '__main__':
    main()
