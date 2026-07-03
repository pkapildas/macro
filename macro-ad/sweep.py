"""
MacroAD — Multi-dataset sweep for RunPod GPU.

Usage:
    python sweep.py --root_path ../dataset --config configs/gpu_sweep.json
    python sweep.py --root_path ../dataset --config configs/default.json --datasets MSL SMAP
"""
import argparse
import subprocess
import sys
import os
import json
import time


DATASETS = ['MSL', 'SMAP', 'PSM', 'SMD', 'SWAT', 'SWAN', 'GECCO']


def run_one(data, config, root_path, gpu):
    """Train + test + evaluate one dataset."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {data}")
    print(f"{'='*60}")

    start = time.time()

    # Train + Test
    cmd = [
        sys.executable, 'run.py',
        '--config', config,
        '--data', data,
        '--root_path', root_path,
        '--mode', 'train_test',
        '--gpu', gpu
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [FAILED] {data} — run.py returned {result.returncode}")
        return None

    # Evaluate
    cmd_eval = [
        sys.executable, 'evaluate.py',
        '--data', data,
        '--root_path', root_path
    ]
    result = subprocess.run(cmd_eval, capture_output=True, text=True)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s")
    print(result.stdout)

    if result.returncode != 0:
        print(f"  [FAILED] {data} — evaluate.py returned {result.returncode}")
        print(result.stderr)
        return None

    return elapsed


def main():
    parser = argparse.ArgumentParser(description='MacroAD — Multi-dataset sweep')
    parser.add_argument('--config', type=str, default='configs/gpu_sweep.json')
    parser.add_argument('--root_path', type=str, default='../dataset')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--gpu', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    args = parser.parse_args()

    print(f"Config: {args.config}")
    print(f"Datasets: {args.datasets}")
    print(f"GPU: {args.gpu}")

    # Clean checkpoints
    for data in args.datasets:
        ckpt_dir = f'./checkpoints/{data}'
        if os.path.exists(ckpt_dir):
            import shutil
            shutil.rmtree(ckpt_dir)
            print(f"  Cleaned {ckpt_dir}")

    results = {}
    total_start = time.time()

    for data in args.datasets:
        elapsed = run_one(data, args.config, args.root_path, args.gpu)
        results[data] = elapsed

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  SWEEP COMPLETE — Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")
    for data, t in results.items():
        status = f"{t:.1f}s" if t else "FAILED"
        print(f"  {data:<10} {status}")


if __name__ == '__main__':
    main()
