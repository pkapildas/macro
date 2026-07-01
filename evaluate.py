"""
MacroAD — Evaluate saved anomaly scores against ground truth labels.

Usage:
    python evaluate.py --data MSL --root_path ../dataset
    python evaluate.py --data MSL --root_path ../dataset --scores_path ./results/MSL/scores.npy
"""
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


def load_labels(data_name, root_path, seq_len=96, batch_size=64):
    """Load test labels from data provider."""
    from data_provider.data_provider import data_provider
    _, test_loader = data_provider(
        root_path=root_path, datasets=data_name,
        batch_size=batch_size, win_size=seq_len, step=seq_len, flag='test'
    )
    labels = []
    for _, label in test_loader:
        labels.append(label.numpy())
    return np.concatenate(labels, axis=0)


def evaluate_scores(scores, labels):
    """Compute all evaluation metrics."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_curve, accuracy_score
    )

    # Flatten if multi-dimensional
    if scores.ndim > 1:
        scores_flat = scores.mean(axis=-1).reshape(-1) if scores.ndim == 2 else scores.reshape(-1)
    else:
        scores_flat = scores

    if labels.ndim > 1:
        labels_flat = labels.mean(axis=-1).reshape(-1) if labels.ndim == 2 else labels.reshape(-1)
    else:
        labels_flat = labels

    # Trim to same length
    min_len = min(len(scores_flat), len(labels_flat))
    scores_flat = scores_flat[:min_len]
    labels_flat = labels_flat[:min_len]

    # Binary labels
    labels_binary = (labels_flat > 0.5).astype(int)

    results = {}

    # AUC-ROC
    try:
        results['AUC_ROC'] = roc_auc_score(labels_binary, scores_flat)
    except:
        results['AUC_ROC'] = 0.0

    # AUC-PR
    try:
        results['AUC_PR'] = average_precision_score(labels_binary, scores_flat)
    except:
        results['AUC_PR'] = 0.0

    # Best F1
    try:
        precisions, recalls, thresholds = precision_recall_curve(labels_binary, scores_flat)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = f1_scores.argmax()
        results['Best_F1'] = f1_scores[best_idx]
        results['Best_Precision'] = precisions[best_idx]
        results['Best_Recall'] = recalls[best_idx]
        results['Best_Threshold'] = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
    except:
        results['Best_F1'] = 0.0
        results['Best_Precision'] = 0.0
        results['Best_Recall'] = 0.0
        results['Best_Threshold'] = 0.0

    # Accuracy at best threshold
    try:
        preds = (scores_flat >= results['Best_Threshold']).astype(int)
        results['Accuracy'] = accuracy_score(labels_binary, preds)
    except:
        results['Accuracy'] = 0.0

    # Additional stats
    results['Anomaly_Ratio'] = labels_binary.mean()
    results['Score_Min'] = scores_flat.min()
    results['Score_Max'] = scores_flat.max()
    results['Score_Mean'] = scores_flat.mean()
    results['N_Samples'] = min_len

    return results


def main():
    parser = argparse.ArgumentParser(description='MacroAD - Evaluate anomaly detection scores')
    parser.add_argument('--data', type=str, default='MSL', help='Dataset name')
    parser.add_argument('--root_path', type=str, default='../dataset', help='Dataset root directory')
    parser.add_argument('--scores_path', type=str, default=None, help='Path to scores.npy (default: ./results/{data}/scores.npy)')
    parser.add_argument('--seq_len', type=int, default=96, help='Window size used during training')
    parser.add_argument('--save_results', action='store_true', help='Save results to CSV')
    args = parser.parse_args()

    # Load scores
    scores_path = args.scores_path or f'./results/{args.data}/scores.npy'
    if not os.path.exists(scores_path):
        print(f"Error: Scores file not found at {scores_path}")
        print("Run training first: python run.py --data MSL --mode train_test")
        sys.exit(1)

    print(f"Loading scores from: {scores_path}")
    scores = np.load(scores_path)
    print(f"Scores shape: {scores.shape}")

    # Load labels
    print(f"Loading labels for {args.data} from: {args.root_path}")
    labels = load_labels(args.data, args.root_path, args.seq_len)
    print(f"Labels shape: {labels.shape}")

    # Evaluate
    print("\n" + "=" * 50)
    print(f"  MacroAD Evaluation Results — {args.data}")
    print("=" * 50)

    results = evaluate_scores(scores, labels)

    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'AUC-ROC':<20} {results['AUC_ROC']:<15.6f}")
    print(f"{'AUC-PR':<20} {results['AUC_PR']:<15.6f}")
    print(f"{'Best-F1':<20} {results['Best_F1']:<15.6f}")
    print(f"{'Precision':<20} {results['Best_Precision']:<15.6f}")
    print(f"{'Recall':<20} {results['Best_Recall']:<15.6f}")
    print(f"{'Accuracy':<20} {results['Accuracy']:<15.6f}")
    print(f"{'Threshold':<20} {results['Best_Threshold']:<15.6f}")
    print("-" * 35)
    print(f"{'Anomaly Ratio':<20} {results['Anomaly_Ratio']:<15.4f}")
    print(f"{'N Samples':<20} {results['N_Samples']:<15d}")
    print(f"{'Score Range':<20} [{results['Score_Min']:.4f}, {results['Score_Max']:.4f}]")

    # Compare with baseline
    baseline = {
        'MSL': {'AUC_ROC': 0.7839, 'Best_F1': 0.3380, 'AUC_PR': 0.2894},
        'SMAP': {'AUC_ROC': 0.5583, 'Best_F1': 0.2730, 'AUC_PR': 0.1336},
        'PSM': {'AUC_ROC': 0.6810, 'Best_F1': 0.4974, 'AUC_PR': 0.4906},
        'SMD': {'AUC_ROC': 0.7823, 'Best_F1': 0.2412, 'AUC_PR': 0.1864},
        'SWAT': {'AUC_ROC': 0.8464, 'Best_F1': 0.6422, 'AUC_PR': 0.5831},
        'SWAN': {'AUC_ROC': 0.6011, 'Best_F1': 0.4906, 'AUC_PR': 0.5023},
        'GECCO': {'AUC_ROC': 0.9867, 'Best_F1': 0.5700, 'AUC_PR': 0.5514},
    }

    if args.data in baseline:
        b = baseline[args.data]
        print(f"\n{'='*50}")
        print(f"  Comparison with CrossAD Baseline")
        print(f"{'='*50}")
        print(f"\n{'Metric':<20} {'Baseline':<12} {'MacroAD':<12} {'Diff':<12}")
        print("-" * 55)
        for metric in ['AUC_ROC', 'AUC_PR', 'Best_F1']:
            diff = results[metric] - b[metric]
            sign = '+' if diff >= 0 else ''
            print(f"{metric:<20} {b[metric]:<12.4f} {results[metric]:<12.4f} {sign}{diff:<12.4f}")

    # Save results
    if args.save_results:
        import pandas as pd
        results_dir = f'./results/{args.data}'
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame([results])
        csv_path = f'{results_dir}/metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")


if __name__ == '__main__':
    main()
