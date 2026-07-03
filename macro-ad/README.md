# MacroAD — Multi-scale Anomaly Detection with Cross-scale Reconstruction and Adaptive Decomposition

A modularized, clean implementation of the CrossAD-v2 architecture for time series anomaly detection.

## Architecture Overview

```
Input [B, T, C]
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Normalization (AdaIN / RevIN / None)           │  ← models/normalization.py
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Learnable Wavelet Decomposition                │  ← models/decomposition.py
│  (4 scales: k=16,8,4,2, low-pass + high-pass)  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Patch Embedding + Positional Encoding          │  ← models/embedding.py
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Mamba Encoder (×2 layers)                      │  ← models/encoder.py
│  (Mamba-1 sequential scan or Mamba-2 multi-head)│
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Temporal Graph Attention (multi-hop)           │  ← models/graph.py
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Context Memory Network                         │  ← models/memory.py
│  (Router → Query Library → Extractor → EMA Bank)│
│  Optional: 3-tier Hierarchical Memory           │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Mamba Decoder (×2 layers)                      │  ← models/decoder.py
│  (Mamba self-mixing + Cross-Attention + FFN)    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Anomaly Scoring                                │  ← models/scoring.py
│  (Scale Fusion / Distribution-Aware)            │
└─────────────────────────────────────────────────┘
    │
    ▼
Output: Anomaly Score [B, T, C]
```

## File Structure

```
macro-ad/
├── README.md                   ← This file
├── run.py                      ← Entry point (train/test/train_test)
├── configs/
│   └── default.json            ← Default model + training configuration
├── models/
│   ├── __init__.py             ← Module exports
│   ├── macro_ad.py             ← Main model class (composes all modules)
│   ├── normalization.py        ← AdaIN, RevIN
│   ├── decomposition.py        ← LearnableWaveletDecomposition
│   ├── embedding.py            ← PatchEmbedding, PositionalEmbedding
│   ├── encoder.py              ← MambaBlock, Mamba2Block, MambaEncoder
│   ├── decoder.py              ← MambaDecoder, DecoderLayer
│   ├── graph.py                ← TemporalGraphAttention
│   ├── memory.py               ← Router, MemoryTier, HierarchicalMemory, ContextNet
│   └── scoring.py              ← ScaleAttentionFusion, DistributionAwareScoring
├── exp/
│   ├── __init__.py
│   └── trainer.py              ← Training loop, validation, testing, early stopping
├── data_provider/
│   └── __init__.py             ← Uses parent project's data_provider
├── evaluation/
│   └── __init__.py             ← Evaluation utilities
├── checkpoints/                ← Saved model weights (created at runtime)
└── results/                    ← Output anomaly scores (created at runtime)
```

## Quick Start

### 1. Install Requirements

```bash
pip install torch numpy pandas scikit-learn tqdm
```

### 2. Prepare Data

Place datasets in `../dataset/` (relative to this folder):
```
../dataset/
├── DETECT_META.csv
└── data/
    ├── MSL/
    ├── SMAP/
    ├── PSM/
    └── ...
```

### 3. Train

```bash
# Train on MSL with default config
python run.py --data MSL --mode train

# Train on specific dataset with custom config
python run.py --config configs/default.json --data SWAT --mode train

# Specify GPU type
python run.py --data MSL --mode train --gpu cuda
python run.py --data MSL --mode train --gpu mps
python run.py --data MSL --mode train --gpu cpu

====== Advanced Usage ======
# MPS-safe (default config — works on Mac)
  python run.py --data MSL --mode train_test --gpu mps --root_path ../dataset

  # CUDA with full model (all 7 improvements)
  python run.py --config configs/cuda_full.json --data MSL --mode train_test --gpu cuda --root_path ../dataset

  Quick Check: Do you have CUDA?

  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
  
  ===== Evironment Setup =====
    Option 1: Google Colab (Free GPU)

  # In a Colab notebook:
  !git clone <your-repo-url>
  %cd CrossAD/macro-ad
  !pip install torch numpy pandas scikit-learn tqdm

  # Upload dataset or mount from Google Drive
  # then run:
  !python run.py --data MSL --mode train_test --gpu cuda --root_path ../dataset

  Option 2: Remote Server / Cloud (AWS, GCP, Lambda Labs)

  # SSH into your GPU server
  ssh user@gpu-server

  # Clone and setup
  git clone <your-repo-url>
  cd CrossAD/macro-ad
  pip install torch numpy pandas scikit-learn tqdm
  
  # Run all datasets on CUDA
  for dataset in MSL SMAP PSM SMD SWAT SWAN GECCO; do
      python run.py --config configs/cuda_full.json --data $dataset --mode train_test --gpu cuda --root_path ../dataset
=======RUNPOD SETUP=======
Steps on RunPod:

  # 1. Clone your repo
  git clone <your-repo-url>
  cd CrossAD/macro-ad

  # 2. Install deps (PyTorch already in template)
  pip install pandas scikit-learn tqdm

  # 3. Upload dataset (or download from Google Drive)
  # Option A: Upload via RunPod UI
  # Option B: gdown from Google Drive
  pip install gdown
  gdown https://drive.google.com/uc?id=1YU_d9kIaP2EubyUhGWOwSKAhJxNntB8W
  unzip -q dataset.zip -d ../dataset

  # 4. Run full model on CUDA
  python run.py --config configs/cuda_full.json --data MSL --mode train_test --gpu cuda --root_path ../dataset

  # 5. Run all datasets
  for dataset in MSL SMAP PSM SMD SWAT SWAN GECCO; do
      echo "=== $dataset ==="
      python run.py --config configs/cuda_full.json --data $dataset --mode train_test --gpu cuda --root_path ../dataset
  done
```

### 4. Test

```bash
# Test with saved checkpoint
python run.py --data MSL --mode test

# Train + Test in one run
python run.py --data MSL --mode train_test

 python run.py --data MSL --mode train_test --root_path ../dataset
```

### 5. Custom Config

Edit `configs/default.json` or create a new config:

```bash
python run.py --config configs/my_config.json --data MSL --mode train_test
```

## Configuration Options

### Core Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 96 | Input window size |
| `patch_len` | 6 | Patch tokenization length |
| `d_model` | 128 | Model embedding dimension |
| `e_layers` | 2 | Encoder layers |
| `d_layers` | 2 | Decoder layers |
| `n_heads` | 4 | Attention heads |
| `ms_kernels` | [16,8,4,2] | Multi-scale decomposition kernels |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_adain` | false | Frequency-conditioned normalization |
| `use_revin` | false | Standard instance normalization |
| `ms_use_detail` | true | High-pass detail coefficients |
| `use_mamba2` | false | Multi-head Mamba (needs CUDA) |
| `use_gnn` | true | Temporal graph attention |
| `use_hier_memory` | false | 3-tier hierarchical memory (needs CUDA) |
| `use_dual_decoder` | false | Prediction path + temporal loss |
| `use_dist_scoring` | false | Mahalanobis + spectral scoring |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_epochs` | 100 | Maximum training epochs |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 2e-4 | AdamW learning rate |
| `patience` | 15 | Early stopping patience |
| `grad_clip` | 1.0 | Gradient clipping max norm |

## Hardware Compatibility

| Component | MPS (Apple) | CUDA | CPU |
|-----------|:-----------:|:----:|:---:|
| Mamba-1 Encoder | Stable | Stable | Stable |
| Mamba-2 Multi-head | Unstable (NaN ~epoch 27) | Stable | Slow |
| T-GAT | Stable | Stable | Stable |
| Hierarchical Memory | Unstable (NaN ~epoch 31) | Stable | Stable |
| Learnable Wavelet | Stable | Stable | Stable |
| AdaIN | Stable | Stable | Stable |
| Dual Decoder | Unstable | Stable | Stable |
| Distribution Scoring | Unstable | Stable | Stable |

**Recommended for MPS:** `use_mamba2=false, use_hier_memory=false, use_dual_decoder=false, use_dist_scoring=false`

**Recommended for CUDA:** All features enabled

## Module Descriptions

### normalization.py
- **RevIN**: Removes distribution info before model, restores after. Simple but erases anomaly-relevant shifts.
- **AdaIN**: FFT-conditioned normalization that preserves frequency-dependent distribution information.

### decomposition.py
- **LearnableWaveletDecomposition**: Replaces fixed avg-pooling with learnable conv filters (low-pass + high-pass). Filters adapt per dataset during training.

### encoder.py
- **MambaBlock**: Original Mamba-1 with HiPPO-initialized A, sequential scan, SiLU gating.
- **Mamba2Block**: Multi-head variant with per-head B/C/A projections for representational diversity.
- **MambaEncoder**: Stacks encoder layers, returns encoded features + hidden states for Router.

### decoder.py
- **MambaDecoder**: Mamba self-mixing + cross-attention to global context + FFN. Reconstructs fine scales from coarse.

### graph.py
- **TemporalGraphAttention**: Learns inter-variable adjacency via Q/K projections, then performs multi-hop message passing.

### memory.py
- **Router**: FFT → Top-K freq → MLP + Mamba state → Gumbel-Softmax → selects queries.
- **MemoryTier**: Single EMA bank with soft attention routing.
- **HierarchicalMemory**: 3 tiers (short/medium/long) with learned fusion gate.
- **ContextNet**: Composes Router + Query Library + Extractor into the full context system.

### scoring.py
- **ScaleAttentionFusion**: Learned weighted combination of per-scale anomaly scores.
- **DistributionAwareScoring**: Mahalanobis distance + spectral divergence for enhanced sensitivity.

## Citation

```
@inproceedings{CrossAD,
  title     = {{CrossAD}: Time Series Anomaly Detection with Cross-scale Associations and Cross-window Modeling},
  author    = {Li, Beibu and Shentu, Qichao and Shu, Yang and Zhang, Hui and Li, Ming and Jin, Ning and Yang, Bin and Guo, Chenjuan},
  booktitle = {NeurIPS},
  year      = {2025}
}
```
