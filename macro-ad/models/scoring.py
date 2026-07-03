"""
Anomaly scoring modules for MacroAD.
Scoring is now handled via calibration-based z-scores in the trainer.
This module is kept minimal for potential future use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
