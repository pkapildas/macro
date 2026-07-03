"""
Training and evaluation pipeline for MacroAD.
"""
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
import time
import math
import json


class Configs:
    def __init__(self, json_path):
        with open(json_path) as f:
            self.__dict__.update(json.load(f))


class EarlyStopping:
    def __init__(self, patience=15, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        # Reject NaN/Inf — never save a broken model
        if math.isnan(val_loss) or math.isinf(val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            self.save_checkpoint(model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, path):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        if self.verbose:
            print('Saving model ...')


class Trainer:
    def __init__(self, model, device, configs, save_path='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.configs = configs
        self.save_path = save_path
        self.train_mean = None
        self.train_std = None

    def train(self, train_loader, vali_loader):
        configs = self.configs
        optimizer = optim.AdamW(self.model.parameters(), lr=configs.learning_rate)
        early_stopping = EarlyStopping(patience=configs.patience, verbose=True)
        grad_clip = getattr(configs, 'grad_clip', 1.0)
        warmup_epochs = getattr(configs, 'warmup_epochs', 0)

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / max(configs.train_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) if warmup_epochs > 0 else None

        for epoch in range(configs.train_epochs):
            self.model.train()
            train_loss = []
            epoch_time = time.time()

            for i, (batch_x, _) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)

                loss, q_dist, _, _ = self.model(batch_x)
                total_loss = loss + 0.1 * q_dist

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                train_loss.append(total_loss.item())
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                optimizer.step()

            train_loss_avg = np.mean(train_loss) if train_loss else float('nan')
            vali_loss = self._validate(vali_loader)

            print(f"Epoch {epoch+1}/{configs.train_epochs} | "
                  f"Train: {train_loss_avg:.7f} | Vali: {vali_loss:.7f} | "
                  f"Time: {time.time()-epoch_time:.1f}s")

            if math.isnan(train_loss_avg):
                print("NaN in training — stopping")
                break

            # If validation is NaN, reload best checkpoint and continue
            if math.isnan(vali_loss):
                nan_count = getattr(self, '_nan_count', 0) + 1
                self._nan_count = nan_count
                print(f"  (Vali NaN #{nan_count} — reloading best checkpoint)")
                if nan_count >= 5:
                    print("  Too many NaN epochs — stopping")
                    break
                # Reload last good checkpoint to reset model state
                ckpt = os.path.join(self.save_path, 'checkpoint.pth')
                if os.path.exists(ckpt):
                    self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
                continue

            early_stopping(vali_loss, self.model, self.save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if scheduler is not None:
                scheduler.step()

        # Load best model
        ckpt = os.path.join(self.save_path, 'checkpoint.pth')
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))

    def _validate(self, vali_loader):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                try:
                    loss, _, _, _ = self.model(batch_x)
                    val = loss.item()
                    if not (math.isnan(val) or math.isinf(val)):
                        total_loss.append(val)
                except RuntimeError:
                    continue
        return np.mean(total_loss) if total_loss else float('nan')

    def calibrate(self, train_loader):
        """Collect per-channel reconstruction error quantiles from training data."""
        self.model.eval()
        errors_list = []
        with torch.no_grad():
            for batch_x, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                recon_errors, _ = self.model.infer(batch_x)
                errors_list.append(recon_errors.cpu().numpy())

        errors = np.concatenate(errors_list, axis=0)  # [N, T, C]
        # Flatten time into samples: [N*T, C]
        errors_flat = errors.reshape(-1, errors.shape[-1])
        # Store quantile thresholds per channel for ranking
        self.train_q50 = np.percentile(errors_flat, 50, axis=0)  # [C]
        self.train_q95 = np.percentile(errors_flat, 95, axis=0)  # [C]
        self.train_q05 = np.percentile(errors_flat, 5, axis=0)   # [C]
        self.train_iqr = self.train_q95 - self.train_q05 + 1e-8  # [C]

    def test(self, test_loader):
        self.model.eval()
        scores_list = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.float().to(self.device)
                recon_scores, _ = self.model.infer(batch_x)
                scores_list.append(recon_scores.cpu().numpy())

        scores = np.concatenate(scores_list, axis=0)  # [N, T, C]

        if self.train_q50 is not None:
            # Robust deviation: how far from training median, scaled by IQR
            deviation_high = (scores - self.train_q50) / self.train_iqr
            deviation_low = (self.train_q50 - scores) / self.train_iqr

            # Clip to bounded range
            deviation_high = np.clip(deviation_high, 0, 5)
            deviation_low = np.clip(deviation_low, 0, 5)

            # Anomaly = unusual in either direction
            z_scores = np.maximum(deviation_high, deviation_low)
        else:
            z_scores = scores

        # Mean across channels
        z_scores = np.mean(z_scores, axis=-1)
        return z_scores
