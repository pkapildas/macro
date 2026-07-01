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

    def train(self, train_loader, vali_loader):
        configs = self.configs
        optimizer = optim.AdamW(self.model.parameters(), lr=configs.learning_rate)
        early_stopping = EarlyStopping(patience=configs.patience, verbose=True)
        grad_clip = getattr(configs, 'grad_clip', 1.0)

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

            if math.isnan(vali_loss) or math.isnan(train_loss_avg):
                print("NaN detected — stopping training")
                break

            early_stopping(vali_loss, self.model, self.save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'checkpoint.pth')))

    def _validate(self, vali_loader):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                loss, _, _, _ = self.model(batch_x)
                total_loss.append(loss.item())
        return np.mean(total_loss)

    def test(self, test_loader):
        self.model.eval()
        scores_list = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.float().to(self.device)
                scores, _ = self.model.infer(batch_x)
                scores_list.append(scores.cpu().numpy())

        scores = np.concatenate(scores_list, axis=0)
        # Average across channels
        scores = np.mean(scores, axis=-1)
        return scores
