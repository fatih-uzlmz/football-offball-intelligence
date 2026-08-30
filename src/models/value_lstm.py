"""Value model V1: LSTM over per-frame attacking-state features.

Input:  [B, T=75, F=11]  (3s window of game state)
Output: P(shot by possession team within 5s after window end)
"""
import torch
import torch.nn as nn


class ValueLSTM(nn.Module):
    def __init__(self, n_features=11, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)
