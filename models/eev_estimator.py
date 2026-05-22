import torch.nn as nn


class EEVEstimator(nn.Module):
    """EEV 모듈 — 현재 train.py에서 비활성화됨 (FIX E 참고)"""
    def __init__(self, pilot_dim=128, eev_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pilot_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, eev_dim),
        )

    def forward(self, pilots):
        return self.net(pilots)
