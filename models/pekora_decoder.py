import torch
import torch.nn as nn
from models.blocks import ResidualBlock

class PekoraDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * 8 * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(16,  8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.refine = nn.Sequential(
            ResidualBlock(8),
            nn.Conv2d(8, 2, 3, padding=1),
        )
    def forward(self, latent):
        x = self.fc(latent).view(-1, 32, 8, 8)
        return torch.sigmoid(self.refine(self.up(x)))