import torch.nn as nn

class PekoraEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2,  16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, latent_dim),
        )
    def forward(self, x):
        return {"latent": self.bottleneck(self.features(x))}