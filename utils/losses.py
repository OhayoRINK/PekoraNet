import torch.nn as nn


# [FIX H] ProgressiveOrderingLoss 완전 제거
# - ordering loss가 reconstruction loss와 목표 충돌
# - latent 벡터 크기 정렬 강제 → gradient 흐름 불안정
# - 기본 reconstruction이 안정화된 후 필요 시 재도입
# 현재는 순수 MSELoss만 사용

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, recon, target):
        return self.mse(recon, target)
