import torch
import numpy as np


def nmse_db_torch(recon, target):
    # CsiNet 논문 기준: DC(0.5) 제거 후 NMSE
    target_c = target - 0.5
    error_c  = recon - target

    mse   = torch.sum(error_c  ** 2, dim=(1, 2, 3))
    power = torch.sum(target_c ** 2, dim=(1, 2, 3))
    nmse  = torch.mean(mse / (power + 1e-8))
    return 10.0 * torch.log10(nmse + 1e-8)


def nmse_db(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_c  = gt - 0.5
    err_c = pred - gt

    mse   = np.sum(err_c ** 2,  axis=(1, 2, 3))
    power = np.sum(gt_c  ** 2,  axis=(1, 2, 3))
    nmse  = np.mean(mse / (power + 1e-8))
    return float(10.0 * np.log10(nmse + 1e-8))


def cosine_similarity(gt: np.ndarray, pred: np.ndarray) -> float:
    # DC(0.5) 제거 후 계산 (CsiNet 논문 기준)
    g = (gt   - 0.5).reshape(gt.shape[0], -1)
    p = (pred - 0.5).reshape(pred.shape[0], -1)
    return float(np.mean(
        np.sum(g * p, axis=1) /
        (np.linalg.norm(g, axis=1) * np.linalg.norm(p, axis=1) + 1e-8)
    ))