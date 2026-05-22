import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset

class COST2100Dataset(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        print("[INFO] MAT keys:", data.keys())
        if   "HT"      in data: H = data["HT"]
        elif "H_train" in data: H = data["H_train"]
        elif "H_test"  in data: H = data["H_test"]
        else: raise ValueError("Cannot find CSI key")
        H = H.astype(np.float32)
        if H.ndim == 2 and H.shape[1] == 2048:
            H = np.transpose(H.reshape(-1, 32, 32, 2), (0, 3, 1, 2))
        elif H.ndim == 4 and H.shape[-1] == 2:
            H = np.transpose(H, (0, 3, 1, 2))
        self.data = torch.tensor(H)
        print(f"[INFO] shape={self.data.shape}  min={self.data.min():.4f}"
              f"  max={self.data.max():.4f}  mean={self.data.mean():.4f}"
              f"  std={self.data.std():.6f}")
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]