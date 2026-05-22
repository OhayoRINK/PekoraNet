import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.pekora_encoder import PekoraEncoder
from models.pekora_decoder import PekoraDecoder
from utils.dataset import COST2100Dataset
from utils.metrics import nmse_db, cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="indoor",
                    choices=["indoor", "outdoor"])
parser.add_argument("--cr", type=str, default="1/4",
                    choices=["1/4", "1/16", "1/32", "1/64"])
parser.add_argument("--ckpt", type=str, default="best",
                    help="best, last, or epoch50 style suffix")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CR_MAP = {"1/4": 512, "1/16": 128, "1/32": 64, "1/64": 32}
latent_dim = CR_MAP[args.cr]
tag = args.cr.replace("/", "_")
ckpt_dir = "./checkpoints"

encoder = PekoraEncoder(latent_dim=latent_dim).to(DEVICE)
decoder = PekoraDecoder(latent_dim=latent_dim).to(DEVICE)

encoder.load_state_dict(
    torch.load(f"{ckpt_dir}/encoder_{tag}_{args.ckpt}.pth", map_location=DEVICE)
)
decoder.load_state_dict(
    torch.load(f"{ckpt_dir}/decoder_{tag}_{args.ckpt}.pth", map_location=DEVICE)
)

encoder.eval()
decoder.eval()

suffix = "in" if args.scenario == "indoor" else "out"
dataset = COST2100Dataset(f"./data/DATA_Htest{suffix}.mat")
loader = DataLoader(dataset, batch_size=200, num_workers=0)

gt_all, pred_all = [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        latent = encoder(batch)["latent"]
        recon = decoder(latent)
        gt_all.append(batch.cpu().numpy())
        pred_all.append(recon.cpu().numpy())

gt = np.concatenate(gt_all)
pred = np.concatenate(pred_all)

print("=" * 60)
print(f"Scenario   : {args.scenario}")
print(f"CR         : {args.cr}  (latent_dim={latent_dim})")
print(f"Checkpoint : {ckpt_dir}/encoder_{tag}_{args.ckpt}.pth")
print(f"NMSE(dB)   : {nmse_db(gt, pred):.3f}")
print(f"Rho        : {cosine_similarity(gt, pred):.5f}")
print("=" * 60)