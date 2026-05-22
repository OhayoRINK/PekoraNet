import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from models.pekora_encoder import PekoraEncoder
from models.pekora_decoder import PekoraDecoder
from utils.dataset import COST2100Dataset
from utils.metrics import nmse_db, cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch-size", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--scenario", type=str, default="indoor",
                    choices=["indoor", "outdoor", "joint"])
parser.add_argument("--cr", type=str, default="1/4",
                    choices=["1/4", "1/16", "1/32", "1/64"])
parser.add_argument("--amp", action="store_true")
parser.add_argument("--save-every", type=int, default=50,
                    help="Save snapshot every N epochs")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CR_MAP = {"1/4": 512, "1/16": 128, "1/32": 64, "1/64": 32}
latent_dim = CR_MAP[args.cr]
tag = args.cr.replace("/", "_")

ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

print("=" * 60)
print(f"Device      : {DEVICE}")
print(f"Scenario    : {args.scenario}")
print(f"CR Setting  : {args.cr}  (latent_dim={latent_dim})")
print(f"Epochs      : {args.epochs}")
print(f"Batch Size  : {args.batch_size}")
print(f"AMP         : {args.amp}")
print(f"Save Every  : {args.save_every} epoch(s)")
print(f"Checkpoint  : {ckpt_dir}/")
print("=" * 60)

encoder = PekoraEncoder(latent_dim=latent_dim).to(DEVICE)
decoder = PekoraDecoder(latent_dim=latent_dim).to(DEVICE)

total_p = sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
print(f"[INFO] Total params: {total_p:,}")

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=args.lr, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-5
)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

if args.scenario == "indoor":
    train_dataset = COST2100Dataset("./data/DATA_Htrainin.mat")
    val_dataset   = COST2100Dataset("./data/DATA_Htestin.mat")
elif args.scenario == "outdoor":
    train_dataset = COST2100Dataset("./data/DATA_Htrainout.mat")
    val_dataset   = COST2100Dataset("./data/DATA_Htestout.mat")
else:
    train_dataset = ConcatDataset([
        COST2100Dataset("./data/DATA_Htrainin.mat"),
        COST2100Dataset("./data/DATA_Htrainout.mat"),
    ])
    val_dataset = ConcatDataset([
        COST2100Dataset("./data/DATA_Htestin.mat"),
        COST2100Dataset("./data/DATA_Htestout.mat"),
    ])

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=False
)

def validate(enc, dec):
    enc.eval()
    dec.eval()
    gt_all, pred_all = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            latent = enc(batch)["latent"]
            recon = dec(latent)
            gt_all.append(batch.cpu().numpy())
            pred_all.append(recon.cpu().numpy())

    gt = np.concatenate(gt_all)
    pred = np.concatenate(pred_all)
    return nmse_db(gt, pred), cosine_similarity(gt, pred)

best_val_nmse = float("inf")
best_epoch = 0
train_start = time.time()

for epoch in range(args.epochs):
    epoch_start = time.time()
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=args.amp):
            latent = encoder(batch)["latent"]
            recon = decoder(latent)
            loss = criterion(recon, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    lr_now = optimizer.param_groups[0]["lr"]

    val_nmse, val_rho = validate(encoder, decoder)
    epoch_time = time.time() - epoch_start

    print(
        f"[Epoch {epoch+1:4d}/{args.epochs}] "
        f"loss={avg_loss:.5f} | "
        f"val NMSE={val_nmse:.2f}dB  "
        f"ρ={val_rho:.4f} | "
        f"lr={lr_now:.2e} | "
        f"{epoch_time:.1f}s"
    )

    if val_nmse < best_val_nmse:
        best_val_nmse = val_nmse
        best_epoch = epoch + 1
        torch.save(encoder.state_dict(), f"{ckpt_dir}/encoder_{tag}_best.pth")
        torch.save(decoder.state_dict(), f"{ckpt_dir}/decoder_{tag}_best.pth")
        print(f"  → Best 모델 저장 (NMSE={best_val_nmse:.2f}dB)")

    if (epoch + 1) % args.save_every == 0:
        torch.save(encoder.state_dict(), f"{ckpt_dir}/encoder_{tag}_epoch{epoch+1}.pth")
        torch.save(decoder.state_dict(), f"{ckpt_dir}/decoder_{tag}_epoch{epoch+1}.pth")
        print(f"  → Epoch {epoch+1} 스냅샷 저장")

torch.save(encoder.state_dict(), f"{ckpt_dir}/encoder_{tag}_last.pth")
torch.save(decoder.state_dict(), f"{ckpt_dir}/decoder_{tag}_last.pth")

total_time = time.time() - train_start

print("\n" + "=" * 60)
print("Training complete.")
print(f"Best epoch      : {best_epoch}")
print(f"Best val NMSE   : {best_val_nmse:.2f} dB")
print(f"Last checkpoint : {ckpt_dir}/encoder_{tag}_last.pth")
print(f"Total time      : {total_time / 60:.1f} min")
print("=" * 60)