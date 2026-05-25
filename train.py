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
parser.add_argument("--save-every", type=int, default=50)
parser.add_argument("--resume-path", type=str, default="")
parser.add_argument("--scheduler", type=str, default="none",
                    choices=["none", "cosine", "step", "plateau"])
parser.add_argument("--lr-min", type=float, default=1e-5)
parser.add_argument("--step-size", type=int, default=100)
parser.add_argument("--step-gamma", type=float, default=0.5)
parser.add_argument("--plateau-patience", type=int, default=20)
parser.add_argument("--plateau-factor", type=float, default=0.5)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CR_MAP = {"1/4": 512, "1/16": 128, "1/32": 64, "1/64": 32}
latent_dim = CR_MAP[args.cr]
tag = args.cr.replace("/", "_")

# ✅ 체크포인트 폴더: ./checkpoints/{scenario}_{tag}_{scheduler}
sched_suffix = f"_{args.scheduler}" if args.scheduler != "none" else ""
ckpt_dir = os.path.join("./checkpoints", f"{args.scenario}_{tag}{sched_suffix}")
log_dir  = "./logs"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
log_path  = os.path.join(log_dir, f"train_{args.scenario}_{tag}{sched_suffix}_{timestamp}.log")

def log(msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log("=" * 60)
log(f"Device      : {DEVICE}")
log(f"Scenario    : {args.scenario}")
log(f"CR Setting  : {args.cr} (latent_dim={latent_dim})")
log(f"Epochs      : {args.epochs}")
log(f"Batch Size  : {args.batch_size}")
log(f"AMP         : {args.amp}")
log(f"Scheduler   : {args.scheduler}")
log(f"Checkpoint  : {ckpt_dir}/")
log(f"Log File    : {log_path}")
resume_str = args.resume_path if args.resume_path else "(none)"
log(f"Resume Path : {resume_str}")
log("=" * 60)

encoder = PekoraEncoder(latent_dim=latent_dim).to(DEVICE)
decoder = PekoraDecoder(latent_dim=latent_dim).to(DEVICE)

total_p = sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
log(f"[INFO] Total params: {total_p:,}")

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=args.lr, weight_decay=1e-4
)

if args.scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )
    log(f"[INFO] Scheduler: CosineAnnealingLR (T_max={args.epochs}, eta_min={args.lr_min})")
elif args.scheduler == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.step_gamma
    )
    log(f"[INFO] Scheduler: StepLR (step_size={args.step_size}, gamma={args.step_gamma})")
elif args.scheduler == "plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.plateau_patience,
        factor=args.plateau_factor, min_lr=args.lr_min
    )
    log(f"[INFO] Scheduler: ReduceLROnPlateau (patience={args.plateau_patience}, factor={args.plateau_factor})")
else:
    scheduler = None
    log("[INFO] Scheduler: none (고정 lr)")

criterion = nn.MSELoss()
scaler    = torch.amp.GradScaler("cuda", enabled=args.amp)

# ----- Dataset -----
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

log(f"[INFO] Train samples: {len(train_dataset)}")
log(f"[INFO] Val samples  : {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                          shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

def validate(enc, dec):
    enc.eval(); dec.eval()
    gt_all, pred_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch  = batch.to(DEVICE)
            latent = enc(batch)["latent"]
            recon  = dec(latent)
            gt_all.append(batch.cpu().numpy())
            pred_all.append(recon.cpu().numpy())
    gt   = np.concatenate(gt_all, axis=0)
    pred = np.concatenate(pred_all, axis=0)
    return nmse_db(gt, pred), cosine_similarity(gt, pred)

# ----- Resume -----
start_epoch   = 0
best_val_nmse = float("inf")
best_epoch    = 0

if args.resume_path:
    ckpt = torch.load(args.resume_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch   = ckpt.get("epoch", 0)
    best_val_nmse = ckpt.get("best_val_nmse", float("inf"))
    best_epoch    = ckpt.get("best_epoch", 0)
    log(f"[INFO] Resumed from {args.resume_path} (epoch={start_epoch}, best={best_val_nmse:.2f}dB at {best_epoch})")

train_start = time.time()

for epoch in range(start_epoch, args.epochs):
    epoch_start = time.time()
    encoder.train(); decoder.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp):
            latent = encoder(batch)["latent"]
            recon  = decoder(latent)
            loss   = criterion(recon, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    lr_now   = optimizer.param_groups[0]["lr"]
    val_nmse, val_rho = validate(encoder, decoder)

    if scheduler is not None:
        if args.scheduler == "plateau":
            scheduler.step(val_nmse)
        else:
            scheduler.step()

    epoch_time = time.time() - epoch_start
    log(
        f"[Epoch {epoch+1:4d}/{args.epochs}] "
        f"loss={avg_loss:.5f} | "
        f"val NMSE={val_nmse:.2f}dB "
        f"ρ={val_rho:.4f} | "
        f"lr={lr_now:.2e} | "
        f"{epoch_time:.1f}s"
    )

    def make_ckpt():
        d = {
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_nmse": best_val_nmse,
            "best_epoch": best_epoch,
        }
        if scheduler is not None:
            d["scheduler"] = scheduler.state_dict()
        return d

    if val_nmse < best_val_nmse:
        best_val_nmse = val_nmse
        best_epoch    = epoch + 1
        torch.save(make_ckpt(), os.path.join(ckpt_dir, "best.pth"))
        log(f"  → Best 모델 저장 (NMSE={best_val_nmse:.2f}dB)")

    if (epoch + 1) % args.save_every == 0:
        torch.save(make_ckpt(), os.path.join(ckpt_dir, f"epoch{epoch+1}.pth"))
        log(f"  → Epoch {epoch+1} 스냅샷 저장")

torch.save(make_ckpt(), os.path.join(ckpt_dir, "last.pth"))
log("  → Last 모델 저장")

total_time = time.time() - train_start
log("\n" + "=" * 60)
log("Training complete.")
log(f"Best epoch     : {best_epoch}")
log(f"Best val NMSE  : {best_val_nmse:.2f} dB")
log(f"Checkpoint dir : {ckpt_dir}/")
log(f"Total time     : {total_time / 60:.1f} min")
log("=" * 60)
