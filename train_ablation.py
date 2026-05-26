import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from models.pekora_encoder import PekoraEncoder
from utils.dataset import COST2100Dataset
from utils.metrics import nmse_db, cosine_similarity

# ── Ablation B: Decoder 변형 선택
DECODER_MAP = {
    "base":         "models.pekora_decoder",
    "wo_residual":  "models.pekora_decoder_b1_wo_residual",
    "rb_x2":        "models.pekora_decoder_b2_rb_x2",
    "plain_conv":   "models.pekora_decoder_b3_plain_conv",
}

# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",     type=int,   default=1000)
parser.add_argument("--batch-size", type=int,   default=200)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--scenario",   type=str,   default="indoor",
                    choices=["indoor", "outdoor"])
parser.add_argument("--cr",         type=str,   default="1/4",
                    choices=["1/4", "1/16", "1/32", "1/64"])
parser.add_argument("--decoder",    type=str,   default="base",
                    choices=list(DECODER_MAP.keys()),
                    help="Ablation B 변형 선택")
parser.add_argument("--amp",        action="store_true")
parser.add_argument("--save-every", type=int,   default=50)
parser.add_argument("--resume-path",type=str,   default="")
args = parser.parse_args()

# ── decoder 동적 import
import importlib
_dec_module = importlib.import_module(DECODER_MAP[args.decoder])
PekoraDecoder = _dec_module.PekoraDecoder

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CR_MAP    = {"1/4": 512, "1/16": 128, "1/32": 64, "1/64": 32}
latent_dim = CR_MAP[args.cr]
tag        = args.cr.replace("/", "")          # e.g. "14"
dec_tag    = args.decoder                      # e.g. "wo_residual"

ckpt_dir = os.path.join("./checkpoints", f"ablationB_{args.scenario}_{tag}_{dec_tag}")
log_dir  = "./logs"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
log_path  = os.path.join(log_dir, f"ablationB_{args.scenario}_{tag}_{dec_tag}_{timestamp}.log")

def log(msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log("=" * 60)
log(f"[Ablation B] Decoder Refinement")
log(f"Device      : {DEVICE}")
log(f"Scenario    : {args.scenario}")
log(f"CR          : {args.cr}  (latent_dim={latent_dim})")
log(f"Decoder     : {args.decoder}  ({DECODER_MAP[args.decoder]})")
log(f"Epochs      : {args.epochs}")
log(f"Batch Size  : {args.batch_size}")
log(f"AMP         : {args.amp}")
log(f"Checkpoint  : {ckpt_dir}/")
log(f"Log File    : {log_path}")
log(f"Resume Path : {args.resume_path if args.resume_path else '(none)'}")
log("=" * 60)

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
encoder = PekoraEncoder(latent_dim=latent_dim).to(DEVICE)
decoder = PekoraDecoder(latent_dim=latent_dim).to(DEVICE)

total_p = sum(p.numel() for p in list(encoder.parameters()) + list(decoder.parameters()))
log(f"[INFO] Total params: {total_p:,}")

optimizer = torch.optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=args.lr, weight_decay=1e-4
)
criterion = nn.MSELoss()
scaler    = torch.amp.GradScaler("cuda", enabled=args.amp)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
if args.scenario == "indoor":
    train_ds = COST2100Dataset("./data/DATA_Htrainin.mat")
    val_ds   = COST2100Dataset("./data/DATA_Htestin.mat")
else:
    train_ds = COST2100Dataset("./data/DATA_Htrainout.mat")
    val_ds   = COST2100Dataset("./data/DATA_Htestout.mat")

log(f"[INFO] Train samples: {len(train_ds)}")
log(f"[INFO] Val samples  : {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  drop_last=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, drop_last=False, num_workers=0)

# ──────────────────────────────────────────────
# Resume
# ──────────────────────────────────────────────
start_epoch = 1
best_nmse   = float("inf")
best_epoch  = 0

if args.resume_path and os.path.isfile(args.resume_path):
    ckpt = torch.load(args.resume_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_nmse   = ckpt.get("best_nmse", float("inf"))
    best_epoch  = ckpt.get("best_epoch", 0)
    log(f"[INFO] Resumed from {args.resume_path}  "
        f"epoch={start_epoch-1}, best={best_nmse:.2f}dB @ ep{best_epoch}")

# ──────────────────────────────────────────────
# Validate
# ──────────────────────────────────────────────
def validate():
    encoder.eval(); decoder.eval()
    gt_all, pred_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch  = batch.to(DEVICE)
            latent = encoder(batch)["latent"]
            recon  = decoder(latent)
            gt_all.append(batch.cpu().numpy())
            pred_all.append(recon.cpu().numpy())
    gt   = np.concatenate(gt_all,   axis=0)
    pred = np.concatenate(pred_all, axis=0)
    return nmse_db(gt, pred), cosine_similarity(gt, pred)

# ──────────────────────────────────────────────
# Train loop
# ──────────────────────────────────────────────
t_total = time.time()

for epoch in range(start_epoch, args.epochs + 1):
    encoder.train(); decoder.train()
    total_loss = 0.0
    t0 = time.time()

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp):
            latent = encoder(batch)["latent"]
            recon  = decoder(latent)
            loss   = criterion(recon, batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_nmse, val_rho = validate()
    elapsed  = time.time() - t0
    lr_now   = optimizer.param_groups[0]["lr"]

    is_best = val_nmse < best_nmse
    if is_best:
        best_nmse  = val_nmse
        best_epoch = epoch
        torch.save({
            "epoch": epoch, "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
            "best_nmse": best_nmse, "best_epoch": best_epoch,
        }, os.path.join(ckpt_dir, "best.pth"))

    best_marker = "  ★ Best" if is_best else ""
    log(f"[Epoch {epoch:4d}/{args.epochs}] "
        f"loss={avg_loss:.5f} | val NMSE={val_nmse:.2f}dB  ρ={val_rho:.4f} | "
        f"lr={lr_now:.2e} | {elapsed:.1f}s{best_marker}")

    # 주기적 snapshot
    if epoch % args.save_every == 0:
        torch.save({
            "epoch": epoch, "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
            "best_nmse": best_nmse, "best_epoch": best_epoch,
        }, os.path.join(ckpt_dir, f"epoch{epoch}.pth"))
        log(f"  → Snapshot saved (epoch {epoch})")

# ──────────────────────────────────────────────
# 마지막 체크포인트 + 결과 출력
# ──────────────────────────────────────────────
torch.save({
    "epoch": args.epochs, "encoder": encoder.state_dict(),
    "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
    "best_nmse": best_nmse, "best_epoch": best_epoch,
}, os.path.join(ckpt_dir, "last.pth"))

total_min = (time.time() - t_total) / 60
log("=" * 60)
log(f"Training complete.")
log(f"Best epoch      : {best_epoch}")
log(f"Best val NMSE   : {best_nmse:.2f} dB")
log(f"Last checkpoint : {ckpt_dir}/last.pth")
log(f"Total time      : {total_min:.1f} min")
log("=" * 60)
