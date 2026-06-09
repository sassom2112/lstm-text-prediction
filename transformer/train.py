"""
SageMaker entry point for GPT-nano training on WikiText-2.

SageMaker conventions used:
  - Data:   $SM_CHANNEL_TRAINING  → /opt/ml/input/data/training/
  - Output: $SM_MODEL_DIR         → /opt/ml/model/
  - Hyperparams via argparse
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import GPTNano


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    # Non-overlapping chunks so each epoch sees all data once without redundancy.
    def __init__(self, tokens: np.ndarray, context_len: int):
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.context_len = context_len
        self.starts = list(range(0, len(self.tokens) - context_len, context_len))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        chunk = self.tokens[s : s + self.context_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = args.training
    train_tokens = np.load(os.path.join(data_dir, "train.npy"))
    val_tokens   = np.load(os.path.join(data_dir, "val.npy"))
    print(f"Train tokens: {len(train_tokens):,}  Val tokens: {len(val_tokens):,}")

    train_ds = TokenDataset(train_tokens, args.context_len)
    val_ds   = TokenDataset(val_tokens,   args.context_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = GPTNano(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_len=args.context_len,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {model.n_params():,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    total_steps = args.epochs * len(train_dl)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr / 10)

    best_val_ppl = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            step += 1
            if (i + 1) % 50 == 0:
                avg = running_loss / 50
                ppl = math.exp(min(avg, 20))
                elapsed = time.time() - t0
                print(f"  epoch {epoch} step {i+1}/{len(train_dl)}  loss={avg:.4f}  ppl={ppl:.1f}  lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")
                running_loss = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
        val_loss /= len(val_dl)
        val_ppl = math.exp(min(val_loss, 20))
        print(f"Epoch {epoch}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))
            print(f"  ✓ saved best model (ppl={best_val_ppl:.1f})")

    # Save config alongside weights
    config = {
        "vocab_size": args.vocab_size,
        "embed_dim":  args.embed_dim,
        "n_heads":    args.n_heads,
        "n_layers":   args.n_layers,
        "context_len": args.context_len,
        "ff_dim":     args.ff_dim,
        "dropout":    0.0,          # disable at inference time
        "encoding":   "gpt2",       # tiktoken encoding name
        "best_val_ppl": round(best_val_ppl, 2),
    }
    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training complete. Best val ppl: {best_val_ppl:.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker-injected channel paths
    parser.add_argument("--training",  default=os.environ.get("SM_CHANNEL_TRAINING", "data/"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR",        "output/"))

    # Hyperparameters
    parser.add_argument("--vocab-size",  type=int,   default=50257)
    parser.add_argument("--embed-dim",   type=int,   default=128)
    parser.add_argument("--n-heads",     type=int,   default=4)
    parser.add_argument("--n-layers",    type=int,   default=4)
    parser.add_argument("--context-len", type=int,   default=256)
    parser.add_argument("--ff-dim",      type=int,   default=512)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--epochs",      type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=3e-4)

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    train(args)
