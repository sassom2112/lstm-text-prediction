"""
Train GPT-nano locally (uses whatever GPU/CPU is available).
Downloads data from S3, trains, saves model.pt + config.json, then
uploads the artifact back to S3 so 04_deploy_endpoint.py can use it.

Usage:
  python infra/train_local.py
"""
import io
import json
import math
import os
import sys
import tarfile
import tempfile
import time

import boto3
import numpy as np

BUCKET    = "543458926995-mnist-digit-models"
S3_PREFIX = "lstm-transformer"
REGION    = "us-west-2"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "transformer", "local_output")


def download_data(s3, local_dir):
    for split in ("train", "val"):
        key  = f"{S3_PREFIX}/data/{split}.npy"
        dest = os.path.join(local_dir, f"{split}.npy")
        print(f"Downloading {key} ...")
        s3.download_file(BUCKET, key, dest)


def upload_artifact(s3, model_dir):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for fname in ("model.pt", "config.json"):
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                tf.add(fpath, arcname=fname)
    buf.seek(0)
    key = f"{S3_PREFIX}/local_output/model.tar.gz"
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    uri = f"s3://{BUCKET}/{key}"
    print(f"Uploaded model artifact → {uri}")
    return uri


def main():
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    # Add transformer/ to path for model import
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "transformer"))
    from model import GPTNano

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Hyperparams — tuned for RTX 3060 6GB
    VOCAB_SIZE  = 50257
    EMBED_DIM   = 128
    N_HEADS     = 4
    N_LAYERS    = 4
    CONTEXT_LEN = 256
    FF_DIM      = 512
    DROPOUT     = 0.1
    BATCH_SIZE  = 16 if device == "cuda" else 8
    EPOCHS      = 8
    LR          = 3e-4

    s3 = boto3.client("s3", region_name=REGION)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as data_dir:
        download_data(s3, data_dir)

        class TokenDataset(Dataset):
            # Non-overlapping chunks: stride=CONTEXT_LEN gives ~9k chunks from 2.3M tokens.
            # Sliding window (stride=1) would give 2.3M samples — far too many per epoch.
            def __init__(self, path, stride=None):
                tokens = np.load(path)
                self.tokens = torch.from_numpy(tokens.astype(np.int64))
                stride = stride or CONTEXT_LEN
                self.starts = list(range(0, len(self.tokens) - CONTEXT_LEN, stride))

            def __len__(self):
                return len(self.starts)

            def __getitem__(self, idx):
                s = self.starts[idx]
                chunk = self.tokens[s : s + CONTEXT_LEN + 1]
                return chunk[:-1], chunk[1:]

        train_dl = DataLoader(
            TokenDataset(os.path.join(data_dir, "train.npy")),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=(device == "cuda"),
        )
        val_dl = DataLoader(
            TokenDataset(os.path.join(data_dir, "val.npy")),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        )
        print(f"Train batches: {len(train_dl)}  Val batches: {len(val_dl)}")

        model = GPTNano(
            vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, n_heads=N_HEADS,
            n_layers=N_LAYERS, context_len=CONTEXT_LEN, ff_dim=FF_DIM, dropout=DROPOUT,
        ).to(device)
        print(f"Parameters: {model.n_params():,}")

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
        total_steps = EPOCHS * len(train_dl)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR / 10)

        best_val_ppl = float("inf")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            t0 = time.time()
            running = 0.0
            for i, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running += loss.item()
                if (i + 1) % 50 == 0:
                    avg = running / 100
                    print(f"  e{epoch} step {i+1}/{len(train_dl)}  loss={avg:.4f}  ppl={math.exp(min(avg,20)):.1f}  {time.time()-t0:.0f}s")
                    running = 0.0

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    _, l = model(x, y)
                    val_loss += l.item()
            val_loss /= len(val_dl)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"Epoch {epoch}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
                print(f"  ✓ saved best (ppl={best_val_ppl:.1f})")

    config = {
        "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM, "n_heads": N_HEADS,
        "n_layers": N_LAYERS, "context_len": CONTEXT_LEN, "ff_dim": FF_DIM,
        "dropout": 0.0, "encoding": "gpt2", "best_val_ppl": round(best_val_ppl, 2),
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    artifact_uri = upload_artifact(s3, OUTPUT_DIR)
    print(f"\nBest val ppl: {best_val_ppl:.1f}")
    print(f"\nSet this in infra/04_deploy_endpoint.py:")
    print(f'  MODEL_DATA_URI = "{artifact_uri}"')


if __name__ == "__main__":
    main()
