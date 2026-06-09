"""
Downloads WikiText-2, tokenizes with tiktoken (GPT-2 BPE), and uploads
train/val/test .npy files to S3.

WikiText-2 stats: ~2.1M train tokens, ~218K val, ~245K test.

Usage:
  pip install tiktoken datasets
  python infra/02_prepare_data.py
"""
import io

import boto3
import numpy as np
import tiktoken
from datasets import load_dataset

BUCKET    = "543458926995-mnist-digit-models"
S3_PREFIX = "lstm-transformer/data"
REGION    = "us-west-2"

enc = tiktoken.get_encoding("gpt2")  # vocab_size = 50,257


def tokenize_texts(texts):
    tokens = []
    for text in texts:
        text = text.strip()
        if text:
            tokens.extend(enc.encode_ordinary(text))
    return np.array(tokens, dtype=np.uint32)


def upload_array(s3, arr, key):
    buf = io.BytesIO()
    np.save(buf, arr)
    size_mb = len(buf.getvalue()) / 1e6
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Uploaded s3://{BUCKET}/{key}  ({len(arr):,} tokens, {size_mb:.1f} MB)")


def main():
    print("Loading WikiText-2 from HuggingFace (Salesforce/wikitext) ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    split_map = {
        "train": ds["train"]["text"],
        "val":   ds["validation"]["text"],
        "test":  ds["test"]["text"],
    }

    s3 = boto3.client("s3", region_name=REGION)

    for split_name, texts in split_map.items():
        print(f"Tokenizing {split_name} ({len(texts):,} lines) ...")
        tokens = tokenize_texts(texts)
        key = f"{S3_PREFIX}/{split_name}.npy"
        upload_array(s3, tokens, key)

    print(f"\nData ready at s3://{BUCKET}/{S3_PREFIX}/")
    print("Next: python infra/03_launch_training.py")


if __name__ == "__main__":
    main()
