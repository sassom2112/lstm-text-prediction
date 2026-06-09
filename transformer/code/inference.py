"""
SageMaker inference handler for GPT-nano.

Expected request:
  POST /invocations
  Content-Type: application/json
  {"prompt": "the quick", "temperature": 0.8, "max_length": 20}

Response:
  {"prompt": "the quick", "generated": "the quick brown fox ...", "top_words": [...]}

top_words reflects the model's next-token distribution *after* encoding the prompt,
before any generation — same contract as the LSTM API.
"""
import json
import os

import tiktoken
import torch
import torch.nn.functional as F

# Re-export model class so it is available at serve time
# (SageMaker adds the code/ directory to PYTHONPATH)
from model import GPTNano


_model = None
_enc = None
_device = "cpu"


def model_fn(model_dir):
    global _model, _enc, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    _enc = tiktoken.get_encoding(cfg.get("encoding", "gpt2"))

    _model = GPTNano(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        context_len=cfg["context_len"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg.get("dropout", 0.0),
    ).to(_device)

    state = torch.load(
        os.path.join(model_dir, "model.pt"),
        map_location=_device,
        weights_only=True,
    )
    _model.load_state_dict(state)
    _model.eval()
    print(f"Model loaded on {_device}  params={_model.n_params():,}")
    return _model


def input_fn(request_body, content_type="application/json"):
    return json.loads(request_body)


def predict_fn(data, model):
    prompt      = data.get("prompt", "").strip()
    temperature = float(data.get("temperature", 0.8))
    max_length  = min(int(data.get("max_length", 20)), 50)

    if not prompt:
        return {"error": "prompt is required"}

    tokens = _enc.encode(prompt)
    if not tokens:
        return {"error": "prompt produced no tokens"}

    ctx_len = model.context_len
    idx = torch.tensor([tokens[-ctx_len:]], dtype=torch.long, device=_device)

    # Next-token distribution after the prompt (for top_words)
    with torch.no_grad():
        logits_full, _ = model(idx)
    logits_next = logits_full[0, -1, :]
    top_probs, top_ids = torch.topk(F.softmax(logits_next, dim=-1), 10)
    top_words = [
        {
            "word": _enc.decode([int(tid)]).strip() or f"[{int(tid)}]",
            "prob": round(float(p), 4),
        }
        for tid, p in zip(top_ids, top_probs)
    ]

    # Generate continuation
    with torch.no_grad():
        out_ids = model.generate(idx, max_new_tokens=max_length, temperature=temperature, top_k=50)
    generated_text = _enc.decode(out_ids[0].tolist())

    return {
        "prompt":    prompt,
        "generated": generated_text,
        "top_words": top_words,
    }


def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction), "application/json"
