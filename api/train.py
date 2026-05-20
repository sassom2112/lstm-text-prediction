"""Train LSTM and save model weights + vocabulary to api/saved/."""
import io, unicodedata, random, re, json, time, copy, os
import numpy as np
import torch
import torch.nn as nn
from lstm_model import LSTM, create_input_tensor

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "spa.txt")

# ── preprocessing ────────────────────────────────────────────────────────────

def unicode_to_ascii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!'?]+", r" ", s)
    return s.strip()

def parse_data(filename):
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [normalize(l.split("\t")[0]) for l in lines]

def add_words_to_dict(word_dict, word_list, sentences):
    for sentence in sentences:
        for word in sentence.split():
            if word not in word_dict:
                word_list.append(word)
                word_dict[word] = len(word_list) - 1

def make_input_tensor(sentence, word_dict, vocab_size):
    return create_input_tensor(sentence, word_dict, vocab_size)

def make_target_tensor(sentence, word_dict, vocab_size):
    words = sentence.split()
    tensor = torch.zeros(len(words), 1, vocab_size)
    for idx in range(1, len(words)):
        w = words[idx]
        if w in word_dict:
            tensor[idx - 1][0][word_dict[w]] = 1
    tensor[len(words) - 1][0][vocab_size - 1] = 1  # EOS
    return tensor

# ── training ─────────────────────────────────────────────────────────────────

def train(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
          device, num_epochs=10, patience=3):
    best_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ["train", "val", "test"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            for inp, tgt in dataloaders[phase]:
                hidden = model.initHidden(1, device)
                inp = inp.unsqueeze(1).to(device)
                tgt = tgt.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    loss = 0
                    for i in range(inp.size(0)):
                        out, hidden = model(inp[i], hidden)
                        loss += criterion(out, tgt[i])
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() / inp.size(0)
            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            ppl = np.exp(epoch_loss)
            print(f"  {phase:5s} loss={epoch_loss:.4f}  ppl={ppl:.1f}")
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_wts = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_wts)
    print(f"\nBest val loss: {best_loss:.4f}  ppl: {np.exp(best_loss):.1f}")
    return model

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sentences = parse_data(DATA_FILE)
    random.seed(42)
    random.shuffle(sentences)
    train_s, val_s, test_s = sentences[:1000], sentences[1000:2000], sentences[2000:3000]

    word_dict, word_list = {}, []
    for split in [train_s, val_s, test_s]:
        add_words_to_dict(word_dict, word_list, split)

    vocab_size = len(word_dict) + 1  # +1 for EOS

    def make_tensors(split):
        return [(make_input_tensor(s, word_dict, vocab_size),
                 make_target_tensor(s, word_dict, vocab_size)) for s in split]

    dataloaders = {"train": make_tensors(train_s), "val": make_tensors(val_s), "test": make_tensors(test_s)}
    dataset_sizes = {k: len(v) for k, v in dataloaders.items()}

    model = LSTM(vocab_size, 128, vocab_size).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Vocab: {len(word_dict):,}  Params: {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    model = train(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device)

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
    with open(os.path.join(SAVE_DIR, "vocab.json"), "w") as f:
        json.dump({"word_dict": word_dict, "word_list": word_list, "vocab_size": vocab_size}, f)
    print(f"\nSaved model + vocab to {SAVE_DIR}/")

if __name__ == "__main__":
    main()
