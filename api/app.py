import os, json
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from lstm_model import LSTM, predict as lstm_predict

app = Flask(__name__)
CORS(app)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab and model once at startup
with open(os.path.join(SAVE_DIR, "vocab.json")) as f:
    vocab = json.load(f)

WORD_DICT = vocab["word_dict"]
WORD_LIST = vocab["word_list"]
VOCAB_SIZE = vocab["vocab_size"]

model = LSTM(VOCAB_SIZE, 128, VOCAB_SIZE)
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model.pt"), map_location=device, weights_only=True))
model.to(device)
model.eval()

print(f"Model loaded on {device}. Vocab size: {VOCAB_SIZE}")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "vocab_size": VOCAB_SIZE, "device": str(device)})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip().lower()
    temperature = float(data.get("temperature", 0.0))
    max_length = min(int(data.get("max_length", 15)), 30)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    # Replace unknown words with closest known word or drop them
    known_words = [w for w in prompt.split() if w in WORD_DICT]
    if not known_words:
        return jsonify({"error": "No words in prompt are in the vocabulary"}), 400
    clean_prompt = " ".join(known_words)

    temperature = max(0.0, min(temperature, 2.0))

    generated, top_words = lstm_predict(
        model, WORD_DICT, WORD_LIST, clean_prompt,
        max_length=max_length, temperature=temperature, device=device
    )

    return jsonify({
        "prompt": clean_prompt,
        "generated": generated,
        "top_words": top_words,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
