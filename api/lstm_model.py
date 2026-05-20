import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.dropout(output[-1])
        output = self.fc(output)
        return output, hidden

    def initHidden(self, batch_size, device):
        return (
            torch.zeros(2, batch_size, self.hidden_size).to(device),
            torch.zeros(2, batch_size, self.hidden_size).to(device),
        )


def create_input_tensor(sentence, word_dictionary, vocab_size):
    words = sentence.split()
    tensor = torch.zeros(len(words), 1, vocab_size)
    for idx, word in enumerate(words):
        if word in word_dictionary:
            tensor[idx][0][word_dictionary[word]] = 1
    return tensor


def predict(model, word_dictionary, word_list, input_sentence,
            max_length=20, temperature=0.0, device="cpu"):
    vocab_size = len(word_dictionary) + 1
    tensor = create_input_tensor(input_sentence, word_dictionary, vocab_size).to(device)
    hidden = model.initHidden(batch_size=1, device=device)

    model.eval()
    with torch.no_grad():
        for i in range(tensor.size(0)):
            output, hidden = model(tensor[i:i+1], hidden)

        generated_words = []
        for _ in range(max_length):
            if temperature == 0.0:
                _, topi = output.topk(1)
                topi = topi[0][0].item()
            else:
                probs = F.softmax(output.squeeze() / temperature, dim=-1)
                topi = torch.multinomial(probs, 1).item()

            if topi == len(word_dictionary):
                break

            word = word_list[topi]
            generated_words.append(word)

            next_tensor = create_input_tensor(word, word_dictionary, vocab_size).to(device)
            output, hidden = model(next_tensor[0:1], hidden)

        # Top-K next word probabilities from the last output
        probs_all = F.softmax(output.squeeze(), dim=-1)
        topk_probs, topk_idx = probs_all.topk(10)
        top_words = [
            {"word": word_list[i] if i < len(word_list) else "<EOS>",
             "prob": round(p.item(), 4)}
            for i, p in zip(topk_idx.cpu().numpy(), topk_probs)
        ]

    generated = input_sentence + (" " + " ".join(generated_words) if generated_words else "")
    return generated.strip(), top_words
