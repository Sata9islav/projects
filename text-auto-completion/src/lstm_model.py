import torch

import torch.nn as nn


class PredictorModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, rnn_type="GRU"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        rnn_model = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = rnn_model(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        emb = self.embedding(x)
        out, _ = self.rnn(emb)

        last_hidden = out[:, -1, :]
        linear_out = self.fc(last_hidden)
        return linear_out

    def generate(self, prefix_ids, max_new_tokens, device=None):
        self.eval()
        with torch.no_grad():
            if device is None:
                device = next(self.parameters()).device

            x = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)

            for _ in range(max_new_tokens):
                logits = self.forward(x)
                next_id = torch.argmax(logits, dim=1)
                x = torch.cat([x, next_id.unsqueeze(1)], dim=1)

            return x.squeeze(0).tolist()
