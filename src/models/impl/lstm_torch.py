import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pad_id: int, hidden_size: int = 64):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        emb = self.drop(self.emb(x))  # (B, L, D)

        # Option simple: on laisse le padding, ça marche (moins optimal que pack_padded, mais stable et suffisant)
        out, (h, c) = self.lstm(emb)
        # h: (num_layers*2, B, hidden) -> on prend les 2 directions de la dernière couche
        h_fw = h[-2]  # (B, hidden)
        h_bw = h[-1]  # (B, hidden)
        h_cat = torch.cat([h_fw, h_bw], dim=1)  # (B, 2*hidden)

        z = self.act(self.fc1(h_cat))
        z = nn.functional.dropout(z, p=0.3, training=self.training)
        logits = self.fc2(z).squeeze(-1)
        return logits
