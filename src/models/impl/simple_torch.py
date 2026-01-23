import torch
import torch.nn as nn


class SimpleMeanPool(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, L)
        emb = self.emb(x)  # (B, L, D)
        mask = (x != self.pad_id).unsqueeze(-1).float()  # (B, L, 1)
        emb = emb * mask
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = emb.sum(dim=1) / denom  # (B, D)
        pooled = self.drop(pooled)
        h = self.act(self.fc1(pooled))
        h = nn.functional.dropout(h, p=0.3, training=self.training)
        logits = self.fc2(h).squeeze(-1)  # (B,)
        return logits
