from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)


def load_split_csv(path: Path):
    df = pd.read_csv(path)
    X = df["text"].astype(str).tolist()
    y = df["target"].astype(int).to_numpy()
    return X, y


@dataclass
class Vocab:
    stoi: dict
    itos: list
    pad_id: int
    unk_id: int


def build_vocab(texts: list[str], max_words: int = 100_000, min_freq: int = 2) -> Vocab:
    # Simple whitespace tokenizer (tweets déjà clean)
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(t.split())

    # Special tokens
    itos = ["<PAD>", "<UNK>"]
    pad_id, unk_id = 0, 1

    # Most common tokens respecting min_freq
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        itos.append(word)
        if len(itos) >= max_words:
            break

    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, pad_id=pad_id, unk_id=unk_id)


def encode_text(text: str, vocab: Vocab) -> list[int]:
    return [vocab.stoi.get(tok, vocab.unk_id) for tok in text.split()]


def infer_max_len_from_train(texts_train: list[str], vocab: Vocab, percentile: float = 95.0,
                            min_len: int = 10, max_cap: int = 80) -> int:
    lengths = []
    for t in texts_train:
        lengths.append(len(encode_text(t, vocab)))
    max_len = int(np.percentile(np.array(lengths), percentile))
    return max(min_len, min(max_len, max_cap))


def pad_batch(seqs: list[list[int]], max_len: int, pad_id: int) -> np.ndarray:
    arr = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        s2 = s[:max_len]
        arr[i, :len(s2)] = s2
    return arr


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: Vocab, max_len: int):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids = encode_text(self.texts[idx], self.vocab)
        x = pad_batch([ids], self.max_len, self.vocab.pad_id)[0]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }, y_pred


def save_confusion_png(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["non_negative(0)", "negative(1)"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_png(y_true, y_proba, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_vocab(path: Path, vocab: Vocab):
    payload = {"itos": vocab.itos, "pad_id": vocab.pad_id, "unk_id": vocab.unk_id}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
