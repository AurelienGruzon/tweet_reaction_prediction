from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class HFDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels  # numpy array int 0/1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], int(self.labels[idx])
