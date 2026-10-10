"""Training dataset over the tokenized prompt/response sequences."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .indexed import IndexedTokens
from .prepare import SEPARATOR


class DistillationDataset(Dataset):
    """Fixed-length (input, label) pairs whose labels cover the response only.

    Each stored sequence is `prompt + [SEPARATOR] + response`. The collated
    batch is `({"input_ids", "attention_mask"}, labels)`, with -100 on prompt
    and padding positions.
    """

    def __init__(self, data_dir, split, max_length, pad_id, num=-1):
        self.tokens = IndexedTokens(os.path.join(data_dir, f"{split}_0"))
        self.max_length = max_length
        self.pad_id = pad_id
        self.num = len(self.tokens) if num == -1 else num

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.tokens[index].astype(int)

    def collate(self, samples):
        batch, length = len(samples), self.max_length
        input_ids = torch.full((batch, length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(batch, length)
        labels = torch.full((batch, length), -100, dtype=torch.long)
        for i, ids in enumerate(samples):
            prompt_len = 1
            separator = np.where(ids == SEPARATOR)[0]
            if separator.size:
                prompt_len = separator[0]
                ids = np.concatenate([ids[:prompt_len], ids[prompt_len + 1:]])
            ids = ids[:length]
            n = len(ids)
            input_ids[i, :n - 1] = torch.tensor(ids[:-1], dtype=torch.long)
            attention_mask[i, :n - 1] = 1.0
            labels[i, :n - 1] = torch.tensor(ids[1:], dtype=torch.long)
            labels[i, :prompt_len - 1] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels
