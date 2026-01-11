import torch

from torch.utils.data import Dataset

from typing import List, Tuple


class TweetRNNDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.samples: List[Tuple[List[int], int]] = []
        for text in texts:
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=130,
                truncation=True,
            )
            if len(token_ids) < seq_len + 1:
                continue

            for i in range(0, len(token_ids) - seq_len):
                window = token_ids[i : i + seq_len + 1]
                context = window[:-1]
                target = window[-1]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
