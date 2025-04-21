import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, token_ids, block_size):
        # token_ids is a single long list/array of ints
        self.tokens = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y

def get_dataloader(tokens, block_size, batch_size, shuffle=True):
    ds = TextDataset(tokens, block_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
