import torch
from torch.utils.data import Dataset, DataLoader
from config import BLOCK_SIZE, BATCH_SIZE

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

def get_dataloader(tokens, shuffle=True):
    """
    Automatically uses BLOCK_SIZE and BATCH_SIZE from config.py
    """
    ds = TextDataset(tokens, BLOCK_SIZE)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
