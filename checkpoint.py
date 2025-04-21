# checkpoint.py

import torch
import os
from config import CHECKPOINT_DIR

def save_checkpoint(model, optimizer, epoch, suffix=""):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fn = f"ckpt_epoch_{epoch}" + (f"_{suffix}" if suffix else "") + ".pt"
    path = os.path.join(CHECKPOINT_DIR, fn)
    torch.save({
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':     epoch
    }, path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(path, model, optimizer=None):
    """
    Loads the checkpoint at `path` into `model`, optionally restoring optimizer state.
    Skips any parameters whose names or shapes don’t match.
    Returns the saved epoch (or None if not present).
    """
    ckpt = torch.load(path, map_location='cpu')
    pretrained = ckpt['model']
    model_dict = model.state_dict()

    # keep only matching keys
    filtered = {
        k: v for k, v in pretrained.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    total, kept = len(pretrained), len(filtered)
    print(f"→ loading {kept}/{total} tensors")
    if kept < total:
        skipped = sorted(set(pretrained) - set(filtered))
        print(f"⚠ skipped {len(skipped)} tensors:")
        for k in skipped:
            print("    ", k)

    model_dict.update(filtered)
    model.load_state_dict(model_dict)

    # restore optimizer if provided
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt.get('epoch', None)

