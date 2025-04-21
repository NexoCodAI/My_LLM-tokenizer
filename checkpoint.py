import torch, os
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
    Partially loads matching weights from `path` into `model`,
    skipping any keys that don't exist or whose shapes differ.
    """
    ckpt = torch.load(path, map_location='cpu')
    pretrained = ckpt['model']
    model_dict = model.state_dict()

    # keep only keys present in model AND same shape
    filtered = {
        k: v for k, v in pretrained.items()
        if (k in model_dict and v.shape == model_dict[k].shape)
    }

    # diagnostics
    total, kept = len(pretrained), len(filtered)
    print(f"→ loading {kept}/{total} tensors")
    if kept < total:
        skipped = sorted(set(pretrained) - set(filtered))
        print(f"⚠ skipped {len(skipped)} tensors:")
        for k in skipped:
            print("    ", k)

    # update & load
    model_dict.update(filtered)
    model.load_state_dict(model_dict)

    # optionally restore optimizer
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt.get('epoch', None)

