from datasets import load_dataset
import os

def download_and_prepare_wikitext2(save_path="data/wikitext-2"):
    """
    Downloads WikiText-2 raw split, concatenates lines into single files.
    Returns paths to train.txt and valid.txt
    """
    os.makedirs(save_path, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    def concat(split):
        lines = [l for l in ds[split]["text"] if l.strip() and not l.startswith("= =")]
        return "\n".join(lines)
    train_txt = concat("train")
    valid_txt = concat("validation")
    train_path = os.path.join(save_path, "train.txt")
    valid_path = os.path.join(save_path, "valid.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_txt)
    with open(valid_path, "w", encoding="utf-8") as f:
        f.write(valid_txt)
    return train_path, valid_path
