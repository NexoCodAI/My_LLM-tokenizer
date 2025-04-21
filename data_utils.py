#!/usr/bin/env python3
import os
import re
import ftfy
import html
import unicodedata
from datasets import load_dataset
from typing import Optional

# -----------------------------------------------------------------------------
# Pull data-prep config from project config
# -----------------------------------------------------------------------------
from config import SAVE_PATH, MAX_SAMPLES, VALIDATION_SPLIT


def clean_text(text: str) -> str:
    """
    Fix encoding, strip HTML, unicode-normalize, remove brackets, collapse spaces.
    """
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"<[^>]+>", " ", text)       # remove any HTML tags
    text = re.sub(r"\[.*?\]", " ", text)     # remove [bracketed] notes
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text.strip()


def write_split(split_dataset, save_path: str, filename: str):
    """
    Given a 🤗 Dataset split, clean and format it, then write to disk under save_path.
    """
    os.makedirs(save_path, exist_ok=True)
    lines = []
    for item in split_dataset:
        prompt = clean_text(item.get("instruction", ""))
        response = clean_text(item.get("output", ""))
        lines.append(f"User: {prompt}\nAssistant: {response}\n")
    out_path = os.path.join(save_path, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(lines)} examples to {out_path}")


def prepare_chat_data(
    save_path: Optional[str] = SAVE_PATH,
    max_samples: Optional[int] = MAX_SAMPLES,
    valid_frac: float = VALIDATION_SPLIT
):
    """
    Load the 'stingning/ultrachat' dataset, clean, split into train/valid,
    and write out 'train.txt' and 'valid.txt' under save_path.
    """
    # ensure target directory exists
    os.makedirs(save_path, exist_ok=True)

    # load the HuggingFace Ultrachat train split
    ds = load_dataset("stingning/ultrachat", split="train")

    # optionally truncate to max_samples
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    # shuffle and split
    ds = ds.shuffle(seed=42)
    n_valid = int(len(ds) * valid_frac)
    n_train = len(ds) - n_valid
    train_ds = ds.select(range(n_train))
    valid_ds = ds.select(range(n_train, n_train + n_valid))

    # write out splits
    write_split(train_ds, save_path, "train.txt")
    write_split(valid_ds, save_path, "valid.txt")


if __name__ == "__main__":
    prepare_chat_data()
