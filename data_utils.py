import os
import re
import ftfy
import html
import unicodedata
from datasets import load_dataset
from typing import Optional

def clean_text(text: str) -> str:
    """
    Fix encoding, strip HTML, unicode‑normalize, remove brackets, collapse spaces.
    """
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"<[^>]+>", " ", text)       # remove any HTML tags
    text = re.sub(r"\[.*?\]", " ", text)       # remove [bracketed] notes
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text.strip()

def write_split(split_dataset, filename: str):
    """
    Given a 🤗 Dataset split, clean and format it, then write to disk.
    """
    lines = []
    for item in split_dataset:
        prompt   = clean_text(item["instruction"])
        response = clean_text(item["output"])
        lines.append(f"User: {prompt}\nAssistant: {response}\n")
    out_path = os.path.join(SAVE_PATH, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(lines)} examples to {out_path}")

def prepare_chat_data(
    save_path: Optional[str] = SAVE_PATH,
    max_samples: Optional[int]   = MAX_SAMPLES,
    valid_frac: float            = VALIDATION_SPLIT
):
    # 1) Make target directory
    os.makedirs(save_path, exist_ok=True)

    # 2) Load the train split
    ds = load_dataset("stingning/ultrachat", split="train")

    # 3) Optionally truncate
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    # 4) Shuffle, then split off validation
    ds = ds.shuffle(seed=42)
    n_valid = int(len(ds) * valid_frac)
    n_train = len(ds) - n_valid

    train_ds = ds.select(range(n_train))
    valid_ds = ds.select(range(n_train, n_train + n_valid))

    # 5) Write out
    write_split(train_ds, "train.txt")
    write_split(valid_ds, "valid.txt")

if __name__ == "__main__":
    prepare_chat_data()
