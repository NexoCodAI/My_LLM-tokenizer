from datasets import load_dataset
import os
import re
import ftfy
import html
import unicodedata

def clean_text(text):
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags
    text = re.sub(r"\[.*?\]", " ", text)  # Remove [brackets]
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def prepare_chat_data(save_path="data/ultrachat-small", split="train", max_samples=50000):
    os.makedirs(save_path, exist_ok=True)
    ds = load_dataset("stingning/ultrachat", split=split)

    lines = []
    for item in ds.select(range(min(len(ds), max_samples))):
        prompt = clean_text(item["instruction"])
        response = clean_text(item["output"])
        lines.append(f"User: {prompt}\nAssistant: {response}\n")

    output_file = os.path.join(save_path, "train.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return output_file

# Usage
prepare_chat_data()
