from datasets import load_dataset
import os
import re
import ftfy
import html

def clean_line(line):
    # fix mojibake / odd unicode
    line = ftfy.fix_text(line)
    # unescape HTML entities
    line = html.unescape(line)
    # remove citation brackets like [1], [12], [citation needed]
    line = re.sub(r"\[.*?\]", "", line)
    # remove any leftover non‑printable/control chars
    line = re.sub(r"[\x00-\x1F\x7F]", " ", line)
    # collapse multiple spaces
    return re.sub(r"\s+", " ", line).strip()

def download_and_prepare_wikitext2(save_path="data/wikitext-2"):
    os.makedirs(save_path, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    def concat(split):
        lines = []
        for l in ds[split]["text"]:
            l = l.strip()
            # skip pure headings or empty
            if not l or l.startswith("="):
                continue
            cl = clean_line(l)
            if cl:
                lines.append(cl)
        # join with newlines
        return "\n".join(lines)

    train_txt = concat("train")
    valid_txt = concat("validation")
    for name, txt in [("train.txt", train_txt), ("valid.txt", valid_txt)]:
        with open(os.path.join(save_path, name), "w", encoding="utf-8") as f:
            f.write(txt)

    return os.path.join(save_path, "train.txt"), os.path.join(save_path, "valid.txt")
