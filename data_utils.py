from datasets import load_dataset
import os
import re
import ftfy
import html
import unicodedata

def clean_line(line):
    # 1) fix mojibake / odd unicode
    line = ftfy.fix_text(line)
    # 2) unescape HTML entities
    line = html.unescape(line)
    # 3) normalize Unicode (NFC composes accents into single codepoints)
    line = unicodedata.normalize('NFC', line)
    # 4) remove HTML tags (e.g. <ref>…</ref>)
    line = re.sub(r"<[^>]+>", " ", line)
    # 5) remove citation brackets like [1], [12], [citation needed]
    line = re.sub(r"\[.*?\]", " ", line)
    # 6) drop any other control or unassigned Unicode chars
    line = "".join(
        ch for ch in line
        if unicodedata.category(ch)[0] != "C"
    )
    # 7) optionally strip out non-printable glyphs (keeps letters, numbers, punctuation, whitespace)
    #    (uncomment if you want strictly ASCII-only)
    # import string
    # line = "".join(ch for ch in line if ch in string.printable)
    # 8) collapse multiple spaces into one
    line = re.sub(r"\s+", " ", line)
    return line.strip()

def download_and_prepare_wikitext2(save_path="data/wikitext-2"):
    os.makedirs(save_path, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def concat(split):
        lines = []
        for raw in ds[split]["text"]:
            l = raw.strip()
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

    return (
        os.path.join(save_path, "train.txt"),
        os.path.join(save_path, "valid.txt"),
    )
