import os
from tokenizers import ByteLevelBPETokenizer

def train_bpe_tokenizer(
    files=["data/ultrachat-small/train.txt"],              # list of paths: [train.txt, ...]
    vocab_size=3_000,
    min_frequency=5,
    save_dir="tokenizer"
):
    """
    Trains a Byte-Level BPE tokenizer on the provided files.
    """
    os.makedirs(save_dir, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<s>", "<pad>", "</s>", "<unk>", "<mask>"
        ]
    )
    tokenizer.save_model(save_dir)
    print(f"✓ Tokenizer trained and saved to {save_dir}/")
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

class BPETokenizer:
    """
    Wraps the files output by train_bpe_tokenizer().
    Usage:
        tok = BPETokenizer("tokenizer/vocab.json","tokenizer/merges.txt")
        ids = tok.encode("hello")
        s   = tok.decode(ids)
    """
    def __init__(self, vocab_file, merges_file):
        self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        # add post‑processing for <s>, </s> if you like (optional)
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>",  self.tokenizer.token_to_id("<s>")),
        )
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
