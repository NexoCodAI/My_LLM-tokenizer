# infer.py

import os
import torch
from model        import LLM
from tokenizer    import BPETokenizer
from evaluate     import sample
from checkpoint   import load_checkpoint
from config       import (
    CHECKPOINT_DIR,
    D_MODEL,
    N_LAYERS,
    N_HEADS,
    BLOCK_SIZE,
    DEVICE
)
from utils        import set_seed, ensure_dir


def infer(
    vocab_dir: str = "tokenizer",
    prompt: str = "The capital of France is",
    gen_length: int = 100
):
    # 0) reproducibility
    set_seed(42)

    # 1) load tokenizer (must match training)
    vocab_file  = os.path.join(vocab_dir, "vocab.json")
    merges_file = os.path.join(vocab_dir, "merges.txt")
    if not (os.path.exists(vocab_file) and os.path.exists(merges_file)):
        raise FileNotFoundError(f"Tokenizer files not found in {vocab_dir}")
    tokenizer = BPETokenizer(vocab_file, merges_file)
    vocab_size = tokenizer.vocab_size

    # 2) find & load latest checkpoint
    ckpt_files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)),
        reverse=True
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint (.pt) files found in {CHECKPOINT_DIR}")
    latest_ckpt = os.path.join(CHECKPOINT_DIR, ckpt_files[0])
    print(f"Loading checkpoint: {latest_ckpt}")

    # 3) build model and tie weights
    model = LLM(vocab_size, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    model.head.weight = model.token_emb.weight

    # 4) load checkpoint (no optimizer here)
    start_epoch = load_checkpoint(latest_ckpt, model, optimizer=None)
    print(f"Resumed from epoch {start_epoch}")
    model.eval()

    # 5) run sampling
    print(f"\nPrompt: {prompt}")
    generated = sample(model, tokenizer, start_text=prompt, length=gen_length, device=DEVICE)
    print(f"\nGenerated text:\n{generated}\n")


if __name__ == "__main__":
    ensure_dir(CHECKPOINT_DIR)
    infer()

