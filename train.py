#!/usr/bin/env python3
import os
import math
import argparse
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Project-specific imports
from config import (
    D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    SCHEDULER, EPOCHS, DEVICE,
    GRADIENT_ACCUM_STEPS, LABEL_SMOOTHING,
    VALID_LOSS_THRESHOLD, PATIENCE,
    CHECKPOINT_DIR
)
from tokenizer    import train_bpe_tokenizer, BPETokenizer
from dataset      import get_dataloader
from model        import LLM
from evaluate     import evaluate
from utils        import set_seed, ensure_dir
from checkpoint   import save_checkpoint


def train(train_path: str, valid_path: str):
    # reproducibility & setup
    set_seed(42)
    ensure_dir(CHECKPOINT_DIR)
    torch.backends.cudnn.benchmark = True

    # initialize AMP scaler
    scaler = GradScaler()

    # 1) train tokenizer if not already there
    if not os.path.exists("tokenizer/vocab.json"):
        train_bpe_tokenizer([train_path], vocab_size=30_000, save_dir="tokenizer")

    # 2) load tokenizer & update VOCAB_SIZE
    tok = BPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    global VOCAB_SIZE
    VOCAB_SIZE = tok.vocab_size

    # 3) read & tokenize whole file into IDs
    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = tok.encode(f.read())
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_ids = tok.encode(f.read())

    # 4) create dataloader
    train_loader = get_dataloader(train_ids, batch_size=BATCH_SIZE, shuffle=True)

    # 5) build model + optimizer + scheduler + loss
    model = LLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    # — weight tying
    model.head.weight = model.token_emb.weight

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER['mode'],
        factor=SCHEDULER['factor'],
        patience=SCHEDULER['patience'],
        min_lr=SCHEDULER['min_lr'],
        verbose=SCHEDULER['verbose']
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val = float('inf')
    patience = 0
    step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            step += 1
            x, y = x.to(DEVICE), y.to(DEVICE)

            # —— AMP half-precision forward + loss
            with autocast():
                logits = model(x)
                loss = criterion(
                    logits.view(-1, VOCAB_SIZE),
                    y.view(-1)
                )

            # scale gradients, backward pass, and optimizer step
            scaler.scale(loss).backward()
            if step % GRADIENT_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

            # periodic evaluation + checkpointing
            if step % EVAL_INTERVAL == 0:
                val_loss, val_ppl = evaluate(model, valid_ids, DEVICE)
                tqdm.write(f"Step {step} → val loss {val_loss:.4f}, ppl {val_ppl:.2f}")

                # scheduler step on validation loss
                scheduler.step(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    save_checkpoint(model, optimizer, epoch, suffix="best")
                    tqdm.write(f"  🎉 New best checkpoint saved!")
                else:
                    patience += 1
                    tqdm.write(f"  Patience {patience}/{PATIENCE}")

                if val_loss < VALID_LOSS_THRESHOLD or patience >= PATIENCE:
                    return

        # end of epoch checkpoint
        save_checkpoint(model, optimizer, epoch)
        tqdm.write(f"Finished epoch {epoch} (best val {best_val:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train LLM on chat data")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing train.txt and valid.txt"
    )
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.txt")
    valid_path = os.path.join(args.data_dir, "valid.txt")

    if not os.path.isfile(train_path) or not os.path.isfile(valid_path):
        raise FileNotFoundError(
            f"train.txt or valid.txt not found in {args.data_dir}"
        )

    ensure_dir(CHECKPOINT_DIR)
    train(train_path, valid_path)


if __name__ == "__main__":
    main()
