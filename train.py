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
    VALID_LOSS_THRESHOLD, EVAL_INTERVAL,
    PATIENCE, CHECKPOINT_DIR
)
from tokenizer    import train_bpe_tokenizer, BPETokenizer
from dataset      import get_dataloader
from model        import LLM
from evaluate     import evaluate
from utils        import set_seed, ensure_dir
from checkpoint   import save_checkpoint
from data_utils   import prepare_chat_data


def train(train_path: str, valid_path: str):
    # reproducibility & setup
    set_seed(42)
    ensure_dir(CHECKPOINT_DIR)
    torch.backends.cudnn.benchmark = True

    scaler = GradScaler()

    # 1) train tokenizer if not already there
    if not os.path.exists("tokenizer/vocab.json"):
        train_bpe_tokenizer([train_path], vocab_size=3_000, save_dir="tokenizer")

    # 2) load tokenizer
    tok = BPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    global VOCAB_SIZE
    VOCAB_SIZE = tok.vocab_size

    # 3) read & tokenize
    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = tok.encode(f.read())
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_ids = tok.encode(f.read())

    # 4) dataloader
    train_loader = get_dataloader(train_ids, shuffle=True)

    # 5) model, optimizer, scheduler, loss
    model = LLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    model.head.weight = model.token_emb.weight

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # — add WARMUP
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUM_STEPS
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    plateau_scheduler = ReduceLROnPlateau(
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

            # Warmup: Linear increase learning rate
            if step <= warmup_steps:
                warmup_lr = LEARNING_RATE * step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            with autocast():
                logits = model(x)
                loss = criterion(
                    logits.view(-1, VOCAB_SIZE),
                    y.view(-1)
                )

            scaler.scale(loss).backward()
            if step % GRADIENT_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

            if step % EVAL_INTERVAL == 0:
                val_loss, val_ppl = evaluate(model, valid_ids, DEVICE)
                tqdm.write(f"Step {step} → val loss {val_loss:.4f}, ppl {val_ppl:.2f}")

                if step > warmup_steps:
                    plateau_scheduler.step(val_loss)  # only use plateau scheduler after warmup

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

        save_checkpoint(model, optimizer, epoch)
        tqdm.write(f"Finished epoch {epoch} (best val {best_val:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train LLM on chat data")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing train.txt and valid.txt"
    )
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.data_dir, "train.txt")) or not os.path.isfile(os.path.join(args.data_dir, "valid.txt")):
        print(f"train.txt or valid.txt not found in {args.data_dir}. Preparing data...")
        prepare_chat_data(save_path=args.data_dir)

    train_path = os.path.join(args.data_dir, "train.txt")
    valid_path = os.path.join(args.data_dir, "valid.txt")

    ensure_dir(CHECKPOINT_DIR)
    train(train_path, valid_path)


if __name__ == "__main__":
    main()
