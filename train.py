# train.py

import os
import math
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Project-specific imports
from config import (
    D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    SCHEDULER, EPOCHS, DEVICE,
    GRADIENT_ACCUM_STEPS, LABEL_SMOOTHING,
    VALIDATION_SPLIT, EVAL_INTERVAL,
    VALID_LOSS_THRESHOLD, PATIENCE,
    CHECKPOINT_DIR
)
from data_utils   import download_and_prepare_wikitext2
from tokenizer    import train_bpe_tokenizer, BPETokenizer
from dataset      import get_dataloader
from model        import LLM
from evaluate     import evaluate
from utils        import set_seed, ensure_dir
from checkpoint   import save_checkpoint


def train(data_path):
    # reproducibility & setup
    set_seed(42)
    ensure_dir(CHECKPOINT_DIR)
    torch.backends.cudnn.benchmark = True

    # initialize AMP scaler
    scaler = GradScaler()

    # 1) prepare data
    train_path, valid_path = download_and_prepare_wikitext2()

    # 2) train tokenizer if not already there
    if not os.path.exists("tokenizer/vocab.json"):
        train_bpe_tokenizer([train_path], vocab_size=30_000, save_dir="tokenizer")

    # 3) load tokenizer & update VOCAB_SIZE
    tok = BPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    global VOCAB_SIZE
    VOCAB_SIZE = tok.vocab_size

    # 4) read & tokenize whole file into IDs
    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = tok.encode(f.read())
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_ids = tok.encode(f.read())

    # 5) narrow down validation for speed
    split = int(len(valid_ids) * VALIDATION_SPLIT)
    valid_ids = valid_ids[:split]

    # 6) create loaders
    train_loader = get_dataloader(train_ids, shuffle=True)

    # 7) build model + optimizer + scheduler + loss
    model = LLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    # — weight tying
    model.head.weight = model.token_emb.weight

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    # compute total update steps
    steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUM_STEPS)
    total_steps = steps_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=SCHEDULER['max_lr'],
        total_steps=total_steps,
        pct_start=SCHEDULER['pct_start'],
        anneal_strategy=SCHEDULER['anneal_strategy']
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
                scaler.step(optimizer)    # applies gradients
                scaler.update()           # updates the scale for next iteration
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

            # periodic evaluation + checkpointing
            if step % EVAL_INTERVAL == 0:
                val_loss, val_ppl = evaluate(model, valid_ids, DEVICE)
                tqdm.write(f"Step {step} → val loss {val_loss:.4f}, ppl {val_ppl:.2f}")

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


if __name__ == "__main__":
    ensure_dir(CHECKPOINT_DIR)
    train()
