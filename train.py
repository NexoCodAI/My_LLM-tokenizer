import os
import math
import torch
from tqdm import tqdm

from config       import *
from data_utils   import download_and_prepare_wikitext2
from tokenizer    import train_bpe_tokenizer, BPETokenizer
from dataset      import get_dataloader
from model        import LLM           # your existing model.py
from evaluate     import evaluate      # your existing evaluate()
from utils        import set_seed, ensure_dir

def train(data_path):
    # reproducibility & setup
    set_seed(42)
    ensure_dir(CHECKPOINT_DIR)
    torch.backends.cudnn.benchmark = True

    # 1) prepare data
    train_path, valid_path = download_and_prepare_wikitext2()
    # 2) train tokenizer if not already there
    if not os.path.exists("tokenizer/vocab.json"):
        train_bpe_tokenizer([train_path], vocab_size=30_000, save_dir="tokenizer")

    # 3) load tokenizer & update global VOCAB_SIZE
    tok = BPETokenizer("tokenizer/vocab.json","tokenizer/merges.txt")
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
    train_loader = get_dataloader(train_ids, BLOCK_SIZE+1, BATCH_SIZE, shuffle=True)

    # 7) build model + optim
    model     = LLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val = float('inf')
    patience = 0
    step = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x,y in pbar:
            step += 1
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1)
            )

            loss.backward()
            if step % GRADIENT_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

            if step % EVAL_INTERVAL == 0:
                val_loss, val_ppl = evaluate(model, valid_ids, BATCH_SIZE, BLOCK_SIZE, DEVICE)
                tqdm.write(f"Step {step} → val loss {val_loss:.4f}, ppl {val_ppl:.2f}")

                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                    ckpt = os.path.join(CHECKPOINT_DIR, "best.pt")
                    torch.save(model.state_dict(), ckpt)
                    tqdm.write(f"  🎉 New best checkpoint saved to {ckpt}!")
                else:
                    patience += 1
                    tqdm.write(f"  Patience {patience}/{PATIENCE}")

                if val_loss < VALID_LOSS_THRESHOLD or patience >= PATIENCE:
                    return

        # end epoch
        epoch_ckpt = os.path.join(CHECKPOINT_DIR, f"epoch{epoch}.pt")
        torch.save(model.state_dict(), epoch_ckpt)
        tqdm.write(f"Finished epoch {epoch} (best val {best_val:.4f})")

if __name__ == "__main__":
    ensure_dir(CHECKPOINT_DIR)
    train("data/wikitext-2/train.txt")

