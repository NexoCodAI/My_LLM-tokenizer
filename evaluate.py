# evaluate.py

import math
import torch
from torch.nn import functional as F
from tqdm import tqdm

from dataset import get_dataloader


def sample(model, tokenizer, start_text, length, device):
    model.eval()
    tokens = tokenizer.encode(start_text)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = tokens.copy()

    for _ in range(length):
        seq = input_ids if input_ids.size(1) <= model.block_size else input_ids[:, -model.block_size:]
        logits = model(seq)                     # (1, T, vocab_size)
        next_logits = logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    return tokenizer.decode(generated)


@torch.no_grad()
def evaluate(model, val_tokens, device):
    """
    Compute average cross-entropy loss and perplexity on val_tokens.
    Automatically uses BLOCK_SIZE and BATCH_SIZE via get_dataloader.
    """
    model.eval()
    val_loader = get_dataloader(val_tokens, shuffle=False)

    total_loss = 0.0
    count = 0

    for x, y in tqdm(val_loader, desc="🔍 Evaluating", leave=False, dynamic_ncols=True, mininterval=10, ncols=80):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction='mean'
        )
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


if __name__ == "__main__":
    import argparse
    import torch
    from model     import LLM
    from tokenizer import SimpleTokenizer
    from config    import D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE, DEVICE

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    samp = subparsers.add_parser('sample')
    samp.add_argument('--ckpt',   type=str, required=True)
    samp.add_argument('--prompt', type=str, required=True)
    samp.add_argument('--length', type=int, default=100)

    valp = subparsers.add_parser('validate')
    valp.add_argument('--ckpt',  type=str, required=True)
    valp.add_argument('--data',  type=str, required=True)
    valp.add_argument('--split', type=float, default=0.1)

    args = parser.parse_args()
    device = DEVICE

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    text = open(args.data, 'r', encoding='utf-8').read()
    tok = SimpleTokenizer([text])
    vocab_size = tok.vocab_size

    model = LLM(vocab_size, D_MODEL, N_LAYERS, N_HEADS, BLOCK_SIZE).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    # ensure weight‑tying during eval/run
    model.head.weight = model.token_emb.weight

    if args.mode == 'sample':
        print(sample(model, tok, args.prompt, args.length, device))
    else:
        tokens = tok.encode(text)
        split_idx = int((1 - args.split) * len(tokens))
        val_tokens = tokens[split_idx:]
        val_loss, val_ppl = evaluate(model, val_tokens, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Perplexity      : {val_ppl:.2f}")

