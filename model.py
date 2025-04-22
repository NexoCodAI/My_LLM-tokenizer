import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(d_model, d_model)
        # Causal mask will be applied in forward

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        # attention scores
        att = (q @ k.transpose(-2,-1)) / (self.head_dim**0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1,1,T,T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.drop = nn.Dropout(0.1)  # 🔥 Dropout after embedding sum
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence length exceeds model block size"
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = tok_emb + pos_emb
        x = self.drop(x)  # 🔥 drop after embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.drop(x)  # 🔥 optional: drop again before head
        logits = self.head(x)
        return logits
