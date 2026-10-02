"""A small character-level GPT on Tiny Shakespeare, trained under Ravex.

    python train.py --steps 2000 --lr 3e-3

The text is fetched once into ``data/`` beside this file. The run's store,
name and metrics endpoint come from the ``RAVEX_*`` environment, so the same
script runs on a laptop or on a node an agent started it on; checkpoints make
it resumable and forkable from any of them.
"""

from __future__ import annotations

import argparse
import math
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ravex

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))


def load_text() -> str:
    path = os.path.join(HERE, "data", "input.txt")
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("fetching Tiny Shakespeare", flush=True)
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, block: int, dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(block, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(block, block, dtype=torch.bool), diagonal=1))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for block in self.blocks:
            x = block(x, self.mask[:t, :t])
        return self.head(self.ln(x))

    @torch.no_grad()
    def sample(self, idx: torch.Tensor, length: int) -> torch.Tensor:
        for _ in range(length):
            logits = self(idx[:, -self.block:])[:, -1]
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = load_text()
    chars = sorted(set(text))
    index = {c: i for i, c in enumerate(chars)}
    data = torch.from_numpy(np.array([index[c] for c in text], dtype=np.int64))
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]

    def batch_of(source: torch.Tensor, generator: torch.Generator):
        starts = torch.randint(len(source) - args.block - 1, (args.batch,), generator=generator)
        x = torch.stack([source[s : s + args.block] for s in starts])
        y = torch.stack([source[s + 1 : s + args.block + 1] for s in starts])
        return x.to(device), y.to(device)

    @ravex.train_loop(backend="moonclip", checkpoint_every=250, keep_last=5, metrics_every=10)
    def run():
        torch.manual_seed(args.seed)
        model = TinyGPT(len(chars), args.block, args.dim, args.layers, args.heads, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
        warmup = max(1, args.steps // 20)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))),
        )
        ravex.track(model=model, optimizer=optimizer, scheduler=schedule)
        print("%d parameters, %d characters, on %s" % (sum(p.numel() for p in model.parameters()), len(text), device), flush=True)

        generator = torch.Generator().manual_seed(args.seed + ravex.step())
        while ravex.step() < args.steps:
            ravex.batch_boundary()
            model.train()
            x, y = batch_of(train, generator)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()

            step = ravex.step()
            ravex.log_metrics({"train/loss": loss, "train/grad_norm": grad_norm, "train/lr": schedule.get_last_lr()[0]})
            if step % 100 == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = batch_of(val, torch.Generator().manual_seed(step))
                    vlogits = model(vx)
                    vloss = F.cross_entropy(vlogits.view(-1, vlogits.shape[-1]), vy.view(-1))
                ravex.log_metrics({"eval/loss": vloss, "eval/bits_per_char": vloss / math.log(2)})
                print("step %d  train %.3f  eval %.3f" % (step, loss.item(), vloss.item()), flush=True)

        model.eval()
        start = torch.tensor([[index["\n"]]], device=device)
        print("".join(chars[i] for i in model.sample(start, 300)[0].tolist()), flush=True)

    run()


if __name__ == "__main__":
    main()
