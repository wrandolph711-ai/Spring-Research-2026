"""
gpt_v2_deep.py  —  Tiny-but-Deep GPT
Very small embedding (64-dim) but 8 transformer layers instead of 4.
Explores whether depth > width on a CPU budget.
After training: side-by-side comparison of 3 temperatures for every prompt.
"""

import os, time, math, pickle, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────
#  TOKENIZER
# ──────────────────────────────────────────────────────────

class CharTokenizer:
    def __init__(self):
        self.stoi, self.itos = {}, {}

    def fit(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        return self

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(i, "?") for i in ids)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos}, f)

    @classmethod
    def load(cls, path):
        t = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        t.stoi, t.itos = d["stoi"], d["itos"]
        return t


# ──────────────────────────────────────────────────────────
#  DATA
# ──────────────────────────────────────────────────────────

def make_batches(ids, block_size, batch_size):
    data = torch.tensor(ids, dtype=torch.long)
    n    = len(data) - block_size
    idx  = torch.randperm(n)
    batches = []
    for start in range(0, n - batch_size, batch_size):
        ix = idx[start : start + batch_size]
        x  = torch.stack([data[i     : i + block_size]     for i in ix])
        y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        batches.append((x, y))
    return batches


def synthetic_python():
    base = ""
    for i in range(60):
        base += f"""
def function_{i}(x, y):
    \"\"\"Compute result for inputs x and y.\"\"\"
    if x < 0:
        raise ValueError("x must be non-negative")
    result = x * y + {i}
    return result

class Node_{i}:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return f"Node_{i}({{self.value}})"

    def to_list(self):
        result = []
        current = self
        while current is not None:
            result.append(current.value)
            current = current.next
        return result

"""
    return (base * 10)[:200_000]


# ──────────────────────────────────────────────────────────
#  MODEL  (deep: 8 layers, narrow: 64-dim embedding)
# ──────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.n_embd   = n_embd
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        def r(t): return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = r(q), r(k), r(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, block_size, dropout)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init)
        params = sum(p.numel() for p in self.parameters())
        print(f"  Parameters: {params:,}  |  Layers: {n_layer}  |  Embd: {n_embd}")

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x    = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ──────────────────────────────────────────────────────────
#  LOSS CURVE  (shows per-layer perplexity annotation)
# ──────────────────────────────────────────────────────────

def save_loss_curve(train_losses, val_losses, out_dir, n_layer, n_embd):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0a0f1e")
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, color="#00d4ff", lw=2, marker="o", ms=5, label="Train")
    ax.plot(epochs, val_losses,   color="#ff6b35", lw=2, marker="s", ms=5, ls="--", label="Val")

    # annotate final perplexity
    final_ppl = math.exp(min(val_losses[-1], 10))
    ax.text(0.98, 0.95, f"Final perplexity: {final_ppl:.1f}",
            transform=ax.transAxes, ha="right", va="top",
            color="#00d4ff", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#0a0f1e", edgecolor="#00d4ff", alpha=0.8))

    ax.text(0.02, 0.05, f"Architecture: {n_layer} layers × {n_embd}-dim",
            transform=ax.transAxes, ha="left", va="bottom",
            color="#888888", fontsize=8)

    ax.set_xlabel("Epoch", color="#aaaaaa")
    ax.set_ylabel("Cross-Entropy Loss", color="#aaaaaa")
    ax.set_title("Tiny-but-Deep GPT — Loss Curve", color="#ffffff", fontweight="bold")
    ax.tick_params(colors="#888888")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2a3a")
    ax.grid(True, color="#0f1a2a", lw=0.7)
    ax.legend(facecolor="#0a0f1e", edgecolor="#1a2a3a", labelcolor="#cccccc")
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Loss curve → {path}")


# ──────────────────────────────────────────────────────────
#  INTERACTIVE  (side-by-side temperature comparison)
# ──────────────────────────────────────────────────────────

def interactive(model, tokenizer, args):
    print("\n" + "▓"*52)
    print("  TINY-BUT-DEEP GPT  —  Temperature Lab")
    print("▓"*52)
    print("  Each prompt generates 3 completions side-by-side:")
    print("  [SAFE t=0.4]  [BALANCED t=0.8]  [WILD t=1.4]")
    print("\n  Commands:  :single   :quit")
    print("▓"*52 + "\n")

    single_mode = False
    single_temp = 0.8

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDone.")
            break

        if not prompt:
            continue
        if prompt == ":quit":
            break
        if prompt == ":single":
            single_mode = not single_mode
            print(f"  Single mode: {'ON' if single_mode else 'OFF'}\n")
            continue
        if prompt.startswith(":temp") and single_mode:
            try:
                single_temp = float(prompt.split()[1])
                print(f"  Temperature → {single_temp}\n")
            except:
                pass
            continue

        encoded = tokenizer.encode(prompt)
        if not encoded:
            print("  (unknown characters)\n")
            continue

        model.eval()
        ctx = torch.tensor([encoded], dtype=torch.long)

        if single_mode:
            out  = model.generate(ctx, max_new_tokens=args.max_tokens,
                                  temperature=single_temp, top_k=40)
            text = tokenizer.decode(out[0].tolist())
            print("\n" + "─"*52)
            print(text)
            print("─"*52 + "\n")
        else:
            temps = [0.4, 0.8, 1.4]
            labels = ["SAFE", "BALANCED", "WILD"]
            results = []
            for t in temps:
                out  = model.generate(ctx.clone(), max_new_tokens=args.max_tokens,
                                      temperature=t, top_k=40)
                results.append(tokenizer.decode(out[0].tolist()))

            print()
            for label, temp, text in zip(labels, temps, results):
                print(f"  ── {label} (temp={temp}) " + "─" * (36 - len(label)))
                print(f"  {text}")
                print()


# ──────────────────────────────────────────────────────────
#  TRAIN
# ──────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'▓'*52}")
    print("  TINY-BUT-DEEP GPT  —  Windows CPU")
    print(f"{'▓'*52}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] Loading dataset ...")
    if args.data == "file":
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"  Loaded: {args.file}  ({len(text):,} chars)")
    else:
        text = synthetic_python()
        print(f"  Synthetic Python corpus ({len(text):,} chars).")

    print("[2/4] Tokenizer ...")
    tok_path = os.path.join(args.out_dir, "tokenizer.pkl")
    if args.resume and os.path.exists(tok_path):
        tok = CharTokenizer.load(tok_path)
        print(f"  Loaded (vocab={tok.vocab_size})")
    else:
        tok = CharTokenizer().fit(text)
        tok.save(tok_path)
        print(f"  Built (vocab={tok.vocab_size})")

    ids   = tok.encode(text)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"  Train: {len(train_ids):,}  |  Val: {len(val_ids):,}")

    print("[3/4] Building deep model ...")
    model = GPT(
        vocab_size = tok.vocab_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        n_embd     = args.n_embd,
        block_size = args.block_size,
        dropout    = args.dropout,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10)

    best_val  = float("inf")
    ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
    best_path = os.path.join(args.out_dir, "checkpoint_best.pt")
    start_ep  = 1

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_ep = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))

    print(f"\n[4/4] Training {args.epochs} epoch(s) ...")
    print(f"  batch={args.batch_size}  block={args.block_size}  "
          f"layers={args.n_layer}  embd={args.n_embd}\n")

    train_losses, val_losses = [], []

    for epoch in range(start_ep, start_ep + args.epochs):
        model.train()
        t0      = time.time()
        batches = make_batches(train_ids, args.block_size, args.batch_size)
        total, steps = 0.0, 0

        for i, (x, y) in enumerate(batches):
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1
            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch} | Step {i+1}/{len(batches)} "
                      f"| Loss {loss.item():.4f}", end="\r")

        train_loss = total / max(steps, 1)

        model.eval()
        data = torch.tensor(val_ids, dtype=torch.long)
        ix   = torch.randint(0, len(data) - args.block_size, (args.batch_size,))
        xv   = torch.stack([data[i     : i + args.block_size]     for i in ix])
        yv   = torch.stack([data[i + 1 : i + args.block_size + 1] for i in ix])
        with torch.no_grad():
            _, vloss = model(xv, yv)
        val_loss = vloss.item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        ppl = math.exp(min(val_loss, 10))
        print(f"  Epoch {epoch:>3}  |  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"PPL: {ppl:.1f}  |  {time.time()-t0:.1f}s")

        def save(path):
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val, "args": vars(args),
            }, path)

        save(ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            save(best_path)
            print(f"         ↳ Best  PPL: {math.exp(min(best_val,10)):.1f}  ✓")

        scheduler.step()

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    save_loss_curve(train_losses, val_losses, args.out_dir, args.n_layer, args.n_embd)
    return model, tok


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tiny-but-Deep GPT")
    p.add_argument("--data",       choices=["file", "synthetic"], default="synthetic")
    p.add_argument("--file",       default=None)
    p.add_argument("--n-layer",    type=int,   default=8)   # deep!
    p.add_argument("--n-head",     type=int,   default=4)
    p.add_argument("--n-embd",     type=int,   default=64)  # narrow!
    p.add_argument("--block-size", type=int,   default=64)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--out-dir",    default="./output_deep")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--max-tokens", type=int,   default=150)
    p.add_argument("--no-interact",action="store_true")
    args = p.parse_args()

    model, tok = train(args)
    if not args.no_interact:
        interactive(model, tok, args)
