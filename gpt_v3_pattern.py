"""
gpt_v3_pattern.py  —  Pattern & Lyric GPT
Bigger model (6 layers, 256-dim). Best for structured repetitive text:
song lyrics, poetry, dialogue scripts, chants, verse.
After training: generates N completions and ranks them by confidence score.
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
#  BUILT-IN CORPUS  (structured verse / pattern text)
# ──────────────────────────────────────────────────────────

def pattern_corpus():
    verses = []
    # nursery-rhyme style repetitive verse
    for i in range(40):
        verses.append(f"""verse {i}
the river runs and the river flows
down to the valley where the tall grass grows
the wind calls out and the wind replies
under the watch of the open skies
and we rise and we fall and we rise again
through the cold and the dark and the driving rain
we carry on to the end of the road
we share the weight of a heavy load

chorus
carry on carry on carry on
the night is long but the light will come
carry on carry on carry on
we are many we are one

""")
    for i in range(30):
        verses.append(f"""act {i} scene one
SPEAKER A: What do you want from me?
SPEAKER B: I want what you promised.
SPEAKER A: I never made any promises.
SPEAKER B: You did. You said you would stay.
SPEAKER A: Things change. People change.
SPEAKER B: Not like this. Not so fast.
SPEAKER A: I am sorry. I truly am.
SPEAKER B: Sorry is not enough this time.

""")
    for i in range(20):
        verses.append(f"""pattern block {i}
one step forward two steps back
find the thread and find the track
pull it tight and pull it through
every old thing becomes new
break the pattern break the mold
stories waiting to be told
shape the words and shape the song
short the night but wide the dawn

""")
    base = "".join(verses)
    return (base * 5)[:250_000]


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


# ──────────────────────────────────────────────────────────
#  MODEL  (6 layers, 256-dim — largest of the three)
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
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

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
    def generate_with_score(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Generate tokens and return (text_ids, avg_log_prob) for confidence ranking."""
        log_probs = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            # record log prob of chosen token
            lp = torch.log(probs[0, next_tok[0, 0]] + 1e-10).item()
            log_probs.append(lp)
            idx = torch.cat([idx, next_tok], dim=1)
        avg_score = sum(log_probs) / len(log_probs)
        return idx, avg_score


# ──────────────────────────────────────────────────────────
#  LOSS CURVE
# ──────────────────────────────────────────────────────────

def save_loss_curve(train_losses, val_losses, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, color="#39ff14", lw=2, marker="o", ms=5, label="Train")
    ax.plot(epochs, val_losses,   color="#ff073a", lw=2, marker="s", ms=5, ls="--", label="Val")
    ax.set_xlabel("Epoch", color="#888888")
    ax.set_ylabel("Loss",  color="#888888")
    ax.set_title("Pattern GPT — Loss Curve", color="#ffffff", fontweight="bold")
    ax.tick_params(colors="#666666")
    for sp in ax.spines.values(): sp.set_edgecolor("#222222")
    ax.grid(True, color="#1a1a1a", lw=0.7)
    ax.legend(facecolor="#111111", edgecolor="#222222", labelcolor="#cccccc")
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Loss curve → {path}")


# ──────────────────────────────────────────────────────────
#  INTERACTIVE  (ranked completions)
# ──────────────────────────────────────────────────────────

def interactive(model, tokenizer, args):
    print("\n" + "█"*52)
    print("  PATTERN GPT  —  Ranked Completions Mode")
    print("█"*52)
    print("  Each prompt generates multiple completions.")
    print("  Results are ranked by model confidence (★).\n")
    print("  Commands:")
    print("    :n 5         — number of completions (default 3)")
    print("    :temp 0.9    — sampling temperature")
    print("    :tokens 100  — output length")
    print("    :quit")
    print("█"*52 + "\n")

    n_completions = args.n_completions
    temperature   = args.temperature
    max_tokens    = args.max_tokens

    def stars(score):
        # score is avg log prob, higher (less negative) = more confident
        # map roughly: > -1 = 5 stars, < -4 = 1 star
        normalized = max(0.0, min(1.0, (score + 4) / 3))
        filled = round(normalized * 5)
        return "★" * filled + "☆" * (5 - filled)

    while True:
        try:
            prompt = input("Pattern> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDone.")
            break

        if not prompt:
            continue
        if prompt == ":quit":
            break
        if prompt.startswith(":n"):
            try:
                n_completions = int(prompt.split()[1])
                print(f"  Completions → {n_completions}\n")
            except:
                print("  Usage: :n 5\n")
            continue
        if prompt.startswith(":temp"):
            try:
                temperature = float(prompt.split()[1])
                print(f"  Temperature → {temperature}\n")
            except:
                print("  Usage: :temp 0.9\n")
            continue
        if prompt.startswith(":tokens"):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"  Tokens → {max_tokens}\n")
            except:
                print("  Usage: :tokens 100\n")
            continue

        encoded = tokenizer.encode(prompt)
        if not encoded:
            print("  (unknown characters)\n")
            continue

        model.eval()
        results = []
        print(f"  Generating {n_completions} completions...\n")

        for i in range(n_completions):
            ctx = torch.tensor([encoded], dtype=torch.long)
            out, score = model.generate_with_score(
                ctx, max_new_tokens=max_tokens,
                temperature=temperature, top_k=40)
            text = tokenizer.decode(out[0].tolist())
            results.append((score, text))

        # sort by confidence descending
        results.sort(key=lambda x: x[0], reverse=True)

        print("  " + "─"*48)
        for rank, (score, text) in enumerate(results, 1):
            rating = stars(score)
            print(f"  #{rank}  {rating}  (score: {score:.3f})")
            print(f"  {text}")
            print("  " + "─"*48)
        print()


# ──────────────────────────────────────────────────────────
#  TRAIN
# ──────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'█'*52}")
    print("  PATTERN GPT  —  Windows CPU")
    print(f"{'█'*52}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] Loading corpus ...")
    if args.data == "file":
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"  Loaded: {args.file}  ({len(text):,} chars)")
    else:
        text = pattern_corpus()
        print(f"  Built-in pattern corpus ({len(text):,} chars).")

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

    print("[3/4] Building model ...")
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

        print(f"  Epoch {epoch:>3}  |  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  |  "
              f"{time.time()-t0:.1f}s")

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
            print(f"         ↳ Best val loss: {best_val:.4f}  ✓")

        scheduler.step()

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    save_loss_curve(train_losses, val_losses, args.out_dir)
    return model, tok


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pattern & Lyric GPT")
    p.add_argument("--data",          choices=["file", "builtin"], default="builtin")
    p.add_argument("--file",          default=None)
    p.add_argument("--n-layer",       type=int,   default=6)
    p.add_argument("--n-head",        type=int,   default=4)
    p.add_argument("--n-embd",        type=int,   default=256)
    p.add_argument("--block-size",    type=int,   default=64)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--out-dir",       default="./output_pattern")
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--temperature",   type=float, default=0.9)
    p.add_argument("--max-tokens",    type=int,   default=120)
    p.add_argument("--n-completions", type=int,   default=3)
    p.add_argument("--no-interact",   action="store_true")
    args = p.parse_args()

    model, tok = train(args)
    if not args.no_interact:
        interactive(model, tok, args)
