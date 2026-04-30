"""
gpt_v1_story.py  —  Story Writer GPT
Trains on prose/fiction text and generates creative writing completions.
Feed it any .txt novel, story collection, or screenplay.
Falls back to a built-in fairy tale corpus if no file is given.
"""

import os, time, math, pickle, argparse, textwrap
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────
#  TOKENIZER  (word-level for more natural prose generation)
# ──────────────────────────────────────────────────────────

class WordTokenizer:
    """
    Word-level tokenizer. Splits on whitespace, keeps punctuation attached.
    Produces more readable prose than char-level tokenizers.
    """
    UNK = "<UNK>"

    def __init__(self, max_vocab=3000):
        self.max_vocab = max_vocab
        self.stoi, self.itos = {}, {}

    def fit(self, text):
        from collections import Counter
        words  = text.split()
        counts = Counter(words)
        vocab  = [self.UNK] + [w for w, _ in counts.most_common(self.max_vocab - 1)]
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        return self

    def encode(self, text):
        unk = self.stoi[self.UNK]
        return [self.stoi.get(w, unk) for w in text.split()]

    def decode(self, ids):
        return " ".join(self.itos.get(i, self.UNK) for i in ids)

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
#  BUILT-IN CORPUS  (fairy tale style)
# ──────────────────────────────────────────────────────────

def fairy_tale_corpus():
    tales = [
        """Once upon a time in a kingdom far away there lived a young girl named Elara.
She had hair as dark as midnight and eyes the color of the deep forest.
Every morning she would walk to the well at the edge of the village and draw water for her family.
One day she found a small silver key lying in the mud beside the well.
She picked it up and held it to the light. It shimmered with a strange glow.
That night she dreamed of a door hidden deep in the forest, a door no one had ever opened.
When she woke the key was warm in her hand as if it had been waiting for her.
She packed a small bundle of bread and set off into the trees.
The forest was quiet except for the sound of leaves and distant birds.
After many hours she found the door standing alone between two ancient oaks.
She placed the key in the lock and turned it slowly. The door swung open with a sigh.
Beyond it lay a garden where flowers glowed with soft golden light.
In the center of the garden sat an old woman weaving a tapestry of stars.
The woman looked up and smiled. I have been waiting a very long time she said.
Elara stepped forward and asked what is this place.
The woman replied this is where stories are kept before they are told.
Every thread is a life and every color is a choice.
Elara watched the tapestry shimmer and saw within it her own face looking back at her.
She understood then that she had a story to finish and turned back toward home.""",

        """The prince had searched every corner of the kingdom for the lost harp.
It was said that whoever played it could heal any sorrow.
His mother the queen had not spoken in three years since the harp was stolen.
He traveled through deserts and mountains and over frozen seas.
At last he came to a tower on a cliff above the crashing waves.
Inside sat a figure wrapped in grey robes who did not turn when he entered.
I know why you have come the figure said. The harp is not what you seek.
The prince said I seek it for my mother who has forgotten how to speak.
The figure stood and turned and the prince saw it was his own reflection.
The reflection said the harp plays only what is already inside you.
Your mother does not need the harp. She needs you to remember her song.
The prince stood very still and then began to hum a melody from his childhood.
It was a simple tune his mother had sung at his bedside every night.
He hummed it softly and then louder and the tower walls began to shake.
When he returned home without the harp the queen looked up at him.
He sat beside her and hummed the tune. She closed her eyes and smiled.
Then slowly she began to hum with him and the room filled with light.
They sat together until evening came and the stars appeared one by one.""",

        """Deep beneath the city there was a library that no one remembered building.
Its shelves stretched upward beyond sight and every book held a different world.
A boy named Cass found the entrance behind a loose brick in the subway wall.
He pushed through and fell into a room lit by floating lanterns.
A librarian appeared from between the shelves. She was very tall and very old.
She wore reading glasses and carried a book the size of a door.
Welcome she said. You are the first visitor in forty seven years.
Cass asked what kind of library is this.
She said it is a library of things that almost happened.
Every book here is a story that was nearly true but never quite was.
Cass pulled a book from the shelf and opened it. Inside was a life.
A version of his own life where he had taken a different street home one evening.
He read for what felt like hours. In the other life he had found a dog.
He had named it Marco and they had been inseparable.
When he looked up the librarian was watching him gently.
She said every person has a shelf here. Would you like to see yours.
Cass followed her deep into the library until they reached a section marked with his name.
The shelf was full. He reached out and touched the spines one by one.
Each one hummed softly like a living thing waiting to be chosen.""",
    ]
    # repeat and shuffle to build a reasonable corpus
    corpus = "\n\n".join(tales * 30)
    return corpus[:250_000]


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
#  MODEL  (wider + more dropout for prose diversity)
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
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
#  LOSS CURVE
# ──────────────────────────────────────────────────────────

def save_loss_curve(train_losses, val_losses, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a0a2e")
    ax.set_facecolor("#1a0a2e")
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, color="#c084fc", lw=2, marker="o", ms=5, label="Train")
    ax.plot(epochs, val_losses,   color="#f9a8d4", lw=2, marker="s", ms=5, ls="--", label="Val")
    ax.set_xlabel("Epoch", color="#e2d9f3")
    ax.set_ylabel("Loss",  color="#e2d9f3")
    ax.set_title("Story Writer — Loss Curve", color="#ffffff", fontweight="bold")
    ax.tick_params(colors="#a78bca")
    for sp in ax.spines.values(): sp.set_edgecolor("#3b1f6b")
    ax.grid(True, color="#2d1a4a", lw=0.7)
    ax.legend(facecolor="#2d1a4a", edgecolor="#3b1f6b", labelcolor="#e2d9f3")
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Loss curve → {path}")


# ──────────────────────────────────────────────────────────
#  INTERACTIVE  (story continuation mode)
# ──────────────────────────────────────────────────────────

def interactive(model, tokenizer, args):
    print("\n" + "★"*52)
    print("  STORY WRITER  —  Creative Continuation Mode")
    print("★"*52)
    print("  Type the start of a sentence or scene.")
    print("  The model will continue your story.\n")
    print("  Commands:")
    print("    :temp 1.2    — raise for wilder stories")
    print("    :words 80    — change output word count")
    print("    :again       — regenerate last prompt")
    print("    :quit        — exit")
    print("★"*52 + "\n")

    temperature = args.temperature
    max_words   = args.max_tokens
    last_prompt = None

    def generate(prompt):
        encoded = tokenizer.encode(prompt)
        if not encoded:
            print("  (unknown words in prompt — try different phrasing)\n")
            return
        ctx = torch.tensor([encoded], dtype=torch.long)
        out = model.generate(ctx, max_new_tokens=max_words,
                             temperature=temperature, top_k=50)
        text = tokenizer.decode(out[0].tolist())
        # wrap for readability
        wrapped = textwrap.fill(text, width=70)
        print("\n" + "─"*52)
        print(wrapped)
        print("─"*52 + "\n")

    while True:
        try:
            prompt = input("Story> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThe end.")
            break

        if not prompt:
            continue
        if prompt == ":quit":
            print("The end.")
            break
        if prompt == ":again":
            if last_prompt:
                model.eval()
                generate(last_prompt)
            else:
                print("  No previous prompt.\n")
            continue
        if prompt.startswith(":temp"):
            try:
                temperature = float(prompt.split()[1])
                print(f"  Temperature → {temperature}\n")
            except (IndexError, ValueError):
                print("  Usage: :temp 1.2\n")
            continue
        if prompt.startswith(":words"):
            try:
                max_words = int(prompt.split()[1])
                print(f"  Output words → {max_words}\n")
            except (IndexError, ValueError):
                print("  Usage: :words 80\n")
            continue

        last_prompt = prompt
        model.eval()
        generate(prompt)


# ──────────────────────────────────────────────────────────
#  TRAIN
# ──────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'★'*52}")
    print("  STORY WRITER GPT  —  Windows CPU")
    print(f"{'★'*52}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] Loading corpus ...")
    if args.data == "file":
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"  Loaded: {args.file}  ({len(text):,} chars)")
    else:
        text = fairy_tale_corpus()
        print(f"  Built-in fairy tale corpus ({len(text.split()):,} words)")

    print("[2/4] Tokenizer ...")
    tok_path = os.path.join(args.out_dir, "tokenizer.pkl")
    if args.resume and os.path.exists(tok_path):
        tok = WordTokenizer.load(tok_path)
        print(f"  Loaded (vocab={tok.vocab_size})")
    else:
        tok = WordTokenizer(max_vocab=args.vocab_size).fit(text)
        tok.save(tok_path)
        print(f"  Built (vocab={tok.vocab_size})")

    ids   = tok.encode(text)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"  Train: {len(train_ids):,} tokens  |  Val: {len(val_ids):,} tokens")

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
        print(f"  Resumed from epoch {ckpt['epoch']}")

    print(f"\n[4/4] Training {args.epochs} epoch(s) ...")
    print(f"  batch={args.batch_size}  block={args.block_size}  "
          f"lr={args.lr}  layers={args.n_layer}\n")

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
    p = argparse.ArgumentParser(description="Story Writer GPT")
    p.add_argument("--data",        choices=["file", "builtin"], default="builtin")
    p.add_argument("--file",        default=None, help="Path to .txt story file")
    p.add_argument("--vocab-size",  type=int,   default=2000)
    p.add_argument("--n-layer",     type=int,   default=4)
    p.add_argument("--n-head",      type=int,   default=4)
    p.add_argument("--n-embd",      type=int,   default=256)
    p.add_argument("--block-size",  type=int,   default=64)
    p.add_argument("--dropout",     type=float, default=0.15)
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--out-dir",     default="./output_story")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens",  type=int,   default=60)
    p.add_argument("--no-interact", action="store_true")
    args = p.parse_args()

    model, tok = train(args)
    if not args.no_interact:
        interactive(model, tok, args)
