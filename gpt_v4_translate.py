"""
gpt_v4_translate.py  —  Translation GPT
Trains on paired sentence examples (English <-> Spanish by default).
After training, type EN: [sentence] and it attempts to output the translation.
Shows confidence score on each attempt.

NOTE: This is a tiny model on CPU. It won't match Google Translate.
It learns PATTERNS between language pairs, not true semantics.
That's the point — you can see exactly what a small model attempts vs gets wrong.
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
#  BUILT-IN CORPUS  (EN <-> ES paired sentences)
# ──────────────────────────────────────────────────────────

def translation_corpus():
    pairs = [
        ("the cat sat on the mat",           "el gato se sentó en el tapete"),
        ("the dog runs in the park",          "el perro corre en el parque"),
        ("she opened the door slowly",        "ella abrió la puerta lentamente"),
        ("he read the book all night",        "él leyó el libro toda la noche"),
        ("the children played outside",       "los niños jugaron afuera"),
        ("we walked along the river",         "caminamos a lo largo del río"),
        ("the sun rises in the morning",      "el sol sale por la mañana"),
        ("the moon shines at night",          "la luna brilla de noche"),
        ("she smiled and said hello",         "ella sonrió y dijo hola"),
        ("the rain fell on the city",         "la lluvia cayó sobre la ciudad"),
        ("he carried the heavy bag",          "él cargó la bolsa pesada"),
        ("the bird sang in the tree",         "el pájaro cantó en el árbol"),
        ("they ate dinner together",          "ellos cenaron juntos"),
        ("the water is cold and clear",       "el agua está fría y clara"),
        ("she wrote a long letter",           "ella escribió una carta larga"),
        ("the old man sat by the fire",       "el anciano se sentó junto al fuego"),
        ("he found the lost key",             "él encontró la llave perdida"),
        ("the flowers bloom in spring",       "las flores florecen en primavera"),
        ("they laughed at the story",         "ellos se rieron del cuento"),
        ("the train arrived on time",         "el tren llegó a tiempo"),
        ("she cooked a delicious meal",       "ella cocinó una comida deliciosa"),
        ("the wind blew through the trees",   "el viento sopló entre los árboles"),
        ("he spoke quietly to the child",     "él habló en voz baja al niño"),
        ("the stars appeared in the sky",     "las estrellas aparecieron en el cielo"),
        ("we waited for a long time",         "esperamos durante mucho tiempo"),
        ("the cat slept on the chair",        "el gato durmió en la silla"),
        ("she drank a cup of tea",            "ella bebió una taza de té"),
        ("he climbed the tall mountain",      "él escaló la montaña alta"),
        ("the library was full of books",     "la biblioteca estaba llena de libros"),
        ("they ran across the field",         "ellos corrieron por el campo"),
        ("the door was locked",               "la puerta estaba cerrada con llave"),
        ("she painted a beautiful picture",   "ella pintó un cuadro hermoso"),
        ("the river flows to the sea",        "el río fluye hacia el mar"),
        ("he told a funny joke",              "él contó un chiste gracioso"),
        ("the baby cried in the night",       "el bebé lloró en la noche"),
        ("they built a small house",          "ellos construyeron una casa pequeña"),
        ("the teacher explained the lesson",  "el maestro explicó la lección"),
        ("she sang a beautiful song",         "ella cantó una canción hermosa"),
        ("the fire burned all night",         "el fuego ardió toda la noche"),
        ("he swam across the lake",           "él nadó a través del lago"),
        ("the horse ran very fast",           "el caballo corrió muy rápido"),
        ("they found the hidden treasure",    "ellos encontraron el tesoro escondido"),
        ("the snow covered the ground",       "la nieve cubrió el suelo"),
        ("she learned to play the piano",     "ella aprendió a tocar el piano"),
        ("the car stopped at the light",      "el carro se detuvo en el semáforo"),
        ("he fixed the broken window",        "él arregló la ventana rota"),
        ("the market opened early today",     "el mercado abrió temprano hoy"),
        ("they watched the sunset together",  "ellos vieron el atardecer juntos"),
        ("the bread smells fresh and warm",   "el pan huele fresco y caliente"),
        ("she found a coin on the street",    "ella encontró una moneda en la calle"),
    ]

    # format every pair as: EN: ... \nES: ...\n\n
    lines = []
    for en, es in pairs:
        lines.append(f"EN: {en}\nES: {es}\n")

    # repeat many times so the model sees enough examples
    corpus = "\n".join(lines) * 25
    return corpus[:150_000]


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
#  MODEL
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
    def generate_with_score(self, idx, max_new_tokens, temperature=0.5, top_k=20):
        log_probs = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            lp = torch.log(probs[0, next_tok[0, 0]] + 1e-10).item()
            log_probs.append(lp)
            idx = torch.cat([idx, next_tok], dim=1)
            # stop early if we hit a newline (end of translation)
            if next_tok[0, 0].item() == 10:  # newline char
                break
        avg_score = sum(log_probs) / max(len(log_probs), 1)
        return idx, avg_score


# ──────────────────────────────────────────────────────────
#  LOSS CURVE
# ──────────────────────────────────────────────────────────

def save_loss_curve(train_losses, val_losses, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, color="#58a6ff", lw=2, marker="o", ms=5, label="Train")
    ax.plot(epochs, val_losses,   color="#f78166", lw=2, marker="s", ms=5, ls="--", label="Val")

    best    = min(val_losses)
    best_ep = epochs[val_losses.index(best)]
    ax.annotate(f"Best {best:.4f}",
                xy=(best_ep, best), xytext=(best_ep + 0.3, best + 0.05),
                color="#ffd700", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#ffd700", lw=0.8))

    ax.set_xlabel("Epoch", color="#8b949e")
    ax.set_ylabel("Loss",  color="#8b949e")
    ax.set_title("Translation GPT — Loss Curve", color="#ffffff", fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_edgecolor("#21262d")
    ax.grid(True, color="#161b22", lw=0.7)
    ax.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9")
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Loss curve → {path}")


# ──────────────────────────────────────────────────────────
#  INTERACTIVE  (translation mode)
# ──────────────────────────────────────────────────────────

def interactive(model, tokenizer, args):
    src_tag = args.src_tag   # e.g. "EN"
    tgt_tag = args.tgt_tag   # e.g. "ES"

    print("\n" + "═"*52)
    print(f"  TRANSLATION GPT  —  {src_tag} → {tgt_tag}")
    print("═"*52)
    print(f"  Type a sentence in {src_tag} and press Enter.")
    print(f"  The model will attempt to output {tgt_tag}.\n")
    print(f"  Example:  the dog runs in the park")
    print(f"  Also try: {tgt_tag}: ... to translate the other way\n")
    print("  Commands:  :tries 3   :quit")
    print("═"*52 + "\n")

    n_tries = args.n_tries

    def confidence_bar(score):
        # score = avg log prob. map to 0-10 bar
        normalized = max(0.0, min(1.0, (score + 5) / 4))
        filled = round(normalized * 10)
        bar = "█" * filled + "░" * (10 - filled)
        pct = round(normalized * 100)
        return f"[{bar}] {pct}%"

    while True:
        try:
            user_input = input(f"{src_tag}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDone.")
            break

        if not user_input:
            continue
        if user_input == ":quit":
            break
        if user_input.startswith(":tries"):
            try:
                n_tries = int(user_input.split()[1])
                print(f"  Attempts → {n_tries}\n")
            except:
                print("  Usage: :tries 3\n")
            continue

        # build the prompt in the format the model was trained on
        # if user already typed "ES: ..." treat as reverse direction
        if user_input.upper().startswith(f"{tgt_tag}:"):
            prompt = f"{user_input}\n{src_tag}:"
            output_tag = src_tag
        else:
            # strip EN: prefix if they typed it
            sentence = user_input
            if user_input.upper().startswith(f"{src_tag}:"):
                sentence = user_input[len(src_tag)+1:].strip()
            prompt = f"{src_tag}: {sentence}\n{tgt_tag}:"
            output_tag = tgt_tag

        encoded = tokenizer.encode(prompt)
        if not encoded:
            print("  (unknown characters in prompt)\n")
            continue

        model.eval()
        results = []

        for _ in range(n_tries):
            ctx = torch.tensor([encoded], dtype=torch.long)
            out, score = model.generate_with_score(
                ctx, max_new_tokens=80,
                temperature=args.temperature, top_k=20)
            full_text = tokenizer.decode(out[0].tolist())

            # extract just the translation part after the target tag
            tag_marker = f"{output_tag}:"
            if tag_marker in full_text:
                translation = full_text.split(tag_marker)[-1].strip()
                # cut at next newline
                translation = translation.split("\n")[0].strip()
            else:
                translation = full_text[len(prompt):].strip().split("\n")[0]

            results.append((score, translation))

        # sort by confidence
        results.sort(key=lambda x: x[0], reverse=True)

        print(f"\n  {output_tag} translations:")
        print("  " + "─"*48)
        for i, (score, text) in enumerate(results, 1):
            bar = confidence_bar(score)
            label = " ← best" if i == 1 else ""
            print(f"  #{i}  {text}")
            print(f"      Confidence: {bar}{label}")
        print()


# ──────────────────────────────────────────────────────────
#  TRAIN
# ──────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'═'*52}")
    print("  TRANSLATION GPT  —  Windows CPU")
    print(f"{'═'*52}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/4] Loading corpus ...")
    if args.data == "file":
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"  Loaded: {args.file}  ({len(text):,} chars)")
        print(f"  Expected format per line:  {args.src_tag}: sentence")
        print(f"                             {args.tgt_tag}: sentence\n")
    else:
        text = translation_corpus()
        print(f"  Built-in EN/ES corpus ({len(text):,} chars, "
              f"{text.count('EN:'):,} pairs)")

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
        print(f"  Resumed from epoch {ckpt['epoch']}")

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
    p = argparse.ArgumentParser(description="Translation GPT")
    p.add_argument("--data",        choices=["file", "builtin"], default="builtin")
    p.add_argument("--file",        default=None,
                   help="Path to .txt file with paired sentences")
    p.add_argument("--src-tag",     default="EN",
                   help="Source language tag (default: EN)")
    p.add_argument("--tgt-tag",     default="ES",
                   help="Target language tag (default: ES)")
    p.add_argument("--n-layer",     type=int,   default=4)
    p.add_argument("--n-head",      type=int,   default=4)
    p.add_argument("--n-embd",      type=int,   default=128)
    p.add_argument("--block-size",  type=int,   default=64)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--out-dir",     default="./output_translate")
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--n-tries",     type=int,   default=3)
    p.add_argument("--no-interact", action="store_true")
    args = p.parse_args()

    model, tok = train(args)
    if not args.no_interact:
        interactive(model, tok, args)