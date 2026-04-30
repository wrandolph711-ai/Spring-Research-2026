# Spring-Research-2026
VGG19 / ViT / ResNet18 — CIFAR-100:
Three image classifiers trained on the CIFAR-100 dataset (100 object categories). Includes a competitive guessing game where a user goes head-to-head against the model to see who can correctly classify images faster.

All models built from scratch — custom tokenizer, causal self-attention, training loop, and interactive mode. No pre-trained weights. Runs fully on CPU.

train_gpt.py — Base GPT:
The foundation model. Trains on any text file or a built-in Python code corpus. After training, drops into an interactive code completion prompt where you type the start of a function and it continues it.
gpt_v1_story.py — Story Writer
Uses a word-level tokenizer instead of character-level for more natural prose output. Trained on fiction/narrative text. Interactive mode lets you type the start of a scene and the model continues the story. Supports :again to regenerate the same prompt with a different result.
gpt_v2_deep.py — Tiny-but-Deep
Explores depth vs width at small scale — 8 transformer layers with only a 64-dim embedding. Every prompt generates three completions side by side at different temperatures (safe / balanced / wild) to demonstrate how sampling temperature affects output.
gpt_v3_pattern.py — Pattern & Lyric
Larger model tuned for structured repetitive text like verse, dialogue, and song lyrics. Interactive mode generates multiple completions and ranks them by model confidence score using average log probability.
gpt_v4_translate.py — Translation
Trained on paired English/Spanish sentence examples. Attempts EN → ES translation with a visual confidence bar on each output. Supports reverse direction (ES → EN) and custom language pairs via --src-tag and --tgt-tag flags.
