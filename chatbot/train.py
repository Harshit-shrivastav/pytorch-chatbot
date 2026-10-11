"""Train the chatbot.

Usage:
    uv run python -m chatbot.train
    uv run python -m chatbot.train --epochs 300
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chatbot.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EMBED_DIM,
    EPOCHS,
    HIDDEN_DIM,
    LEARNING_RATE,
    LR_DECAY,
    MIN_FREQ,
    SEED,
    TEACHER_FORCING,
)
from chatbot.dataset import DEFAULT_DATA_PATH, DialogueDataset, collate_fn, load_pairs
from chatbot.model import build_model
from chatbot.vocabulary import Vocabulary, ensure_nltk_data, tokenize


def train(args: argparse.Namespace) -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # data
    ensure_nltk_data()
    pairs = load_pairs(args.data)
    print(f"Loaded {len(pairs)} conversation pairs from {args.data}")

    inputs = [tokenize(source) for source, _ in pairs]
    targets = [tokenize(target) for _, target in pairs]
    vocab = Vocabulary.from_tokens(inputs + targets, min_freq=MIN_FREQ)
    print(f"Vocabulary size: {len(vocab)} words")

    dataset = DialogueDataset(pairs, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # model
    model = build_model(len(vocab), EMBED_DIM, HIDDEN_DIM, vocab.pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Slow down halfway through: big steps find the region, small steps refine it.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 2), gamma=LR_DECAY
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # training loop
    print(f"\nTraining for {args.epochs} epochs...")
    model.train()
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for source, target in loader:
            source, target = source.to(device), target.to(device)

            optimizer.zero_grad()
            # (batch, target_len, vocab_size) -> the model's guess at every step
            predictions = model(source, target, teacher_forcing=TEACHER_FORCING)

            # Compare step 1..n with the real reply (step 0 is <sos>, not a word).
            output_dim = predictions.size(-1)
            loss = criterion(
                predictions[:, 1:, :].reshape(-1, output_dim),
                target[:, 1:].reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # keep training stable
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(loader)
            print(
                f"  epoch {epoch:>{len(str(args.epochs))}}/{args.epochs}"
                f" | loss {avg_loss:.4f}"
                f" | perplexity {math.exp(avg_loss):8.2f}"
                f" | lr {scheduler.get_last_lr()[0]:.5f}"
            )

    print(f"Done in {time.time() - start:.1f}s")

    # save the vocabulary together with the weights, chat.py needs both
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "itos": vocab.itos,
            "config": {
                "vocab_size": len(vocab),
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "pad_idx": vocab.pad_idx,
            },
        },
        args.checkpoint,
    )
    print(f"\nCheckpoint saved to {args.checkpoint}")

    # a few replies so you can see it working without opening the chat
    model.eval()
    print("\nSample replies:")
    for question in ("hi", "how are you", "tell me a joke"):
        source = torch.tensor([vocab.encode(question)], dtype=torch.long).to(device)
        answer = vocab.decode(model.greedy_decode(source)[0].tolist())
        print(f"  you : {question}")
        print(f"  bot : {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PyTorch chatbot.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="number of training epochs")
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_PATH, help="where to save the model"
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH, help="path to conversations.json"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
