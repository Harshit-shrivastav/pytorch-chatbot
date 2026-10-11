"""Talk to the trained chatbot.

Usage:
    uv run python -m chatbot.chat              # interactive chat
    uv run python -m chatbot.chat --message hi # one-shot reply (great for tests)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chatbot.config import CHECKPOINT_PATH, MAX_REPLY_LENGTH
from chatbot.model import build_model
from chatbot.vocabulary import Vocabulary, ensure_nltk_data

FALLBACK_REPLY = "I am not sure what to say to that. Could you rephrase it?"
EXIT_WORDS = {"quit", "exit", "q", "bye bye"}


def load_chatbot(checkpoint_path: Path) -> tuple[Vocabulary, torch.nn.Module, torch.device]:
    """Rebuild the vocabulary and the network from a checkpoint file."""
    if not checkpoint_path.exists():
        raise SystemExit(
            f"No checkpoint found at '{checkpoint_path}'.\n"
            "Train the model first:  uv run python -m chatbot.train"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = Vocabulary.from_words(checkpoint["itos"])
    config = checkpoint["config"]

    model = build_model(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        pad_idx=config["pad_idx"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return vocab, model.to(device), device


def reply(message: str, vocab: Vocabulary, model: torch.nn.Module, device: torch.device) -> str:
    """Generate one reply for a single message."""
    source = torch.tensor([vocab.encode(message)], dtype=torch.long).to(device)
    token_ids = model.greedy_decode(source, max_len=MAX_REPLY_LENGTH)[0].tolist()
    return vocab.decode(token_ids) or FALLBACK_REPLY


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the trained PyTorch chatbot.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--message", type=str, default=None, help="reply once and exit")
    args = parser.parse_args()

    ensure_nltk_data()
    vocab, model, device = load_chatbot(args.checkpoint)

    if args.message is not None:
        print(reply(args.message, vocab, model, device))
        return

    print("PyBot is ready! Type 'quit' to stop.\n")
    while True:
        try:
            message = input("you : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nPyBot: Goodbye!")
            break
        if not message:
            continue
        if message.lower() in EXIT_WORDS:
            print("PyBot: Goodbye! It was nice chatting with you.")
            break
        print(f"PyBot: {reply(message, vocab, model, device)}")


if __name__ == "__main__":
    main()
