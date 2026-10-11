"""Dataset and collate function that turn the JSON pairs into padded batches."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from chatbot.vocabulary import Vocabulary

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "conversations.json"


def load_pairs(path: Path = DEFAULT_DATA_PATH) -> list[tuple[str, str]]:
    """Load the (user message, bot reply) pairs from the JSON file."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return [(pair["input"], pair["output"]) for pair in data["conversations"]]


class DialogueDataset(Dataset):
    """Each item is a conversation pair already converted to token ids.

    ``source`` ends with <eos> (the encoder just reads it), while ``target`` is
    wrapped in <sos> ... <eos> because the decoder needs a signal to start and
    to stop talking.
    """

    def __init__(self, pairs: list[tuple[str, str]], vocab: Vocabulary) -> None:
        self.vocab = vocab
        self.sources = [vocab.encode(source) for source, _ in pairs]
        self.targets = [vocab.encode_target(target) for _, target in pairs]

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.sources[index], self.targets[index]


def collate_fn(batch: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad every sequence in the batch so they can share one tensor.

    Returns:
        ``source``: (batch, max_source_len) and ``target``: (batch, max_target_len)
    """
    sources, targets = zip(*batch)
    # pad_sequence stacks 1D tensors and fills the gaps with the pad token (0).
    source_tensor = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in sources],
        batch_first=True,
        padding_value=0,
    )
    target_tensor = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in targets],
        batch_first=True,
        padding_value=0,
    )
    return source_tensor, target_tensor
