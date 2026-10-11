"""Text in, numbers out.

Torch works with tensors, so every word needs an index. This module also
downloads the NLTK tokenizer data on first use.
"""

from __future__ import annotations

from collections import Counter

import nltk

# The four tokens below are reserved, they get indexes 0 to 3.
PAD_TOKEN = "<pad>"  # padding, keeps every row of a batch the same length
SOS_TOKEN = "<sos>"  # first input of the decoder
EOS_TOKEN = "<eos>"  # decoder emits this when the reply is finished
UNK_TOKEN = "<unk>"  # a word the model never saw during training


def ensure_nltk_data() -> None:
    """Download the NLTK tokenizer data if it is missing (safe to call often)."""
    for package in ("punkt", "punkt_tab"):
        nltk.download(package, quiet=True)


def tokenize(text: str) -> list[str]:
    """Split a sentence into lowercase word tokens."""
    return nltk.word_tokenize(text.lower())


class Vocabulary:
    """A word <-> integer mapping that always keeps the special tokens first.

    Index meanings:
        0 -> <pad>, 1 -> <sos>, 2 -> <eos>, 3 -> <unk>, 4+ -> normal words
    """

    def __init__(self) -> None:
        self.itos: list[str] = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.stoi: dict[str, int] = {token: i for i, token in enumerate(self.itos)}

    @classmethod
    def from_tokens(cls, token_lists: list[list[str]], min_freq: int = 1) -> Vocabulary:
        """Build a vocabulary from tokenized sentences."""
        vocab = cls()
        counter: Counter[str] = Counter(token for tokens in token_lists for token in tokens)
        for word, freq in sorted(counter.items()):
            if freq >= min_freq and word not in vocab.stoi:
                vocab.stoi[word] = len(vocab.itos)
                vocab.itos.append(word)
        return vocab

    @classmethod
    def from_words(cls, words: list[str]) -> Vocabulary:
        """Rebuild a vocabulary from a saved list of words (used when loading)."""
        vocab = cls()
        for word in words:
            if word not in vocab.stoi:
                vocab.stoi[word] = len(vocab.itos)
                vocab.itos.append(word)
        return vocab

    def encode(self, text: str) -> list[int]:
        """Sentence -> list of ids, finished with <eos>. Unknown words -> <unk>."""
        unk = self.stoi[UNK_TOKEN]
        return [self.stoi.get(token, unk) for token in tokenize(text)] + [self.stoi[EOS_TOKEN]]

    def encode_target(self, text: str) -> list[int]:
        """Reply -> ids, wrapped in <sos> ... <eos>.

        The <sos> at the start is what the decoder gets as its very first input,
        so training and chat time look exactly the same to the model.
        """
        return [self.sos_idx] + self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """List of ids -> sentence, stopping at the first <eos>/<pad>."""
        skip = {UNK_TOKEN, SOS_TOKEN}
        words = []
        for idx in ids:
            if idx in (self.stoi[EOS_TOKEN], self.stoi[PAD_TOKEN]):
                break
            if 0 <= idx < len(self.itos) and self.itos[idx] not in skip:
                words.append(self.itos[idx])
        return " ".join(words)

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.stoi[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)
