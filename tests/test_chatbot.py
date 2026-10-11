"""Small, fast tests that pin down what each part of the pipeline promises.

Run them with:  uv run pytest
"""

from __future__ import annotations

import torch

from chatbot.dataset import DialogueDataset, collate_fn, load_pairs
from chatbot.model import build_model
from chatbot.vocabulary import Vocabulary, ensure_nltk_data, tokenize


def test_tokenize_lowercases() -> None:
    ensure_nltk_data()
    assert tokenize("Hello World!") == ["hello", "world", "!"]


def test_encode_decode_round_trip() -> None:
    ensure_nltk_data()
    vocab = Vocabulary.from_tokens([tokenize("hello world")])

    ids = vocab.encode("hello world")

    assert ids[0] == vocab.stoi["hello"]  # words first...
    assert ids[-1] == vocab.eos_idx  # ...finished by <eos>
    assert vocab.decode(ids) == "hello world"  # and back to text again


def test_unknown_words_become_unk() -> None:
    ensure_nltk_data()
    vocab = Vocabulary.from_tokens([tokenize("hello")])

    ids = vocab.encode("flibbertigibbet")

    assert ids == [vocab.unk_idx, vocab.eos_idx]
    assert vocab.decode(ids) == ""


def test_encode_target_is_wrapped_in_sos_and_eos() -> None:
    ensure_nltk_data()
    vocab = Vocabulary.from_tokens([tokenize("hello")])

    ids = vocab.encode_target("hello")

    assert ids[0] == vocab.sos_idx  # decoder input #1
    assert ids[-1] == vocab.eos_idx  # "stop talking"
    assert vocab.decode(ids) == "hello"


def test_special_tokens_have_stable_indexes() -> None:
    vocab = Vocabulary()
    assert vocab.pad_idx == 0
    assert vocab.sos_idx == 1
    assert vocab.eos_idx == 2
    assert vocab.unk_idx == 3


def test_collate_fn_pads_to_the_longest_sequence() -> None:
    source, target = collate_fn([([4, 5, 6], [4]), ([4], [4, 5])])

    assert source.tolist() == [[4, 5, 6], [4, 0, 0]]  # padded with 0 (<pad>)
    assert target.tolist() == [[4, 0], [4, 5]]
    assert source.dtype == torch.long


def test_dataset_loads_the_bundled_conversations() -> None:
    ensure_nltk_data()
    pairs = load_pairs()

    assert len(pairs) >= 100
    assert all(source and target for source, target in pairs)

    vocab = Vocabulary.from_tokens([tokenize(text) for pair in pairs for text in pair])
    dataset = DialogueDataset(pairs, vocab)
    assert len(dataset) == len(pairs)

    source_ids, target_ids = dataset[0]
    assert source_ids[-1] == vocab.eos_idx
    # regression guard: if <sos> is missing, the model learns the wrong shift
    assert target_ids[0] == vocab.sos_idx


def test_model_forward_returns_one_prediction_per_target_word() -> None:
    model = build_model(vocab_size=20, embed_dim=8, hidden_dim=16)
    source = torch.randint(4, 20, (2, 5))  # no <pad> in these rows
    target = torch.randint(4, 20, (2, 7))

    logits = model(source, target)

    assert logits.shape == (2, 7, 20)


def test_greedy_decode_produces_a_bounded_reply() -> None:
    model = build_model(vocab_size=20, embed_dim=8, hidden_dim=16)
    model.eval()
    source = torch.randint(4, 20, (1, 4))

    reply = model.greedy_decode(source, max_len=10)

    assert reply.shape[0] == 1  # one reply for the one input
    assert reply.shape[1] <= 10  # never longer than max_len
    assert reply.max() < 20  # only real vocabulary ids
