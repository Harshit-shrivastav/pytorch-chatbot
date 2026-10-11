"""The model: an encoder-decoder (seq2seq) with Bahdanau attention.

The encoder reads the input sentence and returns one hidden state per word.
On every decoding step the attention layer mixes those states into a context
vector, the decoder LSTM uses it to guess the next word, and the loop stops
once <eos> is produced.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Reads the input sentence and compresses it into hidden states."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # src: (batch, src_len)  ->  outputs: (batch, src_len, hidden)
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Decoder(nn.Module):
    """Produces one reply word at a time, paying attention to the input."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pad_idx = pad_idx
        # Bahdanau attention: score = v * tanh(W * [encoder_output ; decoder_hidden])
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_v = nn.Linear(hidden_dim, 1, bias=False)
        # LSTM input = the new word (embed_dim) + the context (hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(embed_dim + hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def attend(
        self, encoder_outputs: torch.Tensor, hidden: torch.Tensor, src: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix the encoder outputs into one context vector for this step."""
        # hidden: (batch, hidden) -> (batch, src_len, hidden)
        hidden_expanded = hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        energy = torch.tanh(self.attention(torch.cat([encoder_outputs, hidden_expanded], dim=2)))
        scores = self.attention_v(energy).squeeze(2)  # (batch, src_len)

        # Never look at padding tokens: give them a huge negative score.
        scores = scores.masked_fill(src == self.pad_idx, float("-inf"))
        weights = F.softmax(scores, dim=1)  # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden)
        return context, weights

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_token: (batch,)
        embedded = self.dropout(self.embedding(input_token))  # (batch, embed_dim)
        context, weights = self.attend(encoder_outputs, hidden[-1], src)

        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # (batch, 1, E + H)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded], dim=1)
        )  # (batch, vocab_size)
        return prediction, hidden, cell, weights


class Seq2Seq(nn.Module):
    """Glues an encoder and a decoder together for training and inference."""

    def __init__(
        self, encoder: Encoder, decoder: Decoder, sos_idx: int = 1, eos_idx: int = 2
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx  # <sos> kicks off the decoder
        self.eos_idx = eos_idx  # <eos> means "reply finished"

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing: float = 0.5
    ) -> torch.Tensor:
        """Predict the whole reply at once during training.

        Args:
            src:  (batch, src_len)  the input sentence
            tgt:  (batch, tgt_len)  the reply the model should learn, starts with <sos>
            teacher_forcing: probability of feeding the *correct* word to the next step

        Returns:
            logits of shape (batch, tgt_len, vocab_size)
        """
        batch, tgt_len = tgt.shape
        outputs = torch.zeros(batch, tgt_len, self.decoder.fc_out.out_features, device=src.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        # The first word the decoder sees is always <sos>.
        input_token = tgt[:, 0]
        for step in range(1, tgt_len):
            prediction, hidden, cell, _ = self.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, src
            )
            outputs[:, step] = prediction

            use_teacher = random.random() < teacher_forcing
            best_guess = prediction.argmax(dim=1)
            input_token = tgt[:, step] if use_teacher else best_guess

        return outputs

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 20) -> torch.Tensor:
        """Generate a reply word by word (used by the chat script)."""
        batch = src.size(0)
        encoder_outputs, hidden, cell = self.encoder(src)

        input_token = torch.full((batch,), self.sos_idx, dtype=torch.long, device=src.device)
        collected = []

        for _ in range(max_len):
            prediction, hidden, cell, _ = self.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, src
            )
            input_token = prediction.argmax(dim=1)  # most likely next word
            collected.append(input_token)
            if (input_token == self.eos_idx).all():  # the reply is finished
                break

        if not collected:
            return torch.empty(batch, 0, dtype=torch.long, device=src.device)
        return torch.stack(collected, dim=1)


def build_model(
    vocab_size: int,
    embed_dim: int,
    hidden_dim: int,
    pad_idx: int = 0,
    sos_idx: int = 1,
    eos_idx: int = 2,
) -> Seq2Seq:
    """Create a fresh encoder-decoder model."""
    encoder = Encoder(vocab_size, embed_dim, hidden_dim, pad_idx)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim, pad_idx)
    return Seq2Seq(encoder, decoder, sos_idx, eos_idx)
