# PyTorch Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A chatbot written in PyTorch. It is a sequence to sequence model: an LSTM
encoder reads what you type, an LSTM decoder writes a reply one word at a time,
and a Bahdanau attention layer decides which input words matter at each step.

The dataset ships with the repo, so there is nothing to download. Training runs
on CPU in a couple of minutes and the whole project is managed with `uv`.

```text
you : hi
PyBot: hello ! what can i help you with ?

you : where does harshit study
PyBot: iit madras , he is doing a bs in data science and applications .

you : tell me a joke
PyBot: why do programmers prefer dark mode ? because light attracts bugs !
```

The bot answers questions about me: what I build, where I studied, what I have
shipped, and where to find me. Swap the data file and it can talk about
anything else.

## Features

- Encoder, decoder and attention written from scratch, no `nn.Transformer`, no
  hidden framework magic
- 180 conversation pairs included, no external dataset to fetch
- One checkpoint file holds the weights, the vocabulary and the dimensions, so
  `chat.py` needs nothing else
- Fixed random seed, the same command gives the same model
- CPU only PyTorch wheels, about 190 MB instead of several GB
- `ruff` for linting, 9 unit tests, and a GitHub Actions workflow that runs
  all of it on every push

## How it works

```text
"what has harshit built"
        |
        v
   encoder LSTM -> one hidden state per word
        |
        v
   attention -> weighted mix of those states (the context)
        |
        v
   decoder LSTM -> next word -> next word -> ... -> <eos>
```

1. **Tokenize**. NLTK splits the sentence into lowercase words.
2. **Encode**. Each word becomes an integer, then a learned 64 dim vector.
3. **Compress**. The encoder LSTM returns a hidden state for every input word.
4. **Attend**. At each step the attention layer mixes those states into one
   context vector, ignoring padding.
5. **Decode**. Starting from `<sos>`, the decoder emits one word at a time
   until it produces `<eos>` or hits 20 words.

A note on `<sos>`: the decoder needs to know when to start talking. Training
feeds `<sos>` as its first input and chat time does the same. Without that the
model learns a word shift it never sees at inference, and replies fall apart on
the very first word. I ran into exactly that, and there is a regression test
for it in `tests/test_chatbot.py`.

## Quickstart

Needs [`uv`](https://docs.astral.sh/uv/getting-started/installation/), Python 3.10 or newer.

```bash
uv sync                          # creates .venv, pulls CPU-only PyTorch
uv run python -m chatbot.train   # trains, writes checkpoints/pytorch_chatbot.pt
uv run python -m chatbot.chat    # talk to it
```

Other commands:

```bash
uv run python -m chatbot.train --epochs 500          # train longer
uv run python -m chatbot.chat --message "hi there"   # one reply, no prompt
uv run pytest -q                                     # unit tests
uv run ruff check .                                  # lint
```

## Project structure

```text
pytorch-chatbot/
├── chatbot/
│   ├── __init__.py     # version string
│   ├── __main__.py     # python -m chatbot prints usage
│   ├── config.py       # hyperparameters
│   ├── vocabulary.py   # text <-> indexes, special tokens, NLTK tokenizer
│   ├── dataset.py      # reads the JSON pairs, pads them into batches
│   ├── model.py        # Encoder, Decoder (attention), Seq2Seq
│   ├── train.py        # training loop, checkpoint saving
│   └── chat.py         # loads the checkpoint, interactive prompt
├── data/
│   └── conversations.json   # the training data, edit this
├── tests/
│   └── test_chatbot.py      # uv run pytest
├── checkpoints/             # written by training, git ignored
├── .github/workflows/ci.yml # lint, test, train, chat on push
├── pyproject.toml           # dependencies, uv, ruff
└── uv.lock
```

## Changing what it knows

Open [`data/conversations.json`](data/conversations.json) and add a pair. No
code change needed:

```json
{ "input": "what is 2 + 2", "output": "4, of course !" }
```

Then train again with `uv run python -m chatbot.train`.

The knobs live in [`chatbot/config.py`](chatbot/config.py):

| Setting | Default | Effect |
| --- | --- | --- |
| `EPOCHS` | `300` | More epochs, lower loss. Past a point it only memorises harder. |
| `HIDDEN_DIM` | `128` | Bigger LSTM, more capacity, slower epochs. |
| `LEARNING_RATE` | `0.005` | Too high and it diverges, too low and it crawls. |
| `LR_DECAY` | `0.5` | Multiplier applied to the learning rate halfway through. |
| `TEACHER_FORCING` | `0.5` | Chance of feeding the correct word instead of the model's own guess. |
| `MIN_FREQ` | `1` | Drops words that appear less often than this. |

## Limitations

- 180 pairs is a small dataset. The model memorises the questions it was
  trained on and wanders on anything else.
- No memory. Every message is answered on its own, context does not carry over.
- Replies are lowercase and tokenized, so you get `hello ! how can i help you ?`
  rather than `Hello! How can I help you?`. Formatting the output is a
  separate problem from generating it.
- Greedy decoding picks the most likely word every time, so the bot is
  deterministic. Sampling would give more variety and more nonsense.

## Things worth adding

- A larger corpus, for example the Cornell Movie Dialogs
- BLEU or perplexity evaluation on a held out split
- Temperature / top-k sampling instead of greedy decoding
- A Transformer decoder to replace the LSTM

## Author

Harshit Shrivastav, Delhi.

- Portfolio: [me.h-s.site](https://me.h-s.site)
- GitHub: [github.com/Harshit-shrivastav](https://github.com/Harshit-shrivastav)
- LinkedIn: [linkedin.com/in/harshits-shrivastav](https://linkedin.com/in/harshits-shrivastav)
- YouTube: [youtube.com/@pyrogrammers](https://youtube.com/@pyrogrammers)
- Kaggle: [kaggle.com/theharshitshrivastav](https://www.kaggle.com/theharshitshrivastav)

## License

MIT, see [LICENSE](LICENSE).
