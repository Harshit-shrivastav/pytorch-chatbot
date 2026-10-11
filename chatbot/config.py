"""Hyperparameters used by train.py and chat.py.

Change a value here, then run the training again.
"""

from __future__ import annotations

from pathlib import Path

# model
EMBED_DIM = 64  # size of the vector that represents each word
HIDDEN_DIM = 128  # size of the LSTM memory
MIN_FREQ = 1  # words that appear less often than this are ignored

# training
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.005
LR_DECAY = 0.5  # multiply the learning rate by this after half of the epochs
TEACHER_FORCING = 0.5  # chance of feeding the model the correct word during training
SEED = 42  # makes runs reproducible

# files
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "pytorch_chatbot.pt"
MAX_REPLY_LENGTH = 20  # longest reply the decoder will generate
