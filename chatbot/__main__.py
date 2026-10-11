"""``uv run python -m chatbot`` prints this instead of doing anything useful."""

from chatbot import __version__

print(
    f"""
PyBot v{__version__}
A tiny seq2seq chatbot built with PyTorch.

Train the model :  uv run python -m chatbot.train
Chat with it    :  uv run python -m chatbot.chat
"""
)
