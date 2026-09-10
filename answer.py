"""Answer a rules question using only the retrieved chunks as context."""

import sys

import ollama

from search import search

CHAT_MODEL = "qwen2.5:7b"


def format_context(results):
