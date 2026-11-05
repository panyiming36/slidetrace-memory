"""Token accounting utilities backed by the same real tokenizer everywhere."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=8)
def get_tokenizer(model: str = "gpt-4o-mini"):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    return len(get_tokenizer(model).encode(text or ""))


def truncate_to_tokens(text: str, limit: int, model: str = "gpt-4o-mini") -> str:
    if limit < 0:
        raise ValueError("token limit must be non-negative")
    encoding = get_tokenizer(model)
    return encoding.decode(encoding.encode(text or "")[:limit])
