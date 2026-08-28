from __future__ import annotations
import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")

def tokenize(text: str)->list[str]:
    """
      Convert text into normalized lexical tokens
      Technical identifiers such as:
        AUTH-401
        RedisConnectionError
        PG-08006
    are kept as single tokens when possible.

    """
    return [
        match.group(0).lower()
        for match in _TOKEN_PATTERN.finditer(text)
    ]