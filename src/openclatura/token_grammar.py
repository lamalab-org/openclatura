"""Shared lexical token grammar for rendered nomenclature text.

This module intentionally stays small and dependency-light. Renderers should
emit graph ownership metadata directly; this scanner only splits already
rendered terms into stable visible tokens and provides shared locant
predicates for final metadata placement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

ELEMENT_LOCANTS = frozenset({"N", "O", "P", "S"})

_LEXICAL_TOKEN_RE = re.compile(
    r"""
    [A-Za-z]+(?:\^[0-9]+)?
    |
    \d+(?:,\d+)*(?:\([0-9,]+\))?
    """,
    re.VERBOSE,
)
_LOCANT_RE = re.compile(r"(?:\d+|[NOPS])(?:,(?:\d+|[NOPS]))*(?:\([0-9,]+\))?")


class TokenBindingLike(Protocol):
    """Minimal structural protocol for token bindings."""

    text: str
    token_kind: str
    locants: tuple[str, ...]


@dataclass(frozen=True)
class LexicalToken:
    """One visible token span in a rendered name."""

    text: str
    start: int
    end: int


def lexical_token_spans(text: str, offset: int = 0) -> tuple[LexicalToken, ...]:
    """Return lexical token spans for visible nomenclature text."""

    return tuple(
        LexicalToken(match.group(0), offset + match.start(), offset + match.end())
        for match in _LEXICAL_TOKEN_RE.finditer(text)
    )


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Return visible lexical token strings from ``text``."""

    return tuple(token.text for token in lexical_token_spans(text))


def binding_term_tokens(text: str) -> tuple[str, ...]:
    """Return visible token strings for a rendered binding term."""

    return lexical_tokens(text)


def normalize_name_text(text: str) -> str:
    """Return the comparison form used for rendered names and binding terms."""

    return text.lower().replace(" ", "").replace("(", "").replace(")", "")


def locant_tokens_in_text(text: str) -> tuple[str, ...]:
    """Return locant-like tokens visible in ``text``."""

    return tuple(token for token in lexical_tokens(text) if is_locant_token(token))


def is_locant_token(token: str, *, allow_element: bool = True) -> bool:
    """Return whether ``token`` is a numeric or element locant token."""

    token = token.strip().strip("'").strip('"')
    if not token:
        return False
    if token in ELEMENT_LOCANTS:
        return allow_element
    return bool(_LOCANT_RE.fullmatch(token))


def is_locant_binding_token(token_binding: TokenBindingLike) -> bool:
    """Return whether an emitted token binding represents a locant."""

    if token_binding.token_kind == "locant":
        return True
    return bool(token_binding.locants) and is_locant_token(token_binding.text)
