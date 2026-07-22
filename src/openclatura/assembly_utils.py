"""Small grammar utilities shared by assembly formatters."""

from .locants import parse_locant as parse_locant


def needs_hyphen(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if (
        right[0].isdigit()
        or right.startswith("N,")
        or right.startswith("N-")
        or right.startswith("N',")
        or right.startswith("N'-")
    ):
        return True
    if left[-1].isdigit():
        return True
    return False


def is_fully_enclosed(text: str) -> bool:
    if not text.startswith("(") or not text.endswith(")"):
        return False
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and index < len(text) - 1:
            return False
    return depth == 0
