"""Small grammar utilities shared by assembly formatters."""

import re


def parse_locant(locant):
    text = str(locant)
    match = re.match(r"^(\d+)([a-zA-Z]*)$", text.split("(")[0])
    if match:
        return (1, float(match.group(1)), match.group(2))
    if any(char.isdigit() for char in text):
        numbers = re.findall(r"\d+", text)
        return (1, float(numbers[0]) if numbers else 0.0, text)
    return (2, 0.0, text)


_STEREO_DESCRIPTOR_START = re.compile(r"\(\d+[A-Za-z']*(?:[RS]|[EZ])(?:,\d+[A-Za-z']*(?:[RS]|[EZ]))*\)")


def needs_hyphen(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if (
        right[0].isdigit()
        or right.startswith("N,")
        or right.startswith("N-")
        or right.startswith("N',")
        or right.startswith("N'-")
        # A stereo descriptor opening the right-hand side is a separate
        # italicised element and takes a hyphen: `1-methyl-(5'E)-1'-butyl…`.
        or _STEREO_DESCRIPTOR_START.match(right)
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
