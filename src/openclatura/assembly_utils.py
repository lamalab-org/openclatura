"""Small grammar utilities shared by assembly formatters."""

import re
from functools import lru_cache

_SUPERSCRIPT_TO_ASCII = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_LOCANT_RE = re.compile(r"^(\d+)([a-zA-Z]*)(\^\d+|[⁰¹²³⁴-⁹]*)$")


def parse_locant(locant):
    """Return a sortable representation of a locant.

    Honors a ``display`` attribute when one is present, so DisplayLocant sorts
    by what it renders as rather than by its underlying integer.
    """

    display = getattr(locant, "display", None)
    text = str(locant) if display is None else str(display)
    return _parse_locant_text(text)


@lru_cache(maxsize=4096)
def _parse_locant_text(text: str):
    """Parse the rendered form, which is all the result depends on.

    Memoising the caller instead would be wrong: DisplayLocant is an int
    subclass, so DisplayLocant(4, "4a") is equal to and hashes like 4 while
    parsing differently. The rendered text is the real argument.
    """

    # P-14.3.5 cites an unsuperscripted locant before its superscripted
    # counterpart, 3a before 3a¹, and compares locants by value. The
    # interior-atom locants of P-25.3.3.3 carry that superscript, and matching
    # letters alone sent them to the fallback below, which compares the whole
    # label against a bare suffix - so "5a¹" was ranked against "a", a digit
    # sorted ahead of a letter, and the superscripted locant came first.
    #
    # The distance is its own integer field rather than part of the suffix
    # text. Folding it into the suffix fixes 5a before 5a¹ but still orders the
    # distances as characters, which transposes a two-digit one: "¹⁰" sorts
    # before "²", so 5a¹⁰ would precede 5a².
    match = _LOCANT_RE.match(text.split("(")[0])
    if match:
        distance = match.group(3).lstrip("^").translate(_SUPERSCRIPT_TO_ASCII)
        return (1, float(match.group(1)), match.group(2), int(distance) if distance else 0)
    if any(char.isdigit() for char in text):
        numbers = re.findall(r"\d+", text)
        return (1, float(numbers[0]) if numbers else 0.0, text, 0)
    # P-14.3.5: italic-letter locants (N, O, S, N') are cited before numerals: N,4-dimethylbenzamide.
    return (0, 0.0, text, 0)


_STEREO_DESCRIPTOR_START = re.compile(r"\(\d+[A-Za-z']*(?:[RS]|[EZ])(?:,\d+[A-Za-z']*(?:[RS]|[EZ]))*\)")


def needs_hyphen(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if (
        right[0].isdigit()
        or re.match(r"^[NOSP]'*[,-]", right) is not None  # N-, N,N'-, N'''-, O-, S-
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
