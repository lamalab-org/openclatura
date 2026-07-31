"""
openclatura/rules/stems.py

Chain-length stems for IUPAC nomenclature.
Used for alkane parent names, substituent names, and ring stems.

References:
- IUPAC 2013 Recommendations, P-23.2.1 (chain length names)
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from re import Pattern


@dataclass(frozen=True)
class Stem:
    length: int  # Number of carbons (or skeletal atoms)
    stem: str  # Bare stem, e.g. "meth", "eth", "prop"
    retained: bool  # True for 1-4 (meth/eth/prop/but); False for systematic (pent+)


# Stems 1-4 are retained (non-systematic) names.
# Stems 5+ are derived from Greek/Latin numerical roots.
# Retained and established spellings through 30. Larger stems are generated
# from the basic numerical terms in Blue Book P-14.2.1.
STEMS: dict[int, Stem] = {
    1: Stem(1, "meth", retained=True),
    2: Stem(2, "eth", retained=True),
    3: Stem(3, "prop", retained=True),
    4: Stem(4, "but", retained=True),
    5: Stem(5, "pent", retained=False),
    6: Stem(6, "hex", retained=False),
    7: Stem(7, "hept", retained=False),
    8: Stem(8, "oct", retained=False),
    9: Stem(9, "non", retained=False),
    10: Stem(10, "dec", retained=False),
    11: Stem(11, "undec", retained=False),
    12: Stem(12, "dodec", retained=False),
    13: Stem(13, "tridec", retained=False),
    14: Stem(14, "tetradec", retained=False),
    15: Stem(15, "pentadec", retained=False),
    16: Stem(16, "hexadec", retained=False),
    17: Stem(17, "heptadec", retained=False),
    18: Stem(18, "octadec", retained=False),
    19: Stem(19, "nonadec", retained=False),
    20: Stem(20, "icos", retained=False),  # 2013 PIN; older lit uses "eicos"
    21: Stem(21, "henicos", retained=False),
    22: Stem(22, "docos", retained=False),
    23: Stem(23, "tricos", retained=False),
    24: Stem(24, "tetracos", retained=False),
    25: Stem(25, "pentacos", retained=False),
    26: Stem(26, "hexacos", retained=False),
    27: Stem(27, "heptacos", retained=False),
    28: Stem(28, "octacos", retained=False),
    29: Stem(29, "nonacos", retained=False),
    30: Stem(30, "triacont", retained=False),
}

MIN_STEM_LENGTH = 1
MAX_STEM_LENGTH = 1000

_UNITS = {
    1: "hen",
    2: "do",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
}

_TENS = {
    1: "deca",
    2: "icosa",
    3: "triaconta",
    4: "tetraconta",
    5: "pentaconta",
    6: "hexaconta",
    7: "heptaconta",
    8: "octaconta",
    9: "nonaconta",
}

_HUNDREDS = {
    1: "hecta",
    2: "dicta",
    3: "tricta",
    4: "tetracta",
    5: "pentacta",
    6: "hexacta",
    7: "heptacta",
    8: "octacta",
    9: "nonacta",
}


def _validate_length(length: int) -> None:
    if isinstance(length, bool) or not isinstance(length, int):
        raise ValueError("Stem length must be an integer from 1 through 1000")
    if not MIN_STEM_LENGTH <= length <= MAX_STEM_LENGTH:
        raise ValueError("Stem length must be from 1 through 1000")


def _under_one_hundred(value: int) -> str:
    """Return the basic numerical term for a value from 1 through 99."""

    if value == 11:
        return "undeca"

    units = value % 10
    tens = value // 10
    unit_term = _UNITS.get(units, "")
    tens_term = _TENS.get(tens, "")

    # The initial i of icosa is elided after a vowel (P-14.2.1.2), e.g.
    # do + icosa -> docosa, but hen + icosa -> henicosa.
    if unit_term and tens == 2 and unit_term[-1] in "aeiou":
        tens_term = tens_term[1:]
    return unit_term + tens_term


def _numerical_term(length: int) -> str:
    """Build the P-14.2.1 basic numerical term for ``length``."""

    _validate_length(length)
    if length == 1000:
        return "kilia"

    hundreds, remainder = divmod(length, 100)
    parts = []
    if remainder:
        parts.append(_under_one_hundred(remainder))
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    return "".join(parts)


def get(length: int) -> Stem:
    """Return the chain stem for a supported skeletal-atom count."""

    _validate_length(length)
    return _get_cached(length)


@lru_cache(maxsize=MAX_STEM_LENGTH)
def _get_cached(length: int) -> Stem:
    """Return a validated stem while caching generated values."""

    if length in STEMS:
        return STEMS[length]
    numerical_term = _numerical_term(length)
    return Stem(length, numerical_term.removesuffix("a"), retained=False)


def stem_for(length: int) -> str:
    """Return just the stem string for a supported chain length."""

    return get(length).stem


def _iter_supported_stems() -> Iterator[Stem]:
    """Yield supported stems without retaining a second full collection."""

    for length in range(MIN_STEM_LENGTH, MAX_STEM_LENGTH + 1):
        yield get(length)


@lru_cache(maxsize=1)
def _alkyl_suffix_index() -> dict[str, Stem]:
    """Lazily map supported terminal alkyl text to its stem."""

    return {f"{stem.stem}yl": stem for stem in _iter_supported_stems()}


@lru_cache(maxsize=1)
def _basic_prefix_index() -> dict[str, int]:
    """Lazily map supported basic numerical prefixes to atom counts."""

    return {f"{stem.stem}a": stem.length for stem in _iter_supported_stems()}


@lru_cache(maxsize=1)
def _terminal_alkyl_pattern() -> Pattern[str]:
    """Compile a longest-first matcher for supported terminal alkyl text."""

    alternatives = sorted((re.escape(suffix) for suffix in _alkyl_suffix_index()), key=len, reverse=True)
    return re.compile(f"({'|'.join(alternatives)})$")


def terminal_stem(name: str) -> Stem | None:
    """Return the stem represented by the longest terminal alkyl suffix.

    Reverse-search data is constructed only on the first call. An empty or
    unrecognized string returns ``None``; a non-string input raises
    ``ValueError`` because it cannot represent nomenclature text.
    """

    if not isinstance(name, str):
        raise ValueError("Terminal stem name must be a string")
    if not name:
        return None
    match = _terminal_alkyl_pattern().search(name)
    if match is None:
        return None
    return _alkyl_suffix_index()[match.group(1)]


def length_for_basic_prefix(prefix: str) -> int | None:
    """Return the atom count represented by a basic numerical prefix.

    Reverse-search data is constructed only on the first call. An empty or
    unrecognized string returns ``None``; a non-string input raises
    ``ValueError`` because it cannot represent nomenclature text.
    """

    if not isinstance(prefix, str):
        raise ValueError("Basic numerical prefix must be a string")
    if not prefix:
        return None
    return _basic_prefix_index().get(prefix)
