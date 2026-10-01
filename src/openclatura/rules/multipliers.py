"""
openclatura/rules/multipliers.py

Multiplicative prefixes for IUPAC nomenclature.
Used for repeated substituents, multiple bonds, and identical structural features.

References:
- IUPAC 2013 Recommendations, P-14.2 (multiplicative prefixes)
"""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Multiplier:
    count: int
    basic: str  # Used for simple substituents: di, tri, tetra...
    complex: str  # Used for substituted substituents: bis, tris, tetrakis...


# "basic" prefixes are used for simple repeated units, e.g. "dimethyl", "trichloro".
# "complex" prefixes (bis, tris, ...) are used when the substituent itself contains
# locants, multiplying prefixes, or would cause ambiguity, e.g.
#   "bis(2-chloroethyl)" not "di(2-chloroethyl)"
#   "tris(hydroxymethyl)" not "trihydroxymethyl"

MULTIPLIERS: dict[int, Multiplier] = {
    2: Multiplier(2, "di", "bis"),
    3: Multiplier(3, "tri", "tris"),
    4: Multiplier(4, "tetra", "tetrakis"),
    5: Multiplier(5, "penta", "pentakis"),
    6: Multiplier(6, "hexa", "hexakis"),
    7: Multiplier(7, "hepta", "heptakis"),
    8: Multiplier(8, "octa", "octakis"),
    9: Multiplier(9, "nona", "nonakis"),
    10: Multiplier(10, "deca", "decakis"),
    11: Multiplier(11, "undeca", "undecakis"),
    12: Multiplier(12, "dodeca", "dodecakis"),
    13: Multiplier(13, "trideca", "tridecakis"),
    14: Multiplier(14, "tetradeca", "tetradecakis"),
    15: Multiplier(15, "pentadeca", "pentadecakis"),
    16: Multiplier(16, "hexadeca", "hexadecakis"),
    17: Multiplier(17, "heptadeca", "heptadecakis"),
    18: Multiplier(18, "octadeca", "octadecakis"),
    19: Multiplier(19, "nonadeca", "nonadecakis"),
    20: Multiplier(20, "icosa", "icosakis"),
}

MIN_MULTIPLIER_COUNT = 2
MAX_MULTIPLIER_COUNT = 1000

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


def _validate_numerical_count(count: int, *, minimum: int = 1) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"Numerical count must be an integer from {minimum} through {MAX_MULTIPLIER_COUNT}")
    if not minimum <= count <= MAX_MULTIPLIER_COUNT:
        raise ValueError(f"Numerical count must be from {minimum} through {MAX_MULTIPLIER_COUNT}")


def _under_one_hundred(value: int) -> str:
    if value == 11:
        return "undeca"

    units = value % 10
    tens = value // 10
    unit_term = _UNITS.get(units, "")
    tens_term = _TENS.get(tens, "")
    if unit_term and tens == 2 and unit_term[-1] in "aeiou":
        tens_term = tens_term[1:]
    return unit_term + tens_term


def numerical_term(count: int) -> str:
    """Return the P-14.2.1 basic numerical term for counts through 1000."""

    _validate_numerical_count(count)
    if count == 1000:
        return "kilia"

    hundreds, remainder = divmod(count, 100)
    parts = []
    if remainder:
        parts.append(_under_one_hundred(remainder))
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    return "".join(parts)


def get(count: int) -> Multiplier:
    """Return an explicit or generated multiplier for a supported count."""

    try:
        _validate_numerical_count(count, minimum=MIN_MULTIPLIER_COUNT)
    except ValueError:
        raise KeyError(count) from None
    return _get_cached(count)


@lru_cache(maxsize=MAX_MULTIPLIER_COUNT)
def _get_cached(count: int) -> Multiplier:
    if count in MULTIPLIERS:
        return MULTIPLIERS[count]
    term = numerical_term(count)
    return Multiplier(count, term, f"{term.removesuffix('a')}akis")


def basic(count: int) -> str:
    """Return the basic multiplicative prefix (di, tri, tetra, ...)."""
    return get(count).basic


def complex_(count: int) -> str:
    """Return the complex multiplicative prefix (bis, tris, tetrakis, ...).
    Trailing underscore avoids shadowing the `complex` builtin.
    """
    return get(count).complex


# --------------------------------------------------------------------------- #
# Reading prefixes back off a name
# --------------------------------------------------------------------------- #
# The namer writes multipliers (count -> prefix); anything that *parses* a name
# needs the inverse.  Deriving it from the same table keeps the two directions
# from drifting apart, and means a parser accepts every prefix the namer emits.

COUNTS_BY_PREFIX: dict[str, int] = {
    prefix: mult.count for mult in MULTIPLIERS.values() for prefix in (mult.basic, mult.complex)
}


@lru_cache(maxsize=1)
def _counts_by_prefix() -> dict[str, int]:
    generated = {
        prefix: count
        for count in range(max(MULTIPLIERS) + 1, MAX_MULTIPLIER_COUNT + 1)
        for prefix in (basic(count), complex_(count))
    }
    return {**COUNTS_BY_PREFIX, **generated}


@lru_cache(maxsize=1)
def _prefixes_longest_first() -> tuple[str, ...]:
    return tuple(sorted(_counts_by_prefix(), key=len, reverse=True))


def count_for(prefix: str) -> int | None:
    """Count for one multiplicative prefix (``di`` and ``bis`` -> 2), else ``None``."""
    return _counts_by_prefix().get(prefix)


def candidate_splits(name: str):
    """Yield every ``(count, rest)`` reading of a leading multiplicative prefix,
    longest prefix first.

    A leading prefix is genuinely ambiguous out of context — ``triazole`` starts
    with ``tri`` but is not three of anything — so callers that can validate
    ``rest`` should walk these and take the first reading that checks out, rather
    than committing to the longest match.
    """

    counts_by_prefix = _counts_by_prefix()
    for prefix in _prefixes_longest_first():
        if name.startswith(prefix):
            yield counts_by_prefix[prefix], name[len(prefix) :]
