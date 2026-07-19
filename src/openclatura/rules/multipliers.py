"""
openclatura/rules/multipliers.py

Multiplicative prefixes for IUPAC nomenclature.
Used for repeated substituents, multiple bonds, and identical structural features.

References:
- IUPAC 2013 Recommendations, P-14.2 (multiplicative prefixes)
"""

from dataclasses import dataclass


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


def get(count: int) -> Multiplier:
    """Look up a multiplier by count. Raises KeyError if out of range."""
    return MULTIPLIERS[count]


def basic(count: int) -> str:
    """Return the basic multiplicative prefix (di, tri, tetra, ...)."""
    return MULTIPLIERS[count].basic


def complex_(count: int) -> str:
    """Return the complex multiplicative prefix (bis, tris, tetrakis, ...).
    Trailing underscore avoids shadowing the `complex` builtin.
    """
    return MULTIPLIERS[count].complex
