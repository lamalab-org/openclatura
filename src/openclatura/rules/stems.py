"""
bluenamer/rules/stems.py

Chain-length stems for IUPAC nomenclature.
Used for alkane parent names, substituent names, and ring stems.

References:
- IUPAC 2013 Recommendations, P-23.2.1 (chain length names)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stem:
    length: int  # Number of carbons (or skeletal atoms)
    stem: str  # Bare stem, e.g. "meth", "eth", "prop"
    retained: bool  # True for 1-4 (meth/eth/prop/but); False for systematic (pent+)


# Stems 1-4 are retained (non-systematic) names.
# Stems 5+ are derived from Greek/Latin numerical roots.
# Coverage up to 30; extend as needed (IUPAC defines stems well beyond this).
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


def get(length: int) -> Stem:
    """Look up a stem by chain length. Raises KeyError if out of range."""
    return STEMS[length]


def stem_for(length: int) -> str:
    """Return just the stem string for a given chain length."""
    return STEMS[length].stem
