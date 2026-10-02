# openclatura/rules/bonds.py
from dataclasses import dataclass

from . import multipliers


@dataclass(frozen=True)
class BondType:
    key: str
    order: int
    saturated_suffix: str
    suffix: str
    needs_locant: bool


BONDS: dict[str, BondType] = {
    "single": BondType(
        key="single",
        order=1,
        saturated_suffix="an",
        suffix="an",
        needs_locant=False,
    ),
    "double": BondType(
        key="double",
        order=2,
        saturated_suffix="",
        suffix="en",
        needs_locant=True,
    ),
    "triple": BondType(
        key="triple",
        order=3,
        saturated_suffix="",
        suffix="yn",
        needs_locant=True,
    ),
}

PARENT_TERMINAL_VOWEL: str = "e"


def get(key: str) -> BondType:
    return BONDS[key]


def unsaturation_infix(bond_key: str, count: int) -> str:
    """The parent-stem infix citing ``count`` bonds of ``bond_key``.

    One bond is just the bare suffix (``hex`` + ``en`` -> ``hex-1-ene``); several
    take the multiplicative prefix, joined to the stem by the interfix ``a``
    (``hexa-1,3-dien``).  The prefix is read off the shared multiplier table
    rather than a private copy, so every count supported by the shared
    P-14.2.1 numerical-term generator is spellable here too.
    """

    bt = BONDS[bond_key]
    if count == 1:
        return bt.suffix
    if not bt.needs_locant:
        raise ValueError(f"{bond_key} bonds are not cited with a multiplicity")
    try:
        prefix = multipliers.basic(count)
    except KeyError:
        raise ValueError(f"no multiplicative prefix for {count} {bond_key} bonds") from None
    return f"a{prefix}{bt.suffix}"
