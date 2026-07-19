# openclatura/rules/bonds.py
from dataclasses import dataclass


@dataclass(frozen=True)
class BondType:
    key: str
    order: int
    saturated_suffix: str
    suffix: str
    needs_locant: bool
    multi_infix: dict[int, str]


BONDS: dict[str, BondType] = {
    "single": BondType(
        key="single",
        order=1,
        saturated_suffix="an",
        suffix="an",
        needs_locant=False,
        multi_infix={},
    ),
    "double": BondType(
        key="double",
        order=2,
        saturated_suffix="",
        suffix="en",
        needs_locant=True,
        multi_infix={
            2: "adien",
            3: "atrien",
            4: "atetraen",
            5: "apentaen",
            6: "ahexaen",
            7: "aheptaen",
            8: "aoctaen",
            9: "anonaen",
            10: "adecaen",
        },
    ),
    "triple": BondType(
        key="triple",
        order=3,
        saturated_suffix="",
        suffix="yn",
        needs_locant=True,
        multi_infix={
            2: "adiyn",
            3: "atriyn",
            4: "atetrayn",
            5: "apentayn",
            6: "ahexayn",
            7: "aheptayn",
            8: "aoctayn",
            9: "anonayn",
            10: "adecayn",
        },
    ),
}

PARENT_TERMINAL_VOWEL: str = "e"


def get(key: str) -> BondType:
    return BONDS[key]


def unsaturation_infix(bond_key: str, count: int) -> str:
    bt = BONDS[bond_key]
    if count == 1:
        return bt.suffix
    return bt.multi_infix[count]
