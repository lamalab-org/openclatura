# openclatura/rules/elements.py
from dataclasses import dataclass

from ..naming_data import load_json_table


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    atomic_number: int
    standard_valence: int
    hw_stem: str | None
    hw_priority: int | None
    substituent_prefix: str | None
    fusion_special_priority: int | None = None
    fusion_general_priority: int | None = None
    mancude_pi_capacity: int = 1
    mancude_forced_single: bool = False
    fusion_supported: bool = False


def _load_elements() -> dict[str, Element]:
    table = load_json_table("elements.json")
    if table.get("schema_version") != 1 or not isinstance(table.get("elements"), list):
        raise ValueError("elements.json must use schema version 1 and contain an elements list")
    result: dict[str, Element] = {}
    for row in table["elements"]:
        element = Element(**row)
        if element.symbol in result:
            raise ValueError(f"duplicate element symbol {element.symbol!r}")
        if element.mancude_forced_single and element.mancude_pi_capacity:
            raise ValueError(f"forced-single element {element.symbol!r} cannot have pi capacity")
        result[element.symbol] = element
    return result


ELEMENTS: dict[str, Element] = _load_elements()


def get(symbol: str) -> Element:
    return ELEMENTS[symbol]


def is_known(symbol: str) -> bool:
    return symbol in ELEMENTS


# Skeletal-replacement ("a") prefix -> element symbol: ``oxa`` is an oxygen,
# ``aza`` a nitrogen.  The namer writes these from ``Element.hw_stem``; inverting
# the same table is what lets a parser read back exactly what the namer can emit.
SYMBOLS_BY_HW_STEM: dict[str, str] = {
    element.hw_stem: element.symbol for element in ELEMENTS.values() if element.hw_stem
}

# Shared chemistry classification derived once from the checked-in table.
MANCUDE_FORCED_SINGLE_SYMBOLS: frozenset[str] = frozenset(
    element.symbol for element in ELEMENTS.values() if element.mancude_forced_single
)
