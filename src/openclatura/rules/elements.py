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
    mancude_forced_single: bool = False
    fusion_supported: bool = False
    mancude_bonding_limit: int | None = None
    mancude_charged_bonding_limits: tuple[tuple[int, int], ...] = ()

    def mancude_limit_for_charge(self, charge: int) -> int | None:
        """Resolve an explicit fixed-valence limit, leaving donor/lambda sites alone."""

        if self.mancude_bonding_limit is None or charge == 0:
            return self.mancude_bonding_limit
        for state, limit in self.mancude_charged_bonding_limits:
            if state == charge:
                return limit
        raise ValueError(f"unsupported fixed-valence charge {charge:+d} for {self.symbol}")


def _load_elements() -> dict[str, Element]:
    table = load_json_table("elements.json")
    if table.get("schema_version") != 1 or not isinstance(table.get("elements"), list):
        raise ValueError("elements.json must use schema version 1 and contain an elements list")
    result: dict[str, Element] = {}
    for row in table["elements"]:
        element = Element(
            **{key: value for key, value in row.items() if key != "mancude_charged_bonding_limits"},
            mancude_charged_bonding_limits=tuple(
                tuple(entry) for entry in row.get("mancude_charged_bonding_limits", ())
            ),
        )
        if element.mancude_bonding_limit is not None and element.mancude_bonding_limit < element.standard_valence:
            raise ValueError(f"mancude bonding limit is below standard valence for {element.symbol}")
        charged_limits = element.mancude_charged_bonding_limits
        if charged_limits and element.mancude_bonding_limit is None:
            raise ValueError(f"charged mancude limits require a neutral limit for {element.symbol}")
        if any(
            len(entry) != 2 or type(entry[0]) is not int or entry[0] == 0 or type(entry[1]) is not int or entry[1] < 0
            for entry in charged_limits
        ) or len({entry[0] for entry in charged_limits}) != len(charged_limits):
            raise ValueError(f"invalid charged mancude bonding limits for {element.symbol}")
        if element.symbol in result:
            raise ValueError(f"duplicate element symbol {element.symbol!r}")
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
