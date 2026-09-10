"""OPSIN-verified two-fragment heteroatom/carbon path constructions.

P-25.4.2.3 puts the senior simple bridge first, without elision. This tier
has one terminal heteroatom fragment, which is senior to the carbon fragment.
Entries describe the path in that citation direction, not completed-system
numbering order. Do not infer unverified homologues from these constructions.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

from ..naming_data import load_json_table

ConstructionKey = tuple[tuple[str, ...], tuple[int, ...]]


@cache
def localized_carbon_bridge_bond_orders() -> frozenset[tuple[int, ...]]:
    """Finite localized bridge grammar; other paths use completed-system dehydro."""
    rows = load_json_table("fusion_composite_bridges.json")["localized_carbon_bridge_bond_orders"]
    if any(
        not row or 2 not in row or any(type(order) is not int or order not in {1, 2} for order in row) for row in rows
    ):
        raise ValueError("localized carbon bridges require single/double path bond orders")
    return frozenset(tuple(row) for row in rows)


@dataclass(frozen=True, slots=True)
class CompositeBridgeConstruction:
    heteroatom: str
    hetero_prefix: str
    carbon_prefix: str
    carbon_bond_orders: tuple[int, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return (self.heteroatom,) + ("C",) * (len(self.carbon_bond_orders) + 1)

    @property
    def internal_bond_orders(self) -> tuple[int, ...]:
        return (1, *self.carbon_bond_orders)

    @property
    def prefix(self) -> str:
        return f"({self.hetero_prefix}{self.carbon_prefix})"

    @property
    def unsaturation_locants(self) -> tuple[str, ...]:
        # Internal locants belong to the carbon fragment, not the whole path.
        return tuple(str(i) for i, order in enumerate(self.carbon_bond_orders, 1) if order == 2)


@cache
def composite_bridge_constructions() -> Mapping[ConstructionKey, CompositeBridgeConstruction]:
    return composite_bridge_constructions_from_data(load_json_table("fusion_composite_bridges.json"))


def composite_bridge_constructions_from_data(data: dict) -> Mapping[ConstructionKey, CompositeBridgeConstruction]:
    if data.get("schema_version") != 1:
        raise ValueError("unsupported composite bridge construction schema version")
    rows = data.get("constructions")
    if not isinstance(rows, list):
        raise ValueError("composite bridge constructions must be a list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("composite bridge construction must be a mapping")
        for field in ("heteroatom", "hetero_prefix", "carbon_prefix"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"composite bridge construction requires {field}")
        if row["heteroatom"] not in {"O", "S", "Se", "Te", "N", "P", "As", "Sb", "Bi"}:
            raise ValueError("composite bridge must start with a supported heteroatom fragment")
        orders = row.get("carbon_bond_orders")
        if not isinstance(orders, list) or any(type(order) is not int or order not in {1, 2} for order in orders):
            raise ValueError("composite carbon fragment requires single or double path bonds")
        construction = CompositeBridgeConstruction(
            row["heteroatom"], row["hetero_prefix"], row["carbon_prefix"], tuple(orders)
        )
        key = construction.symbols, construction.internal_bond_orders
        if key in indexed:
            raise ValueError("duplicate composite bridge construction graph")
        indexed[key] = construction
    return MappingProxyType(indexed)
