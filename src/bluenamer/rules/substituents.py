"""Compatibility accessors for prefix-only functional-group rows."""

from dataclasses import dataclass

from ..nomenclature import RULES


@dataclass(frozen=True)
class Substituent:
    key: str
    prefix: str
    needs_locant: bool


def _to_substituent(key: str) -> Substituent:
    rule = RULES.functional_groups.get(key)
    if rule.role != "prefix" or rule.prefix is None:
        raise KeyError(key)
    return Substituent(key=rule.key, prefix=rule.prefix, needs_locant=rule.needs_locant)


SUBSTITUENTS: dict[str, Substituent] = {
    key: _to_substituent(key)
    for key, rule in RULES.functional_groups.by_key.items()
    if rule.role == "prefix" and rule.prefix
}


def get(key: str) -> Substituent:
    return SUBSTITUENTS[key]


def is_known(key: str) -> bool:
    return key in SUBSTITUENTS
