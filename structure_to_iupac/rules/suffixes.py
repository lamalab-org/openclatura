"""Compatibility accessors for principal functional-group suffix rows."""

from dataclasses import dataclass

from ..nomenclature import RULES


@dataclass(frozen=True)
class CharacteristicGroup:
    key: str
    seniority: int
    suffix: str
    suffix_with_locant: bool
    prefix: str | None
    multi_suffix: str | None


def _to_characteristic_group(key: str) -> CharacteristicGroup:
    rule = RULES.functional_groups.get(key)
    if rule.seniority is None or rule.suffix is None:
        raise KeyError(key)
    return CharacteristicGroup(
        key=rule.key,
        seniority=rule.seniority,
        suffix=rule.suffix,
        suffix_with_locant=rule.suffix_with_locant,
        prefix=rule.prefix,
        multi_suffix=rule.multi_suffix,
    )


GROUPS: dict[str, CharacteristicGroup] = {
    key: _to_characteristic_group(key) for key in RULES.functional_groups.principal_keys()
}


def get(key: str) -> CharacteristicGroup:
    return GROUPS[key]


def most_senior(keys: list[str]) -> CharacteristicGroup:
    return min((GROUPS[k] for k in keys), key=lambda group: group.seniority)

