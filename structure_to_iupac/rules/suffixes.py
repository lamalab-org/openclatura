"""Compatibility accessors for principal functional-group suffix rows."""

from dataclasses import dataclass

from ..nomenclature import RULES
from ..principal_suffixes import render_principal_suffix


@dataclass(frozen=True)
class CharacteristicGroup:
    key: str
    seniority: int
    suffix: str
    suffix_with_locant: bool
    prefix: str | None
    multi_suffix: object | None
    suffix_multiplier_positions: tuple[int, ...]

    def render_suffix(self, count: int = 1) -> str:
        return render_principal_suffix(RULES.functional_groups.get(self.key), count)


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
        suffix_multiplier_positions=rule.suffix_multiplier_positions,
    )


GROUPS: dict[str, CharacteristicGroup] = {
    key: _to_characteristic_group(key) for key in RULES.functional_groups.principal_keys()
}


def get(key: str) -> CharacteristicGroup:
    return GROUPS[key]


def most_senior(keys: list[str]) -> CharacteristicGroup:
    return min((GROUPS[k] for k in keys), key=lambda group: group.seniority)
