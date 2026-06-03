"""Data-backed ring descriptor rendering."""

import re

from .nomenclature import RULES
from .rules import multipliers, stems

_VON_BAEYER_DESCRIPTOR_RE = re.compile(r"^(?P<prefix>[a-z]+)cyclo\[")


def render_ring_descriptor(kind: str, descriptor_numbers: tuple[int, ...]) -> str:
    template = RULES.rings.descriptor_templates[kind]
    numbers = ".".join(str(number) for number in descriptor_numbers)
    return template.format(numbers=numbers)


def render_von_baeyer_descriptor(bridge_count: int, descriptor_body: str) -> str:
    prefix = RULES.rings.polycycle_prefixes.get(bridge_count)
    if prefix is None:
        prefix = f"{_basic_cycle_prefix(bridge_count + 1)}cyclo"
    return prefix + descriptor_body


def von_baeyer_cycle_count(descriptor: str | None) -> int | None:
    """Return the cycle count for a von Baeyer descriptor prefix."""

    if not descriptor:
        return None
    match = _VON_BAEYER_DESCRIPTOR_RE.match(descriptor)
    if match is None:
        return None
    prefix = match.group("prefix")
    for bridge_count, rendered in RULES.rings.polycycle_prefixes.items():
        if rendered == f"{prefix}cyclo":
            return int(bridge_count) + 1
    for count, multiplier in multipliers.MULTIPLIERS.items():
        if prefix == multiplier.basic:
            return count
    for count, stem in stems.STEMS.items():
        if prefix == _basic_cycle_prefix(count):
            return count
    return None


def is_von_baeyer_descriptor(descriptor: str | None) -> bool:
    """Return whether descriptor is an extended von Baeyer descriptor."""

    cycle_count = von_baeyer_cycle_count(descriptor)
    return cycle_count is not None and cycle_count >= 3


def von_baeyer_kind(cycle_count: int) -> str:
    """Return the descriptor kind label for a von Baeyer cycle count."""

    prefix = RULES.rings.polycycle_prefixes.get(cycle_count - 1)
    if prefix is None:
        prefix = f"{_basic_cycle_prefix(cycle_count)}cyclo"
    return prefix


def _basic_cycle_prefix(count: int) -> str:
    """Return a basic numerical prefix for cycle-count descriptors."""

    if count in multipliers.MULTIPLIERS:
        return multipliers.basic(count)
    return stems.get(count).stem + "a"
