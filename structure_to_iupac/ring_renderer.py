"""Data-backed ring descriptor rendering."""

from .nomenclature import RULES


def render_ring_descriptor(kind: str, descriptor_numbers: tuple[int, ...]) -> str:
    template = RULES.rings.descriptor_templates[kind]
    numbers = ".".join(str(number) for number in descriptor_numbers)
    return template.format(numbers=numbers)


def render_von_baeyer_descriptor(bridge_count: int, descriptor_body: str) -> str:
    prefix = RULES.rings.polycycle_prefixes.get(bridge_count, "polycyclo")
    return prefix + descriptor_body
