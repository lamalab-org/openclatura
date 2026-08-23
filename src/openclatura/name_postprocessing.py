"""Data-driven final name post-processing."""

from .nomenclature import RULES


def apply_data_postprocessing(name: str) -> str:
    """Apply ordered literal post-processing rules from the nomenclature registry."""

    for rule in RULES.postprocess.literal_replacements:
        name = name.replace(rule.pattern, rule.replacement)
    return name
