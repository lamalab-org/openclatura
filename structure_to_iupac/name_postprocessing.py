"""Data-driven final name post-processing."""

import re

from .nomenclature import RULES


def apply_data_postprocessing(name: str) -> str:
    """Apply ordered post-processing rules from the nomenclature registry."""

    for old, new in RULES.postprocess.literal_replacements:
        name = name.replace(old, new)
    for rule in RULES.postprocess.regex_replacements:
        name = re.sub(rule.pattern, rule.replacement, name)
    return RULES.postprocess.exact_replacements.get(name.strip(), name)


def apply_acyl_amido_postprocessing(name: str) -> str:
    """Apply acyl-amino to amido contractions from data."""

    for acyl in RULES.postprocess.acyl_amido_terms:
        name = re.sub(rf"(?<!\))(?<!\])\b\(([^()]*{acyl})\)amino\b", rf"\1amido", name)
        name = re.sub(rf"(?<!\))(?<!\])\b([^()]*{acyl})amino\b", rf"\1amido", name)
    return name
