"""Programmatic rendering for principal functional-group suffixes."""

from .nomenclature import FunctionalGroupRule
from .rules import multipliers


def render_principal_suffix(rule: FunctionalGroupRule, count: int) -> str:
    """Render a principal-group suffix for one or more equivalent groups.

    ``FunctionalGroupRule.suffix_multiplier_positions`` stores the suffix
    grammar: the word positions that receive a multiplicative prefix.  This
    keeps suffix spelling extensible without requiring fixed ``di...`` strings
    for every group and count.
    """

    if not rule.suffix:
        return ""
    if count <= 1:
        return rule.suffix
    words = rule.suffix.split()
    positions = set(rule.suffix_multiplier_positions or (0,))
    multiplier = multipliers.basic(count)
    rendered = [f"{multiplier}{word}" if idx in positions else word for idx, word in enumerate(words)]
    return " ".join(rendered)


def principal_suffix_terms(rule: FunctionalGroupRule, counts: tuple[int, ...] = (1, 2, 3)) -> tuple[str, ...]:
    """Return representative suffix terms for traces and documentation."""

    if not rule.suffix:
        return ()
    return tuple(dict.fromkeys(render_principal_suffix(rule, count) for count in counts))
