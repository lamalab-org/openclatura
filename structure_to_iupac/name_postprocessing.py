"""Data-driven final name post-processing."""

import re
from dataclasses import dataclass

from .nomenclature import RULES

OPTIONAL_ONE_LOCANT_PREFIX_RE = re.compile(r"(?P<prefix>\bmethyl |\b\S+yl | |\))1-\(")


@dataclass(frozen=True)
class PostprocessingRuleInventoryItem:
    owner: str
    kind: str
    pattern: str
    replacement: str
    category: str = "migration"
    reason: str = ""


def postprocessing_rule_inventory() -> tuple[PostprocessingRuleInventoryItem, ...]:
    """Return classified compatibility post-processing rules for migration."""

    items = []
    for rule in RULES.postprocess.literal_replacements:
        items.append(
            PostprocessingRuleInventoryItem(
                "compatibility.literal",
                "literal",
                rule.pattern,
                rule.replacement,
                rule.category,
                rule.reason,
            )
        )
    for rule in RULES.postprocess.regex_replacements:
        items.append(
            PostprocessingRuleInventoryItem(
                "compatibility.regex",
                "regex",
                rule.pattern,
                rule.replacement,
                rule.category,
                rule.reason,
            )
        )
    for rule in RULES.postprocess.exact_replacements:
        items.append(
            PostprocessingRuleInventoryItem(
                "compatibility.exact",
                "exact",
                rule.pattern,
                rule.replacement,
                rule.category,
                rule.reason,
            )
        )
    for term in RULES.postprocess.acyl_amido_terms:
        items.append(
            PostprocessingRuleInventoryItem(
                "functional_group.acyl_amido",
                "term",
                term,
                "amido",
                "grammar",
                "Convert acyl-amino constructions to valid amido prefix grammar.",
            )
        )
    for suffix in RULES.postprocess.n_substituted_functional_suffixes:
        items.append(
            PostprocessingRuleInventoryItem(
                "attachment.n_substituted",
                "suffix",
                suffix,
                "N-qualified",
                "grammar",
                "Mark nested substituent attachment as N-substitution for functional prefixes.",
            )
        )
    return tuple(items)


def apply_data_postprocessing(name: str) -> str:
    """Apply ordered post-processing rules from the nomenclature registry."""

    for rule in RULES.postprocess.literal_replacements:
        name = name.replace(rule.pattern, rule.replacement)
    for rule in RULES.postprocess.regex_replacements:
        name = re.sub(rule.pattern, rule.replacement, name)
    stripped = name.strip()
    for rule in RULES.postprocess.exact_replacements:
        if stripped == rule.pattern:
            return rule.replacement
    return name


def apply_acyl_amido_postprocessing(name: str) -> str:
    """Apply acyl-amino to amido contractions from data."""

    for acyl in RULES.postprocess.acyl_amido_terms:
        name = re.sub(
            rf"(?<!\))(?<!\])\b\(([^()]*{acyl})\)amino\b",
            lambda match: _acyl_amido_replacement(match.group(1)),
            name,
        )
        name = re.sub(
            rf"(?<!\))(?<!\])\b([^()]*{acyl})amino\b",
            lambda match: _acyl_amido_replacement(match.group(1)),
            name,
        )
    return name


def _acyl_amido_replacement(acyl_prefix: str) -> str:
    if acyl_prefix.startswith("N,") or acyl_prefix.startswith("N-"):
        return f"{acyl_prefix}amino"
    return f"{acyl_prefix}amido"


def apply_connection_boundary_postprocessing(name: str) -> str:
    """Normalize connection-sensitive prefix boundaries from data rules."""

    name = _qualify_n_substituted_functional_prefixes(name)
    return _elide_optional_one_locants(name)


def _qualify_n_substituted_functional_prefixes(name: str) -> str:
    if not RULES.postprocess.n_substituted_functional_suffixes:
        return name
    suffix_pattern = "|".join(re.escape(suffix) for suffix in RULES.postprocess.n_substituted_functional_suffixes)
    pattern = re.compile(rf"\(\(([^()]+)\)([^()]*?(?:{suffix_pattern}))\)")
    return pattern.sub(r"(N-\1\2)", name)


def _elide_optional_one_locants(name: str) -> str:
    """Drop optional ``1-`` before parenthesized prefixes unless it prevents ambiguity."""

    result = []
    pos = 0
    for match in OPTIONAL_ONE_LOCANT_PREFIX_RE.finditer(name):
        open_idx = match.end() - 1
        result.append(name[pos : match.start()])
        if _connection_tail_needs_one_locant(name, open_idx):
            result.append(match.group(0))
        else:
            result.append(match.group("prefix") + "(")
        pos = match.end()
    result.append(name[pos:])
    return "".join(result)


def _connection_tail_needs_one_locant(name: str, open_idx: int) -> bool:
    close_idx = _matching_close_paren(name, open_idx)
    if close_idx is None:
        return False
    tail = name[close_idx + 1 : close_idx + 80].lower()
    normalized_tail = re.sub(r"[^a-z]", "", tail)
    return any(stem.replace("-", "").lower() in normalized_tail for stem in RULES.assembly.connection_boundary_parent_stems)


def _matching_close_paren(name: str, open_idx: int) -> int | None:
    depth = 0
    for idx in range(open_idx, len(name)):
        char = name[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None
