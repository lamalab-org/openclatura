"""Data-driven final name post-processing."""

import re

from .nomenclature import RULES

OPTIONAL_ONE_LOCANT_PREFIX_RE = re.compile(r"(?P<prefix>\bmethyl |\b\S+yl | |\))1-\(")


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
