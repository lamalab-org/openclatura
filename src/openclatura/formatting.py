"""Name-fragment formatting helpers used by the naming pipeline."""

import re

from .namer_config import ALKYL_OXY_PREFIXES
from .rules import multipliers, stems


def is_fully_enclosed(s: str) -> bool:
    """Return true when a name fragment is already fully parenthesized."""

    if not s.startswith("(") or not s.endswith(")"):
        return False
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return False
    return depth == 0


def strip_outer_parentheses(name: str) -> str:
    """Remove one balanced outer parenthesis pair from a fragment."""

    if name.startswith("(") and name.endswith(")"):
        return name[1:-1]
    return name


def is_complex_prefix(name: str) -> bool:
    """Return true when a substituent prefix needs protective parentheses."""

    return "(" in name or name[0].isdigit() or "-" in name or " " in name or _starts_with_multiplier(name)


def _starts_with_multiplier(name: str) -> bool:
    """Return true when a prefix already begins with a multiplicative prefix."""

    return any(name.startswith(mult.basic) for mult in multipliers.MULTIPLIERS.values())


def format_multiplier(name: str, count: int, safe_enclose: bool = False) -> str:
    """Apply simple or complex multiplicative prefixes to a substituent name."""

    is_complex = is_complex_prefix(name)
    if count == 1:
        if (safe_enclose or is_complex) and not is_fully_enclosed(name):
            return f"({name})"
        return name
    mult = multipliers.complex_(count) if is_complex else multipliers.basic(count)
    if is_complex and not is_fully_enclosed(name):
        return f"{mult}({name})"
    return f"{mult}{name}"


def count_names(names: list[str]) -> dict[str, int]:
    """Count repeated substituent fragments before multiplier formatting."""

    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return counts


def format_counted_prefixes(names: list[str]) -> str:
    """Format repeated substituent fragments with correct multipliers."""

    counts = count_names(names)
    safe = len(counts) > 1 or any(is_complex_prefix(name) for name in counts)
    return "".join(format_multiplier(name, count, safe_enclose=safe) for name, count in sorted(counts.items()))


def oxy_prefix_from_branch(branch: str) -> str:
    """Return an oxy prefix for a named branch."""

    retained = ALKYL_OXY_PREFIXES.get(branch)
    if retained:
        return retained
    branch = strip_outer_parentheses(branch)
    retained = ALKYL_OXY_PREFIXES.get(branch)
    if retained:
        return retained
    substituted_alkoxy = substituted_alkoxy_prefix(branch)
    if substituted_alkoxy:
        return substituted_alkoxy
    if is_complex_prefix(branch):
        return f"(({branch})oxy)"
    return f"({branch}oxy)"


def substituted_alkoxy_prefix(branch: str) -> str | None:
    """Contract substituted acyclic alkyl branches to alkoxy prefixes."""

    if "hydroxy" not in branch:
        return None
    for stem in stems.STEMS.values():
        terminal = f"{stem.stem}yl"
        replacement = f"{stem.stem}oxy"
        if branch.endswith(terminal):
            prefix = branch[: -len(terminal)]
            return f"({prefix}{replacement})" if prefix else replacement
    return None


def format_element_substituent(stereo_prefix: str, branch: str, suffix: str, is_double: bool = False) -> str:
    """Attach a named branch to an element substituent suffix."""

    branch = strip_outer_parentheses(branch)
    suffix_text = suffix + ("idene" if is_double else "")
    if is_complex_prefix(branch):
        return f"({stereo_prefix}({branch}){suffix_text})"
    return f"({stereo_prefix}{branch}{suffix_text})"


def ensure_stereo_descriptor_boundary(name: str) -> str:
    """Ensure stereodescriptor groups are separated from following name stems."""

    descriptor = r"\((?:\d+[A-Za-z]*[RS]|[EZ])(?:,(?:\d+[A-Za-z]*[RS]|[EZ]))*\)"
    return re.sub(rf"({descriptor})(?=[A-Za-z])", r"\1-", name)
