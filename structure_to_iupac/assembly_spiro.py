"""Spiro-specific assembly formatting."""

import re

from .assembly_parts import AssemblyParts
from .nomenclature import RULES
from .rules import elision
from .spiro_assembly import SpiroAssembly


SPIRO_SUBSTITUENT_RE = re.compile(r"^\[SPIRO\]-(\d+)-(.*)$")
AMBIGUOUS_CONNECTION_SUBSTITUENT_STEMS = RULES.assembly.ambiguous_connection_substituent_stems


def split_spiro_substituents(parts: AssemblyParts) -> list[SpiroAssembly]:
    spiro_subs = []
    normal_subs = []
    for sub in parts.substituents:
        if sub.spiro is not None:
            spiro_subs.append(sub.spiro)
            continue
        match = SPIRO_SUBSTITUENT_RE.match(sub.name)
        if match:
            side_prefixes, side_parent_name = extract_spiro_side_prefixes(match.group(2))
            spiro_subs.append(
                SpiroAssembly(
                    parent_locant=str(sub.locants[0]),
                    side_locant=match.group(1),
                    side_parent_name=side_parent_name,
                    side_prefixes=tuple(side_prefixes),
                )
            )
        else:
            normal_subs.append(sub)
    parts.substituents = normal_subs
    return spiro_subs


def format_spiro_core(stem_str: str, unsat_str: str, terminal_e: str, spiro_subs: list[SpiroAssembly]) -> tuple[str, str]:
    if not spiro_subs:
        return stem_str + unsat_str + terminal_e, terminal_e
    core_name = stem_str + unsat_str + "e"
    side_prefixes = []
    for spiro in spiro_subs:
        s_name = spiro.side_parent_name
        side_prefixes.extend(spiro.side_prefixes)
        if core_name.startswith("spiro["):
            # A second spiro operation needs full dispiro numbering.  Do not
            # compose invalid nested ``spiro[spiro[...]]`` strings; keep the
            # already named spiro core and let the remaining side radical stay
            # as a normal prefix until the polyspiro renderer owns that case.
            continue
        if _spiro_side_parent_needs_parentheses(s_name):
            s_name_str = f"({s_name})"
        else:
            s_name_str = s_name
        core_name = f"spiro[{core_name}-{spiro.parent_locant},{spiro.side_locant}'-{s_name_str}]"

    if terminal_e and terminal_e != "e":
        if ("yl" in terminal_e or elision.is_vowel_start(terminal_e.lstrip("-0123456789,"))) and core_name.endswith("e"):
            core_name = core_name[:-1]
        core_name += terminal_e
    if side_prefixes:
        core_name = "-".join(side_prefixes) + core_name
    return core_name, ""


def extract_spiro_side_prefixes(side_name: str) -> tuple[list[str], str]:
    """Move simple side-ring substituents to primed spiro prefixes."""

    parent_aliases = (
        ("1-azacyclopropane", "aziridine"),
        ("1-oxacyclopropane", "oxirane"),
        ("1-thiacyclopropane", "thiirane"),
    )
    parent_names = (
        "tetracyclo",
        "tricyclo",
        "bicyclo",
        "1-oxacyclobutane",
        "1-azacyclobutane",
        "1-thiacyclobutane",
        "aziridine",
        "oxirane",
        "thiirane",
        "cyclopropane",
    )
    normalized = side_name.replace("-aziridine", "aziridine")
    parent = None
    prefix_text = ""
    for alias, retained_parent in parent_aliases:
        if normalized.endswith(alias):
            parent = retained_parent
            prefix_text = normalized[: -len(alias)].rstrip("-")
            break
    if parent is None:
        parent = _extract_polycyclic_parent(normalized)
    if parent is None:
        parent = next((candidate for candidate in parent_names if normalized.endswith(candidate)), None)
        if parent is None:
            return [], normalized
        prefix_text = normalized[: -len(parent)].rstrip("-")
    elif normalized != parent and normalized.endswith(parent):
        prefix_text = normalized[: -len(parent)].rstrip("-")
    match = re.match(r"^([0-9,]+)-(.+)$", prefix_text)
    if not match:
        return [], parent
    locants, substituent = match.groups()
    primed_locants = ",".join(f"{loc}'" for loc in locants.split(","))
    return [f"{primed_locants}-{substituent}"], parent


def _extract_polycyclic_parent(name: str) -> str | None:
    for marker in ("tetracyclo[", "tricyclo[", "bicyclo["):
        idx = name.rfind(marker)
        if idx > 0 and name[idx - 1] == "-":
            return name[idx:]
    return None


def _spiro_side_parent_needs_parentheses(side_parent_name: str) -> bool:
    if re.search(r"(?:^|\d+-[a-z]+)(?:bi|tri|tetra)cyclo\[", side_parent_name):
        return False
    return "-" in side_parent_name or bool(re.search(r"\d", side_parent_name))
