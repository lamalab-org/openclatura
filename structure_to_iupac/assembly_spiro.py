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
            side_prefixes, side_parent_name, side_suffixes = extract_spiro_side_prefixes(match.group(2))
            spiro_subs.append(
                SpiroAssembly(
                    parent_locant=str(sub.locants[0]),
                    side_locant=match.group(1),
                    side_parent_name=side_parent_name,
                    side_prefixes=tuple(side_prefixes),
                    side_suffixes=tuple(side_suffixes),
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
    if len(spiro_subs) == 2 and not core_name.startswith("spiro["):
        return _format_dispiro_core(core_name, terminal_e, spiro_subs), ""
    side_prefixes = []
    side_suffixes = []
    for spiro in spiro_subs:
        s_name = spiro.side_parent_name
        side_prefixes.extend(spiro.side_prefixes)
        side_suffixes.extend(_prime_side_suffixes(spiro.side_suffixes, "'"))
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
        core_name += _merge_terminal_and_side_suffixes(terminal_e, side_suffixes)
    elif side_suffixes:
        core_name += _format_side_suffixes(side_suffixes)
    if side_prefixes:
        side_prefixes = _prime_replacement_prefixes_for_primed_component(core_name, side_prefixes)
        core_name = "-".join(side_prefixes) + core_name
    core_name = _prime_inline_replacement_prefixes_for_primed_component(core_name)
    return core_name, ""


def _format_dispiro_core(core_name: str, terminal_e: str, spiro_subs: list[SpiroAssembly]) -> str:
    first, second = sorted(spiro_subs, key=lambda spiro: (int(spiro.parent_locant), spiro.side_parent_name))
    first_side = _spiro_side_name(first.side_parent_name)
    second_side = _spiro_side_name(second.side_parent_name)
    core = (
        f"dispiro[{first_side}-{first.side_locant},{first.parent_locant}'-"
        f"{core_name}-{second.parent_locant}',{second.side_locant}''-{second_side}]"
    )
    side_prefixes = []
    side_suffixes = []
    side_prefixes.extend(_reprime_side_prefixes(first.side_prefixes, "'"))
    side_prefixes.extend(_reprime_side_prefixes(second.side_prefixes, "''"))
    side_suffixes.extend(_prime_side_suffixes(first.side_suffixes, "'"))
    side_suffixes.extend(_prime_side_suffixes(second.side_suffixes, "''"))
    if terminal_e and terminal_e != "e":
        if ("yl" in terminal_e or elision.is_vowel_start(terminal_e.lstrip("-0123456789,"))) and core.endswith("e"):
            core = core[:-1]
        core += _merge_terminal_and_side_suffixes(terminal_e, side_suffixes)
    elif side_suffixes:
        core += _format_side_suffixes(side_suffixes)
    if side_prefixes:
        side_prefixes = _prime_replacement_prefixes_for_primed_component(core, side_prefixes)
        core = "-".join(side_prefixes) + core
    return core


def _spiro_side_name(side_parent_name: str) -> str:
    if _spiro_side_parent_needs_parentheses(side_parent_name):
        return f"({side_parent_name})"
    return side_parent_name


def _reprime_side_prefixes(prefixes: tuple[str, ...], prime: str) -> list[str]:
    if prime == "'":
        return list(prefixes)
    return [prefix.replace("'", prime) for prefix in prefixes]


def _prime_replacement_prefixes_for_primed_component(core_name: str, prefixes: list[str]) -> list[str]:
    if "spiro[cyclopropane-" not in core_name:
        return prefixes
    return [
        re.sub(r"^([0-9,]+)-(oxa|aza|thia)$", lambda match: f"{match.group(1)}'-{match.group(2)}", prefix)
        for prefix in prefixes
    ]


def _prime_inline_replacement_prefixes_for_primed_component(core_name: str) -> str:
    if "spiro[cyclopropane-" not in core_name:
        return core_name
    return re.sub(r"(^|-)([0-9,]+)-(oxa|aza|thia)spiro\[", r"\1\2'-\3spiro[", core_name)


def _prime_side_suffixes(suffixes: tuple[tuple[str, str], ...], prime: str) -> list[tuple[str, str]]:
    return [(f"{locant}{prime}", suffix) for locant, suffix in suffixes]


def _merge_terminal_and_side_suffixes(terminal_e: str, side_suffixes: list[tuple[str, str]]) -> str:
    if not side_suffixes:
        return terminal_e
    match = re.fullmatch(r"-([0-9,']+)-(ol|one)", terminal_e)
    if not match:
        return terminal_e + _format_side_suffixes(side_suffixes)
    main_locants = match.group(1).split(",")
    main_suffix = match.group(2)
    same_suffix = [(locant, suffix) for locant, suffix in side_suffixes if suffix == main_suffix]
    other_suffix = [(locant, suffix) for locant, suffix in side_suffixes if suffix != main_suffix]
    if not same_suffix:
        return terminal_e + _format_side_suffixes(side_suffixes)
    locants = main_locants + [locant for locant, _ in same_suffix]
    multiplier = {2: "di", 3: "tri", 4: "tetra"}.get(len(locants), "")
    merged_suffix = f"{multiplier}{main_suffix}" if multiplier else main_suffix
    return f"-{','.join(locants)}-{merged_suffix}" + _format_side_suffixes(other_suffix)


def _format_side_suffixes(side_suffixes: list[tuple[str, str]]) -> str:
    return "".join(f"-{locant}-{suffix}" for locant, suffix in side_suffixes)


def extract_spiro_side_prefixes(side_name: str) -> tuple[list[str], str, tuple[tuple[str, str], ...]]:
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
        "cyclobutane",
        "cyclopropane",
    )
    normalized = side_name.replace("-aziridine", "aziridine")
    normalized, side_suffixes = _extract_side_suffixes(normalized)
    parent = None
    prefix_text = ""
    for alias, retained_parent in parent_aliases:
        if normalized.endswith(alias):
            parent = retained_parent
            prefix_text = normalized[: -len(alias)].rstrip("-")
            break
    if parent is None:
        parent = _extract_polycyclic_parent(normalized)
        if parent is not None and normalized.endswith(parent):
            prefix_text = normalized[: -len(parent)].rstrip("-")
    if parent is None:
        parent = next((candidate for candidate in parent_names if normalized.endswith(candidate)), None)
        if parent is None:
            return [], normalized, tuple(side_suffixes)
        prefix_text = normalized[: -len(parent)].rstrip("-")
    elif normalized != parent and normalized.endswith(parent):
        prefix_text = normalized[: -len(parent)].rstrip("-")
    match = re.match(r"^([0-9,]+)-(.+)$", prefix_text)
    if not match:
        return [], parent, tuple(side_suffixes)
    locants, substituent = match.groups()
    primed_locants = ",".join(f"{loc}'" for loc in locants.split(","))
    return [f"{primed_locants}-{substituent}"], parent, tuple(side_suffixes)


def _extract_side_suffixes(side_name: str) -> tuple[str, list[tuple[str, str]]]:
    match = re.fullmatch(r"(.+?)an-([0-9,]+)-(ol|one)", side_name)
    if not match:
        return side_name, []
    stem, locants, suffix = match.groups()
    return stem + "ane", [(locant, suffix) for locant in locants.split(",")]


def _extract_polycyclic_parent(name: str) -> str | None:
    for marker in ("tetracyclo[", "tricyclo[", "bicyclo["):
        idx = name.rfind(marker)
        if idx > 0 and name[idx - 1] == "-":
            return name[idx:]
        if idx > 0 and re.fullmatch(r"(?:[0-9,]+-)?(?:oxa|aza|thia)", name[:idx]):
            return name[idx:]
    return None


def _spiro_side_parent_needs_parentheses(side_parent_name: str) -> bool:
    if re.search(r"(?:^|\d+-[a-z]+)(?:bi|tri|tetra)cyclo\[", side_parent_name):
        return False
    return "-" in side_parent_name or bool(re.search(r"\d", side_parent_name))
