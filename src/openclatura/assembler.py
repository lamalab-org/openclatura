# openclatura/assembler.py

import re

from .assembly_charge import (
    positive_parent_n_charges,
)
from .assembly_parent import (
    apply_replacement_prefix,
    format_parent_tail,
    format_substituent_tail,
    parent_stem_and_terminal,
    promote_acyl_substituent_name,
    promote_benzene_retained_name,
    promote_retained_functional_parent,
    promote_retained_substituent_name,
)
from .assembly_parts import AssemblyParts
from .assembly_prefixes import format_replacement_prefixes, format_substituent_prefixes
from .assembly_spiro import format_spiro_core, split_spiro_substituents
from .assembly_utils import needs_hyphen, parse_locant
from .formatting import format_multiplier
from .fused_ion_templates import consume_fused_ion_operation, select_fused_ion_operation
from .name_assembly import NameAssemblyResult, rewrite_history_trace_data, token_span_trace_data
from .name_bindings import refresh_name_atom_bindings, refresh_parent_binding
from .name_postprocessing import apply_data_postprocessing


def _post_process_name(name: str) -> str:
    name = apply_data_postprocessing(name)
    name = name.replace("iminoamino", "diazenyl")
    name = name.replace("aminoimino", "hydrazono")
    return apply_data_postprocessing(name)


def _add_indicated_hydrogen_prefix(parts: AssemblyParts, core_name: str) -> str:
    additive_hydrogens = [
        locant
        for operation in parts.hydro_operations
        if operation.operation_kind == "additive_hydrogen"
        for locant in operation.locants
    ]
    indicated_hydrogens = [
        locant
        for operation in parts.hydro_operations
        if operation.operation_kind == "indicated_hydrogen"
        for locant in operation.locants
    ] or parts.indicated_hydrogens
    if not indicated_hydrogens and not additive_hydrogens:
        return core_name
    if positive_parent_n_charges(parts):
        # The ium suffix states the hydrogen on its own nitrogen, but a second,
        # neutral ring NH still has to be cited: 4-imino-1H-pyrimidin-3-ium.
        cationic_locants = {charge.locant for charge in positive_parent_n_charges(parts)}
        indicated_hydrogens = [locant for locant in indicated_hydrogens if locant not in cationic_locants]
        if not additive_hydrogens and not indicated_hydrogens:
            return core_name
    if indicated_hydrogens:
        indicated_hydrogens = sorted(set(indicated_hydrogens), key=parse_locant)
        core_name = _drop_stem_indicated_hydrogen(core_name, indicated_hydrogens)
        core_name = ",".join(f"{locant}H" for locant in indicated_hydrogens) + "-" + core_name
    if additive_hydrogens:
        additive_hydrogens = sorted(set(additive_hydrogens), key=parse_locant)
        separator = "-" if core_name[:1].isdigit() else ""
        hydro = format_multiplier("hydro", len(additive_hydrogens))

        stated = parts.retained_parent_metadata.indicated_hydrogen_count if parts.retained_parent_metadata else 0
        cited = len(re.findall(r"\d+[a-z]?H(?=[,-])", core_name))
        if cited and cited < stated:
            # The template counts its inherent saturation, but the stem has been
            # rewritten to its mancude parent and states only the cited H.
            stated = cited
        if len(additive_hydrogens) + max(
            len(indicated_hydrogens), stated
        ) == parts.parent_length and not core_name.startswith("spiro["):
            return f"{hydro}{separator}{core_name}"
        core_name = f"{','.join(additive_hydrogens)}-{hydro}{separator}{core_name}"
    return core_name


def _drop_stem_indicated_hydrogen(core_name: str, indicated_hydrogens: list[str]) -> str:
    """Drop a stem's built-in ``1H-``; the cited set replaces it."""

    match = re.match(r"^(\d+[a-z]?H(?:,\d+[a-z]?H)*)-", core_name)
    if match is None:
        return core_name
    return core_name[match.end() :]


def _move_added_hydrogen_to_suffix(parts: AssemblyParts, core_name: str, suffix_str: str) -> tuple[str, str]:
    """P-14.7: added hydrogen follows the suffix locant -- quinolin-4(1H)-one."""

    added = {
        locant
        for operation in parts.hydro_operations
        if operation.key == "added_hydrogen"
        for locant in operation.locants
    }
    if not added:
        return core_name, suffix_str
    cite = ",".join(f"{locant}H" for locant in sorted(added, key=parse_locant))
    if not core_name.startswith(cite + "-"):
        return core_name, suffix_str
    match = re.match(r"^-(\d+[a-z]?)-", suffix_str)
    if match is None:
        return core_name, suffix_str
    return core_name[len(cite) + 1 :], f"-{match.group(1)}({cite})-{suffix_str[match.end() :]}"


def _add_stereo_prefix(parts: AssemblyParts, final_word: str) -> str:
    if not parts.stereo_features:
        return final_word
    unique_stereo = []
    seen = set()
    for feature in parts.stereo_features:
        if feature not in seen:
            seen.add(feature)
            unique_stereo.append(feature)
    unlocanted_descriptors = {descriptor for locant, descriptor in unique_stereo if not locant}
    if unlocanted_descriptors:
        unique_stereo = [
            feature for feature in unique_stereo if not (feature[0] == "1" and feature[1] in unlocanted_descriptors)
        ]
    sorted_stereo = sorted(unique_stereo, key=lambda f: parse_locant(f[0]) if f[0] else (0, ""))
    stereo_str = "(" + ",".join(f"{loc}{st}" if loc else st for loc, st in sorted_stereo) + ")-"
    return stereo_str + final_word


def _add_relative_stereo_prefix(parts: AssemblyParts, final_word: str) -> str:
    if not parts.relative_stereo_prefixes:
        return final_word
    prefixes = []
    seen = set()
    for prefix in parts.relative_stereo_prefixes:
        if prefix not in seen:
            prefixes.append(prefix)
            seen.add(prefix)
    return "".join(f"{prefix}-" for prefix in prefixes) + final_word


def _front_modifier_sort_key(name: str) -> str:
    """Alphanumeric ordering key: ignore the italic ``tert-``/``sec-`` prefixes."""

    key = name
    for prefix in ("tert-", "sec-"):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return key.lstrip("([").lower()


def _add_front_modifiers(parts: AssemblyParts, final_word: str) -> str:
    if not parts.front_modifiers:
        return final_word
    mods = parts.front_modifiers
    locants = parts.front_modifier_locants
    have_locants = len(locants) == len(mods) and all(loc is not None for loc in locants)

    if have_locants and len(set(mods)) > 1:
        by_name: dict[str, list[str]] = {}
        for mod, loc in zip(mods, locants):
            by_name.setdefault(mod, []).append(loc)
        entries = []
        for name in sorted(by_name, key=_front_modifier_sort_key):
            group_locants = sorted(by_name[name], key=lambda loc: (len(loc), loc))
            locant_str = ",".join(group_locants)
            count = len(group_locants)
            rendered = format_multiplier(name, count, safe_enclose=True) if count > 1 else name
            entries.append(f"{locant_str}-{rendered}")
        return f"{' '.join(entries)} {final_word}"
    counts: dict[str, int] = {}
    for mod in mods:
        counts[mod] = counts.get(mod, 0) + 1
    front_words = [format_multiplier(m, c, safe_enclose=True) if c > 1 else m for m, c in sorted(counts.items())]
    return f"{' '.join(front_words)} {final_word}"


def post_process_name(name: str) -> str:
    return _post_process_name(name)


def post_process_rewrite_rules():
    """Return shared post-processing rewrites for metadata-aware assembly paths."""

    return (("post_process_name", _post_process_name),)


def assemble_name_raw(parts: AssemblyParts) -> str:
    fused_ion_candidate = select_fused_ion_operation(parts)
    if fused_ion_candidate is not None:
        consume_fused_ion_operation(parts, fused_ion_candidate)

    promote_acyl_substituent_name(parts)
    promote_retained_substituent_name(parts)
    promote_benzene_retained_name(parts)
    promote_retained_functional_parent(parts)
    spiro_subs = split_spiro_substituents(parts)
    prefix_str = format_substituent_prefixes(parts, spiro_subs)
    a_prefix_str = format_replacement_prefixes(parts)
    if (parts.retained_substituent_name is not None or parts.is_acyl_substituent) and parts.name_atom_bindings:
        refresh_parent_binding(parts)
    if parts.retained_absorbs_principal_group and parts.name_atom_bindings:
        refresh_parent_binding(parts)
    if fused_ion_candidate is not None and fused_ion_candidate.rendered_name is not None:
        core_name = fused_ion_candidate.rendered_name
    elif parts.retained_substituent_name is not None:
        # The retained prefix is the whole word -- skeleton, branch and ``yl``.
        core_name = parts.retained_substituent_name
    else:
        stem_str, terminal_e = parent_stem_and_terminal(parts)
        stem_str = apply_replacement_prefix(stem_str, a_prefix_str)
        if parts.is_substituent:
            stem_str, unsat_str, terminal_e, suffix_str = format_substituent_tail(
                parts, stem_str, terminal_e, spiro_subs
            )
        else:
            stem_str, unsat_str, terminal_e, suffix_str = format_parent_tail(parts, stem_str, terminal_e, spiro_subs)

        core_name, terminal_e, suffix_str = format_spiro_core(stem_str, unsat_str, terminal_e, spiro_subs, suffix_str)
        core_name = _add_indicated_hydrogen_prefix(parts, core_name)
        core_name, suffix_str = _move_added_hydrogen_to_suffix(parts, core_name, suffix_str)
        core_name += suffix_str
    parent_needs_prefix_hyphen = bool(
        prefix_str and positive_parent_n_charges(parts) and parts.retained_name and parts.indicated_hydrogens
    )
    final_word = (
        prefix_str + "-" + core_name
        if prefix_str and (needs_hyphen(prefix_str, core_name) or parent_needs_prefix_hyphen)
        else prefix_str + core_name
    )
    final_word = _add_stereo_prefix(parts, final_word)
    final_word = _add_relative_stereo_prefix(parts, final_word)
    final_word = _add_front_modifiers(parts, final_word)
    return final_word


def assemble_name(parts: AssemblyParts) -> str:
    return assemble_name_result(parts).text


def assemble_name_result(parts: AssemblyParts) -> NameAssemblyResult:
    """Assemble a name while preserving final atom/bond binding metadata."""

    if not parts.name_atom_bindings:
        refresh_name_atom_bindings(parts)
    raw_name = assemble_name_raw(parts)
    result = NameAssemblyResult.from_raw_name(raw_name, parts.name_atom_bindings, postprocess=post_process_name)
    parts.name_atom_bindings = list(result.bindings)
    parts.name_token_spans = token_span_trace_data(result)
    parts.name_rewrite_history = rewrite_history_trace_data(result)
    return result
