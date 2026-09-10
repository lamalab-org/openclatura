# openclatura/assembler.py

import re
from dataclasses import replace

from .additive import parent_indicated_hydrogen_quota
from .assembly_charge import (
    positive_parent_n_charges,
    prepare_fusion_charge_assembly,
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
from .name_operations import HydroOperation
from .name_postprocessing import apply_data_postprocessing


def _post_process_name(name: str) -> str:
    name = apply_data_postprocessing(name)
    name = name.replace("iminoamino", "diazenyl")
    name = name.replace("aminoimino", "hydrazono")
    return apply_data_postprocessing(name)


_STEM_INDICATED_H = re.compile(r"^((?:\d+[a-z]?H,)*\d+[a-z]?H)-")


def _normalize_indicated_hydrogen_quota(parts: AssemblyParts, core_name: str) -> str:
    """Hold the parent's indicated-hydrogen citation to its quota (P-31.1.4.2.1).

    A mancude ring system has one indicated hydrogen for every pi-capable
    skeletal atom its maximum matching cannot pair, and no more. Component
    donors, intrinsic carbon witnesses and retained template defaults each cite
    the hydrogen they own, so together they can claim more than that. The
    surplus is not indicated hydrogen: beside a suffix that consumed a parent pi
    bond it is added hydrogen cited with the suffix (P-31.1.4.2.4), and
    otherwise it is a hydro prefix. Respelling it here keeps the structure
    untouched -- only how the same saturation is spelled changes, so
    ``1H,3H,7H,9H-purine-2,6,8-trione`` becomes
    ``1H-purine-2,6,8(3H,7H,9H)-trione`` and ``1H,4H-pyrrolo[3,2-b]pyrrole``
    becomes ``1,4-dihydropyrrolo[3,2-b]pyrrole``.
    """

    quota = parent_indicated_hydrogen_quota(parts)
    if quota is None:
        return core_name
    stem_match = _STEM_INDICATED_H.match(core_name)
    stem_cited = stem_match.group(1).split(",") if stem_match else []
    citations = [
        operation
        for operation in parts.hydro_operations
        if operation.operation_kind == "indicated_hydrogen" and operation.key != "added_hydrogen"
    ]
    cited = sorted(
        {locant[:-1] for locant in stem_cited}
        | {locant for operation in citations for locant in operation.locants}
        | set(parts.indicated_hydrogens),
        key=parse_locant,
    )
    if len(cited) <= quota:
        return core_name
    surplus = set(cited[quota:])

    bonded: dict[str, set[str]] = {}
    for left, right in parts.parent_bond_ids_by_locants:
        bonded.setdefault(left, set()).add(right)
        bonded.setdefault(right, set()).add(left)
    suffix_locants = {str(locant) for locant in parts.principal_group.locants} if parts.principal_group else set()
    added = {locant for locant in surplus if bonded.get(locant, set()) & suffix_locants}
    if parts.parent_charges:
        # A charged parent states hydrogen through its own ium/ide suffixes, and
        # added hydrogen cited against one of those would read as 3(1H)-ium.
        added = set()
    hydro = surplus - added

    # A locant already carrying another hydrogen role is spelled by that role;
    # re-spelling it here would state the same hydrogen twice or drop it.
    claimed = {
        locant
        for operation in parts.hydro_operations
        if operation.operation_kind == "additive_hydrogen" or operation.key == "added_hydrogen"
        for locant in operation.locants
    }
    if surplus & claimed:
        return core_name
    # Hydro prefixes come in pairs, so only respell what is spellable.
    existing_hydro = sum(
        len(operation.locants)
        for operation in parts.hydro_operations
        if operation.operation_kind == "additive_hydrogen"
    )
    if (existing_hydro + len(hydro)) % 2:
        return core_name

    respelled_by_operation: set[str] = set()
    for operation in citations:
        if not set(operation.locants) & surplus:
            continue
        index = parts.hydro_operations.index(operation)
        if set(operation.locants) <= added:
            parts.hydro_operations[index] = replace(
                operation,
                key="added_hydrogen",
                reason="Saturation past the parent's indicated hydrogen is added hydrogen cited with the suffix.",
            )
        elif set(operation.locants) <= hydro:
            parts.hydro_operations[index] = replace(
                operation,
                key="additive_hydrogen",
                reason="Saturation past the parent's indicated hydrogen is a hydro prefix.",
                operation_kind="additive_hydrogen",
            )
        else:
            return core_name
        respelled_by_operation.update(operation.locants)
    if added - respelled_by_operation:
        # The stem spells these itself, so this operation only tells the suffix
        # mover which locants to relocate: an "added_hydrogen" kind keeps the
        # prefix renderer from citing them a second time.
        parts.hydro_operations.append(
            HydroOperation(
                key="added_hydrogen",
                reason="Saturation past the parent's indicated hydrogen is added hydrogen cited with the suffix.",
                locants=tuple(sorted(added - respelled_by_operation, key=parse_locant)),
                atom_ids=tuple(
                    parts.parent_atom_ids_by_locant[locant]
                    for locant in sorted(added - respelled_by_operation, key=parse_locant)
                    if locant in parts.parent_atom_ids_by_locant
                ),
                operation_kind="added_hydrogen",
            )
        )
    if hydro - respelled_by_operation:
        pending = sorted(hydro - respelled_by_operation, key=parse_locant)
        parts.hydro_operations.append(
            HydroOperation(
                key="additive_hydrogen",
                reason="Saturation past the parent's indicated hydrogen is a hydro prefix.",
                locants=tuple(pending),
                atom_ids=tuple(
                    parts.parent_atom_ids_by_locant[locant]
                    for locant in pending
                    if locant in parts.parent_atom_ids_by_locant
                ),
                operation_kind="additive_hydrogen",
            )
        )
    parts.indicated_hydrogens = [locant for locant in parts.indicated_hydrogens if locant not in hydro]
    if stem_match:
        # Added hydrogen stays spelled in the stem here; the suffix mover
        # relocates it once the suffix locants are known.
        core_name = _respell_indicated_hydrogen(core_name, hydro) or core_name
    parent = parts.parent_hydride
    if (hydro or added) and parent is not None and parent.binding_term:
        # The parent's binding term carries its own spelling of the citation, and
        # the name/graph binding audit checks that term against the final name.
        # Hydro locants leave the stem outright; added hydrogen leaves it too,
        # relocated to the suffix by _move_added_hydrogen_to_suffix below.
        respelled = _respell_indicated_hydrogen(parent.binding_term, hydro | added)
        if respelled is not None:
            metadata = parent.metadata
            if metadata is not None:
                # The hydride metadata counts the parent's own indicated
                # hydrogen; leaving the respelled locants in it would let the
                # hydro prefix elide its locants as though the ring were fully
                # saturated, which reads as an ambiguous name.
                kept_h = tuple(locant for locant in metadata.default_indicated_h if locant not in (hydro | added))
                metadata = replace(
                    metadata,
                    default_indicated_h=kept_h,
                    indicated_hydrogen_count=min(metadata.indicated_hydrogen_count, len(kept_h)),
                )
                parts.retained_parent_metadata = metadata
            parts.parent_hydride = replace(parent, parent_name=respelled, hydride_metadata=metadata)
            # Bindings were built before rendering, so the parent's term has to
            # be rebuilt from the respelling for the final audit to match.
            refresh_parent_binding(parts)
    return core_name


def _respell_indicated_hydrogen(text: str, respelled: set[str]) -> str | None:
    """Drop locants now spelled as hydro from a leading indicated-hydrogen run."""

    match = _STEM_INDICATED_H.match(text)
    if match is None:
        return None
    remaining = [locant for locant in match.group(1).split(",") if locant[:-1] not in respelled]
    rest = text[match.end() :]
    return f"{','.join(remaining)}-{rest}" if remaining else rest


def _add_indicated_hydrogen_prefix(parts: AssemblyParts, core_name: str, *, allow_locant_elision: bool = True) -> str:
    core_name = _normalize_indicated_hydrogen_quota(parts, core_name)
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
        added_only = {
            locant
            for operation in parts.hydro_operations
            if operation.key == "added_hydrogen"
            for locant in operation.locants
        } >= set(indicated_hydrogens)
        if not added_only:
            # Added hydrogen (cited with the suffix) leaves the stem's own indicated hydrogen in place.
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
        parent = parts.parent_hydride
        delta = parts.parent_bond_delta
        # A fused parent can already contain saturated sites (notably neutral
        # three-connected N). Full hydrogenation consumes all parent pi bonds,
        # not necessarily one hydrogen for every skeletal atom.
        fully_hydrogenated_fusion = (
            parent is not None
            and parent.uses_fusion_plan
            and delta is not None
            and delta.compatible
            and not delta.implied_multiple_bond_ids
            and not delta.additional_multiple_bond_ids
            and set(delta.hydrogenated_edges) == {edge for edge, order in delta.assignment.orders if order == 2}
        )
        if (
            fully_hydrogenated_fusion
            or len(additive_hydrogens) + max(len(indicated_hydrogens), stated) == parts.parent_length
        ) and allow_locant_elision:
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
    match = re.match(r"^-(\d+[a-z]?(?:,\d+[a-z]?)*)-", suffix_str)
    if match is None:
        return core_name, suffix_str
    # The citation run may follow a hydro prefix -- 3,4-dihydro-1H-quinoline ->
    # 3,4-dihydroquinolin-2(1H)-one -- and it may also mix the parent's own
    # indicated hydrogen with the suffix's added hydrogen, as in
    # 1H-purine-2,6,8(3H,7H,9H)-trione. Only the added hydrogen moves.
    run = re.search(r"(?:^|(?<=hydro-))((?:\d+[a-z]?H,)*\d+[a-z]?H)-", core_name)
    if run is None:
        return core_name, suffix_str
    cited = run.group(1).split(",")
    moving = [locant for locant in cited if locant[:-1] in added]
    if not moving:
        return core_name, suffix_str
    kept = [locant for locant in cited if locant[:-1] not in added]
    cite = ",".join(moving)
    head = core_name[: run.start()]
    rest = core_name[run.end() :]
    if kept:
        core = f"{head}{','.join(kept)}-{rest}"
    else:
        head = head.rstrip("-")
        joiner = "-" if head and rest[:1].isdigit() else ""
        core = f"{head}{joiner}{rest}"
    return core, f"-{match.group(1)}({cite})-{suffix_str[match.end() :]}"


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
            rendered = format_multiplier(name, count)
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
    if prepare_fusion_charge_assembly(parts):
        refresh_name_atom_bindings(parts)
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
        spiro_parent_terminal = (
            terminal_e if parts.parent_hydride is not None and parts.parent_hydride.uses_fusion_plan else "e"
        )
        stem_str = apply_replacement_prefix(stem_str, a_prefix_str)
        if parts.is_substituent:
            stem_str, unsat_str, terminal_e, suffix_str = format_substituent_tail(
                parts, stem_str, terminal_e, spiro_subs
            )
        else:
            stem_str, unsat_str, terminal_e, suffix_str = format_parent_tail(parts, stem_str, terminal_e, spiro_subs)

        core_name, terminal_e, suffix_str = format_spiro_core(
            stem_str, unsat_str, terminal_e, spiro_subs, suffix_str, parent_terminal_e=spiro_parent_terminal
        )
        core_name = _add_indicated_hydrogen_prefix(parts, core_name, allow_locant_elision=not spiro_subs)
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
