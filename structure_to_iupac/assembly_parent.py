"""Parent stem, unsaturation, substituent-tail, and suffix formatting."""

from .assembly_parts import AssemblyParts, ParentChargeItem
from .assembly_charge import (
    append_charge_suffixes_to_terminal,
    has_ionic_retained_parent,
    has_retained_like_parent,
    inferred_ionic_retained_parent,
    parent_charge_name_operations,
    positive_parent_n_charges,
)
from .assembly_utils import parse_locant
from .nomenclature import RULES
from .principal_suffixes import render_principal_suffix
from .ring_renderer import render_ring_descriptor
from .retained_specs import retained_parent_spec
from .rules import bonds, elision, stems
from .suffix_stack import suffix_operation_spelling


UNSATURATION_ORDER = RULES.assembly.unsaturation_order
AMBIGUOUS_CONNECTION_SUBSTITUENT_STEMS = RULES.assembly.ambiguous_connection_substituent_stems


def promote_benzene_retained_name(parts: AssemblyParts) -> None:
    if parts.is_ring and not parts.is_bicycle and not parts.is_spiro and parts.parent_length == 6:
        if (
            len(parts.unsaturations) == 1
            and parts.unsaturations[0].bond_key == "double"
            and len(parts.unsaturations[0].locants) == 3
        ):
            locs = sorted([parse_locant(l)[1] for l in parts.unsaturations[0].locants])
            if locs == [1.0, 3.0, 5.0]:
                if not parts.a_prefixes:
                    parts.retained_name = "benzene"
                    parts.unsaturations = []


def parent_stem_and_terminal(parts: AssemblyParts) -> tuple[str, str]:
    terminal_e = bonds.PARENT_TERMINAL_VOWEL

    inferred_ionic_parent = inferred_ionic_retained_parent(parts)
    if has_ionic_retained_parent(parts):
        stem_str = RULES.charges.retained_ionic_n_parents[parts.retained_name]
        terminal_e = ""
    elif inferred_ionic_parent:
        stem_str = inferred_ionic_parent
        terminal_e = ""
    elif parts.retained_name:
        retained_spec = retained_parent_spec(parts.retained_name)
        if parts.is_substituent and retained_spec and retained_spec.substituent_stem is not None:
            stem_str = retained_spec.substituent_stem
            terminal_e = retained_spec.substituent_terminal or ""
        else:
            if parts.retained_name.endswith("e"):
                stem_str = parts.retained_name[:-1]
                terminal_e = "e"
            else:
                stem_str = parts.retained_name
                terminal_e = ""
    else:
        stem_str = stems.stem_for(parts.parent_length)
        if parts.is_bicycle:
            x, y, z = parts.bicycle_xyz
            stem_str = render_ring_descriptor("bicyclo", (x, y, z)) + stem_str
        elif parts.is_spiro:
            x, y = parts.spiro_xy
            stem_str = render_ring_descriptor("spiro", (x, y)) + stem_str
        elif parts.is_polycycle:
            if parts.polycycle_descriptor:
                stem_str = parts.polycycle_descriptor + stem_str
            else:
                v = parts.parent_length
                if v >= 4:
                    a = max(1, (v - 2) - 2)
                    b = 1
                    c = 1
                    if a + b + c + 2 != v:
                        a = v - 2 - b - c
                    stem_str = f"tricyclo[{a}.{b}.{c}.0^{{1,3}}]" + stem_str
                else:
                    stem_str = f"tricyclo[1.1.0.0^{{1,3}}]" + stem_str
        elif parts.is_ring:
            stem_str = "cyclo" + stem_str
    return stem_str, terminal_e


def apply_replacement_prefix(stem_str: str, a_prefix_str: str) -> str:
    if a_prefix_str:
        if elision.is_vowel_start(stem_str) and a_prefix_str.endswith("a"):
            a_prefix_str = a_prefix_str[:-1]
        stem_str = a_prefix_str + stem_str
    return stem_str


def format_unsaturations(parts: AssemblyParts, stem_str: str) -> tuple[str, str]:
    sorted_unsats = sorted(parts.unsaturations, key=lambda u: UNSATURATION_ORDER.get(u.bond_key, 99))
    unsat_parts = []
    base_infixes = []
    for unsaturation in sorted_unsats:
        count = len(unsaturation.locants) or 1
        infix = bonds.unsaturation_infix(unsaturation.bond_key, count)
        base_infix = infix[1:] if infix.startswith("a") else infix
        base_infixes.append((unsaturation, base_infix))
    if base_infixes and not elision.is_vowel_start(base_infixes[0][1]):
        stem_str += "a"
    for unsaturation, base_infix in base_infixes:
        if unsaturation.locants:
            loc_str = ",".join(sorted(unsaturation.locants, key=parse_locant))
            unsat_parts.append(f"-{loc_str}-{base_infix}")
        else:
            unsat_parts.append(base_infix)
    return stem_str, "".join(unsat_parts)


def substituent_suffix_word(parts: AssemblyParts) -> str:
    if parts.is_triple_attach:
        return "ylidyne"
    if parts.is_double_attach:
        return "ylidene"
    return "yl"


def always_print_substituent_locant(parts: AssemblyParts) -> bool:
    if parts.parent_length == 1:
        return False
    if parts.is_bicycle or parts.is_spiro or parts.is_polycycle:
        return True
    if parts.is_ring and (parts.a_prefixes or (parts.retained_name and parts.retained_name != "benzene")):
        return True
    retained_spec = retained_parent_spec(parts.retained_name)
    if retained_spec and retained_spec.attachment_policy.print_substituent_locant:
        return True
    stem_str, _ = parent_stem_and_terminal(parts)
    return any(stem_str.endswith(stem) for stem in AMBIGUOUS_CONNECTION_SUBSTITUENT_STEMS)


def format_substituent_tail(
    parts: AssemblyParts, stem_str: str, terminal_e: str, spiro_subs
) -> tuple[str, str, str, str]:
    suffix_yl = substituent_suffix_word(parts)
    always_print_locant = bool(spiro_subs) or always_print_substituent_locant(parts)
    if parts.retained_name == "benzene":
        terminal_e = "yl"
    elif (str(parts.attachment_locant) != "1" or parts.unsaturations or always_print_locant) and parts.parent_length > 1:
        terminal_e = f"-{parts.attachment_locant}-{suffix_yl}"
    else:
        terminal_e = suffix_yl

    unsat_str = ""
    if not has_retained_like_parent(parts) and parts.unsaturations:
        stem_str, unsat_str = format_unsaturations(parts, stem_str)
    elif not has_retained_like_parent(parts) and not parts.unsaturations:
        if parts.parent_length > 1 and (
            str(parts.attachment_locant) != "1"
            or parts.is_bicycle
            or parts.is_spiro
            or parts.is_polycycle
            or always_print_locant
        ):
            unsat_str = "an"
    terminal_e = append_charge_suffixes_to_terminal(parts, terminal_e)
    return stem_str, unsat_str, terminal_e, ""


def format_principal_suffix(parts: AssemblyParts, terminal_e: str, spiro_subs) -> tuple[str, str]:
    if not parts.principal_group:
        return terminal_e, ""
    group = RULES.functional_groups.get(parts.principal_group.key)
    locs = sorted(parts.principal_group.locants, key=parse_locant)
    has_spiro_subs = bool(spiro_subs)
    omit_locant = parts.parent_length == 1
    if not omit_locant and len(locs) == 1 and str(locs[0]) == "1":
        if not group.suffix_with_locant:
            omit_locant = True
        elif (
            parts.is_ring
            and not parts.unsaturations
            and not parts.is_bicycle
            and not parts.is_spiro
            and not parts.is_polycycle
            and not has_spiro_subs
            and not parts.retained_name
        ):
            omit_locant = True

    suffix_text = render_principal_suffix(group, len(locs))

    if elision.is_vowel_start(suffix_text):
        terminal_e = ""
    if group.suffix_with_locant and locs and not omit_locant:
        return terminal_e, f"-{','.join(map(str, locs))}-{suffix_text}"
    return terminal_e, suffix_text


def format_parent_tail(parts: AssemblyParts, stem_str: str, terminal_e: str, spiro_subs) -> tuple[str, str, str, str]:
    unsat_str = ""
    if not has_retained_like_parent(parts):
        if not parts.unsaturations:
            unsat_str = bonds.get("single").saturated_suffix
        else:
            stem_str, unsat_str = format_unsaturations(parts, stem_str)
    terminal_e, suffix_str = format_principal_suffix(parts, terminal_e, spiro_subs)
    charge_operations = parent_charge_name_operations(parts)
    if charge_operations:
        terminal_e = ""
        suffix_str = "".join(
            f"-{','.join(operation.locants)}-{suffix_operation_spelling(operation)}" for operation in charge_operations
        ) + suffix_str
    return stem_str, unsat_str, terminal_e, suffix_str
