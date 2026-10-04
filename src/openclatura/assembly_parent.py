"""Parent stem, unsaturation, substituent-tail, and suffix formatting."""

from .assembly_charge import (
    append_charge_suffixes_to_terminal,
    has_ionic_retained_parent,
    has_retained_like_parent,
    inferred_ionic_retained_parent,
    parent_charge_name_operations,
)
from .assembly_parts import AssemblyParts, SubstituentItem
from .assembly_prefixes import substituent_sort_key
from .assembly_utils import parse_locant
from .formatting import is_complex_prefix, strip_outer_parentheses
from .nomenclature import RULES, RetainedChainParentRule
from .principal_suffixes import render_principal_suffix
from .retained_specs import retained_parent_spec
from .ring_parent import ParentHydrideKind
from .ring_renderer import render_ring_descriptor
from .rules import bonds, elision, multipliers, stems
from .suffix_stack import suffix_operation_spelling

UNSATURATION_ORDER = RULES.assembly.unsaturation_order
AMBIGUOUS_CONNECTION_SUBSTITUENT_STEMS = RULES.assembly.ambiguous_connection_substituent_stems


def _halide_word(key: str) -> str | None:
    """The halide word an acyl-halide suffix ends in, or ``None`` for a ring one."""

    rule = RULES.functional_groups.get(key)
    systematic_acyl = RULES.assembly.substituent_attachment_suffixes["acyl"]
    if rule is None or not rule.suffix.startswith(f"{systematic_acyl} "):
        return None
    return rule.suffix[len(systematic_acyl) + 1 :]


def _retained_acyl_halide_parents() -> dict[tuple[int, str, int], RetainedChainParentRule]:
    """P-65.5.1: the retained acyl word plus the halide -- ``acetyl chloride``."""

    parents: dict[tuple[int, str, int], RetainedChainParentRule] = {}
    for key in RULES.assembly.acid_halide_suffix_keys:
        halide = _halide_word(key)
        if halide is None:
            continue
        for (length, kind), (stem, ending) in RULES.retained.chain_substituent_stems.items():
            if kind == "acyl":
                parents[(length, key, 1)] = RetainedChainParentRule(f"{stem}{ending} {halide}")
    return parents


def _retained_ring_acyl_halide_parents() -> dict[tuple[str, str], str]:
    """The ring counterpart: ``benzoyl chloride``, from benzene's acyl word."""

    parents: dict[tuple[str, str], str] = {}
    for key in RULES.assembly.acid_halide_suffix_keys:
        if not key.startswith("ring_"):
            continue
        halide = _halide_word(key[len("ring_") :])
        if halide is None:
            continue
        for (retained_name, group_key), acyl in RULES.retained.functional_parents.items():
            if group_key == "acyl":
                parents[(retained_name, key)] = f"{acyl} {halide}"
    return parents


def _retained_carbonyl_derivative_parents() -> dict[tuple[str, str], str]:
    """``benzaldehyde hydrazone``: a derivative keeps its carbonyl parent's name."""

    owner_of_suffix = {rule.suffix: key for key, rule in RULES.functional_groups.by_key.items()}
    parents: dict[tuple[str, str], str] = {}
    for key in RULES.functional_groups.keys_with_family("hydrazone"):
        rule = RULES.functional_groups.get(key)
        base_suffix, _, derivative = rule.suffix.partition(" ")
        base_key = owner_of_suffix.get(base_suffix)
        if base_key is None or not derivative:
            continue
        for (retained_name, group_key), retained in RULES.retained.functional_parents.items():
            if group_key == base_key:
                parents[(retained_name, key)] = f"{retained} {derivative}"
    return parents


def _retained_aldehyde_derivative_chain_parents() -> dict[tuple[int, str, int], RetainedChainParentRule]:
    """``acetaldehyde hydrazone``: an aldehyde derivative keeps the retained aldehyde word."""

    aldehyde_suffix = RULES.functional_groups.get("aldehyde").suffix
    parents: dict[tuple[int, str, int], RetainedChainParentRule] = {}
    for key, rule in RULES.functional_groups.by_key.items():
        if not rule.suffix:
            continue
        base_suffix, _, derivative = rule.suffix.partition(" ")
        if base_suffix != aldehyde_suffix or not derivative:
            continue
        for length, word in RULES.retained.aldehyde_chain_words.items():
            parents[(length, key, 1)] = RetainedChainParentRule(
                f"{word} {derivative}",
                allows_carbon_substitution=False,
            )
    return parents


RETAINED_CHAIN_PARENT_RULES = {
    **RULES.retained.chain_parents,
    **_retained_acyl_halide_parents(),
    **_retained_aldehyde_derivative_chain_parents(),
}
RETAINED_FUNCTIONAL_PARENTS = {
    **RULES.retained.functional_parents,
    **_retained_ring_acyl_halide_parents(),
    **_retained_carbonyl_derivative_parents(),
}


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


def _retained_chain_parent(parts: AssemblyParts) -> tuple[str, SubstituentItem | None] | None:
    """The retained name for an acyclic acid-family parent, and the branch it absorbs.
    Declines on unsaturation or replacement prefixes; the retained names are all saturated."""

    group = parts.principal_group
    if group is None or parts.is_ring or parts.unsaturations or parts.a_prefixes:
        return None

    if any(charge.charge < 0 for charge in parts.parent_charges):
        return None
    locants = [str(locant) for locant in group.locants]
    # One group sits at C1; two span the chain's ends.
    expected = ["1"] if len(locants) == 1 else ["1", str(parts.parent_length)]
    if sorted(locants, key=int) != expected:
        return None
    absorbed = _absorbed_retained_chain_branch(parts, group.key, len(locants))
    if absorbed is not None:
        return absorbed
    rule = RETAINED_CHAIN_PARENT_RULES.get((parts.parent_length, group.key, len(locants)))
    if rule is None:
        return None
    retained = rule.name
    has_carbon_substituent = any(str(locant).isdigit() for item in parts.substituents for locant in item.locants)
    if has_carbon_substituent and not rule.allows_carbon_substitution:
        retained = rule.substituted_name
        if retained is None:
            return None
    return retained, None


def _absorbed_retained_chain_branch(
    parts: AssemblyParts, group_key: str, group_count: int
) -> tuple[str, SubstituentItem] | None:
    """A retained acid name that spells its own C1 branch as well as its group."""

    if len(parts.substituents) != 1:
        return None
    branch = parts.substituents[0]
    if len(branch.locants) != 1:
        return None
    key = (parts.parent_length, group_key, group_count, branch.name, str(branch.locants[0]))
    retained = RULES.retained.substituted_chain_parents.get(key)
    if retained is not None:
        return retained, branch
    branch_name = strip_outer_parentheses(branch.name)
    for (length, key_group, count, ending, locant), word in RULES.retained.chain_branch_endings.items():
        if (length, key_group, count, locant) != (parts.parent_length, group_key, group_count, str(branch.locants[0])):
            continue
        if branch_name.endswith(ending) and not branch_name.startswith(("N-", "N,")):
            return f"{branch_name[: -len(ending)]}{word}", branch
    return None


def promote_retained_functional_parent(parts: AssemblyParts) -> None:
    """Fold a principal group into the retained parent name where one exists."""

    group = parts.principal_group
    if group is None or parts.is_substituent or parts.retained_absorbs_principal_group:
        return
    retained = RETAINED_FUNCTIONAL_PARENTS.get((parts.retained_name or "", group.key))
    if retained is not None and [str(locant) for locant in group.locants] != ["1"]:
        retained = None
    absorbed: SubstituentItem | None = None
    if retained is None:
        chain_parent = _retained_chain_parent(parts)
        if chain_parent is None:
            return
        retained, absorbed = chain_parent
    if group.has_positive_nitrogen and len(group.locants) == 1:
        rule = RULES.functional_groups.get(group.key)
        neutral_suffix = rule.retained_suffix or rule.suffix
        charged_suffix = rule.positive_nitrogen_retained_suffix or rule.positive_nitrogen_suffix
        if neutral_suffix and charged_suffix and retained.endswith(neutral_suffix):
            retained = retained[: -len(neutral_suffix)] + charged_suffix
    parts.retained_name = retained
    parts.retained_absorbs_principal_group = True
    if retained == "guanidine":
        # The guanidine C=N is cited as an unlocanted (E)/(Z); a ``1`` would point at a substituent's bond.
        parts.stereo_features = [
            ("" if str(locant) == "1" else locant, descriptor) for locant, descriptor in parts.stereo_features
        ]
    if absorbed is not None:
        parts.substituents = [item for item in parts.substituents if item is not absorbed]
        parts.retained_absorbed_substituents = [*parts.retained_absorbed_substituents, absorbed]


_SIMPLE_N_LIGANDS = frozenset(
    {
        "methyl",
        "ethyl",
        "propyl",
        "butyl",
        "phenyl",
        "benzyl",
        "hydroxy",
        "methoxy",
        "ethoxy",
        "cyano",
        "fluoro",
        "chloro",
        "amino",
    }
)


def promote_carbamimidoyl_substituent_name(parts: AssemblyParts) -> None:
    """P-66.4.1.1.1.4: an ``-C(=NR)NR2`` branch is carbamimidoyl, with N/N' substituents."""

    if (
        not parts.is_substituent
        or parts.retained_substituent_name is not None
        or parts.is_double_attach
        or parts.is_triple_attach
        or parts.is_ring
        or parts.retained_name
        or parts.principal_group
        or parts.unsaturations
        or parts.a_prefixes
        or parts.parent_charges
        or parts.parent_length != 1
        or str(parts.attachment_locant) != "1"
        or len(parts.substituents) != 2
        or substituent_attachment_kind(parts) != "single"
    ):
        return
    amine = imine = None
    for item in parts.substituents:
        text = strip_outer_parentheses(item.name)
        if text.endswith("imino") and imine is None:
            imine = text[: -len("imino")]
        elif text.endswith("amino") and amine is None:
            amine = text[: -len("amino")]
        else:
            return
    if amine is None or imine is None:
        return
    prefixes = []
    for ligands, locant in ((amine, "N"), (imine, "N'")):
        if not ligands:
            continue
        if "(" in ligands or ligands.startswith(("N-", "N,")):
            return
        count = next((n for n, rest in multipliers.candidate_splits(ligands) if rest in _SIMPLE_N_LIGANDS), 1)
        prefixes.append((ligands, f"{','.join([locant] * count)}-{ligands}"))
    prefixes.sort(key=lambda item: substituent_sort_key(item[0]))
    parts.retained_substituent_name = f"{'-'.join(text for _, text in prefixes)}carbamimidoyl"
    # The C=N is the group's own bond, cited as an unlocanted (E)/(Z) in front of the word.
    parts.stereo_features = [
        ("" if str(locant) == "1" else locant, descriptor) for locant, descriptor in parts.stereo_features
    ]
    parts.retained_absorbed_substituents = [*parts.retained_absorbed_substituents, *parts.substituents]
    parts.substituents = []


def promote_retained_substituent_name(parts: AssemblyParts) -> None:
    """Fold a substituent's own branch into a retained prefix where one exists."""

    promote_carbamimidoyl_substituent_name(parts)
    if (
        not parts.is_substituent
        or parts.retained_substituent_name is not None
        or parts.is_double_attach
        or parts.is_triple_attach
        or parts.is_ring
        or parts.retained_name
        or parts.principal_group
        or parts.unsaturations
        or parts.a_prefixes
        or parts.parent_charges
        or len(parts.substituents) != 1
    ):
        return
    branch = parts.substituents[0]
    if len(branch.locants) != 1:
        return
    retained = RULES.retained.substituents.get(
        (
            parts.parent_length,
            str(parts.attachment_locant),
            branch.name,
            str(branch.locants[0]),
            substituent_attachment_kind(parts),
        )
    )
    if retained is None:
        retained = _contracted_acyl_branch_name(parts, branch)
    if retained is None:
        return
    parts.retained_substituent_name = retained
    parts.retained_absorbed_substituents = [*parts.retained_absorbed_substituents, branch]
    parts.substituents = []


def _contracted_acyl_branch_name(parts: AssemblyParts, branch: SubstituentItem) -> str | None:
    """The retained word an acyl group and its C1 branch contract to, if any."""

    if parts.parent_length != 1 or str(parts.attachment_locant) != "1":
        return None
    branch_name = strip_outer_parentheses(branch.name)
    # An N-locant in front would read as a substituent on the contracted word's own
    # parent hydride, so those keep the systematic spelling.
    if branch_name.startswith(("N-", "N,")):
        return None
    kind = substituent_attachment_kind(parts)
    for (ending, branch_kind), word in RULES.retained.acyl_branch_endings.items():
        if branch_kind == kind and branch_name.endswith(ending):
            return f"{branch_name[: -len(ending)]}{word}"
    return None


def promote_acyl_substituent_name(parts: AssemblyParts) -> None:
    """P-65.3.1: recognise an ``R-CO-`` branch as an acyl group."""

    if (
        not parts.is_substituent
        or parts.is_acyl_substituent
        or parts.is_double_attach
        or parts.is_triple_attach
        or parts.is_ring
        or parts.retained_name
        or parts.retained_substituent_name is not None
        or parts.principal_group
        or parts.a_prefixes
        or parts.parent_charges
        or parts.hydro_operations
        or str(parts.attachment_locant) != "1"
    ):
        return
    carbonyls = [
        item for item in parts.substituents if item.name == "oxo" and [str(locant) for locant in item.locants] == ["1"]
    ]
    if len(carbonyls) != 1:
        return
    # C1 is the carbonyl carbon: ``eth-1-enoyl`` names nothing.
    if any("1" in {str(locant) for locant in unsaturation.locants} for unsaturation in parts.unsaturations):
        return
    if (parts.parent_length, "acyl") in RULES.retained.chain_substituent_stems and parts.unsaturations:
        return
    carbonyl = carbonyls[0]
    parts.is_acyl_substituent = True
    parts.substituents = [item for item in parts.substituents if item is not carbonyl]
    parts.retained_absorbed_substituents = [*parts.retained_absorbed_substituents, carbonyl]


def parent_stem_and_terminal(parts: AssemblyParts) -> tuple[str, str]:
    terminal_e = bonds.PARENT_TERMINAL_VOWEL

    inferred_ionic_parent = inferred_ionic_retained_parent(parts)
    parent_hydride = parts.parent_hydride
    if parent_hydride is not None and parent_hydride.hydride_kind in {
        ParentHydrideKind.SYSTEMATIC_FUSION,
        ParentHydrideKind.SKELETAL_REPLACEMENT_FUSION,
        ParentHydrideKind.BRIDGED_FUSION,
    }:
        if parts.is_substituent and parent_hydride.derivative_stem:
            stem_str = parent_hydride.derivative_stem
            terminal_e = ""
        else:
            stem_str, terminal_e = parent_hydride.assembly_stem_and_terminal(parts.parent_length) or ("", "")
    elif has_ionic_retained_parent(parts):
        stem_str = RULES.charges.retained_ionic_n_parents[parts.retained_name]
        terminal_e = ""
    elif inferred_ionic_parent:
        stem_str = inferred_ionic_parent
        terminal_e = ""
    elif parts.retained_name:
        # Assembly may promote a retained parent to a functional parent
        # (benzene -> benzoic acid).  That operation-local spelling takes
        # precedence over the immutable base-parent identity.
        retained_name = parts.retained_name
        retained_spec = retained_parent_spec(retained_name)
        if (
            parts.is_substituent
            and not parts.parent_charges
            and retained_spec
            and retained_spec.substituent_stem is not None
        ):
            stem_str = retained_spec.substituent_stem
            terminal_e = retained_spec.substituent_terminal or ""
        else:
            if retained_name.endswith("e"):
                stem_str = retained_name[:-1]
                terminal_e = "e"
            else:
                stem_str = retained_name
                terminal_e = ""
        if parts.parent_lambda_conventions:
            lambda_prefix = ",".join(
                f"{locant}lambda^{parts.parent_lambda_conventions[locant]}"
                for locant in sorted(parts.parent_lambda_conventions, key=parse_locant)
            )
            stem_str = f"{lambda_prefix}-{stem_str}"
    elif parent_hydride is not None and (
        hydride_spelling := parent_hydride.assembly_stem_and_terminal(parts.parent_length)
    ):
        stem_str, terminal_e = hydride_spelling
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
                raise ValueError("polycyclic parent has no audited descriptor")
        elif parts.is_ring:
            stem_str = "cyclo" + stem_str
    return stem_str, terminal_e


def apply_replacement_prefix(stem_str: str, a_prefix_str: str) -> str:
    """Join skeletal replacement prefixes without vowel elision (P-16.7.2(b))."""
    return a_prefix_str + stem_str


def format_unsaturations(parts: AssemblyParts, stem_str: str, *, omit_locants: bool = False) -> tuple[str, str]:
    sorted_unsats = sorted(parts.unsaturations, key=lambda u: UNSATURATION_ORDER.get(u.bond_key, 99))
    unsat_parts = []
    base_infixes = []
    for unsaturation in sorted_unsats:
        count = len(unsaturation.locants) or 1
        infix = bonds.unsaturation_infix(unsaturation.bond_key, count)
        base_infix = infix[1:] if infix.startswith("a") else infix
        base_infixes.append((unsaturation, count, base_infix))
    if base_infixes:
        first_unsaturation, first_count, first_infix = base_infixes[0]
        # The interfix ``a`` introduces a multiplying prefix, and elides only where the
        # two vowels actually meet: an intervening locant set keeps the ``a``.
        if first_count > 1 and (first_unsaturation.locants or not elision.is_vowel_start(first_infix)):
            stem_str += "a"
    base_infixes = [(unsaturation, infix) for unsaturation, _count, infix in base_infixes]
    for unsaturation, base_infix in base_infixes:
        if unsaturation.locants and not omit_locants and unsaturation.bond_key not in parts.elided_unsaturation_locants:
            loc_str = ",".join(sorted(unsaturation.locants, key=parse_locant))
            unsat_parts.append(f"-{loc_str}-{base_infix}")
        else:
            unsat_parts.append(base_infix)
    return stem_str, "".join(unsat_parts)


def substituent_attachment_kind(parts: AssemblyParts) -> str:
    """Which attachment a substituent spells; the retained tables key off this."""

    if parts.is_triple_attach:
        return "triple"
    if parts.is_double_attach:
        return "double"
    if parts.is_acyl_substituent:
        return "acyl"
    return "single"


def substituent_suffix_word(parts: AssemblyParts) -> str:
    """The word a substituent name ends in, for the attachment it spells."""

    suffixes = RULES.assembly.substituent_attachment_suffixes
    kind = substituent_attachment_kind(parts)
    # ``formyl`` is the whole one-carbon group; a branch on that carbon leaves no
    # chain to name, so the branch carries it: ``(thiophen-2-yl)carbonyl``.
    if kind == "acyl" and parts.parent_length == 1 and parts.substituents:
        return suffixes["acyl_on_branch"]
    return suffixes[kind]


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


def two_carbon_locants_are_redundant(parts: AssemblyParts) -> bool:
    """A two-carbon substituent has one place for its bond and its attachment: ``ethenyl``, ``ethynyl``."""

    if parts.is_ring or parts.parent_length != 2 or str(parts.attachment_locant) != "1":
        return False
    if len(parts.unsaturations) != 1:
        return False
    return [str(locant) for locant in parts.unsaturations[0].locants] == ["1"]


def format_substituent_tail(
    parts: AssemblyParts, stem_str: str, terminal_e: str, spiro_subs
) -> tuple[str, str, str, str]:
    suffix_yl = substituent_suffix_word(parts)
    charge_operations = parent_charge_name_operations(parts)
    if parts.is_acyl_substituent:
        # Attached at its own carbonyl carbon, so no attachment locant is cited.
        if suffix_yl == RULES.assembly.substituent_attachment_suffixes["acyl_on_branch"]:
            return "", "", suffix_yl, ""
        retained_stem = RULES.retained.chain_substituent_stems.get(
            (parts.parent_length, substituent_attachment_kind(parts))
        )
        if retained_stem is not None:
            return retained_stem[0], "", retained_stem[1], ""
        if parts.unsaturations:
            stem_str, unsat_str = format_unsaturations(parts, stem_str)
        else:
            unsat_str = "an"
        return stem_str, unsat_str, suffix_yl, ""
    always_print_locant = bool(spiro_subs) or always_print_substituent_locant(parts)
    if not always_print_locant and two_carbon_locants_are_redundant(parts):
        stem_str, unsat_str = format_unsaturations(parts, stem_str, omit_locants=True)
        return stem_str, unsat_str, append_charge_suffixes_to_terminal(parts, suffix_yl), ""
    if parts.parent_charges and (parts.parent_length > 1 or charge_operations):
        # A charge suffix and an attachment suffix are separate operations:
        # ``benzen-2-ide-1-yl``, not the unparseable ``phen-2-ideyl``.
        terminal_e = f"-{parts.attachment_locant}-{suffix_yl}"
    elif parts.retained_name == "benzene":
        terminal_e = "yl"
    elif (
        str(parts.attachment_locant) != "1" or parts.unsaturations or always_print_locant
    ) and parts.parent_length > 1:
        terminal_e = f"-{parts.attachment_locant}-{suffix_yl}"
    else:
        terminal_e = suffix_yl

    unsat_str = ""
    if not has_retained_like_parent(parts) and parts.unsaturations:
        stem_str, unsat_str = format_unsaturations(
            parts,
            stem_str,
            omit_locants=parts.parent_length == 1,
        )
    elif not has_retained_like_parent(parts) and not parts.unsaturations:
        if charge_operations and (
            parts.parent_length > 1 or any(operation.charge < 0 for operation in charge_operations)
        ):
            # Charged substituents derive from the complete parent hydride:
            # ``propan-1-ide-1-yl`` rather than ``prop-1-ideyl``.
            unsat_str = "an"
        elif parts.parent_length > 1 and (
            str(parts.attachment_locant) != "1"
            or parts.is_bicycle
            or parts.is_spiro
            or parts.is_polycycle
            or always_print_locant
        ):
            unsat_str = "an"
    terminal_e = append_charge_suffixes_to_terminal(parts, terminal_e)
    return stem_str, unsat_str, terminal_e, ""


def _sole_group_locant_is_redundant(parts: AssemblyParts) -> bool:
    """Whether a lone group's ``1`` locant tells the reader nothing."""

    ring_substituents = any(
        not str(locant).startswith(("N", "O", "S", "P")) for item in parts.substituents for locant in item.locants
    )
    if ring_substituents or parts.a_prefixes:
        return False
    if parts.is_ring:
        return not parts.unsaturations and parts.retained_name == "benzene"
    # ``ethanol``, ``ethenol``: a two-carbon chain's sole group can only sit at C1.
    return not parts.is_substituent and parts.parent_length == 2 and not parts.parent_charges


def format_principal_suffix(parts: AssemblyParts, terminal_e: str, spiro_subs) -> tuple[str, str]:
    if not parts.principal_group:
        return terminal_e, ""
    group = RULES.functional_groups.get(parts.principal_group.key)
    locs = sorted(parts.principal_group.locants, key=parse_locant)
    has_spiro_subs = bool(spiro_subs)
    positive_carbon_charge = any(charge.symbol == "C" and charge.charge > 0 for charge in parts.parent_charges)
    omit_locant = parts.parent_length == 1 and not positive_carbon_charge
    if not omit_locant and not positive_carbon_charge and len(locs) == 1 and str(locs[0]) == "1":
        if (
            not group.suffix_with_locant
            or (
                parts.is_ring
                and not parts.unsaturations
                and not parts.is_bicycle
                and not parts.is_spiro
                and not parts.is_polycycle
                and not has_spiro_subs
                and not parts.retained_name
            )
            or _sole_group_locant_is_redundant(parts)
        ):
            omit_locant = True

    suffix_text = render_principal_suffix(
        group,
        len(locs),
        positive_nitrogen=parts.principal_group.has_positive_nitrogen,
    )
    if parts.principal_suffix_modifiers and group.key in RULES.functional_groups.keys_with_family("hydrazone"):
        modifier_text = _format_principal_suffix_modifiers(parts)
        if modifier_text and suffix_text.endswith("hydrazone"):
            suffix_text = suffix_text[: -len("hydrazone")] + f"{modifier_text}hydrazone"

    if elision.is_vowel_start(suffix_text):
        terminal_e = ""
    if positive_carbon_charge and group.key == "carboxylate" and locs:
        return terminal_e, f"-{','.join(map(str, locs))}-{suffix_text}"
    if parts.elide_principal_group_locants:
        omit_locant = True
    if group.suffix_with_locant and locs and not omit_locant:
        return terminal_e, f"-{','.join(map(str, locs))}-{suffix_text}"
    return terminal_e, suffix_text


def _format_principal_suffix_modifiers(parts: AssemblyParts) -> str:
    grouped: dict[tuple[str, tuple[str, ...]], int] = {}
    for modifier in parts.principal_suffix_modifiers:
        key = (modifier.name, tuple(modifier.locants))
        grouped[key] = grouped.get(key, 0) + 1
    rendered = []
    show_locants = len(parts.principal_suffix_modifiers) > 1
    for name, locants in sorted(grouped):
        locant_text = ""
        if show_locants and locants:
            locant_text = f"{','.join(locants)}-"
        count = grouped[(name, locants)]
        # A modifier carrying its own locants (N'-methyl-N-propylcarbamimidoyl) is enclosed.
        text = f"({name})" if is_complex_prefix(name) and not name.startswith("(") else name
        if count > 1:
            repeated_locants = locants * count
            locant_text = f"{','.join(repeated_locants)}-" if repeated_locants else ""
            rendered.append(f"{locant_text}{multipliers.basic(count)}{text}")
        else:
            rendered.append(f"{locant_text}{text}")
    return "-".join(rendered) if show_locants and len(rendered) > 1 else "".join(rendered)


def _parent_unsaturation_locants_are_redundant(parts: AssemblyParts) -> bool:
    """P-31.1.4.2.4: ``ethene``, ``ethyne`` and an unsubstituted monounsaturated monocycle
    (``cyclohexene``) have only one place for their bond."""

    if parts.is_spiro_component:
        return False
    if len(parts.unsaturations) != 1 or len(parts.unsaturations[0].locants) != 1 or parts.a_prefixes:
        return False
    if not parts.is_ring:
        # OPSIN reads ethenol/ethenamine/ethenone but not ethenal or ethenethial.
        safe_suffixes = {"alcohol", "thiol", "amine", "ketone", "olate", "thiolate", "aminium"}
        return parts.parent_length == 2 and (
            parts.principal_group is None or parts.principal_group.key in safe_suffixes
        )
    return (
        not parts.is_bicycle
        and not parts.is_spiro
        and not parts.is_polycycle
        and not parts.substituents
        and parts.principal_group is None
        and not parts.parent_charges
    )


def format_parent_tail(parts: AssemblyParts, stem_str: str, terminal_e: str, spiro_subs) -> tuple[str, str, str, str]:
    unsat_str = ""
    if not has_retained_like_parent(parts):
        if not parts.unsaturations:
            unsat_str = bonds.get("single").saturated_suffix
        else:
            stem_str, unsat_str = format_unsaturations(
                parts, stem_str, omit_locants=_parent_unsaturation_locants_are_redundant(parts) and not spiro_subs
            )
    if parts.retained_absorbs_principal_group:
        # The retained name already spells the group (``phenol``).  ``terminal_e`` stays:
        # it is the retained name's own final vowel, and only a rendered suffix elides.
        suffix_str = ""
    else:
        terminal_e, suffix_str = format_principal_suffix(parts, terminal_e, spiro_subs)
    charge_operations = parent_charge_name_operations(parts)
    if charge_operations:
        terminal_e = ""
        suffix_str = (
            "".join(
                f"-{','.join(operation.locants)}-{suffix_operation_spelling(operation)}"
                for operation in charge_operations
            )
            + suffix_str
        )
    return stem_str, unsat_str, terminal_e, suffix_str
