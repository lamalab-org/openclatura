"""Characteristic-group prefix collection for component naming."""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace

from .assembly_parts import NameTokenBinding, SubstituentItem, rendered_substituent_text
from .assembly_prefixes import substituent_sort_key
from .chalcogen_roles import ChalcogenLigandRole, DerivativeKind, FunctionalFamily
from .chalcogen_vocabulary import (
    central_linkage_prefix,
    chalcogen_substituent_suffix,
    resolve_central_acid_rule,
    simple_group_key,
)
from .formatting import (
    format_center_ligands,
    format_element_substituent,
    format_multiplier,
    is_complex_prefix,
    oxy_prefix_from_branch,
    strip_outer_parentheses,
)
from .group_atom_roles import amide_nitrogen, ester_or_peroxy_single_oxygen
from .molecule import Molecule, bond_ids_within
from .naming_protocols import RecursiveSubgraphNamer
from .nomenclature import RULES, FunctionalGroupCapability
from .perception import PerceivedGroup
from .rules import multipliers
from .subgraph_tools import subgraph_component

PrefixHandler = Callable[["PrefixContext", PerceivedGroup], str]


@dataclass(frozen=True)
class PrefixContext:
    mol: Molecule
    parent_path: list[int]
    sub_exclude: set[int]
    branch_namer: RecursiveSubgraphNamer


def ester_prefix_from_group(
    mol: Molecule,
    group: PerceivedGroup,
    sub_exclude: set[int],
    suffix_text: str,
    branch_namer: RecursiveSubgraphNamer,
) -> str:
    """Return an alkoxycarbonyl or alkoxysulfonyl-style prefix."""

    single_o = ester_or_peroxy_single_oxygen(mol, group)
    if single_o is None:
        return ""
    r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in group.atoms_involved), None)
    if r_group_c is None:
        return ""
    branch_name = branch_namer(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
    if not branch_name:
        return ""
    branch_name = rendered_substituent_text(branch_name)
    return f"({oxy_prefix_from_branch(branch_name)}{suffix_text})"


def amide_prefix_from_group(
    mol: Molecule, group: PerceivedGroup, sub_exclude: set[int], branch_namer: RecursiveSubgraphNamer
) -> str:
    """Return carbamoyl or carbamothioyl prefix text for an amide-like group."""

    single_n = amide_nitrogen(mol, group)
    if single_n is None:
        return ""
    n_subs = [n for n in mol.get_neighbors(single_n) if n not in group.atoms_involved and mol.atoms[n].symbol != "H"]
    base = group.prefix or ""
    if not base:
        return ""
    if not n_subs:
        return base
    sub_names = [
        rendered_substituent_text(branch_namer(mol, x, sub_exclude | {single_n}, upstream_atom=single_n))
        for x in n_subs
    ]
    return f"({format_center_ligands(sub_names, sort_key=substituent_sort_key)}{base})"


def peroxy_ester_prefix_from_group(
    mol: Molecule,
    group: PerceivedGroup,
    sub_exclude: set[int],
    branch_namer: RecursiveSubgraphNamer,
) -> str:
    """Return a ``…peroxycarbonyl`` prefix for a peroxy ester.

    ``oxycarbonyl`` spells one single-bonded oxygen, and a peroxy ester has
    two, so ``-C(=O)-O-O-tBu`` came out as ``tert-butoxycarbonyl`` -- an oxygen
    short, and a carbon long once the parent absorbed the acyl carbon anyway.
    """

    single_o = ester_or_peroxy_single_oxygen(mol, group)
    if single_o is None:
        return ""
    r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in group.atoms_involved), None)
    if r_group_c is None:
        return ""
    branch_name = branch_namer(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
    if not branch_name:
        return ""
    branch_name = strip_outer_parentheses(rendered_substituent_text(branch_name))
    return f"({format_multiplier(branch_name, 1)}peroxycarbonyl)"


def ester_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    if group.key in RULES.prefixes.peroxy_ester_groups:
        return peroxy_ester_prefix_from_group(context.mol, group, context.sub_exclude, context.branch_namer)
    return ester_prefix_from_group(context.mol, group, context.sub_exclude, "carbonyl", context.branch_namer)


def central_ester_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    """Render an ester prefix from its resolved central-acid vocabulary."""

    descriptor = group.descriptor
    if descriptor is not None:
        linkers = descriptor.ligands_with_role(ChalcogenLigandRole.CARBON_LINK)
        base = central_linkage_prefix(descriptor)
        if len(linkers) == 1 and base:
            linker = linkers[0]
            attachment = linker.attachment_atom
            if attachment is not None:
                branch = rendered_substituent_text(
                    context.branch_namer(
                        context.mol,
                        attachment,
                        context.sub_exclude | {linker.atom},
                        upstream_atom=linker.atom,
                    )
                )
                ligand = format_element_substituent(
                    "",
                    branch,
                    chalcogen_substituent_suffix(linker.element),
                )
                return f"({strip_outer_parentheses(ligand)}{base})"
    return ester_prefix_from_group(
        context.mol,
        group,
        context.sub_exclude,
        group.prefix or "",
        context.branch_namer,
    )


def central_chalcogen_linkage_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    """Render a two-chalcogen central linkage as a substitutive prefix."""

    descriptor = group.descriptor
    if descriptor is None:
        return group.prefix or ""
    double_ligands = descriptor.ligands_with_role(ChalcogenLigandRole.DOUBLE_BONDED)
    bridge = tuple(ligand for ligand in descriptor.ligands if ligand not in double_ligands)
    base = central_linkage_prefix(descriptor)
    if len(bridge) != 2 or not base:
        return group.prefix or ""
    first, terminal = bridge
    terminal_text = ""
    if terminal.role is ChalcogenLigandRole.HYDROGEN_BEARING:
        key = simple_group_key(FunctionalFamily.HYDROXY, DerivativeKind.ALCOHOL, terminal.element)
        terminal_text = RULES.functional_groups.cited_prefix_for(key) if key is not None else ""
    elif terminal.role in {ChalcogenLigandRole.CARBON_LINK, ChalcogenLigandRole.HETEROATOM_LINK}:
        attachment = terminal.attachment_atom
        if attachment is not None:
            branch = rendered_substituent_text(
                context.branch_namer(
                    context.mol,
                    attachment,
                    context.sub_exclude | {terminal.atom},
                    upstream_atom=terminal.atom,
                )
            )
            terminal_text = format_element_substituent(
                "",
                branch,
                chalcogen_substituent_suffix(terminal.element),
            )
    if not terminal_text:
        return group.prefix or ""
    linked = format_element_substituent(
        "",
        terminal_text,
        chalcogen_substituent_suffix(first.element),
    )
    return f"({linked}{base})"


def amide_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return amide_prefix_from_group(context.mol, group, context.sub_exclude, context.branch_namer)


def amidine_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    """P-66.4.1.1.1.4: ``-C(=NR')NR2`` is (N-R)(N'-R')carbamimidoyl."""

    mol = context.mol
    carbon = next((a for a in group.atoms_involved if mol.atoms[a].is_carbon), None)
    if carbon is None:
        return "carbamimidoyl"
    nitrogens = [n for n in group.atoms_involved if mol.atoms[n].symbol == "N"]
    nitrogens.sort(key=lambda n: mol.get_bond(n, carbon).order)  # amine N first, imino N' second
    # The C=N is the group's own bond, cited as an unlocanted (E)/(Z) in front of the word.
    imine_bond = mol.get_bond(nitrogens[-1], carbon) if nitrogens else None
    stereo = f"({imine_bond.stereo})-" if imine_bond and imine_bond.order == 2 and imine_bond.stereo else ""
    locants_by_name: dict[str, list[str]] = {}
    for nitrogen, locant in zip(nitrogens, ("N", "N'"), strict=False):
        for x in mol.get_neighbors(nitrogen):
            if x == carbon or mol.atoms[x].symbol == "H":
                continue
            name = rendered_substituent_text(
                context.branch_namer(mol, x, context.sub_exclude | {nitrogen}, upstream_atom=nitrogen)
            )
            locants_by_name.setdefault(name, []).append(locant)
    if not locants_by_name:
        return f"({stereo}carbamimidoyl)" if stereo else "carbamimidoyl"
    prefixes = []
    for name in sorted(locants_by_name, key=substituent_sort_key):
        locants = locants_by_name[name]
        prefixes.append(
            f"{','.join(locants)}-{format_multiplier(name, len(locants), safe_enclose=is_complex_prefix(name))}"
        )
    return f"({stereo}{'-'.join(prefixes)}carbamimidoyl)"


def hydrazide_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    """P-66.3.1.3: ``-C(=O)-NR-NR2`` is (1-R-2-R)hydrazine-1-carbonyl; unsubstituted, hydrazinecarbonyl."""

    mol = context.mol
    nitrogens = [n for n in group.atoms_involved if mol.atoms[n].symbol == "N"]
    carbon = next(
        (
            a
            for a in group.atoms_involved
            if mol.atoms[a].is_carbon and any(mol.get_bond(a, n) is not None for n in nitrogens)
        ),
        None,
    )
    if carbon is None or len(nitrogens) != 2:
        return "hydrazinecarbonyl"
    nitrogens.sort(key=lambda n: mol.get_bond(n, carbon) is None)  # acyl-bound N first (locant 1)
    locants_by_name: dict[str, list[str]] = {}
    for nitrogen, locant in zip(nitrogens, ("1", "2"), strict=True):
        for x in mol.get_neighbors(nitrogen):
            if x in group.atoms_involved or x == carbon or mol.atoms[x].symbol == "H":
                continue
            name = rendered_substituent_text(
                context.branch_namer(mol, x, context.sub_exclude | {nitrogen}, upstream_atom=nitrogen)
            )
            locants_by_name.setdefault(name, []).append(locant)
    if not locants_by_name:
        return "hydrazinecarbonyl"
    prefixes = []
    for name in sorted(locants_by_name, key=substituent_sort_key):
        locants = locants_by_name[name]
        prefixes.append(
            f"{','.join(locants)}-{format_multiplier(name, len(locants), safe_enclose=is_complex_prefix(name))}"
        )
    return f"({'-'.join(prefixes)}hydrazine-1-carbonyl)"


def central_hydrazide_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    """Render a central hydrazide as a substituted central amide prefix."""

    descriptor = group.descriptor
    if descriptor is None or descriptor.family is not FunctionalFamily.CENTRAL_ACID:
        return group.prefix or ""
    center = descriptor.centers[0]
    nitrogens = [
        atom
        for atom in group.atoms_involved
        if context.mol.atoms[atom].symbol == "N" and context.mol.get_bond(center, atom) is not None
    ]
    if len(nitrogens) != 1:
        return group.prefix or ""
    nitrogen = nitrogens[0]
    substituents = [
        atom for atom in context.mol.get_neighbors(nitrogen) if atom != center and context.mol.atoms[atom].symbol != "H"
    ]
    amide_key, amide_rule = resolve_central_acid_rule(replace(descriptor, derivative=DerivativeKind.AMIDE))
    base = amide_rule.prefix if amide_rule is not None else RULES.functional_groups.cited_prefix_for(amide_key)
    if not base:
        return ""
    if not substituents:
        return base
    names = [
        rendered_substituent_text(
            context.branch_namer(
                context.mol,
                atom,
                context.sub_exclude | {nitrogen},
                upstream_atom=nitrogen,
            )
        )
        for atom in substituents
    ]
    return f"({format_center_ligands(names, sort_key=substituent_sort_key)}{base})"


def iminium_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    nitrogens = [n for n in group.atoms_involved if context.mol.atoms[n].symbol == "N"]
    if not nitrogens:
        return "iminio"
    iminium_n = nitrogens[0]
    n_subs = [
        n
        for n in context.mol.get_neighbors(iminium_n)
        if n != group.attachment_carbon and n not in group.atoms_involved and context.mol.atoms[n].symbol != "H"
    ]
    if not n_subs:
        return "iminio"
    sub_names = [
        rendered_substituent_text(
            context.branch_namer(
                context.mol,
                n_sub,
                context.sub_exclude | {iminium_n},
                upstream_atom=iminium_n,
            )
        )
        for n_sub in n_subs
    ]
    return f"({format_center_ligands(sub_names, sort_key=substituent_sort_key)}iminio)"


def hydrazine_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    """Render C-N-N hydrazines as graph-bound hydrazinyl prefixes."""

    nitrogens = [n for n in group.atoms_involved if context.mol.atoms[n].symbol == "N"]
    attached = next(
        (n for n in nitrogens if context.mol.get_bond(n, group.attachment_carbon) is not None),
        None,
    )
    if attached is None:
        return "hydrazinyl"
    terminal = next((n for n in nitrogens if n != attached), None)
    if terminal is None:
        return "hydrazinyl"
    substituents: list[tuple[str, int]] = []
    substituents.extend(
        (
            "N",
            n,
        )
        for n in context.mol.get_neighbors(attached)
        if n != group.attachment_carbon and n not in group.atoms_involved and context.mol.atoms[n].symbol != "H"
    )
    substituents.extend(
        (
            "N'",
            n,
        )
        for n in context.mol.get_neighbors(terminal)
        if n != attached and n not in group.atoms_involved and context.mol.atoms[n].symbol != "H"
    )
    if not substituents:
        return "hydrazinyl"
    prefix_items = []
    for locant, n_sub in substituents:
        owner = attached if locant == "N" else terminal
        branch = context.branch_namer(context.mol, n_sub, context.sub_exclude | {owner}, upstream_atom=owner)
        if not branch:
            continue
        branch = strip_outer_parentheses(branch)
        prefix_items.append((locant, branch))
    if not prefix_items:
        return "hydrazinyl"
    prefixes = []
    for (locant, branch), count in Counter(prefix_items).items():
        locants = ",".join([locant] * count)
        branch_text = branch
        if count > 1:
            branch_text = f"{multipliers.basic(count)}{branch}"
        elif is_complex_prefix(branch_text):
            branch_text = f"({branch_text})"
        prefixes.append(f"{locants}-{branch_text}")
    return f"({'-'.join(prefixes)}hydrazinyl)"


def imino_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    """Render an imino nitrogen, carrying its own substituent when it has one."""

    nitrogen = next(iter(group.atoms_involved), None)
    if nitrogen is None:
        return "imino"
    ligands = [
        n
        for n in context.mol.get_neighbors(nitrogen)
        if n != group.attachment_carbon and context.mol.atoms[n].symbol != "H"
    ]
    if len(ligands) != 1:
        return "imino"
    branch = context.branch_namer(context.mol, ligands[0], context.sub_exclude | {nitrogen}, upstream_atom=nitrogen)
    branch = strip_outer_parentheses(rendered_substituent_text(branch))
    if not branch:
        return "imino"
    return f"({branch}imino)"


def sulfonyl_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return ester_prefix_from_group(context.mol, group, context.sub_exclude, "sulfonyl", context.branch_namer) or "sulfo"


def static_prefix_handler(name: str) -> PrefixHandler:
    return lambda _context, _group: name


def acyl_leaving_group_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    if group.resolved_rule is not None and group.resolved_rule.has_capability(
        FunctionalGroupCapability.ACYL_LEAVING_GROUP
    ):
        return f"({group.prefix})" if group.prefix else ""
    return RULES.functional_groups.cited_prefix_for(group.key) or ""


def direct_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return RULES.functional_groups.cited_prefix_for(group.key) or ""


def fallback_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    if group.attachment_carbon not in context.parent_path:
        return ""
    return group.prefix or ""


PREFIX_HANDLERS: dict[str, PrefixHandler] = {}
PREFIX_HANDLERS.update(dict.fromkeys(RULES.functional_groups.keys_with_family("ester_like"), ester_prefix_handler))
PREFIX_HANDLERS.update(dict.fromkeys(RULES.functional_groups.keys_with_family("amide_like"), amide_prefix_handler))
PREFIX_HANDLERS.update(
    {key: static_prefix_handler("carboxy") for key in RULES.functional_groups.keys_with_family("carboxy_prefix")}
)
PREFIX_HANDLERS.update(
    {key: static_prefix_handler("cyano") for key in RULES.functional_groups.keys_with_family("cyano_prefix")}
)
PREFIX_HANDLERS.update(
    dict.fromkeys(
        RULES.functional_groups.keys_with_capability(FunctionalGroupCapability.ACYL_LEAVING_GROUP),
        acyl_leaving_group_prefix_handler,
    )
)
PREFIX_HANDLERS.update(
    {
        key: static_prefix_handler("hydroperoxycarbonyl")
        for key in RULES.functional_groups.keys_with_family("peroxy_acid")
    }
)
PREFIX_HANDLERS.update(dict.fromkeys(RULES.functional_groups.keys_with_family("sulfonyl"), sulfonyl_prefix_handler))
PREFIX_HANDLERS.update(dict.fromkeys(RULES.functional_groups.keys_with_family("direct_prefix"), direct_prefix_handler))
PREFIX_HANDLERS["hydrazide"] = hydrazide_prefix_handler
PREFIX_HANDLERS["ring_hydrazide"] = hydrazide_prefix_handler
PREFIX_HANDLERS["amidine"] = amidine_prefix_handler
PREFIX_HANDLERS["ring_amidine"] = amidine_prefix_handler
PREFIX_HANDLERS["imino_prefix"] = imino_prefix_handler
PREFIX_HANDLERS["iminium"] = iminium_prefix_handler
PREFIX_HANDLERS["hydrazine"] = hydrazine_prefix_handler


def prefix_from_group(context: PrefixContext, group: PerceivedGroup) -> str:
    handler = PREFIX_HANDLERS.get(group.key)
    if handler is None and group.resolved_rule is not None:
        central_linkage = (
            group.descriptor is not None
            and group.descriptor.family is FunctionalFamily.CENTRAL_ACID
            and len(group.descriptor.ligands)
            - len(group.descriptor.ligands_with_role(ChalcogenLigandRole.DOUBLE_BONDED))
            == 2
        )
        if central_linkage:
            handler = central_chalcogen_linkage_prefix_handler
        elif group.resolved_rule.has_capability(
            FunctionalGroupCapability.CENTRAL_ACID
        ) and group.resolved_rule.has_capability(FunctionalGroupCapability.HYDRAZIDE):
            handler = central_hydrazide_prefix_handler
        elif group.resolved_rule.has_capability(
            FunctionalGroupCapability.CENTRAL_ACID
        ) and group.resolved_rule.has_capability(FunctionalGroupCapability.ESTER_LIKE):
            handler = central_ester_prefix_handler
        elif group.resolved_rule.has_capability(FunctionalGroupCapability.ESTER_LIKE):
            handler = ester_prefix_handler
        elif group.resolved_rule.has_capability(FunctionalGroupCapability.AMIDE_LIKE):
            handler = amide_prefix_handler
        elif group.resolved_rule.has_capability(FunctionalGroupCapability.ACYL_LEAVING_GROUP):
            handler = acyl_leaving_group_prefix_handler
    if handler is None:
        handler = fallback_prefix_handler
    return handler(context, group)


def collect_component_prefix_substituents(
    mol: Molecule,
    prefix_groups: list[PerceivedGroup],
    parent_path: list[int],
    sub_exclude: set[int],
    branch_namer: RecursiveSubgraphNamer,
) -> tuple[dict[int, list[SubstituentItem]], set[int]]:
    """Collect characteristic groups cited as prefixes on the component parent."""

    main_set = set(parent_path)
    subst_mapping: dict[int, list[SubstituentItem]] = {}
    handled_prefix_atoms = set()
    context = PrefixContext(mol=mol, parent_path=parent_path, sub_exclude=sub_exclude, branch_namer=branch_namer)

    for group in prefix_groups:
        if (
            group.key in RULES.functional_groups.keys_with_family("prefix_skip")
            or group.attachment_carbon not in main_set
        ):
            continue

        name = prefix_from_group(context, group)
        if name:
            trace_atoms = set(group.atoms_involved)
            for atom_idx in group.atoms_involved:
                for neighbor in mol.get_neighbors(atom_idx):
                    if (
                        neighbor not in group.atom_ids
                        and neighbor not in main_set
                        and neighbor not in context.sub_exclude
                        and mol.atoms[neighbor].symbol != "H"
                    ):
                        trace_atoms.update(subgraph_component(mol, neighbor, context.sub_exclude | {atom_idx}))
            trace_bonds = bond_ids_within(mol, trace_atoms | {group.attachment_carbon})
            subst_mapping.setdefault(group.attachment_carbon, []).append(
                SubstituentItem(
                    name=name,
                    locants=[],
                    atom_ids=trace_atoms,
                    bond_ids=trace_bonds,
                    charge_atom_ids={atom_idx for atom_idx in trace_atoms if mol.atoms[atom_idx].charge != 0},
                    emitted_tokens=functional_prefix_tokens(mol, group, name, trace_atoms, trace_bonds),
                )
            )
            handled_prefix_atoms.update(group.atoms_involved)

    return subst_mapping, handled_prefix_atoms


def functional_prefix_tokens(
    mol: Molecule,
    group: PerceivedGroup,
    name: str,
    atom_ids: set[int],
    bond_ids: set[int],
) -> tuple[NameTokenBinding, ...]:
    """Return graph-bound tokens emitted by a functional-prefix renderer."""

    charge_atoms = {atom_idx for atom_idx in atom_ids if mol.atoms[atom_idx].charge != 0}
    return tuple(
        NameTokenBinding(
            text=token_text,
            token_kind="prefix",
            source="functional_prefix_renderer",
            grammar_role=group.key,
            binding_key=f"prefix:{group.key}",
            atom_ids=set(atom_ids),
            bond_ids=set(bond_ids),
            charge_atom_ids=set(charge_atoms),
        )
        for token_text in _prefix_lexical_tokens(name)
    )


def _prefix_lexical_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _PREFIX_TOKEN_RE.finditer(strip_outer_parentheses(text)))


_PREFIX_TOKEN_RE = re.compile(r"[A-Za-z]+(?:\^[0-9]+)?|\d+(?:,\d+)*(?:'\")?|[0-9]+(?:\([0-9,]+\))?")
