"""Characteristic-group prefix collection for component naming."""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from .assembly_parts import NameTokenBinding, SubstituentItem
from .formatting import format_counted_prefixes, is_complex_prefix, oxy_prefix_from_branch, strip_outer_parentheses
from .group_atom_roles import amide_nitrogen, ester_or_peroxy_single_oxygen
from .molecule import Molecule
from .naming_protocols import RecursiveSubgraphNamer
from .nomenclature import RULES
from .perception import PerceivedGroup
from .rules import multipliers
from .subgraph_tools import subgraph_component
from .trace_helpers import bond_ids_within

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
    return f"({oxy_prefix_from_branch(branch_name)}{suffix_text})"


def amide_prefix_from_group(
    mol: Molecule, group: PerceivedGroup, sub_exclude: set[int], branch_namer: RecursiveSubgraphNamer
) -> str:
    """Return carbamoyl or carbamothioyl prefix text for an amide-like group."""

    single_n = amide_nitrogen(mol, group)
    if single_n is None:
        return ""
    n_subs = [n for n in mol.get_neighbors(single_n) if n not in group.atoms_involved and mol.atoms[n].symbol != "H"]
    base = RULES.functional_groups.prefix_for(group.key) or ""
    if not base:
        return ""
    if not n_subs:
        return base
    sub_names = [branch_namer(mol, x, sub_exclude | {single_n}, upstream_atom=single_n) for x in n_subs]
    return f"({format_counted_prefixes(sub_names)}{base})"


def ester_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return ester_prefix_from_group(context.mol, group, context.sub_exclude, "carbonyl", context.branch_namer)


def amide_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return amide_prefix_from_group(context.mol, group, context.sub_exclude, context.branch_namer)


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
        context.branch_namer(context.mol, n_sub, context.sub_exclude | {iminium_n}, upstream_atom=iminium_n)
        for n_sub in n_subs
    ]
    return f"({format_counted_prefixes(sub_names)}iminio)"


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


def sulfonyl_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return ester_prefix_from_group(context.mol, group, context.sub_exclude, "sulfonyl", context.branch_namer) or "sulfo"


def static_prefix_handler(name: str) -> PrefixHandler:
    return lambda _context, _group: name


def acid_halide_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return RULES.functional_groups.cited_prefix_for(group.key) or ""


def direct_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    return RULES.functional_groups.cited_prefix_for(group.key) or ""


def fallback_prefix_handler(context: PrefixContext, group: PerceivedGroup) -> str:
    if group.attachment_carbon not in context.parent_path:
        return ""
    return RULES.functional_groups.prefix_for(group.key) or ""


PREFIX_HANDLERS: dict[str, PrefixHandler] = {}
PREFIX_HANDLERS.update({key: ester_prefix_handler for key in RULES.functional_groups.keys_with_family("ester_like")})
PREFIX_HANDLERS.update({key: amide_prefix_handler for key in RULES.functional_groups.keys_with_family("amide_like")})
PREFIX_HANDLERS.update(
    {key: static_prefix_handler("carboxy") for key in RULES.functional_groups.keys_with_family("carboxy_prefix")}
)
PREFIX_HANDLERS.update(
    {key: static_prefix_handler("cyano") for key in RULES.functional_groups.keys_with_family("cyano_prefix")}
)
PREFIX_HANDLERS.update(
    {key: acid_halide_prefix_handler for key in RULES.functional_groups.keys_with_family("acid_halide")}
)
PREFIX_HANDLERS.update(
    {
        key: static_prefix_handler("hydroperoxycarbonyl")
        for key in RULES.functional_groups.keys_with_family("peroxy_acid")
    }
)
PREFIX_HANDLERS.update({key: sulfonyl_prefix_handler for key in RULES.functional_groups.keys_with_family("sulfonyl")})
PREFIX_HANDLERS.update(
    {key: direct_prefix_handler for key in RULES.functional_groups.keys_with_family("direct_prefix")}
)
PREFIX_HANDLERS["iminium"] = iminium_prefix_handler
PREFIX_HANDLERS["hydrazine"] = hydrazine_prefix_handler


def prefix_from_group(context: PrefixContext, group: PerceivedGroup) -> str:
    handler = PREFIX_HANDLERS.get(group.key, fallback_prefix_handler)
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
