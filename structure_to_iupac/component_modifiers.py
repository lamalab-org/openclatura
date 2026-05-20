"""Data-driven component modifiers attached after parent numbering."""

from collections.abc import Callable

from .assembly_parts import AssemblyParts
from .formatting import strip_outer_parentheses
from .group_atom_roles import ester_or_peroxy_single_oxygen
from .locants import parse_locant
from .molecule import Molecule
from .nomenclature import RULES
from .perception import PerceivedGroup
from .subgraph_tools import subgraph_component
from .trace_helpers import add_substituent_trace, bond_ids_within

BranchNamer = Callable[..., str]


def add_component_front_modifiers(
    mol: Molecule,
    parts: AssemblyParts,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    sub_exclude: set[int],
    branch_namer: BranchNamer,
) -> None:
    """Add ester/sulfonate front modifiers such as the alcohol component name."""

    if principal_key not in RULES.functional_groups.keys_with_family("front_modifier"):
        return
    for group in perceived_groups:
        if group.key != principal_key:
            continue
        single_o = ester_or_peroxy_single_oxygen(mol, group)
        if single_o is None:
            continue
        r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in group.atoms_involved), None)
        if r_group_c is None:
            continue
        branch_name = branch_namer(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
        if branch_name:
            parts.front_modifiers.append(strip_outer_parentheses(branch_name))
            parts.front_modifier_atom_ids.update(subgraph_component(mol, r_group_c, sub_exclude | {single_o}))


def n_substituent_locant(
    principal_key: str, principal_group_count: int, nitrogen_count: int, nitrogen_index: int, global_index: int
) -> str:
    """Return the N/N' locant prefix for a principal-group nitrogen."""

    if principal_key == "hydrazine":
        return "N" if nitrogen_index == 0 else "N'"
    if principal_key in RULES.functional_groups.keys_with_family("hydrazone"):
        return "N"
    if principal_group_count == 1 and nitrogen_count == 1:
        return "N"
    return "N" + "'" * global_index


def add_component_n_substituents(
    mol: Molecule,
    parts: AssemblyParts,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    numbered_path: list[int],
    get_loc,
    sub_exclude: set[int],
    branch_namer: BranchNamer,
) -> None:
    """Add N-substituent prefixes and N/N' locants for principal groups."""

    if principal_key is None:
        return
    principal_groups = [g for g in perceived_groups if g.key == principal_key and g.attachment_carbon in numbered_path]
    principal_groups.sort(key=lambda g: parse_locant(get_loc(g.attachment_carbon)))

    n_idx_global = 0
    for group in principal_groups:
        c_idx = group.attachment_carbon
        nitrogens = [n for n in group.atoms_involved if mol.atoms[n].symbol == "N"]
        nitrogens.sort(key=lambda n: mol.get_bond(n, c_idx) is not None, reverse=True)
        for n_idx_local, single_n in enumerate(nitrogens):
            n_substituents = [
                n
                for n in mol.get_neighbors(single_n)
                if n != c_idx and n not in group.atoms_involved and mol.atoms[n].symbol != "H"
            ]
            if not n_substituents:
                n_idx_global += 1
                continue

            loc_prefix = n_substituent_locant(
                principal_key, len(principal_groups), len(nitrogens), n_idx_local, n_idx_global
            )
            for n_sub in n_substituents:
                branch_name, branch_trace = branch_namer(
                    mol, n_sub, sub_exclude | {single_n}, upstream_atom=single_n, return_trace=True
                )
                if branch_name:
                    branch_atoms = subgraph_component(mol, n_sub, sub_exclude | {single_n})
                    add_substituent_trace(
                        parts,
                        branch_name,
                        loc_prefix,
                        branch_atoms,
                        bond_ids_within(mol, branch_atoms | {single_n}),
                        branch_trace,
                    )
            n_idx_global += 1
