"""Data-driven component modifiers attached after parent numbering."""

from typing import Literal, Protocol, overload

from .assembly_parts import AssemblyParts, SubstituentItem
from .formatting import strip_outer_parentheses
from .group_atom_roles import ester_or_peroxy_single_oxygen
from .locants import parse_locant
from .molecule import DecisionTrace, Molecule
from .nomenclature import RULES
from .perception import PerceivedGroup
from .subgraph_tools import subgraph_component
from .trace_helpers import add_substituent_trace, bond_ids_within, decision_trace_data


class BranchNamer(Protocol):
    """Recursive branch namer with simple and traced/tree return modes."""

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[False] = False,
        return_tree: Literal[False] = False,
        decision_trace: DecisionTrace | None = None,
    ) -> str: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[True],
        return_tree: Literal[False] = False,
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[str, list[dict]]: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[True],
        return_tree: Literal[True],
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[str, list[dict], dict | None]: ...

    @overload
    def __call__(
        self,
        mol: Molecule,
        start_idx: int,
        exclude_atoms: set[int],
        *,
        upstream_atom: int | None = None,
        return_trace: Literal[False] = False,
        return_tree: Literal[True],
        decision_trace: DecisionTrace | None = None,
    ) -> tuple[str, dict | None]: ...


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
            modifier_atoms = subgraph_component(mol, r_group_c, sub_exclude | {single_o})
            parts.front_modifiers.append(strip_outer_parentheses(branch_name))
            parts.front_modifier_atom_ids.update(modifier_atoms)
            parts.front_modifier_charge_atom_ids.update(_charged_atoms(mol, modifier_atoms))


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
                if n != c_idx
                and n not in group.atoms_involved
                and mol.atoms[n].symbol != "H"
                and not _is_principal_hydrazone_carbon(mol, principal_key, single_n, n)
            ]
            if not n_substituents:
                n_idx_global += 1
                continue

            loc_prefix = n_substituent_locant(
                principal_key, len(principal_groups), len(nitrogens), n_idx_local, n_idx_global
            )
            for n_sub in n_substituents:
                branch_decisions = DecisionTrace()
                branch_name, branch_trace, branch_tree = _nitrogen_substituent_name(
                    mol, single_n, n_sub, sub_exclude, branch_namer, branch_decisions
                )
                if branch_name:
                    branch_atoms = subgraph_component(mol, n_sub, sub_exclude | {single_n})
                    nested_decisions = decision_trace_data(branch_decisions)
                    if _use_hydrazone_suffix_modifier(parts, principal_key):
                        parts.principal_suffix_modifiers.append(
                            SubstituentItem(
                                branch_name,
                                [],
                                atom_ids=branch_atoms,
                                bond_ids=bond_ids_within(mol, branch_atoms | {single_n}),
                                charge_atom_ids=_charged_atoms(mol, branch_atoms),
                                trace_segments=branch_trace,
                                nested_decisions=nested_decisions,
                                substituent_tree=branch_tree,
                            )
                        )
                        continue
                    add_substituent_trace(
                        parts,
                        branch_name,
                        loc_prefix,
                        branch_atoms,
                        bond_ids_within(mol, branch_atoms | {single_n}),
                        _charged_atoms(mol, branch_atoms),
                        branch_trace,
                        nested_decisions,
                        substituent_tree=branch_tree,
                    )
            n_idx_global += 1


def _charged_atoms(mol: Molecule, atom_ids: set[int]) -> set[int]:
    """Return formally charged atoms from an already named graph fragment."""

    return {atom_idx for atom_idx in atom_ids if mol.atoms[atom_idx].charge != 0}


def _nitrogen_substituent_name(
    mol: Molecule,
    nitrogen: int,
    substituent: int,
    sub_exclude: set[int],
    branch_namer: BranchNamer,
    decision_trace: DecisionTrace | None = None,
) -> tuple[str, list, dict | None]:
    """Render graph-bound N-substituents on principal nitrogen groups."""

    bond = mol.get_bond(nitrogen, substituent)
    if (
        bond is not None
        and bond.order == 2
        and mol.atoms[substituent].symbol == "N"
        and not [n for n in mol.get_neighbors(substituent) if n != nitrogen and mol.atoms[n].symbol != "H"]
    ):
        return "imino", [], None
    return branch_namer(
        mol,
        substituent,
        sub_exclude | {nitrogen},
        upstream_atom=nitrogen,
        return_trace=True,
        return_tree=True,
        decision_trace=decision_trace,
    )


def _use_hydrazone_suffix_modifier(parts: AssemblyParts, principal_key: str | None) -> bool:
    """Avoid ambiguous N-prefixes when a hydrazone parent already has ring N atoms."""

    if principal_key not in RULES.functional_groups.keys_with_family("hydrazone"):
        return False
    return any(symbol == "N" for symbol in parts.parent_atom_symbols_by_locant.values())


def _is_principal_hydrazone_carbon(mol: Molecule, principal_key: str | None, nitrogen: int, neighbor: int) -> bool:
    if principal_key not in RULES.functional_groups.keys_with_family("hydrazone"):
        return False
    if not mol.atoms[neighbor].is_carbon:
        return False
    bond = mol.get_bond(nitrogen, neighbor)
    return bond is not None and bond.order == 2
