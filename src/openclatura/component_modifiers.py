"""Data-driven component modifiers attached after parent numbering."""

from .assembly_parts import AssemblyParts, NameTokenBinding, SubstituentItem
from .formatting import strip_outer_parentheses
from .graph_queries import bond_ids_within, charged_atom_ids
from .group_atom_roles import ester_or_peroxy_single_oxygen
from .locants import parse_locant
from .molecule import DecisionTrace, Molecule
from .naming_protocols import RecursiveSubgraphNamer
from .nomenclature import RULES
from .perception import PerceivedGroup
from .subgraph_tools import subgraph_component
from .substituent_tokens import graph_bound_substituent_tokens
from .trace_helpers import add_substituent_trace, decision_trace_data


def add_component_front_modifiers(
    mol: Molecule,
    parts: AssemblyParts,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    sub_exclude: set[int],
    branch_namer: RecursiveSubgraphNamer,
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
            parts.front_modifier_charge_atom_ids.update(charged_atom_ids(mol, modifier_atoms))


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
    branch_namer: RecursiveSubgraphNamer,
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
                    branch_exclude = sub_exclude | {single_n}
                    branch_atoms = subgraph_component(mol, n_sub, branch_exclude)
                    nested_decisions = decision_trace_data(branch_decisions)
                    emitted_tokens = graph_bound_substituent_tokens(
                        mol,
                        n_sub,
                        branch_atoms,
                        branch_name,
                        single_n,
                        branch_exclude,
                        branch_namer,
                    )
                    emitted_tokens = _with_n_substituent_locant_token(
                        emitted_tokens,
                        loc_prefix,
                        single_n,
                        bond_ids_within(mol, {single_n, n_sub}),
                    )
                    if _use_hydrazone_suffix_modifier(parts, principal_key):
                        parts.principal_suffix_modifiers.append(
                            SubstituentItem(
                                branch_name,
                                [],
                                atom_ids=branch_atoms,
                                bond_ids=bond_ids_within(mol, branch_atoms | {single_n}),
                                charge_atom_ids=charged_atom_ids(mol, branch_atoms),
                                emitted_tokens=emitted_tokens,
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
                        charged_atom_ids(mol, branch_atoms),
                        branch_trace,
                        nested_decisions,
                        emitted_tokens,
                        substituent_tree=branch_tree,
                    )
            n_idx_global += 1


def _with_n_substituent_locant_token(
    emitted_tokens: tuple[NameTokenBinding, ...],
    locant: str,
    nitrogen_atom: int,
    branch_bonds: set[int],
) -> tuple[NameTokenBinding, ...]:
    """Bind N-substituent locants to the principal nitrogen atom."""

    locant_token = NameTokenBinding(
        text=locant,
        token_kind="locant",
        source="n_substituent_locant",
        grammar_role="n_substituent",
        binding_key="prefix:n_substituent_locant",
        atom_ids={nitrogen_atom},
        bond_ids=set(branch_bonds),
        locants=(locant,),
    )
    return (
        locant_token,
        *tuple(token for token in emitted_tokens if not (token.token_kind == "locant" and token.text == locant)),
    )


def _nitrogen_substituent_name(
    mol: Molecule,
    nitrogen: int,
    substituent: int,
    sub_exclude: set[int],
    branch_namer: RecursiveSubgraphNamer,
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
