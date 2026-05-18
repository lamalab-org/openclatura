# structure-to-iupac/api.py

import re

from .formatting import format_counted_prefixes, strip_outer_parentheses
from .group_atom_roles import amide_nitrogen
from .molecule import DecisionTrace, Molecule, NameAnalysis, TracePhase
from .naming_context import ComponentNamingState, NamingIntent
from .perception import perceive_groups, PerceivedGroup
from .chains import find_all_carbon_paths, find_ring_systems, get_cyclic_atoms
from .parent_selection import select_principal_parent
from .assembler import assemble_name, assemble_name_raw, post_process_name
from .assembly_parts import AssemblyParts, SubstituentItem
from .assembly_spiro import extract_spiro_side_prefixes
from .component_namer import name_component as _name_component_impl
from .engine import DEFAULT_NAMING_ENGINE
from .graph_io import get_connected_components, read_smiles
from .heteroatom_subgraphs import name_heteroatom_subgraph
from .ionic_naming import apply_anionic_parent_names, apply_cationic_imino_names
from .namer_config import (
    SALT_METAL_NAMES,
    SPECIAL_COMPONENT_NAMES,
)
from .nomenclature import RULES
from .operations import infer_operations
from .parent_pipeline import build_parent_assembly_plan, resolve_retained_parent
from .spiro_assembly import SpiroAssembly
from .subgraph_tools import (
    add_indicated_hydrogens as _add_indicated_hydrogens,
    add_parent_features as _add_subgraph_parent_features,
    emit_bond_stereo as _emit_bond_stereo,
    find_spiro_side_pair as _find_spiro_side_pair,
    spiro_side_component as _spiro_side_component,
    subgraph_component as _subgraph_component,
)
from .trace_helpers import (
    add_substituent_trace as _add_substituent_trace,
    assembly_trace_segments as _assembly_trace_segments,
    bond_ids_within as _bond_ids_within,
    functional_group_trace_data as _functional_group_trace_data,
    trace_decision as _trace_decision,
)


def _direct_subgraph_prefix(mol: Molecule, start_idx: int, component: set[int]) -> str:
    """Return a direct functional-group prefix when the subgraph is one group.

    Blue Book references: P-63 through P-67 for direct detachable prefixes such
    as nitro, cyano, carboxy, carbamoyl, and halo-carbonyl prefixes.
    """

    for group in perceive_groups(mol):
        if start_idx in group.atoms_involved and group.atoms_involved.issubset(component):
            if group.key in RULES.prefixes.amide_like_groups:
                substituted = _direct_amide_subgraph_prefix(mol, group, component)
                if substituted:
                    return substituted
            prefix = RULES.functional_groups.direct_subgraph_prefix_for(group.key)
            if prefix:
                return prefix
    return ""


def _direct_amide_subgraph_prefix(mol: Molecule, group: PerceivedGroup, component: set[int]) -> str:
    """Return substituted carbamoyl/carbamothioyl for direct amide subgraphs."""

    amide_n = amide_nitrogen(mol, group)
    if amide_n is None:
        return ""
    n_substituents = [
        n
        for n in mol.get_neighbors(amide_n)
        if n in component and n not in group.atoms_involved and mol.atoms[n].symbol != "H"
    ]
    base = RULES.functional_groups.prefix_for(group.key) or ""
    if not base:
        return ""
    if not n_substituents:
        return base
    outside_component = set(mol.atoms.keys()) - component
    branch_names = []
    for n_sub in n_substituents:
        branch = name_subgraph(
            mol,
            n_sub,
            outside_component | set(group.atoms_involved) | {amide_n},
            upstream_atom=amide_n,
        )
        branch = strip_outer_parentheses(branch)
        if branch:
            branch_names.append(branch)
    if not branch_names:
        return base
    return f"({format_counted_prefixes(branch_names)}{base})"


def _find_acyclic_subgraph_paths(
    mol: Molecule, start_idx: int, component: set[int], cyclic_atoms: set[int], sub_exclude: set[int]
) -> list[list[int]]:
    """Find carbon paths available for recursive acyclic substituent naming.

    Blue Book references: P-44 and P-45 for parent hydride selection in
    substituent names.
    """

    valid_nodes = {n for n in component if n not in cyclic_atoms and mol.atoms[n].is_carbon and n not in sub_exclude}
    paths = []

    def dfs_sub(curr, path, visited_nodes):
        neighbors = [n for n in mol.get_neighbors(curr) if n in valid_nodes and n not in visited_nodes]
        if not neighbors:
            if start_idx in path:
                paths.append(path)
            return
        for n in neighbors:
            dfs_sub(n, path + [n], visited_nodes | {n})

    endpoints = [n for n in valid_nodes if sum(1 for x in mol.get_neighbors(n) if x in valid_nodes) <= 1]
    start_nodes = endpoints if endpoints else valid_nodes
    for start in start_nodes:
        dfs_sub(start, [start], {start})
    return paths


def _select_subgraph_parent(mol: Molecule, start_idx: int, component: set[int], sub_exclude: set[int]):
    """Select parent candidates for a recursive substituent component.

    Blue Book references: P-44, P-45, P-52, and P-53 for parent selection in
    chains, rings, fused systems, and retained parents.
    """

    cyclic_atoms = get_cyclic_atoms(mol, sub_exclude)
    if start_idx in cyclic_atoms:
        ring_systems = find_ring_systems(mol, sub_exclude)
        valid_rings = [rs for rs in ring_systems if start_idx in rs.atoms]
        if not valid_rings:
            return None
        selection = select_principal_parent(mol, [], valid_rings, [start_idx])
        if selection is None:
            return None
        return selection.with_fixed_start(selection.requires_fixed_substituent_start)

    paths = _find_acyclic_subgraph_paths(mol, start_idx, component, cyclic_atoms, sub_exclude)
    if not paths:
        return None
    selection = select_principal_parent(mol, paths, [], [start_idx])
    if selection is None:
        return None
    return selection.with_fixed_start(False)


def _spiro_subgraph_assembly(mol: Molecule, c_idx: int, sub_comp: set[int]) -> SpiroAssembly:
    """Name a side ring as structured spiro assembly data.

    Blue Book references: P-24 and P-52.3; the attachment atom is temporarily
    represented as silicon so the side ring can be named independently, then the
    silane marker is stripped back out before assembly rendering.
    """

    sub_mol = Molecule()
    for n in sub_comp:
        atom = mol.atoms[n]
        symbol = "Si" if n == c_idx else atom.symbol
        sub_mol.add_atom(
            symbol=symbol,
            idx=n,
            charge=atom.charge,
            stereo=atom.stereo,
            is_aromatic=atom.is_aromatic,
            explicit_h_count=atom.explicit_h_count,
            total_h_count=atom.total_h_count,
        )
    for n in sub_comp:
        for nxt in mol.get_neighbors(n):
            if nxt in sub_comp and n < nxt:
                bond = mol.get_bond(n, nxt)
                sub_mol.add_bond(u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring)

    sub_name_raw = name_component(sub_mol, sub_comp, is_substituent=False)
    match = re.search(r"(?:(^|-)(\d+)-)?sil[a]?", sub_name_raw)
    if not match:
        return SpiroAssembly(parent_locant="", side_locant="1", side_parent_name=sub_name_raw)

    loc = match.group(2) if match.group(2) else "1"
    if match.group(2):
        sub_name_clean = re.sub(rf"(^|-){loc}-sil[a]?-?", r"\1", sub_name_raw)
    else:
        sub_name_clean = re.sub(r"sil[a]?-?", "", sub_name_raw)

    sub_name_clean = sub_name_clean.replace("--", "-").strip("-")
    sub_name_clean = sub_name_clean.replace("-cyclo", "cyclo")
    if not sub_name_clean:
        sub_name_clean = "methane"
    side_prefixes, side_parent_name = extract_spiro_side_prefixes(sub_name_clean)
    return SpiroAssembly(
        parent_locant="",
        side_locant=loc,
        side_parent_name=side_parent_name,
        side_prefixes=tuple(side_prefixes),
    )


def _spiro_subgraph_name(mol: Molecule, c_idx: int, sub_comp: set[int]) -> str:
    """Compatibility marker for older assembly callers."""

    spiro = _spiro_subgraph_assembly(mol, c_idx, sub_comp)
    return f"[SPIRO]-{spiro.side_locant}-{spiro.side_parent_name}"


def _collect_subgraph_substituents(
    mol: Molecule,
    candidate_path: list[int],
    sub_perceived: list[PerceivedGroup],
    sub_exclude: set[int],
) -> dict[int, list[SubstituentItem]]:
    """Collect prefixes attached to a recursive subgraph parent.

    Blue Book references: P-14.2, P-16.5, P-44, P-61 through P-67, and P-24
    for multiplicative prefixes, complex prefixes, parent substituents, and
    spiro side-ring substituents.
    """

    main_set = set(candidate_path)
    subst_mapping: dict[int, list[SubstituentItem]] = {}
    sub_handled_atoms = set()

    for group in sub_perceived:
        if group.attachment_carbon in main_set and not group.is_principal_candidate:
            rule = RULES.functional_groups.by_key.get(group.key)
            name = rule.prefix if rule and rule.role == "prefix" else ""
            if name:
                subst_mapping.setdefault(group.attachment_carbon, []).append(
                    SubstituentItem(
                        name=name,
                        locants=[],
                        atom_ids=set(group.atoms_involved),
                        bond_ids=_bond_ids_within(mol, set(group.atoms_involved) | {group.attachment_carbon}),
                    )
                )
                sub_handled_atoms.update(group.atoms_involved)

    for c_idx in candidate_path:
        n_subs = [
            n_idx
            for n_idx in mol.get_neighbors(c_idx)
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude
        ]

        spiro_pair = _find_spiro_side_pair(mol, c_idx, n_subs, main_set, sub_exclude)
        if spiro_pair:
            sub_comp = _spiro_side_component(mol, c_idx, spiro_pair[0], main_set, sub_exclude)
            subst_mapping.setdefault(c_idx, []).append(
                SubstituentItem(
                    name="",
                    locants=[],
                    atom_ids=sub_comp - {c_idx},
                    bond_ids=_bond_ids_within(mol, sub_comp),
                    spiro=_spiro_subgraph_assembly(mol, c_idx, sub_comp),
                )
            )
            sub_handled_atoms.update(sub_comp - {c_idx})
            n_subs = [n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude:
                branch_name, branch_trace = name_subgraph(
                    mol, n_idx, sub_exclude | main_set, upstream_atom=c_idx, return_trace=True
                )
                if branch_name:
                    branch_atoms = _subgraph_component(mol, n_idx, sub_exclude | main_set)
                    branch_with_attachment = branch_atoms | {c_idx}
                    subst_mapping.setdefault(c_idx, []).append(
                        SubstituentItem(
                            name=branch_name,
                            locants=[],
                            atom_ids=branch_atoms,
                            bond_ids=_bond_ids_within(mol, branch_with_attachment),
                            trace_segments=branch_trace,
                        )
                    )

    return subst_mapping


def _add_subgraph_substituents(parts: AssemblyParts, subst_mapping: dict[int, list[SubstituentItem]], get_loc) -> None:
    """Add collected substituent prefixes to assembly parts.

    Blue Book references: P-14.2 and P-16.5 for locants, multiplicative
    prefixes, and complex substituent citation.
    """

    for c_idx, items in subst_mapping.items():
        locant = get_loc(c_idx)
        for item in items:
            _add_substituent_trace(
                parts,
                item.name,
                locant,
                item.atom_ids,
                item.bond_ids,
                item.trace_segments,
                spiro=item.spiro,
            )


def _finalize_subgraph_name(name: str, parts: AssemblyParts) -> str:
    """Apply recursive-substituent wrapping rules to an assembled name.

    Blue Book references: P-13.6 and P-16.5 for substituent suffix citation and
    parentheses around complex substituent prefixes.
    """

    if name == "phenyl" and not parts.substituents:
        return name
    if (
        (name.endswith("yl") or name.endswith("ylidene") or name.endswith("ylidyne"))
        and not parts.substituents
        and not parts.unsaturations
        and str(parts.attachment_locant) == "1"
        and not name.startswith("bicyclo")
        and not name.startswith("spiro")
        and not name.startswith("tricyclo")
    ):
        return name
    return f"({name})"


def _assemble_parent_name(
    mol: Molecule,
    parts: AssemblyParts,
    numbered_path: list[int],
    get_loc,
    *,
    finalize_subgraph: bool = False,
    apply_special_component_names: bool = False,
) -> str:
    """Assemble a parent name and apply shared post-assembly charge rules."""

    name = assemble_name_raw(parts)
    if finalize_subgraph:
        name = _finalize_subgraph_name(name, parts)
    name = apply_anionic_parent_names(name, mol, numbered_path, get_loc, parts.retained_name)
    name = apply_cationic_imino_names(name, mol)
    if apply_special_component_names:
        name = SPECIAL_COMPONENT_NAMES.get(name, name)
    return post_process_name(name)


def name_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int = None,
    return_trace: bool = False,
):
    """Name a recursive substituent subgraph attached to the current parent.

    Blue Book references: P-13.6, P-14.2, P-16.5, P-61, P-62, P-63, P-65,
    P-66, and P-67.  Extendable prefix vocabularies are loaded from
    ``data/namer_rules.json``.
    """

    start_atom = mol.atoms[start_idx]
    cyclic_atoms_global = get_cyclic_atoms(mol, exclude_atoms)

    if not start_atom.is_carbon and start_idx not in cyclic_atoms_global:
        name = name_heteroatom_subgraph(mol, start_idx, exclude_atoms, upstream_atom, name_subgraph)
        if name is not None:
            return (name, []) if return_trace else name

    component = _subgraph_component(mol, start_idx, exclude_atoms)
    direct_prefix = _direct_subgraph_prefix(mol, start_idx, component)
    if direct_prefix:
        return (direct_prefix, []) if return_trace else direct_prefix

    sub_exclude = set(mol.atoms.keys()) - component
    parent_selection = _select_subgraph_parent(mol, start_idx, component, sub_exclude)
    if parent_selection is None:
        return ("", []) if return_trace else ""

    retained_name_val, locant_maps = resolve_retained_parent(
        mol,
        parent_selection.primary_path,
        parent_selection.is_ring,
        parent_selection.is_bicycle,
        parent_selection.is_polycycle,
    )
    sub_perceived = perceive_groups(mol)
    subst_mapping = _collect_subgraph_substituents(mol, parent_selection.primary_path, sub_perceived, sub_exclude)
    parent_plan = build_parent_assembly_plan(
        mol,
        parent_selection,
        NamingIntent.subgraph(
            start_idx,
            upstream_atom,
            fixed_start=parent_selection.fixed_start_required,
        ),
        subst_mapping,
        locant_maps,
        retained_name_val,
    )
    numbered_path = parent_plan.numbered_path
    get_loc = parent_plan.get_loc
    parts = parent_plan.parts
    _emit_bond_stereo(mol, parts, numbered_path, get_loc, sub_exclude, upstream_atom)
    _add_indicated_hydrogens(mol, parts, numbered_path, get_loc)
    _add_subgraph_substituents(parts, subst_mapping, get_loc)
    _add_subgraph_parent_features(
        mol,
        parts,
        numbered_path,
        get_loc,
        parent_selection.is_bicycle,
        parent_selection.is_spiro,
        parent_selection.is_polycycle,
    )

    name = _assemble_parent_name(mol, parts, numbered_path, get_loc, finalize_subgraph=True)
    if return_trace:
        return name, _assembly_trace_segments(parts)
    return name


def name_component(
    mol: Molecule,
    component_atoms: set[int],
    is_substituent: bool = False,
    return_trace: bool = False,
    decision_trace: DecisionTrace | None = None,
):
    """Name one connected component or recursive component of a molecule."""

    return _name_component_impl(
        mol,
        component_atoms,
        is_substituent=is_substituent,
        return_trace=return_trace,
        decision_trace=decision_trace,
        name_subgraph=name_subgraph,
        name_spiro_subgraph=_spiro_subgraph_assembly,
        assemble_parent_name=_assemble_parent_name,
    )


def name_smiles_with_trace(smiles: str) -> tuple[str, list[dict]]:
    """Return a generated name and AssemblyParts-derived trace annotations.

    Blue Book references are carried by each trace segment. This API keeps the
    original ``name_smiles`` function unchanged while exposing atom and bond IDs
    selected during parent, prefix, unsaturation, and suffix assembly.
    """

    return DEFAULT_NAMING_ENGINE.name_smiles_with_trace(smiles)


def analyze_smiles(smiles: str) -> NameAnalysis:
    """Return a generated name with structure annotations and decision traces.

    The default ``name_smiles`` path stays minimal. This analysis API turns on
    explainability and records the major choices in the pipeline: parsing,
    component splitting, functional-group perception, priority, parent
    selection, numbering, and final assembly.
    """

    return DEFAULT_NAMING_ENGINE.analyze_smiles(smiles)

def name_smiles(smiles: str) -> str:
    """Return an IUPAC-style name for a SMILES string.

    Blue Book references: P-13 for name construction, P-44/P-45 for parent
    selection and numbering, and P-72 for ordering disconnected ionic
    components.  Component-order metal names are data-backed in
    ``data/namer_rules.json``.
    """

    return DEFAULT_NAMING_ENGINE.name_smiles(smiles)
