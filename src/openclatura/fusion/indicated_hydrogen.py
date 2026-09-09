"""Component-proved carbon hydrogen conventions and fusion relocation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache

from ..locants import SystemLocant, system_locant_sort_key
from ..molecule import Molecule
from ..retained_graph_model import RetainedGraphAtomTemplate, merge_parent_bond_classes
from .charges import fusion_charge_lone_pair_sites, protonated_pi_nitrogen_atoms
from .mancude import (
    _nitrogen_composition_parent_model,
    _single_site_parent_model,
    compare_actual_parent_to_implied_parent,
    indicated_hydrogen_parent_bond_model,
    saturated_nitrogen_hydrogen_sites,
)
from .model import (
    FusionComponentMatch,
    FusionComponentSpec,
    FusionGraph,
    FusionGraphAtom,
    FusionGraphBond,
    FusionNameAst,
    ParentBondModel,
)
from .numbering import parent_bond_model, parent_pi_capable_atom_ids


def _component_graph(spec: FusionComponentSpec, atoms: tuple[RetainedGraphAtomTemplate, ...]) -> FusionGraph:
    ids = {atom.locant: index for index, atom in enumerate(atoms)}
    return FusionGraph(
        atoms=tuple(
            FusionGraphAtom(
                ids[atom.locant],
                atom.symbol,
                atom.charge,
                pi_capacity=atom.resolved_pi_capacity,
                forced_single=atom.forced_single,
                saturated=atom.saturated,
            )
            for atom in atoms
        ),
        bonds=tuple(
            FusionGraphBond(tuple(ids[locant] for locant in bond.locants), bond.bond_class) for bond in spec.bonds
        ),
    )


@lru_cache(maxsize=512)
def component_parent_atoms(spec: FusionComponentSpec) -> tuple[RetainedGraphAtomTemplate, ...]:
    """Release a declared carbon H tautomer without changing component pi capacity."""

    atoms = spec.atoms
    template = spec.template
    if template.mancude_double_bonds is None:
        return atoms
    movable = {
        atom.locant
        for atom in atoms
        if atom.locant in template.default_indicated_h
        and atom.symbol == "C"
        and atom.charge == 0
        and atom.saturated
        and not atom.forced_single
        and atom.pi_capacity != 0
    }
    if not movable or any(bond.bond_class not in {"aromatic", "mancude", "fusion"} for bond in spec.bonds):
        return atoms
    relaxed = tuple(
        replace(atom, saturated=False, default_h=False) if atom.locant in movable else atom for atom in atoms
    )
    original_model = parent_bond_model(_component_graph(spec, atoms))
    relaxed_model = parent_bond_model(_component_graph(spec, relaxed))
    if original_model.maximum_non_cumulative_double_bonds != template.mancude_double_bonds:
        return atoms
    if relaxed_model.maximum_non_cumulative_double_bonds != template.mancude_double_bonds:
        return atoms
    movable_ids = {index for index, atom in enumerate(atoms) if atom.locant in movable}
    if not any(
        all(order == 1 for edge, order in assignment.orders if movable_ids.intersection(edge))
        for assignment in relaxed_model.allowed_kekule_assignments
    ):
        return atoms
    return relaxed


@lru_cache(maxsize=512)
def component_h_locants(spec: FusionComponentSpec, symbol: str) -> frozenset[str]:
    """Eligible intrinsic H roles from the isolated component's pi budget."""
    atoms = component_parent_atoms(spec)
    graph = _component_graph(spec, atoms)
    model = parent_bond_model(graph)
    declared_count = spec.template.mancude_double_bonds
    if declared_count is not None and model.maximum_non_cumulative_double_bonds != declared_count:
        return frozenset()
    candidates = set()
    for assignment in model.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        candidates.update(
            atoms[atom.id].locant
            for atom in graph.atoms
            if atom.symbol == symbol
            and atom.pi_capacity
            and not atom.forced_single
            and not atom.saturated
            and atom.id not in paired
        )
    return frozenset(candidates)


def _component_carbon_h_locants(spec: FusionComponentSpec) -> frozenset[str]:
    return component_h_locants(spec, "C")


@lru_cache(maxsize=512)
def _component_carbon_pi_locants(spec: FusionComponentSpec) -> frozenset[str]:
    """Candidate roles, not H sites: fusion can leave locally paired C unpaired."""

    atoms = component_parent_atoms(spec)
    model = parent_bond_model(_component_graph(spec, atoms))
    declared_count = spec.template.mancude_double_bonds
    if declared_count is not None and model.maximum_non_cumulative_double_bonds != declared_count:
        return frozenset()
    return frozenset(
        atom.locant
        for atom in atoms
        if atom.symbol == "C"
        and not atom.charge
        and atom.resolved_pi_capacity
        and not atom.saturated
        and not atom.forced_single
    )


def _has_component_pi_budget(spec: FusionComponentSpec) -> bool:
    """Accept declared counts or an unmodified carbocyclic matching domain.

    Generated mancude carbocycles need no stored double-bond count: their
    neutral carbon roles and eligible edges define it through maximum matching.
    Missing metadata must not promote fixed saturation into movable carbon H.
    """

    return spec.template.mancude_double_bonds is not None or (
        all(
            atom.symbol == "C"
            and atom.charge == 0
            and atom.resolved_pi_capacity == 1
            and not atom.saturated
            and not atom.forced_single
            for atom in spec.atoms
        )
        and all(bond.bond_class in {"aromatic", "mancude", "fusion"} for bond in spec.bonds)
    )


def intrinsic_carbon_fusion_scope(ast: FusionNameAst, specs: Mapping[int, FusionComponentSpec]) -> bool:
    """Require a connected composition of components with proved pi budgets.

    Join order and component ring count do not establish hydrogen capacity.
    Component and completed-parent maximum matchings establish it instead.
    """

    occurrences = {match.occurrence_id for match in ast.component_occurrences}
    if len(occurrences) < 2 or occurrences != set(specs):
        return False
    if not all(_has_component_pi_budget(spec) for spec in specs.values()):
        return False
    neighbors = {occurrence: set() for occurrence in occurrences}
    for join in ast.joins:
        if join.host_occurrence not in neighbors or join.attached_occurrence not in neighbors:
            return False
        neighbors[join.host_occurrence].add(join.attached_occurrence)
        neighbors[join.attached_occurrence].add(join.host_occurrence)
    visited = set()
    pending = [next(iter(occurrences))]
    while pending:
        occurrence = pending.pop()
        if occurrence not in visited:
            visited.add(occurrence)
            pending.extend(neighbors[occurrence] - visited)
    return visited == occurrences


def component_carbon_h_relocation_scope(ast: FusionNameAst, specs: Mapping[int, FusionComponentSpec]) -> bool:
    """Carbon-only composition preserves the proved component pi capacities.

    Heteroatom donor composition needs its separate intrinsic-H proof and
    explicit component pi budgets before additive operations.
    """

    eligible = intrinsic_carbon_fusion_scope(ast, specs) or all(
        atom.symbol == "C" and atom.charge == 0 for spec in specs.values() for atom in spec.atoms
    )
    if not eligible:
        return False
    if all(component_parent_atoms(spec) == spec.atoms for spec in specs.values()):
        return True
    # Preserve the completed pi budget unless every local maximum matching
    # proves that fusion consumes the released H at a shared junction.
    components = tuple((match, specs[match.occurrence_id]) for match in ast.component_occurrences)
    if _completed_carbon_h_budget(components, False) == _completed_carbon_h_budget(components, True):
        return True
    from .component_hydrogen import component_hydrogen_consumption

    return component_hydrogen_consumption(ast, specs) is not None


@lru_cache(maxsize=512)
def _completed_carbon_h_budget(
    components: tuple[tuple[FusionComponentMatch, FusionComponentSpec], ...], relaxed: bool
) -> int:
    return _completed_carbon_h_model(components, relaxed).maximum_non_cumulative_double_bonds


@lru_cache(maxsize=512)
def _completed_carbon_h_model(
    components: tuple[tuple[FusionComponentMatch, FusionComponentSpec], ...], relaxed: bool
) -> ParentBondModel:
    return parent_bond_model(_component_parent_graph(components, relaxed))


def component_parent_graph(
    ast: FusionNameAst, specs: Mapping[int, FusionComponentSpec], *, relocate_carbon_h: bool
) -> FusionGraph:
    """Project component roles once, shared by planning and the relocation proof."""

    components = tuple((match, specs[match.occurrence_id]) for match in ast.component_occurrences)
    return _component_parent_graph(components, relocate_carbon_h)


@lru_cache(maxsize=512)
def _component_parent_graph(
    components: tuple[tuple[FusionComponentMatch, FusionComponentSpec], ...], relaxed: bool
) -> FusionGraph:
    labels = {}
    edges = {}
    for match, spec in components:
        local_map = match.input_atom_by_locant
        for atom in component_parent_atoms(spec) if relaxed else spec.atoms:
            atom_id = local_map[atom.locant]
            site = FusionGraphAtom(
                atom_id,
                atom.symbol,
                atom.charge,
                pi_capacity=atom.resolved_pi_capacity,
                forced_single=atom.forced_single,
                indicated_h_site=atom.indicated_h_site or atom.default_h,
                saturated=atom.saturated,
            )
            previous = labels.get(atom_id, site)
            if previous.symbol != site.symbol or previous.formal_charge != site.formal_charge:
                raise ValueError("component atom identities disagree at a shared fusion site")
            labels[atom_id] = replace(
                site,
                pi_capacity=min(previous.pi_capacity, site.pi_capacity),
                forced_single=previous.forced_single or site.forced_single,
                indicated_h_site=previous.indicated_h_site or site.indicated_h_site,
                saturated=previous.saturated or site.saturated,
            )
        for bond in spec.bonds:
            edge = tuple(sorted(local_map[locant] for locant in bond.locants))
            merged = merge_parent_bond_classes(edges.get(edge, bond.bond_class), bond.bond_class)
            if merged is None:
                raise ValueError("component bond classes disagree on a shared fusion edge")
            edges[edge] = merged
    return FusionGraph(
        atoms=tuple(labels[atom] for atom in sorted(labels)),
        bonds=tuple(FusionGraphBond(edge, edges[edge]) for edge in sorted(edges)),
    )


def intrinsic_carbon_candidate_atoms(
    ast: FusionNameAst,
    specs: Mapping[int, FusionComponentSpec],
    mol: Molecule,
) -> frozenset[int]:
    """Supply eligible carbon roles for the completed-system matching proof."""

    parent_atoms = {atom for match in ast.component_occurrences for atom in match.input_atom_by_locant.values()}
    blocked = set()
    movable_defaults = set()
    for match in ast.component_occurrences:
        spec = specs[match.occurrence_id]
        relaxed = {atom.locant: atom for atom in component_parent_atoms(spec)}
        for atom in spec.atoms:
            atom_id = match.input_atom_by_locant[atom.locant]
            role = relaxed[atom.locant]
            if not role.resolved_pi_capacity or role.saturated or role.forced_single:
                blocked.add(atom_id)
            elif atom.saturated and not role.saturated:
                movable_defaults.add(atom_id)
    pi_junctions = frozenset(
        atom for atom in movable_defaults - blocked if _is_pi_bearing_carbon_junction(mol, atom, parent_atoms)
    )
    if not intrinsic_carbon_fusion_scope(ast, specs) or not component_carbon_h_relocation_scope(ast, specs):
        return pi_junctions
    if not any(is_intrinsic_carbon_h_site(mol, atom, parent_atoms) for atom in parent_atoms):
        return pi_junctions
    candidates = set()
    local_candidates = set()
    for match in ast.component_occurrences:
        spec = specs[match.occurrence_id]
        candidates.update(match.input_atom_by_locant[locant] for locant in _component_carbon_pi_locants(spec))
        local_candidates.update(match.input_atom_by_locant[locant] for locant in _component_carbon_h_locants(spec))
    components = tuple((match, specs[match.occurrence_id]) for match in ast.component_occurrences)
    model = _completed_carbon_h_model(components, True)
    unpaired_candidates = set()
    for assignment in model.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        unpaired_candidates.update(candidates - blocked - paired)
    return frozenset((local_candidates | unpaired_candidates) - blocked) | pi_junctions


def is_intrinsic_carbon_h_site(mol: Molecule, atom: int, parent_atoms: set[int] | frozenset[int]) -> bool:
    """Recognize a parent CH2 site, including H replaced by branches or bridges."""

    site = mol.atoms[atom]
    if site.symbol != "C" or site.charge or site.is_aromatic:
        return False
    neighbors = mol.get_neighbors(atom)
    external_count = sum(neighbor not in parent_atoms for neighbor in neighbors)
    return (
        len(neighbors) - external_count == 2
        and site.total_h_count + external_count == 2
        and all(mol.get_bond(atom, neighbor).order == 1 for neighbor in neighbors)
    )


def intrinsic_carbon_parent_model(
    mol: Molecule,
    graph: FusionGraph,
    model: ParentBondModel,
    locants: Mapping[int, SystemLocant],
    candidates: frozenset[int],
    *,
    intrinsic_hydrogen_atom_ids: frozenset[int] = frozenset(),
    cited_nitrogen_hydrogen_atom_ids: frozenset[int] = frozenset(),
) -> tuple[ParentBondModel, frozenset[int]]:
    """Choose jointly proved intrinsic CH2 sites without consuming hydro bonds.

    Intrinsic H restricts the maximum-bond assignments themselves. It is not
    an operation deleting a double bond from an otherwise chosen assignment.
    """

    pi_junctions = pi_bearing_fusion_carbon_sites(mol, graph, candidates)
    if pi_junctions:
        # Fusion consumes the component's movable CH2 convention at an
        # observed pi-bearing junction, not through an external substituent.
        graph = replace(
            graph,
            atoms=tuple(
                replace(atom, pi_capacity=1, saturated=False, indicated_h_site=False)
                if atom.id in pi_junctions
                else atom
                for atom in graph.atoms
            ),
        )
        model = indicated_hydrogen_parent_bond_model(graph, intrinsic_hydrogen_atom_ids)
    hydrogen_atoms = intrinsic_hydrogen_atom_ids | cited_nitrogen_hydrogen_atom_ids
    owned_h = hydrogen_atoms | protonated_pi_nitrogen_atoms(mol, graph)
    if not candidates or any(
        mol.atoms[atom].symbol != "C" and mol.atoms[atom].total_h_count and atom not in owned_h for atom in locants
    ):
        return model, frozenset()
    # Resolve fusion-N valence before deciding whether a carbon is intrinsically
    # unpaired. Reassigning N afterwards can turn an additive pair into false H.
    parent_atoms = frozenset(locants)
    if saturated_nitrogen_hydrogen_sites(mol, parent_atoms, hydrogen_atoms):
        # Complete saturated C/N composition owns its hydrogenation already;
        # constraining its donors must not add a competing carbon tautomer.
        return model, frozenset()
    original_model = model
    model = _nitrogen_composition_parent_model(mol, parent_atoms, model, hydrogen_atoms)
    # Cited N informs the carbon witness; without a witness its composition
    # remains owned by the derivative step, not a changed parent bond domain.
    no_carbon_model = original_model if cited_nitrogen_hydrogen_atom_ids - intrinsic_hydrogen_atom_ids else model
    eligible = {atom for atom in candidates if is_intrinsic_carbon_h_site(mol, atom, parent_atoms)}
    oxo_sites = {
        atom
        for atom in candidates
        if mol.atoms[atom].symbol == "C"
        and not mol.atoms[atom].charge
        and len(parent_atoms.intersection(mol.get_neighbors(atom))) == 2
        and all(
            mol.get_bond(atom, other).order == 1 or (mol.atoms[atom].is_aromatic and mol.atoms[other].is_aromatic)
            for other in mol.get_neighbors(atom)
            if other in parent_atoms
        )
        and any(
            other not in parent_atoms and mol.atoms[other].symbol == "O" and mol.get_bond(atom, other).order == 2
            for other in mol.get_neighbors(atom)
        )
    }
    eligible.update(oxo_sites)
    capable_sites = parent_pi_capable_atom_ids(graph)
    carbon_atoms = {atom.id for atom in graph.atoms if atom.symbol == "C" and atom.id in capable_sites}
    occupied_pi_atoms = carbon_atoms | {
        atom
        for edge in model.pi_eligible_edges | model.required_double_bonds
        if mol.get_bond(*edge).order > 1
        for atom in edge
    }
    groups = set()
    for assignment in model.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        unpaired = carbon_atoms - paired
        # A donor can account for the unpaired parent valence only if the
        # assignment also pairs heteroatoms carrying an observed pi bond.
        if intrinsic_hydrogen_atom_ids and occupied_pi_atoms <= paired:
            return no_carbon_model, frozenset()
        if unpaired and unpaired <= eligible:
            groups.add(frozenset(unpaired))
    choices = []
    for sites in groups:
        constrained = _single_site_parent_model(model, sites)
        if constrained.maximum_non_cumulative_double_bonds != model.maximum_non_cumulative_double_bonds:
            continue
        delta = compare_actual_parent_to_implied_parent(
            mol,
            parent_atoms,
            constrained,
            atom_to_locant=locants,
            indicated_hydrogen_atom_ids=hydrogen_atoms,
            externally_unsaturated_atom_ids=oxo_sites,
        )
        if delta is None or not delta.compatible or delta.additional_multiple_bond_ids:
            continue
        # Nitrogen recomposition must not pay for a carbon tautomer by deleting
        # another parent pi bond or hydrogenating an aromatic boundary atom.
        if sum(order == 2 for _, order in delta.assignment.orders) != model.maximum_non_cumulative_double_bonds:
            continue
        paired = {atom for edge, order in delta.assignment.orders if order == 2 for atom in edge}
        if carbon_atoms - paired != sites or not occupied_pi_atoms - sites <= paired:
            continue
        if any(mol.atoms[atom].is_aromatic for edge in delta.hydrogenated_edges for atom in edge):
            continue
        if sites & oxo_sites and not delta.intrinsic_hydro_operations:
            continue
        rank = (len(delta.hydrogenated_edges), tuple(sorted(system_locant_sort_key(locants[atom]) for atom in sites)))
        choices.append((rank, constrained, frozenset() if delta.intrinsic_hydro_operations else sites))
    if not choices:
        return no_carbon_model, frozenset()
    _, constrained, sites = min(choices, key=lambda choice: choice[0])
    return constrained, sites


def _is_pi_bearing_carbon_junction(mol: Molecule, atom: int, parent_atoms: set[int] | frozenset[int]) -> bool:
    value = mol.atoms[atom]
    neighbors = parent_atoms.intersection(mol.get_neighbors(atom))
    return (
        value.symbol == "C"
        and not value.charge
        and len(neighbors) == 3
        and (value.is_aromatic or any(mol.get_bond(atom, other).order == 2 for other in neighbors))
    )


def pi_bearing_fusion_carbon_sites(
    mol: Molecule, graph: FusionGraph, movable_candidates: frozenset[int]
) -> frozenset[int]:
    """Bind component-proved movability to an observed parent-pi C junction."""

    atoms = frozenset(atom.id for atom in graph.atoms)
    return frozenset(
        atom.id
        for atom in graph.atoms
        if atom.id in movable_candidates
        and atom.symbol == "C"
        and not atom.formal_charge
        and atom.saturated
        and not atom.forced_single
        and _is_pi_bearing_carbon_junction(mol, atom.id, atoms)
    )


def intrinsic_parent_lone_pair_sites(mol: Molecule, graph: FusionGraph) -> frozenset[int]:
    """Shared planner/auditor intrinsic donors, including charged derivatives."""

    return aromatic_nitrogen_lone_pair_sites(mol, graph) | fusion_charge_lone_pair_sites(mol, graph)


def aromatic_nitrogen_hydrogen_atoms(mol: Molecule, graph: FusionGraph) -> frozenset[int]:
    """Hydrogen-bearing subset of the neutral aromatic donor roles."""
    return frozenset(
        atom for atom in aromatic_nitrogen_lone_pair_sites(mol, graph) if mol.atoms[atom].total_h_count == 1
    )


def aromatic_nitrogen_lone_pair_sites(mol: Molecule, graph: FusionGraph) -> frozenset[int]:
    """Prove neutral aromatic donors from local sigma valence.

    Replacing donor H with a ligand does not create another parent pi bond.
    Terminal carbonyls compose with non-junction donors through the oxo
    derivative proof. Junction donors and other exocyclic classes retain
    separate proofs.
    """

    atoms = {atom.id for atom in graph.atoms}
    external_multiple = tuple(
        (atom, neighbor)
        for atom in atoms
        for neighbor in mol.get_neighbors(atom)
        if neighbor not in atoms and mol.get_bond(atom, neighbor).order > 1
    )
    if external_multiple and any(mol.atoms[atom].charge for atom in atoms):
        return frozenset()
    if any(
        mol.atoms[atom].symbol != "C"
        or mol.atoms[neighbor].symbol != "O"
        or mol.atoms[atom].charge
        or mol.atoms[neighbor].charge
        or mol.get_bond(atom, neighbor).order != 2
        or len(mol.get_neighbors(neighbor)) != 1
        for atom, neighbor in external_multiple
    ):
        return frozenset()
    return frozenset(
        atom.id
        for atom in graph.atoms
        if atom.symbol == "N"
        and mol.atoms[atom.id].is_aromatic
        and mol.atoms[atom.id].charge == 0
        and len(neighbors := mol.get_neighbors(atom.id)) + mol.atoms[atom.id].total_h_count == 3
        and len(atoms.intersection(neighbors)) in {2, 3}
        and (not external_multiple or len(atoms.intersection(neighbors)) == 2)
        and all(mol.get_bond(atom.id, neighbor).order == 1 for neighbor in neighbors)
    )
