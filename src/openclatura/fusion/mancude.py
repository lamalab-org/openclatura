"""Compare observed parent bonding with a proved mancude parent model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache

from ..locants import SystemLocant, system_locant_sort_key
from ..molecule import Molecule
from ..name_operations import HydroOperation, OxoOperation, UnsaturationOperation
from ..polycycle_topology import normalize_edge
from .model import BondAssignment, FusionGraph, FusionGraphAtom, FusionGraphBond, ParentBondModel


@dataclass(frozen=True, slots=True)
class ParentBondDelta:
    """The smallest graph delta from one allowed parent-bond assignment."""

    assignment: BondAssignment
    implied_multiple_bond_ids: frozenset[int]
    hydrogenated_edges: tuple[tuple[int, int], ...]
    additional_multiple_bond_ids: frozenset[int]
    compatible: bool
    added_hydrogen_operations: tuple[HydroOperation, ...] = ()
    intrinsic_hydro_operations: tuple[HydroOperation, ...] = ()

    @property
    def hydrogenated_atom_ids(self) -> frozenset[int]:
        return frozenset(atom for edge in self.hydrogenated_edges for atom in edge)


@dataclass(frozen=True, slots=True)
class ParentDerivativeState:
    """Typed graph operations separating a parent skeleton from its derivative."""

    bond_delta: ParentBondDelta
    hydro_operations: tuple[HydroOperation, ...] = ()
    unsaturation_operations: tuple[UnsaturationOperation, ...] = ()
    oxo_operations: tuple[OxoOperation, ...] = ()

    @property
    def added_hydrogen_operations(self) -> tuple[HydroOperation, ...]:
        return self.bond_delta.added_hydrogen_operations

    @property
    def intrinsic_hydro_operations(self) -> tuple[HydroOperation, ...]:
        return self.bond_delta.intrinsic_hydro_operations


def indicated_hydrogen_parent_bond_model(
    graph: FusionGraph,
    indicated_hydrogen_atom_ids: set[int] | frozenset[int],
) -> ParentBondModel:
    """Constrain proved intrinsic-H sites before maximizing parent pi bonds.

    Callers own the proof that these sites cannot carry a parent double bond;
    additive hydrogenation and charged-N hydrogen are not intrinsic-H sites.
    Keep the original component atom roles and exact bond constraints intact.
    Search-budget exhaustion propagates to the planner's typed abstention.
    """

    from .numbering import parent_bond_model

    sites = frozenset(indicated_hydrogen_atom_ids)
    if not sites <= {atom.id for atom in graph.atoms}:
        raise ValueError("indicated-hydrogen site is outside the parent graph")
    if any(bond.bond_class == "double" and sites.intersection(bond.atoms) for bond in graph.bonds):
        raise ValueError("indicated-hydrogen site conflicts with a required parent double bond")
    constrained = replace(
        graph,
        atoms=tuple(replace(atom, forced_single=True) if atom.id in sites else atom for atom in graph.atoms),
    )
    return parent_bond_model(constrained)


@lru_cache(maxsize=256)
def _single_site_parent_model(model: ParentBondModel, sites: frozenset[int]) -> ParentBondModel:
    """Reassign pi bonds after a graph-proved single-site constraint.

    The input model already resolves component atom roles into eligible and
    fixed edges. Reconstruct only that bond-domain graph, never relaxing a
    component constraint. Cache immutable domains for repeated derivative audits.
    """

    edges = model.required_single_bonds | model.required_double_bonds | model.pi_eligible_edges
    graph = FusionGraph(
        atoms=tuple(FusionGraphAtom(atom, "C") for atom in sorted({atom for edge in edges for atom in edge})),
        bonds=tuple(
            FusionGraphBond(
                edge,
                "double"
                if edge in model.required_double_bonds
                else "single"
                if edge in model.required_single_bonds
                else "mancude",
            )
            for edge in sorted(edges)
        ),
    )
    return indicated_hydrogen_parent_bond_model(graph, sites)


def _nitrogen_composition_parent_model(
    mol: Molecule,
    atoms: frozenset[int],
    model: ParentBondModel,
    indicated_hydrogen_atom_ids: set[int] | frozenset[int],
) -> ParentBondModel:
    """Resolve neutral fusion-N valence and cited N-H before the bond delta."""

    sites = frozenset(
        atom
        for atom in atoms
        if mol.atoms[atom].symbol == "N"
        and mol.atoms[atom].charge == 0
        and (
            len(atoms.intersection(mol.get_neighbors(atom))) == 3
            or (
                atom in indicated_hydrogen_atom_ids
                and not mol.atoms[atom].is_aromatic
                and _five_membered_single_nitrogen(mol, atoms, atom)
            )
        )
    )
    if not sites or not any(sites.intersection(edge) for edge in model.pi_eligible_edges | model.required_double_bonds):
        return model
    return _single_site_parent_model(model, sites)


def _five_membered_single_nitrogen(mol: Molecule, atoms: frozenset[int], atom: int) -> bool:
    """Prove a two-connected single-bonded N in a five-membered parent ring."""

    neighbors = atoms.intersection(mol.get_neighbors(atom))
    if len(neighbors) != 2 or any(mol.get_bond(atom, other).order != 1 for other in neighbors):
        return False
    first, last = sorted(neighbors)
    return any(
        mol.get_bond(second, third) is not None
        for second in atoms.intersection(mol.get_neighbors(first)) - {atom, last}
        for third in atoms.intersection(mol.get_neighbors(last)) - {atom, first, second}
    )


def compare_actual_parent_to_implied_parent(
    mol: Molecule,
    atom_ids: set[int] | frozenset[int],
    bond_model: ParentBondModel,
    *,
    externally_unsaturated_atom_ids: set[int] | frozenset[int] = frozenset(),
    indicated_hydrogen_atom_ids: set[int] | frozenset[int] = frozenset(),
    atom_to_locant: Mapping[int, str | SystemLocant] | None = None,
    preserve_retained_parent_state: bool = False,
) -> ParentBondDelta | None:
    """Select the allowed Kekulé form requiring the smallest observed delta.

    Aromatic input bonds are representation-independent.  Explicit single
    bonds where the parent permits a double bond are hydrogenation sites;
    explicit multiple bonds where the selected parent has a single bond are
    additional unsaturation. A missing internal double bond incident to an
    externally unsaturated parent atom or an indicated-hydrogen atom is
    consumed by that typed operation rather than misclassified as additive
    hydrogenation. The function only compares graph data and does not infer
    nomenclature from rendered text.

    When completed-system locants are supplied, they break equivalent-form
    ties independently of the input atom numbering. Retained-parent wrappers
    preserve their template's hydrogen convention; ordinary fusion resolves
    neutral nitrogen composition sites before choosing the assignment.
    """

    atoms = frozenset(atom_ids)
    observed = {
        normalize_edge(bond.u, bond.v): bond for bond in mol.bonds.values() if bond.u in atoms and bond.v in atoms
    }
    known_edges = bond_model.required_single_bonds | bond_model.required_double_bonds | bond_model.pi_eligible_edges
    if set(observed) != set(known_edges):
        return None
    if not preserve_retained_parent_state:
        try:
            bond_model = _nitrogen_composition_parent_model(mol, atoms, bond_model, indicated_hydrogen_atom_ids)
        except ValueError:
            # An intrinsic single site cannot override a component's fixed double.
            return None

    candidates: list[tuple[tuple, ParentBondDelta]] = []
    locant_keys = (
        {atom: system_locant_sort_key(str(atom_to_locant[atom])) for atom in atoms}
        if atom_to_locant is not None
        else None
    )
    numbered_edges = (
        sorted(observed, key=lambda edge: tuple(sorted(locant_keys[atom] for atom in edge)))
        if locant_keys is not None
        else None
    )
    for assignment in bond_model.allowed_kekule_assignments:
        expected = {normalize_edge(*edge): order for edge, order in assignment.orders}
        if set(expected) != set(observed):
            continue
        implied_ids: set[int] = set()
        hydrogenated: list[tuple[int, int]] = []
        additional_ids: set[int] = set()
        incompatible = 0
        for edge, bond in observed.items():
            expected_order = expected[edge]
            aromatic = (
                edge in bond_model.pi_eligible_edges
                and mol.atoms[edge[0]].is_aromatic
                and mol.atoms[edge[1]].is_aromatic
            )
            if aromatic:
                # Aromatic inputs are Kekule-representation independent. Any
                # observed multiple bond on a pi-eligible parent edge belongs
                # to the parent, even when another equivalent assignment was
                # selected to calculate the hydrogenation delta.
                if bond.order > 1:
                    implied_ids.add(bond.idx)
                continue
            if bond.order == expected_order:
                if bond.order > 1:
                    implied_ids.add(bond.idx)
            elif bond.order == 1 and expected_order == 2:
                hydrogenated.append(edge)
            elif bond.order > expected_order:
                additional_ids.add(bond.idx)
            else:
                incompatible += 1
        after_indicated_h = tuple(edge for edge in hydrogenated if not set(edge) & indicated_hydrogen_atom_ids)
        residual_hydrogenated = tuple(
            sorted(edge for edge in after_indicated_h if not set(edge) & externally_unsaturated_atom_ids)
        )
        delta = ParentBondDelta(
            assignment=assignment,
            implied_multiple_bond_ids=frozenset(implied_ids),
            hydrogenated_edges=residual_hydrogenated,
            additional_multiple_bond_ids=frozenset(additional_ids),
            compatible=incompatible == 0,
        )
        rank = (
            incompatible,
            len(additional_ids),
            # Select the closest parent Kekule form before allowing an
            # exocyclic operation to consume an incident pi bond. Otherwise
            # the assignment can move a missing bond onto an oxo atom merely
            # to suppress a required hydro citation elsewhere in the parent.
            len(hydrogenated),
            len(after_indicated_h),
            len(after_indicated_h) - len(residual_hydrogenated),
            # Equivalent parent forms must not change their hydro citations
            # when input atoms are reordered. Use the proved system numbering.
            tuple(expected[edge] for edge in numbered_edges)
            if numbered_edges is not None
            else tuple(assignment.orders),
        )
        candidates.append((rank, delta))
    if not candidates:
        return None
    delta = min(candidates, key=lambda item: item[0])[1]
    if preserve_retained_parent_state or atom_to_locant is None or not delta.compatible:
        return delta
    if delta.additional_multiple_bond_ids and externally_unsaturated_atom_ids and not indicated_hydrogen_atom_ids:
        composed = _spiro_oxo_parent_delta(mol, atoms, bond_model, externally_unsaturated_atom_ids, atom_to_locant)
        if composed is not None:
            return composed
    intrinsic = _fixed_carbon_hydro_operations(
        mol, atoms, bond_model, delta, externally_unsaturated_atom_ids, indicated_hydrogen_atom_ids, atom_to_locant
    )
    return replace(delta, intrinsic_hydro_operations=intrinsic) if intrinsic else delta


def _spiro_carbon_sites(mol: Molecule, atoms: frozenset[int]) -> frozenset[int]:
    """Find external ring components attached solely through one saturated C."""

    sites = set()
    for atom in atoms:
        neighbors = set(mol.get_neighbors(atom))
        outside = neighbors - atoms
        if (
            mol.atoms[atom].symbol != "C"
            or mol.atoms[atom].charge
            or len(neighbors & atoms) != 2
            or len(outside) != 2
            or any(mol.get_bond(atom, other).order != 1 for other in neighbors)
        ):
            continue
        reached = set()
        boundary = set()
        pending = [min(outside)]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            adjacent = set(mol.get_neighbors(current))
            boundary.update(adjacent & atoms)
            pending.extend(adjacent - atoms - reached)
        if outside <= reached and boundary == {atom}:
            sites.add(atom)
    return frozenset(sites)


def _spiro_oxo_parent_delta(
    mol: Molecule,
    atoms: frozenset[int],
    model: ParentBondModel,
    oxo_atoms: set[int] | frozenset[int],
    locants: Mapping[int, str | SystemLocant],
) -> ParentBondDelta | None:
    """Compose spiro cyclic ketones with graph-derived residual carbon H sites."""

    spiro = _spiro_carbon_sites(mol, atoms)
    if not spiro or any(mol.atoms[atom].symbol != "C" or mol.atoms[atom].charge for atom in oxo_atoms):
        return None
    forced = frozenset(oxo_atoms) | spiro
    try:
        constrained = _single_site_parent_model(model, forced)
    except ValueError:
        return None
    always_paired = set.intersection(
        *(
            {atom for edge, order in assignment.orders if order == 2 for atom in edge}
            for assignment in model.allowed_kekule_assignments
        )
    )
    candidates = []
    for assignment in constrained.allowed_kekule_assignments:
        paired = {atom for edge, order in assignment.orders if order == 2 for atom in edge}
        added = always_paired - paired - forced
        if any(
            mol.atoms[atom].symbol != "C"
            or mol.atoms[atom].charge
            or mol.atoms[atom].is_aromatic
            or len(atoms.intersection(mol.get_neighbors(atom))) not in {2, 3}
            for atom in added
        ):
            continue
        if any(mol.get_bond(atom, neighbor).order != 1 for atom in added for neighbor in mol.get_neighbors(atom)):
            continue
        delta = compare_actual_parent_to_implied_parent(
            mol,
            atoms,
            replace(constrained, allowed_kekule_assignments=(assignment,)),
            atom_to_locant=locants,
            preserve_retained_parent_state=True,
        )
        if delta is None or not delta.compatible or delta.additional_multiple_bond_ids:
            continue
        ordered = tuple(sorted(added, key=lambda atom: system_locant_sort_key(str(locants[atom]))))
        operations = (
            (
                HydroOperation(
                    key="added_hydrogen",
                    reason="Oxo and spiro valence constraints leave carbon added-H sites in the composed parent.",
                    locants=tuple(str(locants[atom]) for atom in ordered),
                    atom_ids=ordered,
                    bond_ids=tuple(
                        sorted(
                            {
                                mol.get_bond(atom, other).idx
                                for atom in ordered
                                for other in mol.get_neighbors(atom)
                                if other in atoms
                            }
                        )
                    ),
                    operation_kind="indicated_hydrogen",
                ),
            )
            if ordered
            else ()
        )
        rank = (
            len(delta.hydrogenated_edges),
            tuple(sorted(system_locant_sort_key(str(locants[site])) for site in delta.hydrogenated_atom_ids)),
            tuple(system_locant_sort_key(str(locants[atom])) for atom in ordered),
        )
        candidates.append((rank, replace(delta, added_hydrogen_operations=operations)))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _fixed_carbon_hydro_operations(
    mol: Molecule,
    atoms: frozenset[int],
    model: ParentBondModel,
    delta: ParentBondDelta,
    oxo_atoms: set[int] | frozenset[int],
    indicated_atoms: set[int] | frozenset[int],
    locants: Mapping[int, str | SystemLocant],
) -> tuple[HydroOperation, ...]:
    """Cite the proved pair-to-hydro conversion of fixed intrinsic CH2 sites.

    Two is a supported composition scope, not a chemical parity requirement.
    Even cardinality alone does not prove how several components' intrinsic
    H conventions combine. Higher counts still need completed-parent citation
    and round-trip evidence; a configurable search cap is not a chemical bound.
    """

    if not oxo_atoms or indicated_atoms or delta.hydrogenated_edges or delta.additional_multiple_bond_ids:
        return ()
    sites = {
        atom
        for atom in atoms
        if mol.atoms[atom].symbol == "C"
        and not mol.atoms[atom].charge
        and len(atoms.intersection(mol.get_neighbors(atom))) == 2
        and all(
            normalize_edge(atom, other) in model.required_single_bonds
            for other in mol.get_neighbors(atom)
            if other in atoms
        )
    }
    if len(sites) != 2 or not sites.intersection(oxo_atoms):
        return ()
    ordered = tuple(sorted(sites, key=lambda atom: system_locant_sort_key(str(locants[atom]))))
    return (
        HydroOperation(
            key="intrinsic_parent_hydrogen",
            reason="Two fixed carbon parent-H sites are cited together in the oxo derivative.",
            locants=tuple(str(locants[atom]) for atom in ordered),
            atom_ids=ordered,
            operation_kind="additive_hydrogen",
        ),
    )


def parent_derivative_state(
    mol: Molecule,
    atom_ids: set[int] | frozenset[int],
    bond_model: ParentBondModel,
    atom_to_locant: Mapping[int, str | SystemLocant],
    *,
    indicated_hydrogen_atom_ids: set[int] | frozenset[int] = frozenset(),
    preserve_retained_parent_state: bool = False,
) -> ParentDerivativeState | None:
    """Describe every supported bond-state difference from a parent hydride.

    Parent topology is deliberately not inferred here. The supplied bond model
    already owns that proof; this function only records graph-bound operations
    needed to obtain the observed derivative from one allowed parent state.
    """

    atoms = frozenset(atom_ids)
    locants = {atom: str(locant) for atom, locant in atom_to_locant.items()}
    if set(locants) != set(atoms):
        return None

    oxo = []
    for parent_atom in sorted(atoms, key=lambda atom: system_locant_sort_key(locants[atom])):
        for neighbor in mol.get_neighbors(parent_atom):
            if neighbor in atoms or mol.atoms[neighbor].symbol != "O":
                continue
            bond = mol.get_bond(parent_atom, neighbor)
            if bond is None or bond.order != 2:
                continue
            oxo.append(
                OxoOperation(
                    key="oxo",
                    reason="Exocyclic doubly bonded oxygen modifies the fused parent.",
                    locant=locants[parent_atom],
                    parent_atom_id=parent_atom,
                    oxygen_atom_id=neighbor,
                    bond_id=bond.idx,
                )
            )

    delta = compare_actual_parent_to_implied_parent(
        mol,
        atoms,
        bond_model,
        externally_unsaturated_atom_ids={operation.parent_atom_id for operation in oxo},
        indicated_hydrogen_atom_ids=indicated_hydrogen_atom_ids,
        atom_to_locant=atom_to_locant,
        preserve_retained_parent_state=preserve_retained_parent_state,
    )
    if delta is None or not delta.compatible:
        return None

    hydrogenated_atoms = sorted(
        delta.hydrogenated_atom_ids,
        key=lambda atom: system_locant_sort_key(locants[atom]),
    )
    hydrogenated_bonds = tuple(
        sorted(mol.get_bond(*edge).idx for edge in delta.hydrogenated_edges if mol.get_bond(*edge) is not None)
    )
    hydro = (
        (
            HydroOperation(
                key="additive_hydrogen",
                reason="Observed single bonds replace parent-hydride double bonds.",
                locants=tuple(locants[atom] for atom in hydrogenated_atoms),
                atom_ids=tuple(hydrogenated_atoms),
                bond_ids=hydrogenated_bonds,
                operation_kind="additive_hydrogen",
            ),
        )
        if hydrogenated_atoms
        else ()
    )

    unsaturation = []
    for bond_id in sorted(delta.additional_multiple_bond_ids):
        bond = mol.bonds[bond_id]
        ordered_atoms = tuple(sorted((bond.u, bond.v), key=lambda atom: system_locant_sort_key(locants[atom])))
        unsaturation.append(
            UnsaturationOperation(
                key="additional_unsaturation",
                reason="Observed multiple bond is not implied by the parent hydride.",
                locants=(locants[ordered_atoms[0]], locants[ordered_atoms[1]]),
                atom_ids=ordered_atoms,
                bond_id=bond_id,
                bond_order=bond.order,
            )
        )

    return ParentDerivativeState(
        bond_delta=delta,
        hydro_operations=hydro,
        unsaturation_operations=tuple(unsaturation),
        oxo_operations=tuple(oxo),
    )


__all__ = [
    "ParentBondDelta",
    "ParentDerivativeState",
    "indicated_hydrogen_parent_bond_model",
    "compare_actual_parent_to_implied_parent",
    "parent_derivative_state",
]
