"""Compare observed parent bonding with a proved mancude parent model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..locants import SystemLocant, system_locant_sort_key
from ..molecule import Molecule
from ..name_operations import HydroOperation, OxoOperation, UnsaturationOperation
from ..polycycle_topology import normalize_edge
from .model import BondAssignment, ParentBondModel


@dataclass(frozen=True, slots=True)
class ParentBondDelta:
    """The smallest graph delta from one allowed parent-bond assignment."""

    assignment: BondAssignment
    implied_multiple_bond_ids: frozenset[int]
    hydrogenated_edges: tuple[tuple[int, int], ...]
    additional_multiple_bond_ids: frozenset[int]
    compatible: bool

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


def compare_actual_parent_to_implied_parent(
    mol: Molecule,
    atom_ids: set[int] | frozenset[int],
    bond_model: ParentBondModel,
    *,
    externally_unsaturated_atom_ids: set[int] | frozenset[int] = frozenset(),
    indicated_hydrogen_atom_ids: set[int] | frozenset[int] = frozenset(),
    atom_to_locant: Mapping[int, str | SystemLocant] | None = None,
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
    ties independently of the input atom numbering.
    """

    atoms = frozenset(atom_ids)
    observed = {
        normalize_edge(bond.u, bond.v): bond for bond in mol.bonds.values() if bond.u in atoms and bond.v in atoms
    }
    known_edges = bond_model.required_single_bonds | bond_model.required_double_bonds | bond_model.pi_eligible_edges
    if set(observed) != set(known_edges):
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
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def parent_derivative_state(
    mol: Molecule,
    atom_ids: set[int] | frozenset[int],
    bond_model: ParentBondModel,
    atom_to_locant: Mapping[int, str | SystemLocant],
    *,
    indicated_hydrogen_atom_ids: set[int] | frozenset[int] = frozenset(),
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
    "compare_actual_parent_to_implied_parent",
    "parent_derivative_state",
]
