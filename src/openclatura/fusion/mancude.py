"""Compare observed parent bonding with a proved mancude parent model."""

from __future__ import annotations

from dataclasses import dataclass

from ..molecule import Molecule
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


def compare_actual_parent_to_implied_parent(
    mol: Molecule,
    atom_ids: set[int] | frozenset[int],
    bond_model: ParentBondModel,
) -> ParentBondDelta | None:
    """Select the allowed Kekulé form requiring the smallest observed delta.

    Aromatic input bonds are representation-independent.  Explicit single
    bonds where the parent permits a double bond are hydrogenation sites;
    explicit multiple bonds where the selected parent has a single bond are
    additional unsaturation.  The function only compares graph data and does
    not infer nomenclature from rendered text.
    """

    atoms = frozenset(atom_ids)
    observed = {
        normalize_edge(bond.u, bond.v): bond for bond in mol.bonds.values() if bond.u in atoms and bond.v in atoms
    }
    known_edges = bond_model.required_single_bonds | bond_model.pi_eligible_edges
    if set(observed) != set(known_edges):
        return None

    candidates: list[tuple[tuple, ParentBondDelta]] = []
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
                if bond.order > 1 and expected_order > 1:
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
        delta = ParentBondDelta(
            assignment=assignment,
            implied_multiple_bond_ids=frozenset(implied_ids),
            hydrogenated_edges=tuple(sorted(hydrogenated)),
            additional_multiple_bond_ids=frozenset(additional_ids),
            compatible=incompatible == 0,
        )
        rank = (
            incompatible,
            len(additional_ids),
            len(hydrogenated),
            tuple(assignment.orders),
        )
        candidates.append((rank, delta))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


__all__ = ["ParentBondDelta", "compare_actual_parent_to_implied_parent"]
