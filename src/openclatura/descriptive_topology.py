"""Audited, description-only views of ring-system topology.

These projections explain a selected retained or fusion parent in an
alternative nomenclature without participating in parent selection or name
rendering.  They are therefore computed only while explainability metadata is
being built.
"""

from __future__ import annotations

from dataclasses import dataclass

from .locants import retained_locant_sort_key
from .molecule import Molecule, edges_within_atoms
from .polycycle_topology import bicyclo_proof
from .ring_renderer import render_ring_descriptor, von_baeyer_cycle_count
from .von_baeyer import find_von_baeyer_candidates


@dataclass(frozen=True, slots=True)
class VonBaeyerTopologyView:
    """One graph-proven von Baeyer representation of a parent skeleton."""

    descriptor: str
    cycle_count: int
    atom_to_locant: tuple[tuple[int, str], ...]
    proof_source: str


def von_baeyer_topology_view(
    mol: Molecule,
    parent_atoms: set[int] | frozenset[int],
    *,
    preferred_atom_to_locant: dict[int, str] | None = None,
) -> VonBaeyerTopologyView | None:
    """Return an independently audited von Baeyer view when one is bounded.

    Bond orders, aromaticity, charges, and atom types do not alter skeletal
    connectivity, so candidate discovery runs on a saturated topology copy.
    The returned locants belong to this auxiliary representation and are not
    confused with the numbering of the selected retained/fusion parent.
    """

    atoms = frozenset(parent_atoms)
    edges = frozenset(edges_within_atoms(mol, set(atoms)))
    if not atoms or len(edges) - len(atoms) + 1 < 2:
        return None

    bicycle = bicyclo_proof(atoms, edges)
    if bicycle is not None:
        descriptor = render_ring_descriptor("bicyclo", bicycle.descriptor_numbers)
        path = min(
            bicycle.numbering_paths,
            key=lambda item: _preferred_path_key(item, preferred_atom_to_locant),
        )
        return VonBaeyerTopologyView(
            descriptor=descriptor,
            cycle_count=2,
            atom_to_locant=tuple((atom, str(index)) for index, atom in enumerate(path, start=1)),
            proof_source="bicyclo_topology_proof",
        )

    topology = _saturated_topology_copy(mol, atoms)
    candidates = find_von_baeyer_candidates(topology, atoms, edges)
    if not candidates:
        return None
    best_rank = candidates[0].rank
    candidate = min(
        (item for item in candidates if item.rank == best_rank),
        key=lambda item: _preferred_path_key(item.path, preferred_atom_to_locant),
    )
    cycle_count = von_baeyer_cycle_count(candidate.descriptor)
    if cycle_count is None:
        return None
    return VonBaeyerTopologyView(
        descriptor=candidate.descriptor,
        cycle_count=cycle_count,
        atom_to_locant=tuple(
            sorted(
                ((atom, str(locant)) for atom, locant in candidate.numbering.atom_to_locant.items()),
                key=lambda item: int(item[1]),
            )
        ),
        proof_source="audited_von_baeyer_candidate",
    )


def _saturated_topology_copy(mol: Molecule, atoms: frozenset[int]) -> Molecule:
    """Copy an induced parent graph without chemistry irrelevant to topology."""

    topology = Molecule()
    for atom_id in sorted(atoms):
        topology.add_atom("C", idx=atom_id)
    for bond in sorted(mol.bonds.values(), key=lambda item: item.idx):
        if bond.u in atoms and bond.v in atoms:
            topology.add_bond(bond.u, bond.v, idx=bond.idx)
    return topology


def _preferred_path_key(path: tuple[int, ...], preferred_atom_to_locant: dict[int, str] | None) -> tuple:
    """Break equivalent topology ties with the selected parent's numbering."""

    if preferred_atom_to_locant and all(atom in preferred_atom_to_locant for atom in path):
        return tuple(retained_locant_sort_key(preferred_atom_to_locant[atom]) for atom in path)
    return tuple((atom,) for atom in path)


__all__ = ["VonBaeyerTopologyView", "von_baeyer_topology_view"]
