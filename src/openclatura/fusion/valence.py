"""Graph-derived lambda descriptors for neutral systematic-fusion parents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..locants import SystemLocant, system_locant_sort_key
from ..molecule import Molecule


@dataclass(frozen=True, slots=True)
class FusionLambdaDescriptor:
    """One nonstandard neutral bonding number at a completed-system locant."""

    atom_id: int
    locant: SystemLocant
    bonding_number: int
    standard_valence: int

    def __post_init__(self) -> None:
        if self.atom_id < 0:
            raise ValueError("lambda descriptor atom id must be non-negative")
        if self.bonding_number <= self.standard_valence:
            raise ValueError("lambda descriptor requires a bonding number above standard valence")

    @property
    def text(self) -> str:
        return f"{self.locant}lambda^{self.bonding_number}"


def fusion_lambda_descriptors(
    mol: Molecule,
    parent_atom_ids: Iterable[int],
    atom_to_locant: Mapping[int, SystemLocant],
) -> tuple[FusionLambdaDescriptor, ...]:
    """Return neutral lambda annotations using the final parent numbering.

    Bonding number is calculated from the complete molecular graph, including
    exocyclic ligands, while locants come exclusively from the audited fused
    parent map. Charged centers are intentionally outside this neutral layer.
    """

    descriptors: list[FusionLambdaDescriptor] = []
    for atom_id in parent_atom_ids:
        atom = mol.atoms[atom_id]
        if atom.charge or atom.is_carbon:
            continue
        locant = atom_to_locant.get(atom_id)
        if locant is None:
            raise ValueError(f"missing completed-system locant for parent atom {atom_id}")
        bonding_number = (atom.total_h_count or atom.explicit_h_count) + sum(
            bond.order
            for neighbor in mol.get_neighbors(atom_id)
            if (bond := mol.get_bond(atom_id, neighbor)) is not None
        )
        if bonding_number <= atom.element.standard_valence:
            continue
        descriptors.append(
            FusionLambdaDescriptor(
                atom_id=atom_id,
                locant=locant,
                bonding_number=bonding_number,
                standard_valence=atom.element.standard_valence,
            )
        )
    return tuple(
        sorted(
            descriptors,
            key=lambda descriptor: (
                system_locant_sort_key(descriptor.locant),
                descriptor.bonding_number,
            ),
        )
    )
