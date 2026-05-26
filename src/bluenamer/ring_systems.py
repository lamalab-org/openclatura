"""Ring-system fragment mapping helpers.

These helpers separate local ring-system work from global molecule atom IDs.
They are intentionally graph-structural and do not use SMARTS.
"""

from dataclasses import dataclass

from .molecule import Molecule


@dataclass(frozen=True)
class RingSystemFragment:
    """A copied ring-system fragment with local/global atom mappings."""

    atom_indices: tuple[int, ...]
    fragment: Molecule
    old_to_new: dict[int, int]
    new_to_old: dict[int, int]

    def global_atom(self, local_atom: int) -> int:
        """Map a fragment-local atom index to the original molecule atom index."""

        return self.new_to_old[local_atom]

    def local_atom(self, global_atom: int) -> int:
        """Map an original molecule atom index to the fragment-local atom index."""

        return self.old_to_new[global_atom]

    def global_atoms(self, local_atoms: list[int] | tuple[int, ...] | set[int]) -> tuple[int, ...]:
        """Map fragment-local atom indices to original molecule atom indices."""

        return tuple(self.new_to_old[atom_idx] for atom_idx in local_atoms)

    def global_numbering(self, local_numbering: dict[int, int]) -> dict[int, int]:
        """Map a fragment-local numbering dictionary back to original atom IDs."""

        return {self.new_to_old[atom_idx]: locant for atom_idx, locant in local_numbering.items()}


def ring_system_fragment(mol: Molecule, atom_indices: set[int] | list[int] | tuple[int, ...]) -> RingSystemFragment:
    """Return a copied fragment and bidirectional atom mapping for ring atoms."""

    ordered = tuple(sorted(atom_indices))
    fragment = Molecule()
    old_to_new: dict[int, int] = {}
    new_to_old: dict[int, int] = {}

    for new_idx, old_idx in enumerate(ordered):
        atom = mol.atoms[old_idx]
        fragment.add_atom(
            symbol=atom.symbol,
            idx=new_idx,
            charge=atom.charge,
            stereo=atom.stereo,
            is_aromatic=atom.is_aromatic,
            explicit_h_count=atom.explicit_h_count,
            total_h_count=atom.total_h_count,
        )
        old_to_new[old_idx] = new_idx
        new_to_old[new_idx] = old_idx

    ordered_set = set(ordered)
    for bond in mol.bonds.values():
        if bond.u not in ordered_set or bond.v not in ordered_set:
            continue
        fragment.add_bond(
            old_to_new[bond.u],
            old_to_new[bond.v],
            order=bond.order,
            idx=bond.idx,
            stereo=bond.stereo,
            in_small_ring=bond.in_small_ring,
        )

    return RingSystemFragment(atom_indices=ordered, fragment=fragment, old_to_new=old_to_new, new_to_old=new_to_old)
