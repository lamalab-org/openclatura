"""Shared stereochemical descriptor tokens used by renderers and metadata."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .molecule import Molecule

ABSOLUTE_STEREO_DESCRIPTORS = frozenset({"R", "S"})
BOND_STEREO_DESCRIPTORS = frozenset({"E", "Z"})
RELATIVE_STEREO_DESCRIPTORS = frozenset({"cis", "trans"})


@dataclass(frozen=True)
class AbsoluteStereoCitation:
    atom_id: int
    descriptor: str
    modern_descriptor: str
    convention: str = "modern_cip"


def parent_absolute_stereo_citation(mol: "Molecule", atom_id: int, *, is_ring: bool) -> AbsoluteStereoCitation:
    atom = mol.atoms[atom_id]
    modern = atom.stereo
    if modern not in ABSOLUTE_STEREO_DESCRIPTORS:
        raise ValueError("absolute stereo citation requires an assigned R/S center")
    legacy = mol.legacy_cip.get(atom_id)
    # OPSIN's ring-parent P(III) convention differs from its acyclic phosphane
    # and phosphanyl handling. Keep the accurate graph label unchanged.
    if (
        is_ring
        and atom.symbol == "P"
        and atom.charge == 0
        and atom.total_h_count == 0
        and mol.degree(atom_id) == 3
        and all(mol.get_bond(atom_id, neighbor).order == 1 for neighbor in mol.get_neighbors(atom_id))
        and legacy in ABSOLUTE_STEREO_DESCRIPTORS
        and legacy != modern
    ):
        return AbsoluteStereoCitation(atom_id, legacy, modern, "opsin_ring_p_legacy_cip")
    return AbsoluteStereoCitation(atom_id, modern, modern)


SEARCHABLE_STEREO_TOKENS = frozenset(
    descriptor.lower()
    for descriptor in (
        *ABSOLUTE_STEREO_DESCRIPTORS,
        *BOND_STEREO_DESCRIPTORS,
        *RELATIVE_STEREO_DESCRIPTORS,
    )
)


def is_searchable_stereo_token(text: str) -> bool:
    """Return whether a renderer-emitted stereo token may be matched directly."""

    return text.lower() in SEARCHABLE_STEREO_TOKENS
