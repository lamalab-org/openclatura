"""Data-driven component functional-group preprocessing."""

from collections.abc import Callable

from .molecule import Molecule
from .group_atom_roles import (
    amide_nitrogen,
    bridge_oxygen,
    ester_single_oxygen,
    peroxy_ester_single_oxygen,
    sulfonyl_sulfur,
)
from .nomenclature import RULES
from .perception import PerceivedGroup

AtomSelector = Callable[[Molecule, PerceivedGroup], int | None]


NONPARENT_ATOM_SELECTORS: dict[str, AtomSelector] = {"anhydride": bridge_oxygen}
NONPARENT_ATOM_SELECTORS.update(
    {
        key: ester_single_oxygen
        for key in RULES.functional_groups.keys_with_family("ester_like")
        - RULES.functional_groups.keys_with_family("peroxy_ester")
    }
)
NONPARENT_ATOM_SELECTORS.update(
    {key: peroxy_ester_single_oxygen for key in RULES.functional_groups.keys_with_family("peroxy_ester")}
)
NONPARENT_ATOM_SELECTORS.update({key: sulfonyl_sulfur for key in RULES.functional_groups.keys_with_family("sulfonyl")})
NONPARENT_ATOM_SELECTORS.update({key: amide_nitrogen for key in RULES.functional_groups.keys_with_family("amide_like")})


def retarget_external_carbonyl_groups(
    mol: Molecule,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    exclude_atoms: set[int],
    cyclic_atoms_all: set[int],
) -> None:
    """Move exocyclic carbonyl group attachment onto the parent chain atom."""

    for group in perceived_groups:
        if group.key == principal_key or group.key not in RULES.functional_groups.keys_with_family(
            "chain_external_carbonyl"
        ):
            continue
        group_c = group.attachment_carbon
        if group_c in cyclic_atoms_all:
            continue
        adj_c = [n for n in mol.get_neighbors(group_c) if mol.atoms[n].is_carbon and n not in group.atoms_involved]
        if len(adj_c) == 1:
            group.attachment_carbon = adj_c[0]
            group.atoms_involved.add(group_c)
            exclude_atoms.add(group_c)


def exclude_nonparent_group_atoms(
    mol: Molecule, perceived_groups: list[PerceivedGroup], exclude_atoms: set[int], cyclic_atoms_all: set[int]
) -> None:
    """Exclude linker atoms that should not become part of the parent skeleton."""

    for group in perceived_groups:
        selector = NONPARENT_ATOM_SELECTORS.get(group.key)
        atom_idx = selector(mol, group) if selector else None
        if atom_idx is not None and atom_idx not in cyclic_atoms_all:
            exclude_atoms.add(atom_idx)


def principal_involved_atoms(
    perceived_groups: list[PerceivedGroup], principal_key: str | None, parent_path: list[int]
) -> set[int]:
    """Return atoms already consumed by the principal group on the parent."""

    atoms = set()
    if principal_key:
        for group in perceived_groups:
            if group.key == principal_key and group.attachment_carbon in parent_path:
                atoms.update(group.atoms_involved)
    return atoms
