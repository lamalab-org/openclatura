"""Require a citation to identify one locant-labelled skeletal graph."""

from collections.abc import Iterable, Mapping

from ..locants import SystemLocant
from ..molecule import Molecule


def numbered_parent_graphs_agree(mol: Molecule, maps: Iterable[Mapping[int, SystemLocant]]) -> bool:
    """Allow tied atom embeddings only when fixed locants preserve the skeleton.

    Bond orders, observed charges, and hydrogens belong to the subsequent
    parent-state proof. They cannot resolve a citation that assigns different
    skeletal adjacency or elements to the same locants.
    """
    signatures = set()
    for mapping in maps:
        vertices = frozenset((locant, mol.atoms[atom].symbol) for atom, locant in mapping.items())
        edges = frozenset(
            frozenset((mapping[bond.u], mapping[bond.v]))
            for bond in mol.bonds.values()
            if bond.u in mapping and bond.v in mapping
        )
        signatures.add((vertices, edges))
        if len(signatures) > 1:
            return False
    return True
