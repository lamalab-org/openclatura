"""Small, cycle-free queries over the internal molecular graph."""

from collections import deque
from collections.abc import Iterable

from .molecule import Molecule


def bond_order(mol: Molecule, first: int, second: int | None) -> int:
    """Return a bond order, or zero when the second atom/bond is absent."""

    if second is None:
        return 0
    bond = mol.get_bond(first, second)
    return bond.order if bond is not None else 0


def charged_atom_ids(mol: Molecule, atom_ids: Iterable[int]) -> set[int]:
    """Return the selected atoms with a non-zero formal charge."""

    return {atom_idx for atom_idx in atom_ids if mol.atoms[atom_idx].charge != 0}


def bond_ids_within(mol: Molecule, atom_ids: set[int]) -> set[int]:
    """Return bond IDs whose two endpoints are in ``atom_ids``."""

    bond_ids = set()
    for atom_idx in atom_ids:
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in atom_ids and atom_idx < neighbor_idx:
                bond = mol.get_bond(atom_idx, neighbor_idx)
                if bond is not None:
                    bond_ids.add(bond.idx)
    return bond_ids


def edges_within_atoms(mol: Molecule, atom_ids: set[int]) -> set[tuple[int, int]]:
    """Return canonical endpoint pairs for edges induced by ``atom_ids``."""

    return {
        (atom_idx, neighbor_idx)
        for atom_idx in atom_ids
        for neighbor_idx in mol.get_neighbors(atom_idx)
        if neighbor_idx in atom_ids and atom_idx < neighbor_idx
    }


def normalize_edges(edges: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    """Return undirected edges with each endpoint pair in canonical order."""

    return {tuple(sorted((first, second))) for first, second in edges}


def component_atoms_until_blocked(
    mol: Molecule,
    component_atoms: set[int],
    root: int,
    blocked: set[int],
) -> set[int]:
    """Traverse a component from ``root`` without crossing blocked atoms."""

    atoms = set()
    queue = deque([root])
    while queue:
        atom_idx = queue.popleft()
        if atom_idx in atoms:
            continue
        if atom_idx not in component_atoms or atom_idx in blocked:
            return set()
        atoms.add(atom_idx)
        queue.extend(
            neighbor
            for neighbor in mol.get_neighbors(atom_idx)
            if neighbor in component_atoms and neighbor not in blocked
        )
    return atoms
