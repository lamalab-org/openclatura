"""Graph roles for cumulene heteroatom substituents.

The naming layer reaches this module before generic carbonyl-amide fallback.
That keeps fragments such as ``N=C=O`` attached through nitrogen as
isocyanato ligands instead of incorrectly converting them to formamido groups.
"""

from dataclasses import dataclass

from .molecule import Molecule


@dataclass(frozen=True)
class HeterocumuleneLigandRole:
    """A terminal X=C=Y ligand bound through the starting heteroatom."""

    key: str
    prefix: str
    start_atom: int
    central_atom: int
    terminal_atom: int
    atom_ids: frozenset[int]
    reason: str


def nitrogen_heterocumulene_role(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
) -> HeterocumuleneLigandRole | None:
    """Return an N-bound heterocumulene role, if the local graph is unambiguous."""

    if upstream_atom is None or mol.atoms[start_idx].symbol != "N":
        return None
    upstream_bond = mol.get_bond(start_idx, upstream_atom)
    if upstream_bond is None or upstream_bond.order != 1:
        return None
    candidates = []
    for center in mol.get_neighbors(start_idx):
        if center == upstream_atom or center in exclude_atoms or not mol.atoms[center].is_carbon:
            continue
        bond = mol.get_bond(start_idx, center)
        if bond is None or bond.order != 2:
            continue
        terminal = _terminal_double_heteroatom(mol, center, start_idx, exclude_atoms)
        if terminal is not None:
            candidates.append((center, terminal))
    if len(candidates) != 1:
        return None
    center, terminal = candidates[0]
    symbol = mol.atoms[terminal].symbol
    prefix_by_terminal = {"O": "isocyanato", "S": "isothiocyanato", "Se": "isoselenocyanato"}
    prefix = prefix_by_terminal.get(symbol)
    if prefix is None:
        return None
    return HeterocumuleneLigandRole(
        key=prefix,
        prefix=prefix,
        start_atom=start_idx,
        central_atom=center,
        terminal_atom=terminal,
        atom_ids=frozenset({start_idx, center, terminal}),
        reason=f"Matched N=C={symbol} heterocumulene ligand at nitrogen {start_idx}.",
    )


def _terminal_double_heteroatom(
    mol: Molecule,
    center: int,
    start_idx: int,
    exclude_atoms: set[int],
) -> int | None:
    terminals = []
    for neighbor in mol.get_neighbors(center):
        if neighbor == start_idx or neighbor in exclude_atoms:
            continue
        if mol.atoms[neighbor].symbol not in {"O", "S", "Se"}:
            return None
        bond = mol.get_bond(center, neighbor)
        if bond is not None and bond.order == 2:
            terminals.append(neighbor)
        else:
            return None
    return terminals[0] if len(terminals) == 1 else None
