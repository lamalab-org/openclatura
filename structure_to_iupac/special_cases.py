"""Special component naming shortcuts."""

from collections.abc import Callable

from .molecule import Molecule
from .nomenclature import RULES
from .perception import PerceivedGroup

ComponentNamer = Callable[..., str]


def single_atom_component_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Return the name for a one-atom ionic component, when supported."""

    if len(component_atoms) != 1:
        return ""
    atom = mol.atoms[list(component_atoms)[0]]
    if atom.symbol in RULES.ions.single_atom_cations:
        return atom.element.name
    if atom.symbol in RULES.ions.single_atom_anions:
        return RULES.ions.single_atom_anions[atom.symbol]
    if atom.symbol == "N":
        return "azane"
    if atom.symbol == "O":
        return "oxidane"
    return ""


def anhydride_half_name(mol: Molecule, start_c: int, bridge_o: int, component_namer: ComponentNamer) -> str:
    """Name one acid half of an anhydride component."""

    half_atoms = set()
    queue = [start_c]
    visited = {bridge_o}
    while queue:
        curr = queue.pop(0)
        if curr not in half_atoms:
            half_atoms.add(curr)
            visited.add(curr)
            queue.extend([x for x in mol.get_neighbors(curr) if x not in visited])

    sub_mol = Molecule()
    for n in half_atoms:
        atom = mol.atoms[n]
        sub_mol.add_atom(
            symbol=atom.symbol,
            idx=n,
            charge=atom.charge,
            stereo=atom.stereo,
            is_aromatic=atom.is_aromatic,
            explicit_h_count=atom.explicit_h_count,
            total_h_count=atom.total_h_count,
        )
    oh_idx = max(mol.atoms.keys()) + 100
    sub_mol.add_atom(symbol="O", idx=oh_idx)
    sub_mol.add_bond(u=start_c, v=oh_idx, order=1)
    half_atoms.add(oh_idx)

    for n in half_atoms:
        if n == oh_idx:
            continue
        for nxt in mol.get_neighbors(n):
            if nxt in half_atoms and n < nxt:
                bond = mol.get_bond(n, nxt)
                sub_mol.add_bond(
                    u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring
                )

    return component_namer(sub_mol, half_atoms).replace(" acid", "")


def try_name_anhydride_component(
    mol: Molecule,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    component_namer: ComponentNamer,
) -> str:
    """Return an anhydride component name when the component is an anhydride."""

    if principal_key != "anhydride":
        return ""
    for group in perceived_groups:
        if group.key != "anhydride":
            continue
        bridge_o = next((o for o in group.atoms_involved if mol.degree(o) == 2 and mol.atoms[o].symbol == "O"), None)
        if bridge_o is None:
            continue
        c_neighbors = [n for n in mol.get_neighbors(bridge_o) if mol.atoms[n].is_carbon]
        if len(c_neighbors) != 2:
            continue
        name1 = anhydride_half_name(mol, c_neighbors[0], bridge_o, component_namer)
        name2 = anhydride_half_name(mol, c_neighbors[1], bridge_o, component_namer)
        if name1 == name2:
            return f"{name1} anhydride"
        names = sorted([name1, name2])
        return f"{names[0]} {names[1]} anhydride"
    return ""
