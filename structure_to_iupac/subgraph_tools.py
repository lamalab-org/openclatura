"""Shared graph-fragment helpers for parent and substituent assembly."""

from .assembler import AssemblyParts, SubstituentItem, UnsaturationItem
from .chains import get_cyclic_atoms
from .locants import parse_locant
from .molecule import Molecule
from .namer_config import INDICATED_H_RETAINED_NAMES


def emit_bond_stereo(mol, parts, numbered_path, get_loc, exclude_atoms=None, upstream_atom=None):
    """Collect E/Z bond stereochemical descriptors for assembly."""

    if exclude_atoms is None:
        exclude_atoms = set()
    path_set = set(numbered_path)
    cyclic_atoms = get_cyclic_atoms(mol)
    for u_idx in numbered_path:
        for v_idx in mol.get_neighbors(u_idx):
            bond = mol.get_bond(u_idx, v_idx)
            if not bond or not bond.stereo:
                continue
            if bond.in_small_ring:
                continue

            if v_idx == upstream_atom:
                upstream_in_ring = v_idx in cyclic_atoms
                if mol.atoms[v_idx].is_carbon or upstream_in_ring:
                    continue

            loc_str_u = get_loc(u_idx)
            if v_idx in path_set:
                loc_str_v = get_loc(v_idx)
                min_loc = min(loc_str_u, loc_str_v, key=lambda x: parse_locant(x))
            else:
                min_loc = loc_str_u

            if not any(feature[0] == min_loc and feature[1] in ["E", "Z"] for feature in parts.stereo_features):
                parts.stereo_features.append((min_loc, bond.stereo))


def subgraph_component(mol: Molecule, start_idx: int, exclude_atoms: set[int]) -> set[int]:
    """Return the connected recursive substituent component from ``start_idx``."""

    visited = set(exclude_atoms)
    component = set()
    queue = [start_idx]
    while queue:
        curr = queue.pop(0)
        if curr not in visited:
            visited.add(curr)
            component.add(curr)
            queue.extend([n for n in mol.get_neighbors(curr) if n not in visited])
    return component


def find_spiro_side_pair(
    mol: Molecule, c_idx: int, n_subs: list[int], main_set: set[int], sub_exclude: set[int]
) -> tuple[int, int] | None:
    """Find a side-ring pair that forms a spiro substituent at ``c_idx``."""

    for i in range(len(n_subs)):
        for j in range(i + 1, len(n_subs)):
            n1, n2 = n_subs[i], n_subs[j]
            visited = {c_idx}
            queue = [n1]
            while queue:
                curr = queue.pop(0)
                if curr == n2:
                    return n1, n2
                visited.add(curr)
                for nxt in mol.get_neighbors(curr):
                    if nxt not in visited and nxt not in main_set and nxt not in sub_exclude:
                        queue.append(nxt)
    return None


def spiro_side_component(
    mol: Molecule, c_idx: int, side_start: int, main_set: set[int], sub_exclude: set[int]
) -> set[int]:
    """Return the atoms in a side ring used as a spiro substituent."""

    sub_comp = set()
    queue = [side_start]
    visited = {c_idx}
    while queue:
        curr = queue.pop(0)
        if curr not in sub_comp:
            sub_comp.add(curr)
            visited.add(curr)
            for nxt in mol.get_neighbors(curr):
                if nxt not in visited and nxt not in main_set and nxt not in sub_exclude:
                    queue.append(nxt)
    sub_comp.add(c_idx)
    return sub_comp


def subgraph_locant_getter(numbered_path: list[int], locant_map):
    """Create a locant accessor for numbered recursive substituent atoms."""

    def get_loc(idx):
        return locant_map[idx] if locant_map else str(numbered_path.index(idx) + 1)

    return get_loc


def add_indicated_hydrogens(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Add indicated hydrogen locants for retained ring names."""

    if parts.retained_name not in INDICATED_H_RETAINED_NAMES:
        return
    for idx in numbered_path:
        atom = mol.atoms[idx]
        if atom.symbol in ["N", "C"]:
            ring_bonds = [mol.get_bond(idx, n) for n in mol.get_neighbors(idx) if n in numbered_path]
            if sum(b.order for b in ring_bonds) == 2:
                parts.indicated_hydrogens.append(get_loc(idx))


def add_parent_features(
    mol: Molecule,
    parts: AssemblyParts,
    numbered_path: list[int],
    get_loc,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
) -> None:
    """Add replacement prefixes and unsaturation locants to assembly parts."""

    if parts.retained_name:
        return

    for atom_idx in numbered_path:
        atom = mol.atoms[atom_idx]
        if not atom.is_carbon:
            hw_stem = atom.element.hw_stem
            if hw_stem:
                valence = sum(mol.get_bond(atom_idx, n).order for n in mol.get_neighbors(atom_idx))
                loc = get_loc(atom_idx)
                if atom.charge == 0 and valence > atom.element.standard_valence:
                    loc = f"{loc}lambda^{valence}"
                parts.a_prefixes.append(SubstituentItem(name=hw_stem, locants=[loc], atom_ids={atom_idx}))

    seen_bonds = set()
    for u_idx in numbered_path:
        for v_idx in mol.get_neighbors(u_idx):
            if v_idx in numbered_path:
                bond = mol.get_bond(u_idx, v_idx)
                if bond and bond.order > 1 and bond.idx not in seen_bonds:
                    seen_bonds.add(bond.idx)
                    bond_key = "double" if bond.order == 2 else "triple"
                    loc_u_idx = numbered_path.index(u_idx)
                    loc_v_idx = numbered_path.index(v_idx)
                    min_idx, max_idx = min(loc_u_idx, loc_v_idx), max(loc_u_idx, loc_v_idx)

                    loc_u_str = get_loc(u_idx)
                    loc_v_str = get_loc(v_idx)
                    min_loc_str, max_loc_str = (
                        min(loc_u_str, loc_v_str, key=lambda x: parse_locant(x)),
                        max(loc_u_str, loc_v_str, key=lambda x: parse_locant(x)),
                    )

                    if max_idx == min_idx + 1:
                        locant_str = min_loc_str
                    elif min_idx == 0 and max_idx == len(numbered_path) - 1 and not (
                        is_bicycle or is_spiro or is_polycycle
                    ):
                        locant_str = max_loc_str
                    else:
                        locant_str = f"{min_loc_str}({max_loc_str})"

                    existing = next((u for u in parts.unsaturations if u.bond_key == bond_key), None)
                    if existing:
                        existing.locants.append(locant_str)
                        existing.atom_ids.update({u_idx, v_idx})
                        existing.bond_ids.add(bond.idx)
                    else:
                        parts.unsaturations.append(
                            UnsaturationItem(
                                bond_key=bond_key,
                                locants=[locant_str],
                                atom_ids={u_idx, v_idx},
                                bond_ids={bond.idx},
                            )
                        )
