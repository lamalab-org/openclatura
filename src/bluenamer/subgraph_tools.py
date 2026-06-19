"""Shared graph-fragment helpers for parent and substituent assembly."""

from .additive import add_indicated_hydrogens as _add_indicated_hydrogens
from .additive import add_replacement_prefixes
from .assembly_parts import AssemblyParts
from .chains import get_cyclic_atoms
from .locants import parse_locant
from .molecule import Molecule
from .subtractive import add_unsaturations


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

    _add_indicated_hydrogens(mol, parts, numbered_path, get_loc)


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

    add_replacement_prefixes(mol, parts, numbered_path, get_loc)
    add_unsaturations(mol, parts, numbered_path, get_loc, is_bicycle, is_spiro, is_polycycle)
