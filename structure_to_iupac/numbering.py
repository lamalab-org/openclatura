"""Parent numbering selection."""

import re

from .locants import get_atom_locants, get_bond_locants, parse_locant
from .molecule import Molecule
from .namer_config import INDICATED_H_RETAINED_NAMES


def number_parent(
    mol: Molecule,
    candidate_paths: list[list[int]],
    principal_carbons: set[int],
    substituent_mapping: dict[int, list[str]],
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool = False,
    fixed_start: bool = False,
    retained_name: str = None,
) -> list[int]:
    """Choose the preferred numbering for a selected parent skeleton."""

    candidates = []
    if is_bicycle or is_spiro or is_polycycle or fixed_start:
        candidates = candidate_paths
    elif is_ring:
        path = candidate_paths[0]
        for i in range(len(path)):
            rotated = path[i:] + path[:i]
            candidates.append(rotated)
            candidates.append(rotated[::-1])
    else:
        path = candidate_paths[0]
        candidates.append(path)
        candidates.append(path[::-1])

    def compare_paths(p1, p2):
        def evaluate(oriented_path):
            pg = get_atom_locants(oriented_path, principal_carbons)

            het_by_priority = {}
            for a in mol:
                if a.idx in oriented_path and not a.is_carbon:
                    prio = a.element.hw_priority or 99
                    het_by_priority.setdefault(prio, []).append(a.idx)

            het_eval = tuple(
                get_atom_locants(oriented_path, set(het_by_priority[prio]))
                for prio in sorted(het_by_priority.keys())
            )

            charged_tetrazole = retained_name == "tetrazole" and any(
                mol.atoms[a_idx].charge for a_idx in oriented_path
            )
            if retained_name in INDICATED_H_RETAINED_NAMES and not charged_tetrazole:
                sat_atoms = []
                for a_idx in oriented_path:
                    ring_bonds = [mol.get_bond(a_idx, n) for n in mol.get_neighbors(a_idx) if n in oriented_path]
                    if sum(b.order for b in ring_bonds) == 2:
                        sat_atoms.append(a_idx)
                sat_eval = tuple(get_atom_locants(oriented_path, set(sat_atoms)))
            else:
                sat_eval = ()

            sub_idx = set(substituent_mapping.keys())
            pref = get_atom_locants(oriented_path, sub_idx)
            db, tb = get_bond_locants(mol, oriented_path, is_bicycle, is_spiro, is_polycycle)

            if retained_name:
                unsat = []
            else:
                unsat = sorted(db + tb)

            pref_unsat = sorted(pref + unsat)

            def sub_sort_key(name):
                s = name.lower()
                s = re.sub(r"^[\(\[\{\)]+", "", s)
                prefix_pattern = r"^((?:(?:[0-9]+[a-z]*|[nospmc]\'*)(?:,(?:[0-9]+[a-z]*|[nospmc]\'*))*|[ezrs]+|sec|tert|t|s|d|l|m|o|p|alpha|beta|gamma))([-)]+)"
                while True:
                    match = re.match(prefix_pattern, s)
                    if match:
                        s = s[match.end() :]
                        s = re.sub(r"^[\(\[\{\)]+", "", s)
                        continue
                    break
                return s

            alpha_list = []
            for idx in oriented_path:
                if idx in substituent_mapping:
                    loc = oriented_path.index(idx) + 1
                    for item in substituent_mapping[idx]:
                        name = item.name if hasattr(item, "name") else item
                        alpha_list.append((sub_sort_key(name), loc))
            alpha_list.sort(key=lambda x: x[0])
            alpha_eval = tuple(x[1] for x in alpha_list)

            stereo_seq = []
            for idx in oriented_path:
                atom = mol.atoms[idx]
                if atom.stereo:
                    stereo_seq.append(0 if atom.stereo == "R" else 1)
            stereo_eval = tuple(stereo_seq)

            if is_ring:
                return het_eval + (sat_eval, pg, unsat, pref_unsat, alpha_eval, stereo_eval)
            else:
                return (pg,) + het_eval + (sat_eval, unsat, pref_unsat, alpha_eval, stereo_eval)

        ev1 = evaluate(p1)
        ev2 = evaluate(p2)

        for v1, v2 in zip(ev1, ev2):
            if not v1 and not v2:
                continue
            if not v1:
                return 1
            if not v2:
                return -1
            for x, y in zip(v1, v2):
                if x < y:
                    return -1
                if x > y:
                    return 1
            if len(v1) < len(v2):
                return -1
            if len(v1) > len(v2):
                return 1
        return 0

    best = candidates[0]
    for c in candidates[1:]:
        if compare_paths(c, best) < 0:
            best = c
    return best


def choose_parent_numbering(
    mol: Molecule,
    candidate_paths: list[list[int]],
    principal_atoms,
    substituent_mapping: dict[int, list],
    locant_maps,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    retained_name: str | None,
    *,
    fixed_start: bool = False,
) -> tuple[list[int], dict[int, str] | None]:
    """Choose parent numbering from retained locant maps or normal rules."""

    principal_atom_set = set(principal_atoms)
    if locant_maps:

        def evaluate_map(lmap):
            def get_val(idx):
                return parse_locant(lmap[idx])

            principal_eval = sorted([get_val(idx) for idx in principal_atom_set if idx in lmap])
            het_by_priority = {}
            for atom in mol:
                if atom.idx in lmap and not atom.is_carbon:
                    priority = atom.element.hw_priority or 99
                    het_by_priority.setdefault(priority, []).append(atom.idx)
            heteroatom_eval = tuple(
                sorted([get_val(idx) for idx in het_by_priority[priority]])
                for priority in sorted(het_by_priority.keys())
            )
            substituent_eval = sorted([get_val(idx) for idx in set(substituent_mapping.keys()) if idx in lmap])
            return heteroatom_eval + (principal_eval, substituent_eval)

        locant_map = min(locant_maps, key=evaluate_map)
        return list(locant_map.keys()), locant_map

    return (
        number_parent(
            mol,
            candidate_paths,
            principal_atom_set,
            substituent_mapping,
            is_ring,
            is_bicycle,
            is_spiro,
            is_polycycle=is_polycycle,
            fixed_start=fixed_start,
            retained_name=retained_name,
        ),
        None,
    )
