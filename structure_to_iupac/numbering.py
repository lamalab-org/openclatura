from .molecule import Molecule
import re

def _get_atom_locants(oriented_path: list[int], target_indices: set[int]) -> list[int]:
    return sorted([i + 1 for i, atom_idx in enumerate(oriented_path) if atom_idx in target_indices])

def _get_bond_locants(mol: Molecule, oriented_path: list[int], is_bicycle: bool, is_spiro: bool, is_polycycle: bool) -> tuple[list[int], list[int]]:
    db_locants =[]
    tb_locants =[]
    seen_bonds = set()
    for u in oriented_path:
        for v in mol.get_neighbors(u):
            if v in oriented_path:
                bond = mol.get_bond(u, v)
                if bond and bond.order > 1 and bond.idx not in seen_bonds:
                    seen_bonds.add(bond.idx)
                    loc_u = oriented_path.index(u) + 1
                    loc_v = oriented_path.index(v) + 1
                    min_loc, max_loc = min(loc_u, loc_v), max(loc_u, loc_v)
                    
                    if max_loc == min_loc + 1:
                        locant_val = min_loc
                    elif min_loc == 1 and max_loc == len(oriented_path) and not (is_bicycle or is_spiro or is_polycycle):
                        locant_val = max_loc
                    else:
                        locant_val = min_loc
                        
                    if bond.order == 2: db_locants.append(locant_val)
                    elif bond.order == 3: tb_locants.append(locant_val)
                        
    return sorted(db_locants), sorted(tb_locants)

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
    retained_name: str = None
) -> list[int]:
    
    candidates =[]
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
            pg = _get_atom_locants(oriented_path, principal_carbons)
            
            het_by_priority = {}
            for a in mol:
                if a.idx in oriented_path and not a.is_carbon:
                    prio = a.element.hw_priority or 99
                    het_by_priority.setdefault(prio,[]).append(a.idx)
            
            het_eval = tuple(_get_atom_locants(oriented_path, set(het_by_priority[prio])) for prio in sorted(het_by_priority.keys()))
            
            sub_idx = set(substituent_mapping.keys())
            pref = _get_atom_locants(oriented_path, sub_idx)
            db, tb = _get_bond_locants(mol, oriented_path, is_bicycle, is_spiro, is_polycycle)
            
            if retained_name:
                unsat =[]
            else:
                unsat = sorted(db + tb)
                
            pref_unsat = sorted(pref + unsat)
            
            def sub_sort_key(name):
                s = name.lower()
                s = re.sub(r'^[\(\[\{\)]+', '', s)
                prefix_pattern = r'^((?:(?:[0-9]+[a-z]*|[nospmc]\'*)(?:,(?:[0-9]+[a-z]*|[nospmc]\'*))*|[ezrs]+|sec|tert|t|s|d|l|m|o|p|alpha|beta|gamma))([-)]+)'
                while True:
                    prev = s
                    match = re.match(prefix_pattern, s)
                    if match:
                        s = s[match.end():]
                        s = re.sub(r'^[\(\[\{\)]+', '', s)
                        continue
                    break
                return s
            
            alpha_list =[]
            for idx in oriented_path:
                if idx in substituent_mapping:
                    loc = oriented_path.index(idx) + 1
                    for name in substituent_mapping[idx]:
                        alpha_list.append((sub_sort_key(name), loc))
            alpha_list.sort(key=lambda x: x[0])
            alpha_eval = tuple(x[1] for x in alpha_list)
            
            stereo_seq =[]
            for idx in oriented_path:
                atom = mol.atoms[idx]
                if atom.stereo:
                    stereo_seq.append(0 if atom.stereo == 'R' else 1)
            stereo_eval = tuple(stereo_seq)
            
            if is_ring:
                return het_eval + (pg, unsat, pref_unsat, alpha_eval, stereo_eval)
            else:
                return (pg,) + het_eval + (unsat, pref_unsat, alpha_eval, stereo_eval)

        ev1 = evaluate(p1)
        ev2 = evaluate(p2)
        
        for v1, v2 in zip(ev1, ev2):
            if not v1 and not v2: continue
            if not v1: return 1
            if not v2: return -1
            for x, y in zip(v1, v2):
                if x < y: return -1
                if x > y: return 1
            if len(v1) < len(v2): return -1
            if len(v1) > len(v2): return 1
        return 0

    best = candidates[0]
    for c in candidates[1:]:
        if compare_paths(c, best) < 0:
            best = c
    return best