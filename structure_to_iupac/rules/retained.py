# structure-to-iupac/rules/retained.py
from structure_to_iupac.molecule import Molecule
from structure_to_iupac.nomenclature import RULES
import itertools

def _find_two_rings(mol: Molecule, path: list[int], fused: list[int], path_set: set[int], small_ring_size: int):
    """Split a fused bicyclic path into its two component rings.
    Returns (small_ring, big_ring) as ordered atom lists, or (None, None) if cannot split.
    """
    rings =[]
    for start_n in mol.get_neighbors(fused[0]):
        if start_n in path_set and start_n != fused[1]:
            ring_atoms =[fused[0], start_n]
            visited_local = {fused[0], start_n}
            cur = start_n
            while cur != fused[1]:
                nxt = next((x for x in mol.get_neighbors(cur) if x in path_set and x not in visited_local), None)
                if nxt is None:
                    break
                ring_atoms.append(nxt)
                visited_local.add(nxt)
                cur = nxt
                if len(ring_atoms) > 7:
                    break
            if len(ring_atoms) >= 5 and ring_atoms[-1] == fused[1]:
                rings.append(ring_atoms)
    if len(rings) != 2:
        return None, None
    if len(rings[0]) == small_ring_size:
        return rings[0], rings[1]
    elif len(rings[1]) == small_ring_size:
        return rings[1], rings[0]
    return None, None

def _count_double_bonds_in_ring(mol: Molecule, ring_atoms: list[int]) -> int:
    seen = set()
    cnt = 0
    n = len(ring_atoms)
    for i in range(n):
        a, b = ring_atoms[i], ring_atoms[(i + 1) % n]
        bd = mol.get_bond(a, b)
        if bd and bd.idx not in seen:
            seen.add(bd.idx)
            if bd.order == 2:
                cnt += 1
    return cnt

def _has_no_cumulated_double_bonds(mol: Molecule, path: list[int]) -> bool:
    path_set = set(path)
    for a in path:
        db_count = 0
        for b in mol.get_neighbors(a):
            if b in path_set:
                bd = mol.get_bond(a, b)
                if bd and bd.order == 2:
                    db_count += 1
        if db_count > 1:
            return False
    return True

def recognizes_retained_ring(mol: Molecule, path: list[int]) -> bool:
    """Return whether a path matches any retained-ring recognizer."""

    return get_retained_ring(mol, path) is not None


def get_retained_ring(mol: Molecule, path: list[int]) -> tuple[str, list[dict[int, str]] | None] | None:
    size = len(path)
    path_set = set(path)
    
    double_bonds = 0
    total_bonds = 0
    seen_bonds = set()
    
    internal_degrees = {u: 0 for u in path}
    symbols = [mol.atoms[idx].symbol for idx in path]
    n_count = symbols.count("N")
    o_count = symbols.count("O")
    s_count = symbols.count("S")
    
    for u in path:
        for v in mol.get_neighbors(u):
            if v in path_set:
                internal_degrees[u] += 1
                bond = mol.get_bond(u, v)
                if bond and bond.idx not in seen_bonds:
                    seen_bonds.add(bond.idx)
                    total_bonds += 1
                    if bond.order == 2:
                        double_bonds += 1

    deg_counts = tuple(sorted(internal_degrees.values()))
    sig = (size, total_bonds, double_bonds, deg_counts)
    deg3_nodes =[u for u, d in internal_degrees.items() if d == 3]

    data_monocycle = _match_data_monocycle_retained(
        mol,
        path,
        size,
        total_bonds,
        double_bonds,
        symbols,
    )
    if data_monocycle is not None:
        return data_monocycle, None
    
    if _matches_any_retained_signature(
        ("naphthalene", "quinoline", "isoquinoline", "quinazoline", "quinoxaline", "cinnoline"),
        sig,
        symbols,
        deg3_nodes,
        mol,
    ):
        if len(deg3_nodes) == 2 and mol.get_bond(deg3_nodes[0], deg3_nodes[1]) is not None:
            small_ring, big_ring = _find_two_rings(mol, path, deg3_nodes, path_set, 6)
            if small_ring and big_ring and len(small_ring) == 6 and len(big_ring) == 6:
                # Check that one ring is benzene-like (all C, 3 double bonds)
                small_symbols = [mol.atoms[a].symbol for a in small_ring]
                big_symbols =[mol.atoms[a].symbol for a in big_ring]
                small_db = _count_double_bonds_in_ring(mol, small_ring)
                big_db = _count_double_bonds_in_ring(mol, big_ring)
                
                if n_count == 1 and s_count == 0 and o_count == 0:
                    n_idx = next(idx for idx in path if mol.atoms[idx].symbol == "N")
                    # N must be in the heterocyclic ring (the one with N), benzene must be all-C
                    benzene_ring = small_ring if "N" not in small_symbols else (big_ring if "N" not in big_symbols else None)
                    if benzene_ring is None:
                        return None
                    benzene_db = _count_double_bonds_in_ring(mol, benzene_ring)
                    if benzene_db != 3 or not _has_no_cumulated_double_bonds(mol, benzene_ring):
                        return None
                    if any(mol.atoms[a].symbol != "C" for a in benzene_ring):
                        return None
                    
                    idx_n = path.index(n_idx)
                    n_neighbors_in_path =[v for v in mol.get_neighbors(n_idx) if v in path_set]
                    is_quinoline = any(internal_degrees[v] == 3 for v in n_neighbors_in_path)
                    
                    if is_quinoline:
                        rot_path = path[idx_n:] + path[:idx_n]
                        if internal_degrees[rot_path[1]] == 3:
                            rot_path =[rot_path[0]] + rot_path[1:][::-1]
                        if internal_degrees[rot_path[4]] == 3 and internal_degrees[rot_path[9]] == 3:
                            locants = _retained_locants("quinoline")
                            return "quinoline", [{rot_path[i]: locants[i] for i in range(10)}]
                    else:
                        c1 = next(v for v in n_neighbors_in_path if any(internal_degrees[w] == 3 for w in mol.get_neighbors(v) if w in path_set))
                        idx_c1 = path.index(c1)
                        rot_path = path[idx_c1:] + path[:idx_c1]
                        if rot_path[1] != n_idx:
                            rot_path =[rot_path[0]] + rot_path[1:][::-1]
                        if internal_degrees[rot_path[4]] == 3 and internal_degrees[rot_path[9]] == 3:
                            locants = _retained_locants("isoquinoline")
                            return "isoquinoline", [{rot_path[i]: locants[i] for i in range(10)}]
                if n_count == 2 and s_count == 0 and o_count == 0:
                    n_indices =[idx for idx in path if mol.atoms[idx].symbol == "N"]
                    # Both N must be in same ring, the other must be benzene
                    ring_with_n = small_ring if all(n in small_ring for n in n_indices) else (big_ring if all(n in big_ring for n in n_indices) else None)
                    if ring_with_n is None:
                        return None
                    benzene_ring = big_ring if ring_with_n is small_ring else small_ring
                    if any(mol.atoms[a].symbol != "C" for a in benzene_ring):
                        return None
                    if _count_double_bonds_in_ring(mol, benzene_ring) != 3 or not _has_no_cumulated_double_bonds(mol, benzene_ring):
                        return None
                    
                    if all(internal_degrees[n] == 2 for n in n_indices):
                        n1, n2 = n_indices
                        rot_path = path[path.index(n1):] + path[:path.index(n1)]
                        if internal_degrees[rot_path[1]] == 3:
                            rot_path = [rot_path[0]] + rot_path[1:][::-1]
                        
                        if internal_degrees[rot_path[4]] == 3 and internal_degrees[rot_path[9]] == 3:
                            if rot_path[2] == n2:
                                locants = _retained_locants("quinazoline")
                                return "quinazoline", [{rot_path[i]: locants[i] for i in range(10)}]
                            elif rot_path[3] == n2:
                                locants = _retained_locants("quinoxaline")
                                return "quinoxaline",[{rot_path[i]: locants[i] for i in range(10)}]
                            elif rot_path[1] == n2:
                                locants = _retained_locants("cinnoline")
                                return "cinnoline", [{rot_path[i]: locants[i] for i in range(10)}]
                        
                        rot_path2 = path[path.index(n2):] + path[:path.index(n2)]
                        if internal_degrees[rot_path2[1]] == 3:
                            rot_path2 = [rot_path2[0]] + rot_path2[1:][::-1]
                        if internal_degrees[rot_path2[4]] == 3 and internal_degrees[rot_path2[9]] == 3:
                            if rot_path2[2] == n1:
                                locants = _retained_locants("quinazoline")
                                return "quinazoline",[{rot_path2[i]: locants[i] for i in range(10)}]
                            elif rot_path2[3] == n1:
                                locants = _retained_locants("quinoxaline")
                                return "quinoxaline",[{rot_path2[i]: locants[i] for i in range(10)}]
                            elif rot_path2[1] == n1:
                                locants = _retained_locants("cinnoline")
                                return "cinnoline", [{rot_path2[i]: locants[i] for i in range(10)}]
                if n_count == 0:
                    if any(mol.atoms[a].symbol != "C" for a in path):
                        return None
                    maps =[]
                    alpha_nodes =[u for u in path if internal_degrees[u] == 2 and any(internal_degrees[v] == 3 for v in mol.get_neighbors(u) if v in path_set)]
                    for start_idx in alpha_nodes:
                        rot_path = path[path.index(start_idx):] + path[:path.index(start_idx)]
                        if internal_degrees[rot_path[1]] == 3:
                            rot_path = [rot_path[0]] + rot_path[1:][::-1]
                        if internal_degrees[rot_path[4]] == 3 and internal_degrees[rot_path[9]] == 3:
                            locants = _retained_locants("naphthalene")
                            maps.append({rot_path[i]: locants[i] for i in range(10)})
                    if maps:
                        return "naphthalene", maps
                    return None

    if _matches_any_retained_signature(
        (
            "indole", "isoindole", "benzofuran", "benzothiophene",
            "benzothiazole", "1,2-benzothiazole", "benzoxazole",
            "1,2-benzoxazole", "benzimidazole", "indazole",
        ),
        sig,
        symbols,
        deg3_nodes,
        mol,
    ):
        if len(deg3_nodes) == 2 and mol.get_bond(deg3_nodes[0], deg3_nodes[1]) is not None:
            small_ring, big_ring = _find_two_rings(mol, path, deg3_nodes, path_set, 5)
            if small_ring is None or big_ring is None:
                return None
            # Big ring (6) must be benzene
            if any(mol.atoms[a].symbol != "C" for a in big_ring):
                return None
            if _count_double_bonds_in_ring(mol, big_ring) != 3 or not _has_no_cumulated_double_bonds(mol, big_ring):
                return None
            small_symbols = [mol.atoms[a].symbol for a in small_ring]
            small_db = _count_double_bonds_in_ring(mol, small_ring)
            
            if n_count == 1 and s_count == 0 and o_count == 0:
                if "N" not in small_symbols:
                    return None
                if small_db != 1:  # indole/isoindole have 1 double bond in 5-ring
                    return None
                n_idx = next(idx for idx in path if mol.atoms[idx].symbol == "N")
                if internal_degrees[n_idx] != 2:
                    return None
                n_neighbors_in_path =[v for v in mol.get_neighbors(n_idx) if v in path_set]
                is_indole = any(internal_degrees[v] == 3 for v in n_neighbors_in_path)
                
                if is_indole:
                    rot_path = path[path.index(n_idx):] + path[:path.index(n_idx)]
                    if internal_degrees[rot_path[1]] == 3:
                        rot_path = [rot_path[0]] + rot_path[1:][::-1]
                    if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                        locants = _retained_locants("indole")
                        return "indole", [{rot_path[i]: locants[i] for i in range(9)}]
                else:
                    c1 = next(v for v in n_neighbors_in_path if any(internal_degrees[w] == 3 for w in mol.get_neighbors(v) if w in path_set))
                    idx_c1 = path.index(c1)
                    rot_path = path[idx_c1:] + path[:idx_c1]
                    if rot_path[1] != n_idx:
                        rot_path = [rot_path[0]] + rot_path[1:][::-1]
                    if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                        locants = _retained_locants("isoindole")
                        return "isoindole", [{rot_path[i]: locants[i] for i in range(9)}]
            if o_count == 1 and n_count == 0 and s_count == 0:
                if "O" not in small_symbols:
                    return None
                if small_db != 1:
                    return None
                o_idx = next(idx for idx in path if mol.atoms[idx].symbol == "O")
                if internal_degrees[o_idx] != 2:
                    return None
                rot_path = path[path.index(o_idx):] + path[:path.index(o_idx)]
                if internal_degrees[rot_path[1]] == 3:
                    rot_path = [rot_path[0]] + rot_path[1:][::-1]
                if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                    locants = _retained_locants("benzofuran")
                    return "benzofuran",[{rot_path[i]: locants[i] for i in range(9)}]
            if s_count == 1 and n_count == 0 and o_count == 0:
                if "S" not in small_symbols:
                    return None
                if small_db != 1:
                    return None
                s_idx = next(idx for idx in path if mol.atoms[idx].symbol == "S")
                if internal_degrees[s_idx] != 2:
                    return None
                rot_path = path[path.index(s_idx):] + path[:path.index(s_idx)]
                if internal_degrees[rot_path[1]] == 3:
                    rot_path =[rot_path[0]] + rot_path[1:][::-1]
                if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                    locants = _retained_locants("benzothiophene")
                    return "benzothiophene", [{rot_path[i]: locants[i] for i in range(9)}]
            if n_count == 1 and s_count == 1 and o_count == 0:
                if "N" not in small_symbols or "S" not in small_symbols:
                    return None
                if small_db != 1:
                    return None
                s_idx = next(idx for idx in path if mol.atoms[idx].symbol == "S")
                n_idx = next(idx for idx in path if mol.atoms[idx].symbol == "N")
                if internal_degrees[s_idx] == 2 and internal_degrees[n_idx] == 2:
                    rot_path = path[path.index(s_idx):] + path[:path.index(s_idx)]
                    if internal_degrees[rot_path[1]] == 3:
                        rot_path = [rot_path[0]] + rot_path[1:][::-1]
                    if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                        if rot_path[2] == n_idx:
                            locants = _retained_locants("benzothiazole")
                            return "benzothiazole", [{rot_path[i]: locants[i] for i in range(9)}]
                        elif rot_path[1] == n_idx:
                            locants = _retained_locants("1,2-benzothiazole")
                            return "1,2-benzothiazole",[{rot_path[i]: locants[i] for i in range(9)}]
            if n_count == 1 and o_count == 1 and s_count == 0:
                if "N" not in small_symbols or "O" not in small_symbols:
                    return None
                if small_db != 1:
                    return None
                o_idx = next(idx for idx in path if mol.atoms[idx].symbol == "O")
                n_idx = next(idx for idx in path if mol.atoms[idx].symbol == "N")
                if internal_degrees[o_idx] == 2 and internal_degrees[n_idx] == 2:
                    rot_path = path[path.index(o_idx):] + path[:path.index(o_idx)]
                    if internal_degrees[rot_path[1]] == 3:
                        rot_path = [rot_path[0]] + rot_path[1:][::-1]
                    if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                        if rot_path[2] == n_idx:
                            locants = _retained_locants("benzoxazole")
                            return "benzoxazole", [{rot_path[i]: locants[i] for i in range(9)}]
                        elif rot_path[1] == n_idx:
                            locants = _retained_locants("1,2-benzoxazole")
                            return "1,2-benzoxazole", [{rot_path[i]: locants[i] for i in range(9)}]
            if n_count == 2 and o_count == 0 and s_count == 0:
                if small_symbols.count("N") != 2:
                    return None
                if small_db != 1:
                    return None
                n_indices =[idx for idx in path if mol.atoms[idx].symbol == "N"]
                if all(internal_degrees[n] == 2 for n in n_indices):
                    n1, n2 = n_indices
                    rot_path = path[path.index(n1):] + path[:path.index(n1)]
                    if internal_degrees[rot_path[1]] == 3:
                        rot_path =[rot_path[0]] + rot_path[1:][::-1]
                    if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                        if rot_path[2] == n2:
                            locants = _retained_locants("benzimidazole")
                            return "benzimidazole", [{rot_path[i]: locants[i] for i in range(9)}]
                        elif rot_path[1] == n2:
                            locants = _retained_locants("indazole")
                            return "indazole", [{rot_path[i]: locants[i] for i in range(9)}]
                    
                    rot_path2 = path[path.index(n2):] + path[:path.index(n2)]
                    if internal_degrees[rot_path2[1]] == 3:
                        rot_path2 = [rot_path2[0]] + rot_path2[1:][::-1]
                    if internal_degrees[rot_path2[3]] == 3 and internal_degrees[rot_path2[8]] == 3:
                        if rot_path2[2] == n1:
                            locants = _retained_locants("benzimidazole")
                            return "benzimidazole", [{rot_path2[i]: locants[i] for i in range(9)}]
                        elif rot_path2[1] == n1:
                            locants = _retained_locants("indazole")
                            return "indazole",[{rot_path2[i]: locants[i] for i in range(9)}]

    if _matches_any_retained_signature(("indoline", "indane"), sig, symbols, deg3_nodes, mol):
        if len(deg3_nodes) == 2 and mol.get_bond(deg3_nodes[0], deg3_nodes[1]) is not None:
            small_ring, big_ring = _find_two_rings(mol, path, deg3_nodes, path_set, 5)
            if small_ring is None or big_ring is None:
                return None
            # Big ring must be benzene (all-C, 3 double bonds)
            if any(mol.atoms[a].symbol != "C" for a in big_ring):
                return None
            if _count_double_bonds_in_ring(mol, big_ring) != 3 or not _has_no_cumulated_double_bonds(mol, big_ring):
                return None
            small_symbols = [mol.atoms[a].symbol for a in small_ring]
            small_db = _count_double_bonds_in_ring(mol, small_ring)
            
            if n_count == 1 and o_count == 0 and s_count == 0:
                # indoline: 5-ring contains exactly one N, all single bonds in 5-ring (saturated)
                if small_symbols.count("N") != 1:
                    return None
                if small_db != 0:
                    return None
                n_idx = next(idx for idx in path if mol.atoms[idx].symbol == "N")
                if internal_degrees[n_idx] != 2:
                    return None
                # Verify N is in small ring
                if n_idx not in small_ring:
                    return None
                rot_path = path[path.index(n_idx):] + path[:path.index(n_idx)]
                if internal_degrees[rot_path[1]] == 3:
                    rot_path = [rot_path[0]] + rot_path[1:][::-1]
                if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                    locants = _retained_locants("indoline")
                    return "indoline",[{rot_path[i]: locants[i] for i in range(9)}]
            if n_count == 0 and o_count == 0 and s_count == 0:
                if any(mol.atoms[a].symbol != "C" for a in path):
                    return None
                if small_db != 0:
                    return None
                
                maps =[]
                fused_atoms = set(deg3_nodes)
                small_non_fused =[a for a in small_ring if a not in fused_atoms]
                start_candidates =[a for a in small_non_fused if any(n in fused_atoms for n in mol.get_neighbors(a) if n in path_set)]
                
                for start_idx in start_candidates:
                    rot_path = path[path.index(start_idx):] + path[:path.index(start_idx)]
                    if internal_degrees[rot_path[1]] == 3:
                        rot_path = [rot_path[0]] + rot_path[1:][::-1]
                    if internal_degrees[rot_path[3]] == 3 and internal_degrees[rot_path[8]] == 3:
                        locants = _retained_locants("indane")
                        maps.append({rot_path[i]: locants[i] for i in range(9)})
                if maps:
                    return "indane", maps
                return None

    if total_bonds == size:
        if size == 6 and double_bonds == 0:
            if n_count == 1 and o_count == 0 and s_count == 0: return "piperidine", None
            if n_count == 2 and o_count == 0 and s_count == 0:
                n_indices =[i for i in path if mol.atoms[i].symbol == "N"]
                dist = min(abs(path.index(n_indices[0]) - path.index(n_indices[1])), 6 - abs(path.index(n_indices[0]) - path.index(n_indices[1])))
                if dist == 3: return "piperazine", None
            if n_count == 1 and o_count == 1 and s_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                o_idx = next(i for i in path if mol.atoms[i].symbol == "O")
                dist = min(abs(path.index(n_idx) - path.index(o_idx)), 6 - abs(path.index(n_idx) - path.index(o_idx)))
                if dist == 3: return "morpholine", None
            if n_count == 1 and s_count == 1 and o_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                s_idx = next(i for i in path if mol.atoms[i].symbol == "S")
                dist = min(abs(path.index(n_idx) - path.index(s_idx)), 6 - abs(path.index(n_idx) - path.index(s_idx)))
                if dist == 3: return "thiomorpholine", None
        if size == 5 and double_bonds == 0:
            if n_count == 1 and o_count == 0 and s_count == 0: return "pyrrolidine", None
            if o_count == 1 and n_count == 0 and s_count == 0: return "oxolane", None
            if s_count == 1 and n_count == 0 and o_count == 0: return "thiolane", None
            if n_count == 2 and o_count == 0 and s_count == 0:
                n_indices = [i for i in path if mol.atoms[i].symbol == "N"]
                dist = min(abs(path.index(n_indices[0]) - path.index(n_indices[1])), 5 - abs(path.index(n_indices[0]) - path.index(n_indices[1])))
                if dist == 1:
                    n1, n2 = n_indices
                    i1, i2 = path.index(n1), path.index(n2)
                    if (i1 + 1) % 5 == i2:
                        rot = path[i1:] + path[:i1]
                    else:
                        rot = path[i2:] + path[:i2]
                    locants =['1', '2', '3', '4', '5']
                    return "pyrazolidine", [{rot[i]: locants[i] for i in range(5)}]
                if dist == 2:
                    n1, n2 = n_indices
                    i1, i2 = path.index(n1), path.index(n2)
                    forward = (i1 + 2) % 5 == i2
                    if forward:
                        rot = path[i1:] + path[:i1]
                    else:
                        rot = path[i2:] + path[:i2]
                    locants =['1', '2', '3', '4', '5']
                    return "imidazolidine", [{rot[i]: locants[i] for i in range(5)}]
            if n_count == 1 and o_count == 1 and s_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                o_idx = next(i for i in path if mol.atoms[i].symbol == "O")
                dist = min(abs(path.index(n_idx) - path.index(o_idx)), 5 - abs(path.index(n_idx) - path.index(o_idx)))
                if dist == 1:
                    return "isoxazolidine", None
                if dist == 2:
                    return "1,3-oxazolidine", None
            if n_count == 1 and s_count == 1 and o_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                s_idx = next(i for i in path if mol.atoms[i].symbol == "S")
                dist = min(abs(path.index(n_idx) - path.index(s_idx)), 5 - abs(path.index(n_idx) - path.index(s_idx)))
                if dist == 1:
                    return "isothiazolidine", None
                if dist == 2:
                    return "1,3-thiazolidine", None
            
        if size == 6 and double_bonds == 3 and _has_no_cumulated_double_bonds(mol, path):
            if n_count == 0: return "benzene", None
            if n_count == 1: return "pyridine", None
            if n_count == 2:
                n_indices =[i for i in path if mol.atoms[i].symbol == "N"]
                dist = min(abs(path.index(n_indices[0]) - path.index(n_indices[1])), 6 - abs(path.index(n_indices[0]) - path.index(n_indices[1])))
                if dist == 1: return "pyridazine", None
                if dist == 2: return "pyrimidine", None
                if dist == 3: return "pyrazine", None
            if n_count == 3:
                n_indices =[i for i in path if mol.atoms[i].symbol == "N"]
                idx_sorted = sorted([path.index(i) for i in n_indices])
                d1 = idx_sorted[1] - idx_sorted[0]
                d2 = idx_sorted[2] - idx_sorted[1]
                d3 = 6 - (idx_sorted[2] - idx_sorted[0])
                dists = sorted([d1, d2, d3])
                if dists ==[1, 1, 4]: return "1,2,3-triazine", None
                if dists ==[1, 2, 3]: return "1,2,4-triazine", None
                if dists ==[2, 2, 2]: return "1,3,5-triazine", None
                return None
        if size == 5 and double_bonds == 2 and _has_no_cumulated_double_bonds(mol, path):
            if o_count == 1 and n_count == 0 and s_count == 0: return "furan", None
            if s_count == 1 and n_count == 0 and o_count == 0: return "thiophene", None
            if n_count == 1 and o_count == 0 and s_count == 0: return "pyrrole", None
            if n_count == 2 and o_count == 0 and s_count == 0:
                n_indices =[i for i in path if mol.atoms[i].symbol == "N"]
                dist = min(abs(path.index(n_indices[0]) - path.index(n_indices[1])), 5 - abs(path.index(n_indices[0]) - path.index(n_indices[1])))
                if dist == 1: return "pyrazole", None
                return "imidazole", None
            if n_count == 3 and o_count == 0 and s_count == 0:
                n_indices =[i for i in path if mol.atoms[i].symbol == "N"]
                idx_sorted = sorted([path.index(i) for i in n_indices])
                d1 = idx_sorted[1] - idx_sorted[0]
                d2 = idx_sorted[2] - idx_sorted[1]
                d3 = 5 - (idx_sorted[2] - idx_sorted[0])
                dists = sorted([d1, d2, d3])
                if dists ==[1, 1, 3]: return "1,2,3-triazole", None
                if dists ==[1, 2, 2]: return "1,2,4-triazole", None
                return None
            if n_count == 4 and o_count == 0 and s_count == 0:
                return "tetrazole", None
            if n_count == 1 and s_count == 1 and o_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                s_idx = next(i for i in path if mol.atoms[i].symbol == "S")
                dist = min(abs(path.index(n_idx) - path.index(s_idx)), 5 - abs(path.index(n_idx) - path.index(s_idx)))
                if dist == 1: return "isothiazole", None
                return "thiazole", None
            if n_count == 1 and o_count == 1 and s_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                o_idx = next(i for i in path if mol.atoms[i].symbol == "O")
                dist = min(abs(path.index(n_idx) - path.index(o_idx)), 5 - abs(path.index(n_idx) - path.index(o_idx)))
                if dist == 1: return "isoxazole", None
                return "oxazole", None

    if _matches_any_retained_signature(("1,2,3,4-tetrahydronaphthalene",), sig, symbols, deg3_nodes, mol):
        if len(deg3_nodes) == 2 and mol.get_bond(deg3_nodes[0], deg3_nodes[1]) is not None and n_count == 0 and o_count == 0 and s_count == 0:
            fused = deg3_nodes
            ring_a = []
            ring_b =[]
            for start_n in mol.get_neighbors(fused[0]):
                if start_n in path_set and start_n != fused[1]:
                    ring_atoms = [fused[0]]
                    cur = start_n
                    visited_local = {fused[0], cur}
                    ring_atoms.append(cur)
                    while cur != fused[1]:
                        nxt = next((x for x in mol.get_neighbors(cur) if x in path_set and x not in visited_local), None)
                        if nxt is None: break
                        ring_atoms.append(nxt)
                        visited_local.add(nxt)
                        cur = nxt
                        if len(ring_atoms) > 6: break
                    if len(ring_atoms) == 6:
                        if not ring_a: ring_a = ring_atoms
                        elif not ring_b: ring_b = ring_atoms
            
            def count_db(atoms):
                seen2 = set()
                cnt = 0
                for i in range(len(atoms)):
                    a, b = atoms[i], atoms[(i+1) % len(atoms)]
                    bd = mol.get_bond(a, b)
                    if bd and bd.idx not in seen2:
                        seen2.add(bd.idx)
                        if bd.order == 2: cnt += 1
                return cnt
            
            if len(ring_a) == 6 and len(ring_b) == 6:
                db_a = count_db(ring_a)
                db_b = count_db(ring_b)
                # True 1,2,3,4-tetrahydronaphthalene: one ring is benzene (3 db), the other is fully saturated (0 db)
                if (db_a == 3 and db_b == 0 and _has_no_cumulated_double_bonds(mol, ring_a)) or (db_a == 0 and db_b == 3 and _has_no_cumulated_double_bonds(mol, ring_b)):
                    # Need locant_map so locants 5,6,7,8,8a,1,2,3,4,4a are assigned correctly
                    # Saturated ring carries locants 1,2,3,4,4a,8a; benzene ring carries 4a,5,6,7,8,8a
                    sat_ring = ring_a if db_a == 0 else ring_b
                    aromatic_ring = ring_b if db_a == 0 else ring_a
                    fused_atoms = set(deg3_nodes)
                    sat_non_fused =[a for a in sat_ring if a not in fused_atoms]
                    if len(sat_non_fused) == 4:
                        # Build locant map: start at a fused atom that's adjacent to a saturated non-fused atom
                        # to assign locant 4a. Pick the fused atom whose neighbor in sat ring is "first".
                        for start_fused in fused_atoms:
                            sat_neighbors_of_start =[n for n in mol.get_neighbors(start_fused) if n in sat_ring and n not in fused_atoms]
                            if not sat_neighbors_of_start:
                                continue
                            other_fused = next(f for f in fused_atoms if f != start_fused)
                            # Walk through sat_ring: start_fused (=8a) - non_fused1 (=1) - non_fused2 (=2) - non_fused3 (=3) - non_fused4 (=4) - other_fused (=4a)
                            for first_nf in sat_neighbors_of_start:
                                walk = [start_fused, first_nf]
                                visited = {start_fused, first_nf}
                                cur = first_nf
                                while cur != other_fused:
                                    nxt = next((x for x in mol.get_neighbors(cur) if x in sat_ring and x not in visited), None)
                                    if nxt is None: break
                                    walk.append(nxt)
                                    visited.add(nxt)
                                    cur = nxt
                                    if len(walk) > 6: break
                                if len(walk) == 6 and walk[-1] == other_fused:
                                    # Now walk through aromatic ring
                                    arom_walk = [other_fused]
                                    arom_visited = {start_fused, other_fused}
                                    cur = other_fused
                                    while True:
                                        nxt = next((x for x in mol.get_neighbors(cur) if x in aromatic_ring and x not in arom_visited), None)
                                        if nxt is None: break
                                        arom_walk.append(nxt)
                                        arom_visited.add(nxt)
                                        cur = nxt
                                        if len(arom_walk) > 5: break
                                    if len(arom_walk) == 5:
                                        # Locants: start_fused=8a, walk[1]=1, walk[2]=2, walk[3]=3, walk[4]=4, other_fused=4a, arom_walk[1]=5, arom_walk[2]=6, arom_walk[3]=7, arom_walk[4]=8
                                        locant_map = {
                                            walk[0]: '8a', walk[1]: '1', walk[2]: '2', walk[3]: '3', walk[4]: '4', walk[5]: '4a',
                                            arom_walk[1]: '5', arom_walk[2]: '6', arom_walk[3]: '7', arom_walk[4]: '8'
                                        }
                                        return "1,2,3,4-tetrahydronaphthalene",[locant_map]
                    return "1,2,3,4-tetrahydronaphthalene", None

    if _matches_any_retained_signature(("anthracene", "phenanthrene"), sig, symbols, deg3_nodes, mol):
        deg3_edges = sum(1 for u in deg3_nodes for v in mol.get_neighbors(u) if v in deg3_nodes and u < v)
        if _matches_retained_signature("anthracene", sig, symbols, deg3_nodes, mol, deg3_edges=deg3_edges):
            return "anthracene", None
        if _matches_retained_signature("phenanthrene", sig, symbols, deg3_nodes, mol, deg3_edges=deg3_edges):
            return "phenanthrene", None
        
    if _matches_any_retained_signature(("adamantane",), sig, symbols, deg3_nodes, mol):
        if n_count == 0 and o_count == 0 and s_count == 0:
            maps =[]
            for n1 in deg3_nodes:
                neighbors_1 =[n for n in mol.get_neighbors(n1) if n in path_set]
                for n2, n8, n9 in itertools.permutations(neighbors_1):
                    n3 = next((n for n in mol.get_neighbors(n2) if n in path_set and n != n1), None)
                    n7 = next((n for n in mol.get_neighbors(n8) if n in path_set and n != n1), None)
                    n5 = next((n for n in mol.get_neighbors(n9) if n in path_set and n != n1), None)
                    if not (n3 and n7 and n5): continue
                    n10 = next((n for n in mol.get_neighbors(n3) if n in path_set and n in mol.get_neighbors(n7)), None)
                    if not n10: continue
                    n4 = next((n for n in mol.get_neighbors(n3) if n in path_set and n in mol.get_neighbors(n5) and n != n10), None)
                    n6 = next((n for n in mol.get_neighbors(n5) if n in path_set and n in mol.get_neighbors(n7) and n != n10), None)
                    if not (n4 and n6): continue
                    locant_map = {n1: '1', n2: '2', n3: '3', n4: '4', n5: '5', n6: '6', n7: '7', n8: '8', n9: '9', n10: '10'}
                    maps.append(locant_map)
            if maps:
                return "adamantane", maps
            return None

    if _matches_any_retained_signature(("cubane",), sig, symbols, deg3_nodes, mol):
        if n_count == 0 and o_count == 0 and s_count == 0:
            maps = []
            for n1 in path:
                neighbors_1 =[n for n in mol.get_neighbors(n1) if n in path_set]
                for n2, n6, n8 in itertools.permutations(neighbors_1):
                    neighbors_2 =[n for n in mol.get_neighbors(n2) if n in path_set and n != n1]
                    neighbors_6 =[n for n in mol.get_neighbors(n6) if n in path_set and n != n1]
                    neighbors_8 =[n for n in mol.get_neighbors(n8) if n in path_set and n != n1]
                    n5 = next((n for n in neighbors_2 if n in neighbors_6), None)
                    n3 = next((n for n in neighbors_2 if n in neighbors_8), None)
                    n7 = next((n for n in neighbors_6 if n in neighbors_8), None)
                    if not (n5 and n3 and n7): continue
                    neighbors_3 =[n for n in mol.get_neighbors(n3) if n in path_set and n not in (n2, n8)]
                    if not neighbors_3: continue
                    n4 = neighbors_3[0]
                    locant_map = {n1: '1', n2: '2', n3: '3', n4: '4', n5: '5', n6: '6', n7: '7', n8: '8'}
                    maps.append(locant_map)
            if maps:
                return "cubane", maps
            return None

    if _matches_retained_signature("gonane", sig, symbols, deg3_nodes, mol):
        return "gonane", None
    if _matches_retained_signature("pyrene", sig, symbols, deg3_nodes, mol):
        return "pyrene", None
    for spec in RULES.retained.fused_polycycle_specs:
        if not spec.get("match_by_signature"):
            continue
        name = spec["name"]
        if _matches_retained_signature(name, sig, symbols, deg3_nodes, mol):
            return name, None
        
    return None


def _match_data_monocycle_retained(
    mol: Molecule,
    path: list[int],
    size: int,
    total_bonds: int,
    double_bonds: int,
    symbols: list[str],
) -> str | None:
    if total_bonds != size:
        return None
    for spec in RULES.retained.monocycle_specs:
        if size != spec["size"] or double_bonds != spec["double_bonds"]:
            continue
        if spec.get("no_cumulated_double_bonds") and not _has_no_cumulated_double_bonds(mol, path):
            continue
        if not _symbol_counts_match(symbols, spec.get("symbols", {})):
            continue
        expected_distances = spec.get("hetero_distance_multiset")
        if expected_distances is not None and _hetero_distance_multiset(mol, path) != sorted(expected_distances):
            continue
        expected_gaps = spec.get("hetero_gap_multiset")
        if expected_gaps is not None and _hetero_gap_multiset(mol, path) != sorted(expected_gaps):
            continue
        return spec["name"]
    return None


def _retained_fused_spec(name: str) -> dict | None:
    return next((spec for spec in RULES.retained.fused_polycycle_specs if spec["name"] == name), None)


def _retained_locants(name: str) -> list[str]:
    spec = _retained_fused_spec(name)
    if spec is None:
        return []
    return list(spec.get("locants", []))


def _matches_any_retained_signature(
    names: tuple[str, ...],
    sig: tuple[int, int, int, tuple[int, ...]],
    symbols: list[str],
    deg3_nodes: list[int],
    mol: Molecule,
) -> bool:
    return any(_matches_retained_signature(name, sig, symbols, deg3_nodes, mol) for name in names)


def _matches_retained_signature(
    name: str,
    sig: tuple[int, int, int, tuple[int, ...]],
    symbols: list[str],
    deg3_nodes: list[int],
    mol: Molecule,
    *,
    deg3_edges: int | None = None,
) -> bool:
    spec = _retained_fused_spec(name)
    if spec is None:
        return False
    signature = spec["signature"]
    expected_sig = (
        int(signature["size"]),
        int(signature["total_bonds"]),
        int(signature["double_bonds"]),
        _degree_counts_tuple(signature["degree_counts"]),
    )
    if sig != expected_sig:
        return False
    if not _symbol_counts_match(symbols, spec.get("symbols", {})):
        return False
    expected_fused_edges = spec.get("fused_degree3_edges")
    if expected_fused_edges is not None:
        actual = sum(1 for u in deg3_nodes for v in mol.get_neighbors(u) if v in deg3_nodes and u < v)
        if actual != int(expected_fused_edges):
            return False
    expected_deg3_edges = spec.get("deg3_edges")
    if expected_deg3_edges is not None:
        actual = deg3_edges
        if actual is None:
            actual = sum(1 for u in deg3_nodes for v in mol.get_neighbors(u) if v in deg3_nodes and u < v)
        if actual != int(expected_deg3_edges):
            return False
    return True


def _degree_counts_tuple(counts: dict[str, int]) -> tuple[int, ...]:
    values = []
    for degree, count in counts.items():
        values.extend([int(degree)] * int(count))
    return tuple(sorted(values))


def _symbol_counts_match(symbols: list[str], expected: dict[str, int]) -> bool:
    for symbol in ("N", "O", "S"):
        if symbols.count(symbol) != int(expected.get(symbol, 0)):
            return False
    hetero_count = sum(int(value) for value in expected.values())
    return symbols.count("C") == len(symbols) - hetero_count


def _hetero_distance_multiset(mol: Molecule, path: list[int]) -> list[int]:
    hetero_indices = [idx for idx, atom_idx in enumerate(path) if mol.atoms[atom_idx].symbol in {"N", "O", "S"}]
    if len(hetero_indices) < 2:
        return []
    size = len(path)
    distances = []
    for left, right in itertools.combinations(hetero_indices, 2):
        distance = abs(left - right)
        distances.append(min(distance, size - distance))
    return sorted(distances)


def _hetero_gap_multiset(mol: Molecule, path: list[int]) -> list[int]:
    hetero_indices = sorted(idx for idx, atom_idx in enumerate(path) if mol.atoms[atom_idx].symbol in {"N", "O", "S"})
    if len(hetero_indices) < 3:
        return []
    size = len(path)
    gaps = [
        hetero_indices[(idx + 1) % len(hetero_indices)] - hetero_indices[idx]
        if idx + 1 < len(hetero_indices)
        else size - hetero_indices[idx] + hetero_indices[0]
        for idx in range(len(hetero_indices))
    ]
    return sorted(gaps)
