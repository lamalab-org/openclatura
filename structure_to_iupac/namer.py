# structure-to-iupac/api.py

import re
from rdkit import Chem
from .molecule import Molecule
from .perception import perceive_groups, PerceivedGroup
from .chains import find_all_carbon_paths, find_ring_systems, get_cyclic_atoms
from .parent_selection import select_principal_parent
from .assembler import assemble_name, AssemblyParts, PrincipalGroupItem, SubstituentItem, UnsaturationItem
from .rules import suffixes, substituents, stems, retained, multipliers


def parse_locant(l):
    s = str(l)
    match = re.match(r"^(\d+)([a-zA-Z]*)$", s.split("(")[0])
    if match:
        return (1, float(match.group(1)), match.group(2))
    if any(c.isdigit() for c in s):
        nums = re.findall(r"\d+", s)
        return (1, float(nums[0]) if nums else 0.0, s)
    return (2, 0.0, s)


def is_fully_enclosed(s: str) -> bool:
    if not s.startswith("(") or not s.endswith(")"):
        return False
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth == 0 and i < len(s) - 1:
            return False
    return depth == 0


def _format_multiplier(name: str, count: int, safe_enclose: bool = False) -> str:
    is_complex = "(" in name or name[0].isdigit() or "-" in name or " " in name
    if count == 1:
        if (safe_enclose or is_complex) and not is_fully_enclosed(name):
            return f"({name})"
        return name
    mult = multipliers.complex_(count) if is_complex else multipliers.basic(count)
    if is_complex and not is_fully_enclosed(name):
        return f"{mult}({name})"
    return f"{mult}{name}"


def _get_atom_locants(oriented_path: list[int], target_indices: set[int]) -> list[int]:
    return sorted([i + 1 for i, atom_idx in enumerate(oriented_path) if atom_idx in target_indices])


def _get_bond_locants(
    mol: Molecule, oriented_path: list[int], is_bicycle: bool, is_spiro: bool, is_polycycle: bool
) -> tuple[list[int], list[int]]:
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
                    elif (
                        min_loc == 1 and max_loc == len(oriented_path) and not (is_bicycle or is_spiro or is_polycycle)
                    ):
                        locant_val = max_loc
                    else:
                        locant_val = min_loc

                    if bond.order == 2:
                        db_locants.append(locant_val)
                    elif bond.order == 3:
                        tb_locants.append(locant_val)

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
    retained_name: str = None,
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

            het_eval = tuple(
                _get_atom_locants(oriented_path, set(het_by_priority[prio])) for prio in sorted(het_by_priority.keys())
            )

            if retained_name in {
                "pyrrole",
                "imidazole",
                "pyrazole",
                "1,2,3-triazole",
                "1,2,4-triazole",
                "indole",
                "isoindole",
                "indazole",
                "benzimidazole",
                "purine",
                "indene",
                "fluorene",
            }:
                sat_atoms =[]
                for a_idx in oriented_path:
                    ring_bonds =[mol.get_bond(a_idx, n) for n in mol.get_neighbors(a_idx) if n in oriented_path]
                    if sum(b.order for b in ring_bonds) == 2:
                        sat_atoms.append(a_idx)
                sat_eval = tuple(_get_atom_locants(oriented_path, set(sat_atoms)))
            else:
                sat_eval = ()

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
                s = re.sub(r"^[\(\[\{\)]+", "", s)
                prefix_pattern = r"^((?:(?:[0-9]+[a-z]*|[nospmc]\'*)(?:,(?:[0-9]+[a-z]*|[nospmc]\'*))*|[ezrs]+|sec|tert|t|s|d|l|m|o|p|alpha|beta|gamma))([-)]+)"
                while True:
                    prev = s
                    match = re.match(prefix_pattern, s)
                    if match:
                        s = s[match.end() :]
                        s = re.sub(r"^[\(\[\{\)]+", "", s)
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


def read_smiles(smiles: str) -> Molecule:
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
        if rdmol:
            rdmol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(rdmol)
    else:
        try:
            Chem.Kekulize(rdmol, clearAromaticFlags=True)
        except Exception:
            pass

    mol = Molecule()
    if rdmol is None:
        return mol

    Chem.AssignStereochemistry(rdmol, force=True, cleanIt=True)
    chiral_centers = dict(Chem.FindMolChiralCenters(rdmol, includeUnassigned=False))

    for atom in rdmol.GetAtoms():
        stereo = chiral_centers.get(atom.GetIdx())
        if stereo and atom.GetSymbol() == "S" and atom.GetTotalDegree() == 3:
            stereo = "R" if stereo == "S" else "S"
        mol.add_atom(symbol=atom.GetSymbol(), idx=atom.GetIdx(), charge=atom.GetFormalCharge(), stereo=stereo)

    for bond in rdmol.GetBonds():
        stereo = None
        st = bond.GetStereo()
        if st == Chem.rdchem.BondStereo.STEREOE:
            stereo = "E"
        elif st == Chem.rdchem.BondStereo.STEREOZ:
            stereo = "Z"

        in_small_ring = any(bond.IsInRingSize(i) for i in range(3, 8))

        mol.add_bond(
            u=bond.GetBeginAtomIdx(),
            v=bond.GetEndAtomIdx(),
            order=int(bond.GetBondTypeAsDouble()),
            stereo=stereo,
            in_small_ring=in_small_ring,
        )
    return mol


def get_connected_components(mol: Molecule) -> list[set[int]]:
    visited = set()
    components =[]
    for atom in mol:
        if atom.idx not in visited:
            comp = set()
            q = [atom.idx]
            while q:
                curr = q.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    q.extend(mol.get_neighbors(curr))
            components.append(comp)
    return components


def _emit_bond_stereo(mol, parts, numbered_path, get_loc, exclude_atoms=None, upstream_atom=None):
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

            if not any(f[0] == min_loc and f[1] in["E", "Z"] for f in parts.stereo_features):
                parts.stereo_features.append((min_loc, bond.stereo))


def name_subgraph(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int = None) -> str:
    start_atom = mol.atoms[start_idx]

    cyclic_atoms_global = get_cyclic_atoms(mol, exclude_atoms)

    if not start_atom.is_carbon and start_idx not in cyclic_atoms_global:
        if start_atom.symbol == "O":
            is_double = False
            if upstream_atom is not None:
                bond = mol.get_bond(start_idx, upstream_atom)
                if bond and bond.order == 2:
                    is_double = True

            next_atoms =[n for n in mol.get_neighbors(start_idx) if n not in exclude_atoms and n != upstream_atom]
            if not next_atoms:
                if is_double:
                    return "oxo"
                if start_atom.charge == -1:
                    return "oxido"
                return "hydroxy"
            nxt = next_atoms[0]

            s_oxygens =[
                o for o in mol.get_neighbors(nxt) if mol.atoms[o].symbol == "O" and mol.get_bond(nxt, o).order == 2
            ]
            if mol.atoms[nxt].symbol == "S" and len(s_oxygens) >= 1:
                r_group = next((n for n in mol.get_neighbors(nxt) if n != start_idx and n not in s_oxygens), None)
                if r_group is not None:
                    branch = name_subgraph(
                        mol, r_group, exclude_atoms | {start_idx, nxt} | set(s_oxygens), upstream_atom=nxt
                    )
                    if branch:
                        if branch.startswith("(") and branch.endswith(")"):
                            branch = branch[1:-1]
                        suffix = "sulfonyl" if len(s_oxygens) == 2 else "sulfinyl"
                        stereo_prefix = f"({mol.atoms[nxt].stereo})-" if mol.atoms[nxt].stereo else ""
                        return f"({stereo_prefix}{branch}{suffix}oxy)"
                return "sulfooxy"

            c_oxygens =[
                o for o in mol.get_neighbors(nxt) if mol.atoms[o].symbol == "O" and mol.get_bond(nxt, o).order == 2
            ]
            c_sulfurs =[
                s for s in mol.get_neighbors(nxt) if mol.atoms[s].symbol == "S" and mol.get_bond(nxt, s).order == 2
            ]
            if mol.atoms[nxt].is_carbon and len(c_oxygens) == 1:
                r_group = next(
                    (
                        n
                        for n in mol.get_neighbors(nxt)
                        if n != start_idx and n not in c_oxygens and mol.atoms[n].is_carbon
                    ),
                    None,
                )
                if r_group is not None:
                    branch = name_subgraph(
                        mol, r_group, exclude_atoms | {start_idx, nxt} | set(c_oxygens), upstream_atom=nxt
                    )
                    if branch:
                        if branch.startswith("(") and branch.endswith(")"):
                            branch = branch[1:-1]
                        return f"({branch}carbonyloxy)"
                hetero_group = next((n for n in mol.get_neighbors(nxt) if n != start_idx and n not in c_oxygens), None)
                if hetero_group is not None:
                    branch = name_subgraph(
                        mol, hetero_group, exclude_atoms | {start_idx, nxt} | set(c_oxygens), upstream_atom=nxt
                    )
                    if branch:
                        if branch.startswith("(") and branch.endswith(")"):
                            branch = branch[1:-1]
                        return f"({branch}carbonyloxy)"
                return "formyloxy"
            elif mol.atoms[nxt].is_carbon and len(c_sulfurs) == 1:
                r_group = next(
                    (
                        n
                        for n in mol.get_neighbors(nxt)
                        if n != start_idx and n not in c_sulfurs and mol.atoms[n].is_carbon
                    ),
                    None,
                )
                if r_group is not None:
                    branch = name_subgraph(
                        mol, r_group, exclude_atoms | {start_idx, nxt} | set(c_sulfurs), upstream_atom=nxt
                    )
                    if branch:
                        if branch.startswith("(") and branch.endswith(")"):
                            branch = branch[1:-1]
                        return f"({branch}carbonothioyloxy)"
                hetero_group = next((n for n in mol.get_neighbors(nxt) if n != start_idx and n not in c_sulfurs), None)
                if hetero_group is not None:
                    branch = name_subgraph(
                        mol, hetero_group, exclude_atoms | {start_idx, nxt} | set(c_sulfurs), upstream_atom=nxt
                    )
                    if branch:
                        if branch.startswith("(") and branch.endswith(")"):
                            branch = branch[1:-1]
                        return f"({branch}carbonothioyloxy)"
                return "methanethioyloxy"
            elif mol.atoms[nxt].symbol == "O":
                r_group = next((n for n in mol.get_neighbors(nxt) if n != start_idx), None)
                if r_group is not None:
                    branch = name_subgraph(mol, r_group, exclude_atoms | {start_idx, nxt}, upstream_atom=nxt)
                    if branch:
                        if branch.startswith("(") and branch.endswith(")"):
                            branch = branch[1:-1]
                        return f"({branch}peroxy)"
                return "hydroperoxy"

            branch = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
            if branch == "methyl":
                return "methoxy"
            if branch == "ethyl":
                return "ethoxy"
            if branch == "propyl":
                return "propoxy"
            if branch == "isopropyl":
                return "isopropoxy"
            if branch == "butyl":
                return "butoxy"
            if branch == "isobutyl":
                return "isobutoxy"
            if branch == "sec-butyl":
                return "sec-butoxy"
            if branch == "tert-butyl":
                return "tert-butoxy"
            if branch == "phenyl":
                return "phenoxy"
            if branch:
                if branch.startswith("(") and branch.endswith(")"):
                    branch = branch[1:-1]
                is_complex = "(" in branch or branch[0].isdigit() or "-" in branch or " " in branch
                if is_complex:
                    return f"(({branch})oxy)"
                return f"({branch}oxy)"
            return "hydroxy"

        if start_atom.symbol == "N":
            is_double = False
            is_triple = False
            if upstream_atom is not None:
                bond = mol.get_bond(start_idx, upstream_atom)
                if bond and bond.order == 2:
                    is_double = True
                elif bond and bond.order == 3:
                    is_triple = True

            next_atoms =[n for n in mol.get_neighbors(start_idx) if n not in exclude_atoms and n != upstream_atom]
            if not next_atoms:
                if is_double:
                    return "imino"
                if is_triple:
                    return "nitrilo"
                return "amino"

            branches =[]
            for nxt in next_atoms:
                c_oxygens =[
                    o for o in mol.get_neighbors(nxt) if mol.atoms[o].symbol == "O" and mol.get_bond(nxt, o).order == 2
                ]
                c_sulfurs =[
                    s for s in mol.get_neighbors(nxt) if mol.atoms[s].symbol == "S" and mol.get_bond(nxt, s).order == 2
                ]
                if mol.atoms[nxt].is_carbon and len(c_oxygens) == 1:
                    r_group = next(
                        (
                            n
                            for n in mol.get_neighbors(nxt)
                            if n != start_idx and n not in c_oxygens and mol.atoms[n].is_carbon
                        ),
                        None,
                    )
                    if r_group is not None:
                        branch = name_subgraph(
                            mol, r_group, exclude_atoms | {start_idx, nxt} | set(c_oxygens), upstream_atom=nxt
                        )
                        if branch:
                            if branch.startswith("(") and branch.endswith(")"):
                                branch = branch[1:-1]
                            branches.append(f"{branch}carbonyl")
                        else:
                            branches.append("formyl")
                    else:
                        hetero_group = next(
                            (n for n in mol.get_neighbors(nxt) if n != start_idx and n not in c_oxygens), None
                        )
                        if hetero_group is not None:
                            branch = name_subgraph(
                                mol, hetero_group, exclude_atoms | {start_idx, nxt} | set(c_oxygens), upstream_atom=nxt
                            )
                            if branch:
                                if branch.startswith("(") and branch.endswith(")"):
                                    branch = branch[1:-1]
                                if branch.endswith("amino"):
                                    branches.append(f"({branch[:-5]}carbamoyl)")
                                else:
                                    branches.append(f"{branch}carbonyl")
                            else:
                                branches.append("formyl")
                        else:
                            branches.append("formyl")
                elif mol.atoms[nxt].is_carbon and len(c_sulfurs) == 1:
                    r_group = next(
                        (
                            n
                            for n in mol.get_neighbors(nxt)
                            if n != start_idx and n not in c_sulfurs and mol.atoms[n].is_carbon
                        ),
                        None,
                    )
                    if r_group is not None:
                        branch = name_subgraph(
                            mol, r_group, exclude_atoms | {start_idx, nxt} | set(c_sulfurs), upstream_atom=nxt
                        )
                        if branch:
                            if branch.startswith("(") and branch.endswith(")"):
                                branch = branch[1:-1]
                            branches.append(f"{branch}carbonothioyl")
                        else:
                            branches.append("methanethioyl")
                    else:
                        hetero_group = next(
                            (n for n in mol.get_neighbors(nxt) if n != start_idx and n not in c_sulfurs), None
                        )
                        if hetero_group is not None:
                            branch = name_subgraph(
                                mol, hetero_group, exclude_atoms | {start_idx, nxt} | set(c_sulfurs), upstream_atom=nxt
                            )
                            if branch:
                                if branch.startswith("(") and branch.endswith(")"):
                                    branch = branch[1:-1]
                                if branch.endswith("amino"):
                                    branches.append(f"({branch[:-5]}carbamothioyl)")
                                else:
                                    branches.append(f"{branch}carbonothioyl")
                            else:
                                branches.append("methanethioyl")
                        else:
                            branches.append("methanethioyl")
                else:
                    s_oxygens =[
                        o
                        for o in mol.get_neighbors(nxt)
                        if mol.atoms[o].symbol == "O" and mol.get_bond(nxt, o).order == 2
                    ]
                    if mol.atoms[nxt].symbol == "S" and len(s_oxygens) >= 1:
                        r_group = next(
                            (n for n in mol.get_neighbors(nxt) if n != start_idx and n not in s_oxygens), None
                        )
                        if r_group is not None:
                            branch = name_subgraph(
                                mol, r_group, exclude_atoms | {start_idx, nxt} | set(s_oxygens), upstream_atom=nxt
                            )
                            if branch:
                                if branch.startswith("(") and branch.endswith(")"):
                                    branch = branch[1:-1]
                                suffix = "sulfonyl" if len(s_oxygens) == 2 else "sulfinyl"
                                stereo_prefix = f"({mol.atoms[nxt].stereo})-" if mol.atoms[nxt].stereo else ""
                                branches.append(f"{stereo_prefix}{branch}{suffix}")
                            else:
                                branches.append("sulfo")
                        else:
                            branches.append("sulfo")
                    else:
                        br = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
                        if br:
                            branches.append(br)

            if is_double:
                if len(branches) == 1:
                    b = branches[0]
                    if b.startswith("(") and b.endswith(")"):
                        b = b[1:-1]
                    is_complex = "(" in b or b[0].isdigit() or "-" in b or " " in b
                    if is_complex:
                        return f"(({b})imino)"
                    return f"({b}imino)"
                return "imino"

            counts = {}
            for b in branches:
                counts[b] = counts.get(b, 0) + 1

            if len(counts) == 1 and list(counts.values())[0] == 1:
                b = branches[0]
                if b.startswith("(") and b.endswith(")"):
                    b = b[1:-1]
                if b.endswith("carbonyl") or b.endswith("sulfonyl") or b.endswith("sulfinyl") or b.endswith("carbonothioyl"):
                    return f"{b}amino"
                is_complex = "(" in b or b[0].isdigit() or "-" in b or " " in b
                if is_complex:
                    return f"(({b})amino)"
                return f"({b}amino)"

            prefix_parts =[]
            safe = len(counts) > 1 or any("(" in b or b[0].isdigit() or "-" in b or " " in b for b in counts)
            for b, c in sorted(counts.items()):
                prefix_parts.append(_format_multiplier(b, c, safe_enclose=safe))
            prefix = "".join(prefix_parts)
            return f"({prefix}amino)"

        if start_atom.symbol == "S":
            is_double = False
            if upstream_atom is not None:
                bond = mol.get_bond(start_idx, upstream_atom)
                if bond and bond.order == 2:
                    is_double = True

            s_oxygens =[
                o
                for o in mol.get_neighbors(start_idx)
                if mol.atoms[o].symbol == "O" and mol.get_bond(start_idx, o).order == 2
            ]
            s_nitrogens =[
                n
                for n in mol.get_neighbors(start_idx)
                if mol.atoms[n].symbol == "N" and mol.get_bond(start_idx, n).order == 2
            ]
            next_atoms =[
                n
                for n in mol.get_neighbors(start_idx)
                if n not in exclude_atoms and n != upstream_atom and n not in s_oxygens and n not in s_nitrogens
            ]

            stereo_prefix = f"({start_atom.stereo})-" if start_atom.stereo else ""

            if len(s_oxygens) == 1 and len(s_nitrogens) == 1:
                suffix = "sulfonimidoyl"
                if is_double:
                    suffix += "idene"
                if not next_atoms:
                    return f"{stereo_prefix}{suffix}"

                if len(next_atoms) == 1:
                    nxt = next_atoms[0]
                    branch = name_subgraph(
                        mol,
                        nxt,
                        exclude_atoms | {start_idx} | set(s_oxygens) | set(s_nitrogens),
                        upstream_atom=start_idx,
                    )
                    if branch:
                        if branch.startswith("(") and branch.endswith(")"):
                            branch = branch[1:-1]
                        return f"({stereo_prefix}{branch}{suffix})"
                    return f"{stereo_prefix}{suffix}"
                else:
                    branches =[]
                    for nxt in next_atoms:
                        br = name_subgraph(
                            mol,
                            nxt,
                            exclude_atoms | {start_idx} | set(s_oxygens) | set(s_nitrogens),
                            upstream_atom=start_idx,
                        )
                        if br:
                            branches.append(br)
                    for _ in s_oxygens:
                        branches.append("oxo")
                    for _ in s_nitrogens:
                        branches.append("imino")
                    val = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))
                    base_suffix = "sulfanylidene" if is_double else "sulfanyl"
                    counts = {}
                    for b in branches:
                        counts[b] = counts.get(b, 0) + 1
                    prefix_parts =[]
                    safe = len(counts) > 1 or any("(" in b or b[0].isdigit() or "-" in b or " " in b for b in counts)
                    for b, c in sorted(counts.items()):
                        prefix_parts.append(_format_multiplier(b, c, safe_enclose=safe))

                    prefix = "".join(prefix_parts)
                    return f"({stereo_prefix}{prefix}-lambda^{val}-{base_suffix})"

            if len(s_oxygens) >= 1:
                suffix = "sulfonyl" if len(s_oxygens) == 2 else "sulfinyl"
                if is_double:
                    suffix += "idene"
                if not next_atoms:
                    return f"{stereo_prefix}{suffix}"

                if len(next_atoms) == 1:
                    nxt = next_atoms[0]
                    branch = name_subgraph(
                        mol, nxt, exclude_atoms | {start_idx} | set(s_oxygens), upstream_atom=start_idx
                    )
                    if branch:
                        if branch.startswith("(") and branch.endswith(")"):
                            branch = branch[1:-1]
                        return f"({stereo_prefix}{branch}{suffix})"
                    return f"{stereo_prefix}{suffix}"
                else:
                    branches =[]
                    for nxt in next_atoms:
                        br = name_subgraph(
                            mol, nxt, exclude_atoms | {start_idx} | set(s_oxygens), upstream_atom=start_idx
                        )
                        if br:
                            branches.append(br)
                    for _ in s_oxygens:
                        branches.append("oxo")
                    val = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))
                    base_suffix = "sulfanylidene" if is_double else "sulfanyl"
                    counts = {}
                    for b in branches:
                        counts[b] = counts.get(b, 0) + 1
                    prefix_parts =[]
                    safe = len(counts) > 1 or any("(" in b or b[0].isdigit() or "-" in b or " " in b for b in counts)
                    for b, c in sorted(counts.items()):
                        prefix_parts.append(_format_multiplier(b, c, safe_enclose=safe))

                    prefix = "".join(prefix_parts)
                    return f"({stereo_prefix}{prefix}-lambda^{val}-{base_suffix})"

            if not next_atoms:
                if is_double:
                    return "thioxo"
                return f"{stereo_prefix}sulfanyl"

            if len(next_atoms) == 1:
                nxt = next_atoms[0]
                branch = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
                if branch == "methyl":
                    return f"({stereo_prefix}methylsulfanyl)"
                if branch == "ethyl":
                    return f"({stereo_prefix}ethylsulfanyl)"
                if branch == "propyl":
                    return f"({stereo_prefix}propylsulfanyl)"
                if branch == "isopropyl":
                    return f"({stereo_prefix}isopropylsulfanyl)"
                if branch == "butyl":
                    return f"({stereo_prefix}butylsulfanyl)"
                if branch == "isobutyl":
                    return f"({stereo_prefix}isobutylsulfanyl)"
                if branch == "sec-butyl":
                    return f"({stereo_prefix}sec-butylsulfanyl)"
                if branch == "tert-butyl":
                    return f"({stereo_prefix}tert-butylsulfanyl)"
                if branch == "phenyl":
                    return f"({stereo_prefix}phenylsulfanyl)"
                if branch:
                    if branch.startswith("(") and branch.endswith(")"):
                        branch = branch[1:-1]
                    is_complex = "(" in branch or branch[0].isdigit() or "-" in branch or " " in branch
                    if is_double:
                        if is_complex:
                            return f"({stereo_prefix}({branch})sulfanylidene)"
                        return f"({stereo_prefix}{branch}sulfanylidene)"
                    if is_complex:
                        return f"({stereo_prefix}({branch})sulfanyl)"
                    return f"({stereo_prefix}{branch}sulfanyl)"
                if is_double:
                    return f"{stereo_prefix}sulfanylidene"
                return f"{stereo_prefix}sulfanyl"
            else:
                branches =[]
                for nxt in next_atoms:
                    br = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
                    if br:
                        branches.append(br)

                val = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))
                base_suffix = "sulfanylidene" if is_double else "sulfanyl"

                counts = {}
                for b in branches:
                    counts[b] = counts.get(b, 0) + 1
                prefix_parts =[]
                safe = len(counts) > 1 or any("(" in b or b[0].isdigit() or "-" in b or " " in b for b in counts)
                for b, c in sorted(counts.items()):
                    prefix_parts.append(_format_multiplier(b, c, safe_enclose=safe))

                prefix = "".join(prefix_parts)
                return f"({stereo_prefix}{prefix}-lambda^{val}-{base_suffix})"

        if start_atom.symbol == "Se":
            is_double = False
            if upstream_atom is not None:
                bond = mol.get_bond(start_idx, upstream_atom)
                if bond and bond.order == 2:
                    is_double = True

            next_atoms =[n for n in mol.get_neighbors(start_idx) if n not in exclude_atoms and n != upstream_atom]

            stereo_prefix = f"({start_atom.stereo})-" if start_atom.stereo else ""

            if not next_atoms:
                if is_double:
                    return "selenoxo"
                return f"{stereo_prefix}selanyl"

            if len(next_atoms) == 1:
                nxt = next_atoms[0]
                branch = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
                if branch == "methyl":
                    return f"({stereo_prefix}methylselanyl)"
                if branch == "ethyl":
                    return f"({stereo_prefix}ethylselanyl)"
                if branch == "phenyl":
                    return f"({stereo_prefix}phenylselanyl)"
                if branch:
                    if branch.startswith("(") and branch.endswith(")"):
                        branch = branch[1:-1]
                    is_complex = "(" in branch or branch[0].isdigit() or "-" in branch or " " in branch
                    if is_double:
                        if is_complex:
                            return f"({stereo_prefix}({branch})selanylidene)"
                        return f"({stereo_prefix}{branch}selanylidene)"
                    if is_complex:
                        return f"({stereo_prefix}({branch})selanyl)"
                    return f"({stereo_prefix}{branch}selanyl)"
                if is_double:
                    return f"{stereo_prefix}selanylidene"
                return f"{stereo_prefix}selanyl"
            else:
                branches =[]
                for nxt in next_atoms:
                    br = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
                    if br:
                        branches.append(br)

                val = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))
                base_suffix = "selanylidene" if is_double else "selanyl"

                counts = {}
                for b in branches:
                    counts[b] = counts.get(b, 0) + 1
                prefix_parts =[]
                safe = len(counts) > 1 or any("(" in b or b[0].isdigit() or "-" in b or " " in b for b in counts)
                for b, c in sorted(counts.items()):
                    prefix_parts.append(_format_multiplier(b, c, safe_enclose=safe))

                prefix = "".join(prefix_parts)
                return f"({stereo_prefix}{prefix}-lambda^{val}-{base_suffix})"

        if start_atom.symbol == "P":
            is_double = False
            if upstream_atom is not None:
                bond = mol.get_bond(start_idx, upstream_atom)
                if bond and bond.order == 2:
                    is_double = True

            p_oxygens =[
                o
                for o in mol.get_neighbors(start_idx)
                if mol.atoms[o].symbol == "O" and mol.get_bond(start_idx, o).order == 2
            ]
            next_atoms =[
                n
                for n in mol.get_neighbors(start_idx)
                if n not in exclude_atoms and n != upstream_atom and n not in p_oxygens
            ]

            stereo_prefix = f"({start_atom.stereo})-" if start_atom.stereo else ""

            suffix = "phosphoryl" if len(p_oxygens) >= 1 else "phosphanyl"
            if is_double:
                suffix += "idene"

            if not next_atoms:
                return f"{stereo_prefix}{suffix}"

            branches =[]
            for nxt in next_atoms:
                br = name_subgraph(mol, nxt, exclude_atoms | {start_idx} | set(p_oxygens), upstream_atom=start_idx)
                if br:
                    branches.append(br)

            counts = {}
            for b in branches:
                counts[b] = counts.get(b, 0) + 1
            prefix_parts =[]
            safe = len(counts) > 1 or any("(" in b or b[0].isdigit() or "-" in b or " " in b for b in counts)
            for b, c in sorted(counts.items()):
                prefix_parts.append(_format_multiplier(b, c, safe_enclose=safe))

            prefix = "".join(prefix_parts)
            return f"({stereo_prefix}{prefix}{suffix})"

        if start_atom.symbol in ["Si", "B"]:
            is_double = False
            if upstream_atom is not None:
                bond = mol.get_bond(start_idx, upstream_atom)
                if bond and bond.order == 2:
                    is_double = True

            suffix = "silyl" if start_atom.symbol == "Si" else "boryl"
            if is_double:
                suffix += "idene"

            next_atoms =[n for n in mol.get_neighbors(start_idx) if n not in exclude_atoms and n != upstream_atom]
            if not next_atoms:
                return suffix
            branches =[]
            for nxt in next_atoms:
                br = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
                if br:
                    branches.append(br)

            counts = {}
            for b in branches:
                counts[b] = counts.get(b, 0) + 1
            prefix_parts =[]
            safe = len(counts) > 1 or any("(" in b or b[0].isdigit() or "-" in b or " " in b for b in counts)
            for b, c in sorted(counts.items()):
                prefix_parts.append(_format_multiplier(b, c, safe_enclose=safe))

            prefix = "".join(prefix_parts)
            return f"({prefix}{suffix})"

        if start_atom.symbol in["F", "Cl", "Br", "I"]:
            next_atoms =[n for n in mol.get_neighbors(start_idx) if n not in exclude_atoms and n != upstream_atom]
            if not next_atoms:
                return {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}[start_atom.symbol]

            suffix = {"F": "fluoranyl", "Cl": "chloranyl", "Br": "bromanyl", "I": "iodanyl"}[start_atom.symbol]
            branches =[]
            for nxt in next_atoms:
                br = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
                if br:
                    branches.append(br)

            val = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))

            counts = {}
            for b in branches:
                counts[b] = counts.get(b, 0) + 1
            prefix_parts =[]
            safe = len(counts) > 1 or any("(" in b or b[0].isdigit() or "-" in b or " " in b for b in counts)
            for b, c in sorted(counts.items()):
                prefix_parts.append(_format_multiplier(b, c, safe_enclose=safe))

            prefix = "".join(prefix_parts)
            return f"({prefix}lambda^{val}-{suffix})"

    visited = set(exclude_atoms)
    comp = set()
    q = [start_idx]
    while q:
        curr = q.pop(0)
        if curr not in visited:
            visited.add(curr)
            comp.add(curr)
            q.extend([x for x in mol.get_neighbors(curr) if x not in visited])

    sub_exclude = set(mol.atoms.keys()) - comp
    sub_perceived = perceive_groups(mol)

    for g in sub_perceived:
        if start_idx in g.atoms_involved and g.atoms_involved.issubset(comp):
            if g.key == "nitro":
                return "nitro"
            if g.key == "nitroso":
                return "nitroso"
            if g.key == "azido":
                return "azido"
            if g.key == "sulfonic_acid":
                return "sulfo"
            if g.key == "isocyano":
                return "isocyano"
            if g.key == "nitrile":
                return "cyano"
            if g.key == "carboxylic_acid":
                return "carboxy"
            if g.key == "amide":
                return "carbamoyl"
            if g.key == "acid_fluoride":
                return "fluorocarbonyl"
            if g.key == "acid_chloride":
                return "chlorocarbonyl"
            if g.key == "acid_bromide":
                return "bromocarbonyl"
            if g.key == "acid_iodide":
                return "iodocarbonyl"
            if g.key == "isothiocyanato":
                return "isothiocyanato"
            if g.key == "isocyanato":
                return "isocyanato"
            if g.key == "thiocyanato":
                return "thiocyanato"
            if g.key == "cyanato":
                return "cyanato"
            if g.key in substituents.SUBSTITUENTS:
                return substituents.get(g.key).prefix

    cyclic_atoms = get_cyclic_atoms(mol, sub_exclude)

    if start_idx in cyclic_atoms:
        ring_systems = find_ring_systems(mol, sub_exclude)
        valid_rings =[rs for rs in ring_systems if start_idx in rs.atoms]
        if not valid_rings:
            return ""
        best_paths, is_ring, is_bicycle, is_spiro, is_polycycle, xyz, polycycle_descriptor = select_principal_parent(
            mol, [], valid_rings,[start_idx]
        )

        candidate_paths = best_paths
        fixed_start_val = is_bicycle or is_spiro or is_polycycle
    else:
        valid_nodes = {n for n in comp if n not in cyclic_atoms and mol.atoms[n].is_carbon and n not in sub_exclude}
        paths =[]

        def dfs_sub(curr, path, visited_nodes):
            neighbors =[n for n in mol.get_neighbors(curr) if n in valid_nodes and n not in visited_nodes]
            if not neighbors:
                if start_idx in path:
                    paths.append(path)
                return
            for n in neighbors:
                dfs_sub(n, path + [n], visited_nodes | {n})

        endpoints =[n for n in valid_nodes if sum(1 for x in mol.get_neighbors(n) if x in valid_nodes) <= 1]
        start_nodes = endpoints if endpoints else valid_nodes
        for start in start_nodes:
            dfs_sub(start, [start], {start})

        if not paths:
            return ""
        best_paths, is_ring, is_bicycle, is_spiro, is_polycycle, xyz, polycycle_descriptor = select_principal_parent(
            mol, paths, [], [start_idx]
        )
        candidate_paths = best_paths
        fixed_start_val = False

    subst_mapping = {}

    temp_retained = retained.get_retained_ring(mol, candidate_paths[0]) if is_ring else None
    if temp_retained:
        retained_name_val, locant_maps = temp_retained
        if locant_maps is None and (is_bicycle or is_polycycle):
            retained_name_val, locant_maps = None, None
        elif any(mol.atoms[idx].symbol not in["C", "N", "O", "S"] for idx in candidate_paths[0]):
            retained_name_val, locant_maps = None, None
    else:
        retained_name_val, locant_maps = None, None

    main_set = set(candidate_paths[0])

    sub_handled_atoms = set()
    for g in sub_perceived:
        if g.attachment_carbon in main_set and not g.is_principal_candidate:
            name = substituents.get(g.key).prefix if g.key in substituents.SUBSTITUENTS else ""
            if name:
                subst_mapping.setdefault(g.attachment_carbon,[]).append(name)
                sub_handled_atoms.update(g.atoms_involved)

    for c_idx in candidate_paths[0]:
        n_subs =[
            n_idx
            for n_idx in mol.get_neighbors(c_idx)
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude
        ]

        spiro_pairs =[]
        for i in range(len(n_subs)):
            for j in range(i + 1, len(n_subs)):
                n1, n2 = n_subs[i], n_subs[j]
                visited = {c_idx}
                q = [n1]
                in_ring = False
                while q:
                    curr = q.pop(0)
                    if curr == n2:
                        in_ring = True
                        break
                    visited.add(curr)
                    for nxt in mol.get_neighbors(curr):
                        if nxt not in visited and nxt not in main_set and nxt not in sub_exclude:
                            q.append(nxt)
                if in_ring:
                    spiro_pairs.append((n1, n2))
                    break
            if spiro_pairs:
                break

        if spiro_pairs:
            n1, n2 = spiro_pairs[0]
            sub_comp = set()
            q_sub = [n1]
            visited_sub = {c_idx}
            while q_sub:
                curr = q_sub.pop(0)
                if curr not in sub_comp:
                    sub_comp.add(curr)
                    visited_sub.add(curr)
                    for nxt in mol.get_neighbors(curr):
                        if nxt not in visited_sub and nxt not in main_set and nxt not in sub_exclude:
                            q_sub.append(nxt)
            sub_comp.add(c_idx)

            sub_mol = Molecule()
            for n in sub_comp:
                atom = mol.atoms[n]
                sym = "Si" if n == c_idx else atom.symbol
                sub_mol.add_atom(symbol=sym, idx=n, charge=atom.charge, stereo=atom.stereo)
            for n in sub_comp:
                for nxt in mol.get_neighbors(n):
                    if nxt in sub_comp and n < nxt:
                        bond = mol.get_bond(n, nxt)
                        sub_mol.add_bond(
                            u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring
                        )

            sub_name_raw = name_component(sub_mol, sub_comp, is_substituent=False)
            match = re.search(r"(?:(^|-)(\d+)-)?sil[a]?", sub_name_raw)
            if match:
                loc = match.group(2) if match.group(2) else "1"
                if match.group(2):
                    sub_name_clean = re.sub(rf"(^|-){loc}-sil[a]?-?", r"\1", sub_name_raw)
                else:
                    sub_name_clean = re.sub(r"sil[a]?-?", "", sub_name_raw)

                sub_name_clean = sub_name_clean.replace("--", "-").strip("-")
                sub_name_clean = sub_name_clean.replace("-cyclo", "cyclo")
                if not sub_name_clean:
                    sub_name_clean = "methane"
                sub_name = f"[SPIRO]-{loc}-{sub_name_clean}"
            else:
                sub_name = f"[SPIRO]-1-{sub_name_raw}"

            subst_mapping.setdefault(c_idx,[]).append(sub_name)
            sub_handled_atoms.update(sub_comp - {c_idx})
            n_subs =[n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude:
                branch_name = name_subgraph(mol, n_idx, sub_exclude | main_set, upstream_atom=c_idx)
                if branch_name:
                    subst_mapping.setdefault(c_idx,[]).append(branch_name)

    if locant_maps:

        def evaluate_map(lmap):
            def get_val(idx):
                return parse_locant(lmap[idx])

            pg = sorted([get_val(start_idx)])
            het_by_priority = {}
            for a in mol:
                if a.idx in lmap and not a.is_carbon:
                    prio = a.element.hw_priority or 99
                    het_by_priority.setdefault(prio,[]).append(a.idx)
            het_eval = tuple(
                sorted([get_val(idx) for idx in het_by_priority[prio]]) for prio in sorted(het_by_priority.keys())
            )
            sub_idx = set(subst_mapping.keys())
            pref = sorted([get_val(idx) for idx in sub_idx if idx in lmap])
            return het_eval + (pg, pref)

        locant_map = min(locant_maps, key=evaluate_map)
        numbered_path = list(locant_map.keys())
    else:
        locant_map = None
        numbered_path = (
            candidate_paths[0]
            if locant_map
            else number_parent(
                mol,
                candidate_paths,
                {start_idx},
                subst_mapping,
                is_ring,
                is_bicycle,
                is_spiro,
                is_polycycle=is_polycycle,
                fixed_start=fixed_start_val,
                retained_name=retained_name_val,
            )
        )

    def get_loc(idx):
        return locant_map[idx] if locant_map else str(numbered_path.index(idx) + 1)

    attach_locant = get_loc(start_idx)

    is_double_attach = False
    is_triple_attach = False
    if upstream_atom is not None:
        bond_to_parent = mol.get_bond(start_idx, upstream_atom)
        if bond_to_parent:
            if bond_to_parent.order == 2:
                is_double_attach = True
            elif bond_to_parent.order == 3:
                is_triple_attach = True

    parts = AssemblyParts(
        parent_length=len(numbered_path),
        is_ring=is_ring,
        is_bicycle=is_bicycle,
        is_spiro=is_spiro,
        is_polycycle=is_polycycle,
        bicycle_xyz=xyz if is_bicycle else (0, 0, 0),
        spiro_xy=(xyz[0], xyz[1]) if is_spiro else (0, 0),
        polycycle_descriptor=polycycle_descriptor,
        is_substituent=True,
        is_double_attach=is_double_attach,
        is_triple_attach=is_triple_attach,
        attachment_locant=attach_locant,
        retained_name=retained_name_val,
    )

    for i, atom_idx in enumerate(numbered_path):
        if mol.atoms[atom_idx].stereo:
            parts.stereo_features.append((get_loc(atom_idx), mol.atoms[atom_idx].stereo))

    _emit_bond_stereo(mol, parts, numbered_path, get_loc, sub_exclude, upstream_atom)

    if retained_name_val in {
        "pyrrole",
        "imidazole",
        "pyrazole",
        "1,2,3-triazole",
        "1,2,4-triazole",
        "indole",
        "isoindole",
        "indazole",
        "benzimidazole",
        "purine",
        "indene",
        "fluorene",
    }:
        for idx in numbered_path:
            atom = mol.atoms[idx]
            if atom.symbol in ["N", "C"]:
                ring_bonds =[mol.get_bond(idx, n) for n in mol.get_neighbors(idx) if n in numbered_path]
                if sum(b.order for b in ring_bonds) == 2:
                    loc = get_loc(idx)
                    parts.indicated_hydrogens.append(loc)

    for c_idx, names in subst_mapping.items():
        locant = get_loc(c_idx)
        for name in names:
            existing = next((s for s in parts.substituents if s.name == name), None)
            if existing:
                existing.locants.append(locant)
            else:
                parts.substituents.append(SubstituentItem(name=name, locants=[locant]))

    if not retained_name_val:
        for i, atom_idx in enumerate(numbered_path):
            atom = mol.atoms[atom_idx]
            if not atom.is_carbon:
                hw_stem = atom.element.hw_stem
                if hw_stem:
                    val = sum(mol.get_bond(atom_idx, n).order for n in mol.get_neighbors(atom_idx))
                    loc = get_loc(atom_idx)
                    if atom.charge == 0 and val > atom.element.standard_valence:
                        loc = f"{loc}lambda^{val}"
                    parts.a_prefixes.append(SubstituentItem(name=hw_stem, locants=[loc]))

        seen_bonds = set()
        for u_idx in numbered_path:
            for v_idx in mol.get_neighbors(u_idx):
                if v_idx in numbered_path:
                    bond = mol.get_bond(u_idx, v_idx)
                    if bond and bond.order > 1 and bond.idx not in seen_bonds:
                        seen_bonds.add(bond.idx)
                        b_key = "double" if bond.order == 2 else "triple"
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
                        elif (
                            min_idx == 0
                            and max_idx == len(numbered_path) - 1
                            and not (is_bicycle or is_spiro or is_polycycle)
                        ):
                            locant_str = max_loc_str
                        else:
                            locant_str = f"{min_loc_str}({max_loc_str})"

                        existing = next((u for u in parts.unsaturations if u.bond_key == b_key), None)
                        if existing:
                            existing.locants.append(locant_str)
                        else:
                            parts.unsaturations.append(UnsaturationItem(bond_key=b_key, locants=[locant_str]))

    name = assemble_name(parts)

    if name == "phenyl" and not parts.substituents:
        return name
    if (
        (name.endswith("yl") or name.endswith("ylidene") or name.endswith("ylidyne"))
        and not parts.substituents
        and not parts.unsaturations
        and str(attach_locant) == "1"
        and not name.startswith("bicyclo")
        and not name.startswith("spiro")
        and not name.startswith("tricyclo")
    ):
        return name

    return f"({name})"


def name_component(mol: Molecule, component_atoms: set[int], is_substituent: bool = False) -> str:
    if len(component_atoms) == 1:
        atom = mol.atoms[list(component_atoms)[0]]
        if atom.symbol in["Na", "K", "Li", "Mg", "Ca"]:
            return atom.element.name
        if atom.symbol in ["F", "Cl", "Br", "I"]:
            halide_map = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}
            return halide_map[atom.symbol]

    perceived_groups = perceive_groups(mol)
    perceived_groups =[g for g in perceived_groups if g.attachment_carbon in component_atoms]

    candidates =[g.key for g in perceived_groups if g.is_principal_candidate]
    if is_substituent:
        principal_key = None
    else:
        principal_key = suffixes.most_senior(candidates).key if candidates else None

    if principal_key == "anhydride":
        for g in perceived_groups:
            if g.key == "anhydride":
                single_o = next(
                    (o for o in g.atoms_involved if mol.degree(o) == 2 and mol.atoms[o].symbol == "O"), None
                )
                if single_o:
                    c_neighbors =[n for n in mol.get_neighbors(single_o) if mol.atoms[n].is_carbon]
                    if len(c_neighbors) == 2:
                        c1, c2 = c_neighbors

                        def get_half_name(start_c):
                            half_atoms = set()
                            q = [start_c]
                            visited = {single_o}
                            while q:
                                curr = q.pop(0)
                                if curr not in half_atoms:
                                    half_atoms.add(curr)
                                    visited.add(curr)
                                    q.extend([x for x in mol.get_neighbors(curr) if x not in visited])

                            sub_mol = Molecule()
                            for n in half_atoms:
                                atom = mol.atoms[n]
                                sub_mol.add_atom(symbol=atom.symbol, idx=n, charge=atom.charge, stereo=atom.stereo)
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
                                            u=n,
                                            v=nxt,
                                            order=bond.order,
                                            stereo=bond.stereo,
                                            in_small_ring=bond.in_small_ring,
                                        )

                            name = name_component(sub_mol, half_atoms)
                            return name.replace(" acid", "")

                        name1 = get_half_name(c1)
                        name2 = get_half_name(c2)

                        if name1 == name2:
                            return f"{name1} anhydride"
                        else:
                            names = sorted([name1, name2])
                            return f"{names[0]} {names[1]} anhydride"

    exclude_atoms = set(mol.atoms.keys()) - component_atoms

    cyclic_atoms_all = get_cyclic_atoms(mol, set())

    for g in perceived_groups:
        if g.key != principal_key:
            if g.key in[
                "nitrile",
                "ring_nitrile",
                "carboxylic_acid",
                "ring_carboxylic_acid",
                "ester",
                "ring_carboxylate",
                "amide",
                "ring_amide",
                "thioamide",
                "ring_thioamide",
                "acid_fluoride",
                "ring_acid_fluoride",
                "acid_chloride",
                "ring_acid_chloride",
                "acid_bromide",
                "ring_acid_bromide",
                "acid_iodide",
                "ring_acid_iodide",
            ]:
                group_c = g.attachment_carbon
                if group_c in cyclic_atoms_all:
                    continue
                adj_c =[n for n in mol.get_neighbors(group_c) if mol.atoms[n].is_carbon and n not in g.atoms_involved]
                if len(adj_c) == 1:
                    g.attachment_carbon = adj_c[0]
                    g.atoms_involved.add(group_c)
                    exclude_atoms.add(group_c)

    principal_carbons =[]
    prefix_groups =[]
    for g in perceived_groups:
        if g.key == principal_key:
            principal_carbons.append(g.attachment_carbon)
        else:
            prefix_groups.append(g)

    for g in perceived_groups:
        if g.key == "anhydride":
            single_o = next((o for o in g.atoms_involved if mol.degree(o) == 2 and mol.atoms[o].symbol == "O"), None)
            if single_o is not None and single_o not in cyclic_atoms_all:
                exclude_atoms.add(single_o)
        elif g.key in["ester", "carboxylate", "ring_carboxylate", "peroxy_ester", "ring_peroxy_ester"]:
            if g.key in["peroxy_ester", "ring_peroxy_ester"]:
                single_o = next(
                    (
                        o
                        for o in g.atoms_involved
                        if mol.atoms[o].symbol == "O"
                        and mol.get_bond(o, g.attachment_carbon) is None
                        and mol.degree(o) == 2
                    ),
                    None,
                )
            else:
                single_o = next((o for o in g.atoms_involved if mol.degree(o) == 2 or mol.atoms[o].charge == -1), None)
            if single_o is not None and single_o not in cyclic_atoms_all:
                exclude_atoms.add(single_o)
        elif g.key in["sulfonic_acid", "sulfonate"]:
            single_s = next((s for s in g.atoms_involved if mol.atoms[s].symbol == "S"), None)
            if single_s is not None and single_s not in cyclic_atoms_all:
                exclude_atoms.add(single_s)
        elif g.key in["amide", "ring_amide", "thioamide", "ring_thioamide"]:
            single_n = next((n for n in g.atoms_involved if mol.atoms[n].symbol == "N"), None)
            if single_n is not None and single_n not in cyclic_atoms_all:
                exclude_atoms.add(single_n)

    chains = find_all_carbon_paths(mol, exclude_atoms)
    ring_systems = find_ring_systems(mol, exclude_atoms)

    if not chains and not ring_systems:
        return "methane"

    best_paths, is_ring, is_bicycle, is_spiro, is_polycycle, xyz, polycycle_descriptor = select_principal_parent(
        mol, chains, ring_systems, principal_carbons
    )

    parent_set = set(best_paths[0])

    valid_groups =[g for g in perceived_groups if g.attachment_carbon in parent_set]
    perceived_groups = valid_groups

    candidates =[g.key for g in perceived_groups if g.is_principal_candidate]
    if is_substituent:
        principal_key = None
    else:
        principal_key = suffixes.most_senior(candidates).key if candidates else None

    principal_carbons =[g.attachment_carbon for g in perceived_groups if g.key == principal_key]

    prefix_groups =[]
    for g in perceived_groups:
        if g.key != principal_key:
            prefix_groups.append(g)

    subst_mapping = {}
    handled_prefix_atoms = set()

    temp_retained = retained.get_retained_ring(mol, best_paths[0]) if is_ring else None
    if temp_retained:
        retained_name_val, locant_maps = temp_retained
        if locant_maps is None and (is_bicycle or is_polycycle):
            retained_name_val, locant_maps = None, None
        elif any(mol.atoms[idx].symbol not in["C", "N", "O", "S"] for idx in best_paths[0]):
            retained_name_val, locant_maps = None, None
    else:
        retained_name_val, locant_maps = None, None

    principal_involved_atoms = set()
    if principal_key:
        for g in perceived_groups:
            if g.key == principal_key and g.attachment_carbon in best_paths[0]:
                principal_involved_atoms.update(g.atoms_involved)

    main_set = set(best_paths[0])
    base_exclude = set(mol.atoms.keys()) - component_atoms
    sub_exclude = base_exclude | main_set | principal_involved_atoms

    for g in prefix_groups:
        if g.key in[
            "ether", "thioether", "amine", "anhydride", "hydrazine", "hydrazone", "imine",
            "aldehyde_hydrazone", "ring_aldehyde_hydrazone", "aldehyde_imine", "ring_aldehyde_imine"
        ]:
            continue

        if g.attachment_carbon not in main_set:
            continue

        if g.key in["ester", "carboxylate", "ring_carboxylate", "peroxy_ester", "ring_peroxy_ester"]:
            if g.key in["peroxy_ester", "ring_peroxy_ester"]:
                single_o = next(
                    (
                        o
                        for o in g.atoms_involved
                        if mol.atoms[o].symbol == "O"
                        and mol.get_bond(o, g.attachment_carbon) is None
                        and mol.degree(o) == 2
                    ),
                    None,
                )
            else:
                single_o = next((o for o in g.atoms_involved if mol.degree(o) == 2 or mol.atoms[o].charge == -1), None)

            if single_o is not None:
                r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in g.atoms_involved), None)
                if r_group_c is not None:
                    branch_name = name_subgraph(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
                    if branch_name:
                        if branch_name == "methyl":
                            alkoxy = "methoxy"
                        elif branch_name == "ethyl":
                            alkoxy = "ethoxy"
                        elif branch_name == "propyl":
                            alkoxy = "propoxy"
                        elif branch_name == "isopropyl":
                            alkoxy = "isopropoxy"
                        elif branch_name == "butyl":
                            alkoxy = "butoxy"
                        elif branch_name == "isobutyl":
                            alkoxy = "isobutoxy"
                        elif branch_name == "sec-butyl":
                            alkoxy = "sec-butoxy"
                        elif branch_name == "tert-butyl":
                            alkoxy = "tert-butoxy"
                        elif branch_name == "phenyl":
                            alkoxy = "phenoxy"
                        else:
                            if branch_name.startswith("(") and branch_name.endswith(")"):
                                branch_name = branch_name[1:-1]
                            alkoxy = f"({branch_name}oxy)"

                        name = f"({alkoxy}carbonyl)"
                        subst_mapping.setdefault(g.attachment_carbon,[]).append(name)
                        handled_prefix_atoms.update(g.atoms_involved)
            continue

        if g.key in["amide", "ring_amide", "thioamide", "ring_thioamide"]:
            single_n = next((n for n in g.atoms_involved if mol.atoms[n].symbol == "N"), None)
            if single_n is not None:
                n_subs =[
                    n for n in mol.get_neighbors(single_n) if n not in g.atoms_involved and mol.atoms[n].symbol != "H"
                ]
                if not n_subs:
                    name = "carbamoyl" if g.key in ["amide", "ring_amide"] else "carbamothioyl"
                else:
                    sub_names =[
                        name_subgraph(mol, x, sub_exclude | {single_n}, upstream_atom=single_n) for x in n_subs
                    ]
                    counts = {}
                    for sn in sub_names:
                        counts[sn] = counts.get(sn, 0) + 1
                    prefix_parts =[]
                    safe = len(counts) > 1 or any("(" in sn or sn[0].isdigit() or "-" in sn or " " in sn for sn in counts)
                    for sn, c in sorted(counts.items()):
                        prefix_parts.append(_format_multiplier(sn, c, safe_enclose=safe))
                    prefix = "".join(prefix_parts)
                    base_name = "carbamoyl" if g.key in ["amide", "ring_amide"] else "carbamothioyl"
                    name = f"({prefix}{base_name})" if len(n_subs) > 1 else f"({prefix}{base_name})"
                subst_mapping.setdefault(g.attachment_carbon,[]).append(name)
                handled_prefix_atoms.update(g.atoms_involved)
            continue

        if g.key in ["carboxylic_acid", "ring_carboxylic_acid"]:
            subst_mapping.setdefault(g.attachment_carbon,[]).append("carboxy")
            handled_prefix_atoms.update(g.atoms_involved)
            continue

        if g.key in["nitrile", "ring_nitrile"]:
            subst_mapping.setdefault(g.attachment_carbon,[]).append("cyano")
            handled_prefix_atoms.update(g.atoms_involved)
            continue

        if g.key in["acid_fluoride", "ring_acid_fluoride"]:
            subst_mapping.setdefault(g.attachment_carbon,[]).append("(fluorocarbonyl)")
            handled_prefix_atoms.update(g.atoms_involved)
            continue
        if g.key in["acid_chloride", "ring_acid_chloride"]:
            subst_mapping.setdefault(g.attachment_carbon,[]).append("(chlorocarbonyl)")
            handled_prefix_atoms.update(g.atoms_involved)
            continue
        if g.key in ["acid_bromide", "ring_acid_bromide"]:
            subst_mapping.setdefault(g.attachment_carbon,[]).append("(bromocarbonyl)")
            handled_prefix_atoms.update(g.atoms_involved)
            continue
        if g.key in["acid_iodide", "ring_acid_iodide"]:
            subst_mapping.setdefault(g.attachment_carbon,[]).append("(iodocarbonyl)")
            handled_prefix_atoms.update(g.atoms_involved)
            continue
        if g.key in["peroxy_acid", "ring_peroxy_acid"]:
            subst_mapping.setdefault(g.attachment_carbon,[]).append("carboperoxy")
            handled_prefix_atoms.update(g.atoms_involved)
            continue

        if g.key in["sulfonic_acid", "sulfonate"]:
            single_o = next((o for o in g.atoms_involved if mol.degree(o) == 2 and mol.atoms[o].symbol == "O"), None)
            if single_o is not None:
                r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in g.atoms_involved), None)
                if r_group_c is not None:
                    branch_name = name_subgraph(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
                    if branch_name:
                        if branch_name == "methyl":
                            alkoxy = "methoxy"
                        elif branch_name == "ethyl":
                            alkoxy = "ethoxy"
                        elif branch_name == "propyl":
                            alkoxy = "propoxy"
                        elif branch_name == "isopropyl":
                            alkoxy = "isopropoxy"
                        elif branch_name == "butyl":
                            alkoxy = "butoxy"
                        elif branch_name == "isobutyl":
                            alkoxy = "isobutoxy"
                        elif branch_name == "sec-butyl":
                            alkoxy = "sec-butoxy"
                        elif branch_name == "tert-butyl":
                            alkoxy = "tert-butoxy"
                        elif branch_name == "phenyl":
                            alkoxy = "phenoxy"
                        else:
                            if branch_name.startswith("(") and branch_name.endswith(")"):
                                branch_name = branch_name[1:-1]
                            alkoxy = f"({branch_name}oxy)"

                        name = f"({alkoxy}sulfonyl)"
                        subst_mapping.setdefault(g.attachment_carbon,[]).append(name)
                        handled_prefix_atoms.update(g.atoms_involved)
                        continue
            subst_mapping.setdefault(g.attachment_carbon,[]).append("sulfo")
            handled_prefix_atoms.update(g.atoms_involved)
            continue

        if g.key == "isocyano":
            subst_mapping.setdefault(g.attachment_carbon,[]).append("isocyano")
            handled_prefix_atoms.update(g.atoms_involved)
            continue

        if g.key == "isothiocyanato":
            subst_mapping.setdefault(g.attachment_carbon,[]).append("isothiocyanato")
            handled_prefix_atoms.update(g.atoms_involved)
            continue
        if g.key == "isocyanato":
            subst_mapping.setdefault(g.attachment_carbon,[]).append("isocyanato")
            handled_prefix_atoms.update(g.atoms_involved)
            continue
        if g.key == "thiocyanato":
            subst_mapping.setdefault(g.attachment_carbon,[]).append("thiocyanato")
            handled_prefix_atoms.update(g.atoms_involved)
            continue
        if g.key == "cyanato":
            subst_mapping.setdefault(g.attachment_carbon,[]).append("cyanato")
            handled_prefix_atoms.update(g.atoms_involved)
            continue

        if g.attachment_carbon in best_paths[0]:
            name = suffixes.get(g.key).prefix if g.is_principal_candidate else substituents.get(g.key).prefix
            if name:
                subst_mapping.setdefault(g.attachment_carbon,[]).append(name)
                handled_prefix_atoms.update(g.atoms_involved)

    for c_idx in best_paths[0]:
        n_subs =[
            n_idx
            for n_idx in mol.get_neighbors(c_idx)
            if n_idx not in main_set
            and n_idx not in principal_involved_atoms
            and n_idx not in handled_prefix_atoms
            and n_idx not in base_exclude
        ]

        spiro_pairs =[]
        for i in range(len(n_subs)):
            for j in range(i + 1, len(n_subs)):
                n1, n2 = n_subs[i], n_subs[j]
                visited = {c_idx}
                q = [n1]
                in_ring = False
                while q:
                    curr = q.pop(0)
                    if curr == n2:
                        in_ring = True
                        break
                    visited.add(curr)
                    for nxt in mol.get_neighbors(curr):
                        if nxt not in visited and nxt not in main_set and nxt not in base_exclude:
                            q.append(nxt)
                if in_ring:
                    spiro_pairs.append((n1, n2))
                    break
            if spiro_pairs:
                break

        if spiro_pairs:
            n1, n2 = spiro_pairs[0]
            sub_comp = set()
            q_sub =[n1]
            visited_sub = {c_idx}
            while q_sub:
                curr = q_sub.pop(0)
                if curr not in sub_comp:
                    sub_comp.add(curr)
                    visited_sub.add(curr)
                    for nxt in mol.get_neighbors(curr):
                        if nxt not in visited_sub and nxt not in main_set and nxt not in base_exclude:
                            q_sub.append(nxt)
            sub_comp.add(c_idx)

            sub_mol = Molecule()
            for n in sub_comp:
                atom = mol.atoms[n]
                sym = "Si" if n == c_idx else atom.symbol
                sub_mol.add_atom(symbol=sym, idx=n, charge=atom.charge, stereo=atom.stereo)
            for n in sub_comp:
                for nxt in mol.get_neighbors(n):
                    if nxt in sub_comp and n < nxt:
                        bond = mol.get_bond(n, nxt)
                        sub_mol.add_bond(
                            u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring
                        )

            sub_name_raw = name_component(sub_mol, sub_comp, is_substituent=False)
            match = re.search(r"(?:(^|-)(\d+)-)?sil[a]?", sub_name_raw)
            if match:
                loc = match.group(2) if match.group(2) else "1"
                if match.group(2):
                    sub_name_clean = re.sub(rf"(^|-){loc}-sil[a]?-?", r"\1", sub_name_raw)
                else:
                    sub_name_clean = re.sub(r"sil[a]?-?", "", sub_name_raw)

                sub_name_clean = sub_name_clean.replace("--", "-").strip("-")
                sub_name_clean = sub_name_clean.replace("-cyclo", "cyclo")
                if not sub_name_clean:
                    sub_name_clean = "methane"
                sub_name = f"[SPIRO]-{loc}-{sub_name_clean}"
            else:
                sub_name = f"[SPIRO]-1-{sub_name_raw}"

            subst_mapping.setdefault(c_idx,[]).append(sub_name)
            handled_prefix_atoms.update(sub_comp - {c_idx})
            n_subs =[n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in principal_involved_atoms and n_idx not in handled_prefix_atoms:
                branch_name = name_subgraph(mol, n_idx, sub_exclude | main_set, upstream_atom=c_idx)
                if branch_name:
                    subst_mapping.setdefault(c_idx,[]).append(branch_name)

    if locant_maps:

        def evaluate_map(lmap):
            def get_val(idx):
                return parse_locant(lmap[idx])

            pg = sorted([get_val(c) for c in principal_carbons if c in lmap])
            het_by_priority = {}
            for a in mol:
                if a.idx in lmap and not a.is_carbon:
                    prio = a.element.hw_priority or 99
                    het_by_priority.setdefault(prio,[]).append(a.idx)
            het_eval = tuple(
                sorted([get_val(idx) for idx in het_by_priority[prio]]) for prio in sorted(het_by_priority.keys())
            )
            sub_idx = set(subst_mapping.keys())
            pref = sorted([get_val(idx) for idx in sub_idx if idx in lmap])
            return het_eval + (pg, pref)

        locant_map = min(locant_maps, key=evaluate_map)
        numbered_path = list(locant_map.keys())
    else:
        locant_map = None
        numbered_path = (
            best_paths[0]
            if locant_map
            else number_parent(
                mol,
                best_paths,
                principal_carbons,
                subst_mapping,
                is_ring,
                is_bicycle,
                is_spiro,
                is_polycycle=is_polycycle,
                retained_name=retained_name_val,
            )
        )

    def get_loc(idx):
        return locant_map[idx] if locant_map else str(numbered_path.index(idx) + 1)

    parts = AssemblyParts(
        parent_length=len(numbered_path),
        is_ring=is_ring,
        is_bicycle=is_bicycle,
        is_spiro=is_spiro,
        is_polycycle=is_polycycle,
        bicycle_xyz=xyz if is_bicycle else (0, 0, 0),
        spiro_xy=(xyz[0], xyz[1]) if is_spiro else (0, 0),
        polycycle_descriptor=polycycle_descriptor,
        retained_name=retained_name_val,
    )

    for i, atom_idx in enumerate(numbered_path):
        if mol.atoms[atom_idx].stereo:
            parts.stereo_features.append((get_loc(atom_idx), mol.atoms[atom_idx].stereo))

    _emit_bond_stereo(mol, parts, numbered_path, get_loc, base_exclude)

    if retained_name_val in {
        "pyrrole",
        "imidazole",
        "pyrazole",
        "1,2,3-triazole",
        "1,2,4-triazole",
        "indole",
        "isoindole",
        "indazole",
        "benzimidazole",
        "purine",
        "indene",
        "fluorene",
    }:
        for idx in numbered_path:
            atom = mol.atoms[idx]
            if atom.symbol in ["N", "C"]:
                ring_bonds =[mol.get_bond(idx, n) for n in mol.get_neighbors(idx) if n in numbered_path]
                if sum(b.order for b in ring_bonds) == 2:
                    loc = get_loc(idx)
                    parts.indicated_hydrogens.append(loc)

    if principal_key in["ester", "carboxylate", "sulfonate", "ring_carboxylate", "peroxy_ester", "ring_peroxy_ester"]:
        for g in perceived_groups:
            if g.key == principal_key:
                c_idx = g.attachment_carbon
                if g.key in ["peroxy_ester", "ring_peroxy_ester"]:
                    single_o = next(
                        (
                            o
                            for o in g.atoms_involved
                            if mol.atoms[o].symbol == "O" and mol.get_bond(o, c_idx) is None and mol.degree(o) == 2
                        ),
                        None,
                    )
                else:
                    single_o = next(
                        (o for o in g.atoms_involved if mol.degree(o) == 2 or mol.atoms[o].charge == -1), None
                    )

                if single_o is not None:
                    r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in g.atoms_involved), None)
                    if r_group_c is not None:
                        branch_name = name_subgraph(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
                        if branch_name:
                            if branch_name.startswith("(") and branch_name.endswith(")"):
                                branch_name = branch_name[1:-1]
                            parts.front_modifiers.append(branch_name)

    if principal_key in[
        "amide",
        "amine",
        "ring_amide",
        "thioamide",
        "ring_thioamide",
        "hydrazine",
        "hydrazone",
        "imine",
        "aldehyde_hydrazone",
        "ring_aldehyde_hydrazone",
        "aldehyde_imine",
        "ring_aldehyde_imine"
    ]:
        pgs =[g for g in perceived_groups if g.key == principal_key and g.attachment_carbon in numbered_path]
        pgs.sort(key=lambda g: parse_locant(get_loc(g.attachment_carbon)))

        n_idx_global = 0
        for g in pgs:
            c_idx = g.attachment_carbon
            ns =[n for n in g.atoms_involved if mol.atoms[n].symbol == "N"]
            ns.sort(key=lambda n: mol.get_bond(n, c_idx) is not None, reverse=True)
            for single_n in ns:
                n_substituents =[
                    n
                    for n in mol.get_neighbors(single_n)
                    if n != c_idx and n not in g.atoms_involved and mol.atoms[n].symbol != "H"
                ]

                if principal_key == "hydrazine":
                    loc_prefix = "N" if single_n == ns[0] else "N'"
                elif principal_key in["hydrazone", "aldehyde_hydrazone", "ring_aldehyde_hydrazone"]:
                    loc_prefix = "N"
                else:
                    if len(pgs) == 1 and len(ns) == 1:
                        loc_prefix = "N"
                    else:
                        loc_prefix = "N" + "'" * n_idx_global

                for n_sub in n_substituents:
                    branch_name = name_subgraph(mol, n_sub, sub_exclude | {single_n}, upstream_atom=single_n)
                    if branch_name:
                        existing = next((s for s in parts.substituents if s.name == branch_name), None)
                        if existing:
                            existing.locants.append(loc_prefix)
                        else:
                            parts.substituents.append(SubstituentItem(name=branch_name, locants=[loc_prefix]))
                n_idx_global += 1

    if not retained_name_val:
        for i, atom_idx in enumerate(numbered_path):
            atom = mol.atoms[atom_idx]
            if not atom.is_carbon:
                hw_stem = atom.element.hw_stem
                if hw_stem:
                    val = sum(mol.get_bond(atom_idx, n).order for n in mol.get_neighbors(atom_idx))
                    loc = get_loc(atom_idx)
                    if atom.charge == 0 and val > atom.element.standard_valence:
                        loc = f"{loc}lambda^{val}"
                    parts.a_prefixes.append(SubstituentItem(name=hw_stem, locants=[loc]))

    if principal_key:
        locants = sorted([get_loc(c) for c in principal_carbons if c in numbered_path], key=parse_locant)
        parts.principal_group = PrincipalGroupItem(key=principal_key, locants=locants)

    for c_idx, names in subst_mapping.items():
        if c_idx in numbered_path:
            locant = get_loc(c_idx)
            for name in names:
                existing = next((s for s in parts.substituents if s.name == name), None)
                if existing:
                    existing.locants.append(locant)
                else:
                    parts.substituents.append(SubstituentItem(name=name, locants=[locant]))

    if not retained_name_val:
        seen_bonds = set()
        for u_idx in numbered_path:
            for v_idx in mol.get_neighbors(u_idx):
                if v_idx in numbered_path:
                    bond = mol.get_bond(u_idx, v_idx)
                    if bond and bond.order > 1 and bond.idx not in seen_bonds:
                        seen_bonds.add(bond.idx)
                        b_key = "double" if bond.order == 2 else "triple"
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
                        elif (
                            min_idx == 0
                            and max_idx == len(numbered_path) - 1
                            and not (is_bicycle or is_spiro or is_polycycle)
                        ):
                            locant_str = max_loc_str
                        else:
                            locant_str = f"{min_loc_str}({max_loc_str})"

                        existing = next((u for u in parts.unsaturations if u.bond_key == b_key), None)
                        if existing:
                            existing.locants.append(locant_str)
                        else:
                            parts.unsaturations.append(UnsaturationItem(bond_key=b_key, locants=[locant_str]))

    name = assemble_name(parts)

    if name == "1-phenylbenzene":
        return "1,1'-biphenyl"

    return name


def name_smiles(smiles: str) -> str:
    mol = read_smiles(smiles)
    if not mol.atoms:
        return ""
    components = get_connected_components(mol)

    names =[]
    for comp in components:
        comp_name = name_component(mol, comp)
        if comp_name:
            names.append(comp_name)

    metals = {
        "lithium",
        "sodium",
        "potassium",
        "magnesium",
        "calcium",
        "zinc",
        "copper",
        "iron",
        "aluminum",
        "silver",
        "gold",
        "lead",
        "bismuth",
        "cesium",
        "rubidium",
        "barium",
        "strontium",
    }

    def sort_key(name):
        return (0 if name in metals else 1, name)

    names.sort(key=sort_key)

    return " ".join(names)