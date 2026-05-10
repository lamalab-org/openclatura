# structure-to-iupac/api.py

import re
from rdkit import Chem
from .molecule import Molecule
from .perception import perceive_groups, PerceivedGroup
from .chains import find_all_carbon_paths, find_ring_systems, get_cyclic_atoms
from .parent_selection import select_principal_parent
from .assembler import assemble_name, AssemblyParts, PrincipalGroupItem, SubstituentItem, UnsaturationItem
from .rules import suffixes, substituents, stems, retained, multipliers
from .naming_data import mapping, values


INDICATED_H_RETAINED_NAMES = set(values("indicated_hydrogen_retained_names"))
ALKYL_OXY_PREFIXES = mapping("alkyl_oxy_prefixes")
SIMPLE_SULFANYL_PREFIXES = set(values("simple_sulfanyl_prefixes"))
SIMPLE_SELANYL_PREFIXES = set(values("simple_selanyl_prefixes"))
HALOGEN_PREFIXES = mapping("halogen_prefixes")
HALOGEN_LAMBDA_SUFFIXES = mapping("halogen_lambda_suffixes")
RETAINED_RING_ELEMENTS = set(values("retained_ring_elements"))
DIRECT_GROUP_PREFIXES = mapping("direct_group_prefixes")
CHAIN_EXTERNAL_CARBONYL_GROUPS = set(values("chain_external_carbonyl_groups"))
PREFIX_GROUPS_TO_SKIP = set(values("prefix_groups_to_skip"))
ESTER_LIKE_PREFIX_GROUPS = set(values("ester_like_prefix_groups"))
PEROXY_ESTER_GROUPS = set(values("peroxy_ester_groups"))
AMIDE_LIKE_PREFIX_GROUPS = set(values("amide_like_prefix_groups"))
AMIDE_PREFIX_BASES = mapping("amide_prefix_bases")
CARBOXY_PREFIX_GROUPS = set(values("carboxy_prefix_groups"))
CYANO_PREFIX_GROUPS = set(values("cyano_prefix_groups"))
ACID_HALIDE_PREFIXES = mapping("acid_halide_prefixes")
PEROXY_ACID_PREFIX_GROUPS = set(values("peroxy_acid_prefix_groups"))
SULFONYL_PREFIX_GROUPS = set(values("sulfonyl_prefix_groups"))
DIRECT_PREFIX_GROUPS = mapping("direct_prefix_groups")
FRONT_MODIFIER_PRINCIPAL_GROUPS = set(values("front_modifier_principal_groups"))
N_SUBSTITUENT_PRINCIPAL_GROUPS = set(values("n_substituent_principal_groups"))
HYDRAZONE_PRINCIPAL_GROUPS = set(values("hydrazone_principal_groups"))
SPECIAL_COMPONENT_NAMES = mapping("special_component_names")
SINGLE_ATOM_CATIONS = set(values("single_atom_cations"))
SINGLE_ATOM_ANIONS = mapping("single_atom_anions")
SALT_METAL_NAMES = set(values("salt_metal_names"))


def parse_locant(l):
    """Return a sortable representation of a locant string.

    Blue Book references: P-14.3 and P-14.4 for locants and ordering of
    locant sets.
    """

    s = str(l)
    match = re.match(r"^(\d+)([a-zA-Z]*)$", s.split("(")[0])
    if match:
        return (1, float(match.group(1)), match.group(2))
    if any(c.isdigit() for c in s):
        nums = re.findall(r"\d+", s)
        return (1, float(nums[0]) if nums else 0.0, s)
    return (2, 0.0, s)


def is_fully_enclosed(s: str) -> bool:
    """Return true when a name fragment is already fully parenthesized.

    Blue Book references: P-16.5 for enclosing complex prefixes in parentheses.
    """

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
    """Apply simple or complex multiplicative prefixes to a substituent name.

    Blue Book references: P-14.2 and P-16.5 for multiplicative prefixes and
    parenthesized complex substituent prefixes.
    """

    is_complex = "(" in name or name[0].isdigit() or "-" in name or " " in name
    if count == 1:
        if (safe_enclose or is_complex) and not is_fully_enclosed(name):
            return f"({name})"
        return name
    mult = multipliers.complex_(count) if is_complex else multipliers.basic(count)
    if is_complex and not is_fully_enclosed(name):
        return f"{mult}({name})"
    return f"{mult}{name}"


def _strip_outer_parentheses(name: str) -> str:
    """Remove one balanced outer parenthesis pair from a fragment.

    Blue Book references: P-16.5; this keeps complex prefixes enclosed only
    when the final assembly context requires it.
    """

    if name.startswith("(") and name.endswith(")"):
        return name[1:-1]
    return name


def _is_complex_prefix(name: str) -> bool:
    """Return true when a substituent prefix needs protective parentheses.

    Blue Book references: P-16.5 for complex substituent prefix enclosure.
    """

    return "(" in name or name[0].isdigit() or "-" in name or " " in name


def _count_names(names: list[str]) -> dict[str, int]:
    """Count repeated substituent fragments before multiplier formatting.

    Blue Book references: P-14.2 for multiplicative prefixes.
    """

    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return counts


def _format_counted_prefixes(names: list[str]) -> str:
    """Format repeated substituent fragments with correct multipliers.

    Blue Book references: P-14.2 and P-16.5 for simple versus complex
    multiplicative prefixes.
    """

    counts = _count_names(names)
    safe = len(counts) > 1 or any(_is_complex_prefix(name) for name in counts)
    return "".join(_format_multiplier(name, count, safe_enclose=safe) for name, count in sorted(counts.items()))


def _oxy_prefix_from_branch(branch: str) -> str:
    """Return an oxy prefix for a named branch.

    Blue Book references: P-61.2.2.1; extend simple retained forms in
    ``alkyl_oxy_prefixes`` in ``data/namer_rules.json``.
    """

    retained = ALKYL_OXY_PREFIXES.get(branch)
    if retained:
        return retained
    branch = _strip_outer_parentheses(branch)
    return f"({branch}oxy)"


def _format_element_substituent(stereo_prefix: str, branch: str, suffix: str, is_double: bool = False) -> str:
    """Attach a named branch to an element substituent suffix.

    Blue Book references: P-61.2 and P-14.5 for chalcogen and heteroatom
    substituent prefixes with lambda/hydride-derived suffixes.
    """

    branch = _strip_outer_parentheses(branch)
    suffix_text = suffix + ("idene" if is_double else "")
    if _is_complex_prefix(branch):
        return f"({stereo_prefix}({branch}){suffix_text})"
    return f"({stereo_prefix}{branch}{suffix_text})"


def _get_atom_locants(oriented_path: list[int], target_indices: set[int]) -> list[int]:
    """Return numeric locants for target atoms in an oriented parent path.

    Blue Book references: P-14.3 and P-44.1 for assigning locants to parent
    atoms and characteristic groups.
    """

    return sorted([i + 1 for i, atom_idx in enumerate(oriented_path) if atom_idx in target_indices])


def _get_bond_locants(
    mol: Molecule, oriented_path: list[int], is_bicycle: bool, is_spiro: bool, is_polycycle: bool
) -> tuple[list[int], list[int]]:
    """Return double- and triple-bond locants for an oriented parent path.

    Blue Book references: P-31.1 and P-44.1 for unsaturation locants in parent
    hydrides.
    """

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
    """Choose the preferred numbering for a selected parent skeleton.

    Blue Book references: P-14.4, P-44, and P-45; numbering minimizes senior
    characteristic group locants, heteroatom locants, unsaturation locants,
    substituent locants, and finally stereochemical locants.
    """


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

            if retained_name in INDICATED_H_RETAINED_NAMES:
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
    """Parse a SMILES string into the internal graph model.

    Blue Book references: P-13 and P-14 govern name construction downstream;
    this function only preserves graph, bond order, charge, and stereochemical
    features needed by those rules.
    """

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
    """Split a molecule graph into disconnected naming components.

    Blue Book references: P-72 for ionic/disconnected component names.
    """

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
    """Collect E/Z bond stereochemical descriptors for assembly.

    Blue Book references: P-91 and P-93 for stereochemical descriptor citation.
    """

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


def _upstream_bond_order(mol: Molecule, start_idx: int, upstream_atom: int | None) -> int:
    """Return the bond order to the parent atom, or zero for root fragments.

    Blue Book references: P-13.6 and P-61; the bond order determines whether a
    heteroatom fragment is named with prefix forms such as oxo, imino, nitrilo,
    or ylidene-style suffixes.
    """

    if upstream_atom is None:
        return 0
    bond = mol.get_bond(start_idx, upstream_atom)
    return bond.order if bond else 0


def _subgraph_neighbors(
    mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None, extra_exclude=None
) -> list[int]:
    """Return substituent-side neighbors that remain available for recursion.

    Blue Book references: P-13.6 and P-16.5; recursive substituent prefixes are
    constructed only from atoms outside the current parent/excluded fragment.
    """

    blocked = set(extra_exclude or [])
    return [
        n
        for n in mol.get_neighbors(start_idx)
        if n not in exclude_atoms and n != upstream_atom and n not in blocked
    ]


def _double_bonded_neighbors(mol: Molecule, center_idx: int, symbol: str) -> list[int]:
    """Return neighbors of a given element double-bonded to ``center_idx``.

    Blue Book references: P-63, P-65, P-66, and P-67 for oxo, thioxo,
    sulfinyl, sulfonyl, carbonyl, and related prefix forms.
    """

    return [
        n
        for n in mol.get_neighbors(center_idx)
        if mol.atoms[n].symbol == symbol and mol.get_bond(center_idx, n).order == 2
    ]


def _first_substituent_neighbor(mol: Molecule, center_idx: int, excluded: set[int], require_carbon=False) -> int | None:
    """Return the first neighbor outside a local functional group.

    Blue Book references: P-13.6; this identifies the R group used to build a
    recursive substituent prefix.
    """

    for n in mol.get_neighbors(center_idx):
        if n in excluded:
            continue
        if require_carbon and not mol.atoms[n].is_carbon:
            continue
        return n
    return None


def _stereo_prefix(atom) -> str:
    """Format an atom-centered stereochemical prefix for a substituent fragment.

    Blue Book references: P-91 and P-93 for stereochemical descriptor citation.
    """

    return f"({atom.stereo})-" if atom.stereo else ""


def _sulfur_oxo_suffix(oxo_count: int) -> str:
    """Choose sulfinyl or sulfonyl from the number of S=O bonds.

    Blue Book references: P-67.1 for sulfur oxo acid and sulfur oxo substituent
    prefix forms.
    """

    return "sulfonyl" if oxo_count == 2 else "sulfinyl"


def _name_branch_or_none(mol: Molecule, branch_idx: int | None, exclude_atoms: set[int], upstream_atom: int) -> str:
    """Name a recursive branch and normalize one redundant parenthesis layer.

    Blue Book references: P-16.5; complex prefixes are parenthesized by the
    final formatting context.
    """

    if branch_idx is None:
        return ""
    return _strip_outer_parentheses(name_subgraph(mol, branch_idx, exclude_atoms, upstream_atom=upstream_atom))


def _name_carbonyl_like_fragment(
    mol: Molecule,
    center_idx: int,
    attach_idx: int,
    double_atoms: list[int],
    exclude_atoms: set[int],
    branch_suffix: str,
    fallback: str,
    amino_base: str | None = None,
    wrap_result: bool = False,
) -> str:
    """Name carbonyl or thiocarbonyl fragments attached through O or N.

    Blue Book references: P-65 and P-66; carbonyl fragments become retained
    prefix forms such as formyl, carbamoyl, carbonyloxy, and carbonothioyl.
    """

    local_exclude = exclude_atoms | {attach_idx, center_idx} | set(double_atoms)
    branch_idx = _first_substituent_neighbor(mol, center_idx, {attach_idx, *double_atoms}, require_carbon=True)
    if branch_idx is None:
        branch_idx = _first_substituent_neighbor(mol, center_idx, {attach_idx, *double_atoms})

    branch = _name_branch_or_none(mol, branch_idx, local_exclude, upstream_atom=center_idx)
    if not branch:
        return fallback
    if amino_base and branch.endswith("amino"):
        return f"({branch[:-5]}{amino_base})"
    result = f"{branch}{branch_suffix}"
    return f"({result})" if wrap_result else result


def _format_amino_from_branches(branches: list[str], is_double: bool) -> str:
    """Format amino or imino substituent names from N-attached branches.

    Blue Book references: P-62 and P-66 for amino, imino, N-substituted, and
    acylamino-style prefixes.
    """

    if is_double:
        if len(branches) == 1:
            branch = _strip_outer_parentheses(branches[0])
            if _is_complex_prefix(branch):
                return f"(({branch})imino)"
            return f"({branch}imino)"
        return "imino"

    counts = _count_names(branches)
    if len(counts) == 1 and list(counts.values())[0] == 1:
        branch = _strip_outer_parentheses(branches[0])
        if branch.endswith(("carbonyl", "sulfonyl", "sulfinyl", "carbonothioyl")):
            return f"{branch}amino"
        if _is_complex_prefix(branch):
            return f"(({branch})amino)"
        return f"({branch}amino)"

    return f"({_format_counted_prefixes(branches)}amino)"


def _format_lambda_substituent(
    mol: Molecule,
    start_idx: int,
    branches: list[str],
    stereo_prefix: str,
    base_suffix: str,
) -> str:
    """Format hypervalent lambda substituent fragments.

    Blue Book references: P-14.1 and P-14.5 for the lambda convention in
    heteroatom substituent names.
    """

    valence = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))
    return f"({stereo_prefix}{_format_counted_prefixes(branches)}-lambda^{valence}-{base_suffix})"


def _name_oxygen_subgraph(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None) -> str:
    """Name an oxygen-starting substituent fragment.

    Blue Book references: P-61.2.2.1 for alkoxy/aryloxy prefixes, P-63 for oxo
    and hydroxy prefixes, P-65 for acyloxy/carbonyl prefixes, and P-67 for
    sulfonyloxy/sulfinyloxy prefixes.
    """

    is_double = _upstream_bond_order(mol, start_idx, upstream_atom) == 2
    next_atoms = _subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    start_atom = mol.atoms[start_idx]
    if not next_atoms:
        if is_double:
            return "oxo"
        if start_atom.charge == -1:
            return "oxido"
        return "hydroxy"

    nxt = next_atoms[0]
    s_oxygens = _double_bonded_neighbors(mol, nxt, "O")
    if mol.atoms[nxt].symbol == "S" and s_oxygens:
        branch_idx = _first_substituent_neighbor(mol, nxt, {start_idx, *s_oxygens})
        branch = _name_branch_or_none(mol, branch_idx, exclude_atoms | {start_idx, nxt} | set(s_oxygens), nxt)
        if branch:
            return f"({_stereo_prefix(mol.atoms[nxt])}{branch}{_sulfur_oxo_suffix(len(s_oxygens))}oxy)"
        return "sulfooxy"

    c_oxygens = _double_bonded_neighbors(mol, nxt, "O")
    c_sulfurs = _double_bonded_neighbors(mol, nxt, "S")
    if mol.atoms[nxt].is_carbon and len(c_oxygens) == 1:
        return _name_carbonyl_like_fragment(
            mol, nxt, start_idx, c_oxygens, exclude_atoms, "carbonyloxy", "formyloxy", wrap_result=True
        )
    if mol.atoms[nxt].is_carbon and len(c_sulfurs) == 1:
        return _name_carbonyl_like_fragment(
            mol,
            nxt,
            start_idx,
            c_sulfurs,
            exclude_atoms,
            "carbonothioyloxy",
            "methanethioyloxy",
            wrap_result=True,
        )
    if mol.atoms[nxt].symbol == "O":
        branch_idx = _first_substituent_neighbor(mol, nxt, {start_idx})
        branch = _name_branch_or_none(mol, branch_idx, exclude_atoms | {start_idx, nxt}, nxt)
        return f"({branch}peroxy)" if branch else "hydroperoxy"

    branch = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
    if branch:
        if branch in ALKYL_OXY_PREFIXES:
            return ALKYL_OXY_PREFIXES[branch]
        branch = _strip_outer_parentheses(branch)
        if _is_complex_prefix(branch):
            return f"(({branch})oxy)"
        return f"({branch}oxy)"
    return "hydroxy"


def _name_nitrogen_subgraph(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None) -> str:
    """Name a nitrogen-starting substituent fragment.

    Blue Book references: P-62 for amines/imines, P-63 for amino/imino/nitrilo
    prefixes, P-66 for amide-derived prefixes, and P-67 for sulfonamide-like
    prefixes.
    """

    upstream_order = _upstream_bond_order(mol, start_idx, upstream_atom)
    is_double = upstream_order == 2
    is_triple = upstream_order == 3
    next_atoms = _subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    if not next_atoms:
        if is_double:
            return "imino"
        if is_triple:
            return "nitrilo"
        return "amino"

    branches = []
    for nxt in next_atoms:
        c_oxygens = _double_bonded_neighbors(mol, nxt, "O")
        c_sulfurs = _double_bonded_neighbors(mol, nxt, "S")
        if mol.atoms[nxt].is_carbon and len(c_oxygens) == 1:
            branches.append(
                _name_carbonyl_like_fragment(
                    mol, nxt, start_idx, c_oxygens, exclude_atoms, "carbonyl", "formyl", amino_base="carbamoyl"
                )
            )
        elif mol.atoms[nxt].is_carbon and len(c_sulfurs) == 1:
            branches.append(
                _name_carbonyl_like_fragment(
                    mol,
                    nxt,
                    start_idx,
                    c_sulfurs,
                    exclude_atoms,
                    "carbonothioyl",
                    "methanethioyl",
                    amino_base="carbamothioyl",
                )
            )
        else:
            s_oxygens = _double_bonded_neighbors(mol, nxt, "O")
            if mol.atoms[nxt].symbol == "S" and s_oxygens:
                branch_idx = _first_substituent_neighbor(mol, nxt, {start_idx, *s_oxygens})
                branch = _name_branch_or_none(mol, branch_idx, exclude_atoms | {start_idx, nxt} | set(s_oxygens), nxt)
                if branch:
                    branches.append(f"{_stereo_prefix(mol.atoms[nxt])}{branch}{_sulfur_oxo_suffix(len(s_oxygens))}")
                else:
                    branches.append("sulfo")
            else:
                branch = name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx)
                if branch:
                    branches.append(branch)

    return _format_amino_from_branches(branches, is_double)


def _name_sulfur_subgraph(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None) -> str:
    """Name a sulfur-starting substituent fragment.

    Blue Book references: P-67 for sulfur prefixes, P-14.5 for lambda
    descriptors, and P-61.2.2.2 for sulfanyl/sulfanylidene prefixes.
    """

    is_double = _upstream_bond_order(mol, start_idx, upstream_atom) == 2
    s_oxygens = _double_bonded_neighbors(mol, start_idx, "O")
    s_nitrogens = _double_bonded_neighbors(mol, start_idx, "N")
    next_atoms = _subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom, s_oxygens + s_nitrogens)
    stereo_prefix = _stereo_prefix(mol.atoms[start_idx])

    if len(s_oxygens) == 1 and len(s_nitrogens) == 1:
        suffix = "sulfonimidoyl" + ("idene" if is_double else "")
        if not next_atoms:
            return f"{stereo_prefix}{suffix}"
        if len(next_atoms) == 1:
            branch = _name_branch_or_none(
                mol, next_atoms[0], exclude_atoms | {start_idx} | set(s_oxygens) | set(s_nitrogens), start_idx
            )
            return f"({stereo_prefix}{branch}{suffix})" if branch else f"{stereo_prefix}{suffix}"
        branches = [
            br
            for nxt in next_atoms
            if (br := name_subgraph(mol, nxt, exclude_atoms | {start_idx} | set(s_oxygens) | set(s_nitrogens), start_idx))
        ]
        branches.extend(["oxo"] * len(s_oxygens))
        branches.extend(["imino"] * len(s_nitrogens))
        return _format_lambda_substituent(mol, start_idx, branches, stereo_prefix, "sulfanylidene" if is_double else "sulfanyl")

    if s_oxygens:
        suffix = _sulfur_oxo_suffix(len(s_oxygens)) + ("idene" if is_double else "")
        if not next_atoms:
            return f"{stereo_prefix}{suffix}"
        if len(next_atoms) == 1:
            branch = _name_branch_or_none(mol, next_atoms[0], exclude_atoms | {start_idx} | set(s_oxygens), start_idx)
            return f"({stereo_prefix}{branch}{suffix})" if branch else f"{stereo_prefix}{suffix}"
        branches = [
            br
            for nxt in next_atoms
            if (br := name_subgraph(mol, nxt, exclude_atoms | {start_idx} | set(s_oxygens), start_idx))
        ]
        branches.extend(["oxo"] * len(s_oxygens))
        return _format_lambda_substituent(mol, start_idx, branches, stereo_prefix, "sulfanylidene" if is_double else "sulfanyl")

    if not next_atoms:
        return "thioxo" if is_double else f"{stereo_prefix}sulfanyl"

    if len(next_atoms) == 1:
        branch = name_subgraph(mol, next_atoms[0], exclude_atoms | {start_idx}, upstream_atom=start_idx)
        if branch in SIMPLE_SULFANYL_PREFIXES:
            return f"({stereo_prefix}{branch}sulfanyl)"
        if branch:
            return _format_element_substituent(stereo_prefix, branch, "sulfanyl", is_double=is_double)
        return f"{stereo_prefix}{'sulfanylidene' if is_double else 'sulfanyl'}"

    branches = [
        br for nxt in next_atoms if (br := name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx))
    ]
    return _format_lambda_substituent(mol, start_idx, branches, stereo_prefix, "sulfanylidene" if is_double else "sulfanyl")


def _name_chalcogen_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int | None,
    simple_prefixes: set[str],
    element_suffix: str,
    oxo_prefix: str,
) -> str:
    """Name selenium-like chalcogen substituent fragments.

    Blue Book references: P-61.2.2.2 and P-14.5 for chalcogen substituent and
    lambda naming patterns.
    """

    is_double = _upstream_bond_order(mol, start_idx, upstream_atom) == 2
    next_atoms = _subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    stereo_prefix = _stereo_prefix(mol.atoms[start_idx])
    if not next_atoms:
        return oxo_prefix if is_double else f"{stereo_prefix}{element_suffix}"
    if len(next_atoms) == 1:
        branch = name_subgraph(mol, next_atoms[0], exclude_atoms | {start_idx}, upstream_atom=start_idx)
        if branch in simple_prefixes:
            return f"({stereo_prefix}{branch}{element_suffix})"
        if branch:
            return _format_element_substituent(stereo_prefix, branch, element_suffix, is_double=is_double)
        return f"{stereo_prefix}{element_suffix + ('idene' if is_double else '')}"
    branches = [
        br for nxt in next_atoms if (br := name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx))
    ]
    return _format_lambda_substituent(
        mol, start_idx, branches, stereo_prefix, element_suffix + ("idene" if is_double else "")
    )


def _name_phosphorus_subgraph(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None) -> str:
    """Name a phosphorus-starting substituent fragment.

    Blue Book references: P-68 by analogy with heteroatom hydride prefixes and
    P-14.5 for valence-sensitive substituent forms.
    """

    is_double = _upstream_bond_order(mol, start_idx, upstream_atom) == 2
    p_oxygens = _double_bonded_neighbors(mol, start_idx, "O")
    next_atoms = _subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom, p_oxygens)
    stereo_prefix = _stereo_prefix(mol.atoms[start_idx])
    suffix = ("phosphoryl" if p_oxygens else "phosphanyl") + ("idene" if is_double else "")
    if not next_atoms:
        return f"{stereo_prefix}{suffix}"
    branches = [
        br for nxt in next_atoms if (br := name_subgraph(mol, nxt, exclude_atoms | {start_idx} | set(p_oxygens), start_idx))
    ]
    return f"({stereo_prefix}{_format_counted_prefixes(branches)}{suffix})"


def _name_group_13_14_subgraph(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None) -> str:
    """Name simple silicon and boron substituent fragments.

    Blue Book references: P-61.2 and P-14.5 for hydride-derived substituent
    names such as silyl, boryl, and their ylidene analogues.
    """

    is_double = _upstream_bond_order(mol, start_idx, upstream_atom) == 2
    suffix = ("silyl" if mol.atoms[start_idx].symbol == "Si" else "boryl") + ("idene" if is_double else "")
    next_atoms = _subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    if not next_atoms:
        return suffix
    branches = [
        br for nxt in next_atoms if (br := name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx))
    ]
    return f"({_format_counted_prefixes(branches)}{suffix})"


def _name_halogen_subgraph(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int | None) -> str:
    """Name a halogen-starting substituent fragment.

    Blue Book references: P-61.3 for fluoro/chloro/bromo/iodo prefixes and
    P-14.5 for lambda descriptors on substituted halogen centers.
    """

    next_atoms = _subgraph_neighbors(mol, start_idx, exclude_atoms, upstream_atom)
    symbol = mol.atoms[start_idx].symbol
    if not next_atoms:
        return HALOGEN_PREFIXES[symbol]
    branches = [
        br for nxt in next_atoms if (br := name_subgraph(mol, nxt, exclude_atoms | {start_idx}, upstream_atom=start_idx))
    ]
    valence = sum(mol.get_bond(start_idx, n).order for n in mol.get_neighbors(start_idx))
    return f"({_format_counted_prefixes(branches)}lambda^{valence}-{HALOGEN_LAMBDA_SUFFIXES[symbol]})"


def _subgraph_component(mol: Molecule, start_idx: int, exclude_atoms: set[int]) -> set[int]:
    """Return the connected recursive substituent component from ``start_idx``.

    Blue Book references: P-13.6; substituent prefixes are named as separate
    substituent components before being attached to the parent name.
    """

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


def _direct_subgraph_prefix(mol: Molecule, start_idx: int, component: set[int]) -> str:
    """Return a direct functional-group prefix when the subgraph is one group.

    Blue Book references: P-63 through P-67 for direct detachable prefixes such
    as nitro, cyano, carboxy, carbamoyl, and halo-carbonyl prefixes.
    """

    for group in perceive_groups(mol):
        if start_idx in group.atoms_involved and group.atoms_involved.issubset(component):
            if group.key in DIRECT_GROUP_PREFIXES:
                return DIRECT_GROUP_PREFIXES[group.key]
            if group.key in substituents.SUBSTITUENTS:
                return substituents.get(group.key).prefix
    return ""


def _find_acyclic_subgraph_paths(
    mol: Molecule, start_idx: int, component: set[int], cyclic_atoms: set[int], sub_exclude: set[int]
) -> list[list[int]]:
    """Find carbon paths available for recursive acyclic substituent naming.

    Blue Book references: P-44 and P-45 for parent hydride selection in
    substituent names.
    """

    valid_nodes = {n for n in component if n not in cyclic_atoms and mol.atoms[n].is_carbon and n not in sub_exclude}
    paths = []

    def dfs_sub(curr, path, visited_nodes):
        neighbors = [n for n in mol.get_neighbors(curr) if n in valid_nodes and n not in visited_nodes]
        if not neighbors:
            if start_idx in path:
                paths.append(path)
            return
        for n in neighbors:
            dfs_sub(n, path + [n], visited_nodes | {n})

    endpoints = [n for n in valid_nodes if sum(1 for x in mol.get_neighbors(n) if x in valid_nodes) <= 1]
    start_nodes = endpoints if endpoints else valid_nodes
    for start in start_nodes:
        dfs_sub(start, [start], {start})
    return paths


def _select_subgraph_parent(mol: Molecule, start_idx: int, component: set[int], sub_exclude: set[int]):
    """Select parent candidates for a recursive substituent component.

    Blue Book references: P-44, P-45, P-52, and P-53 for parent selection in
    chains, rings, fused systems, and retained parents.
    """

    cyclic_atoms = get_cyclic_atoms(mol, sub_exclude)
    if start_idx in cyclic_atoms:
        ring_systems = find_ring_systems(mol, sub_exclude)
        valid_rings = [rs for rs in ring_systems if start_idx in rs.atoms]
        if not valid_rings:
            return None
        best_paths, is_ring, is_bicycle, is_spiro, is_polycycle, xyz, polycycle_descriptor = select_principal_parent(
            mol, [], valid_rings, [start_idx]
        )
        return best_paths, is_ring, is_bicycle, is_spiro, is_polycycle, xyz, polycycle_descriptor, (
            is_bicycle or is_spiro or is_polycycle
        )

    paths = _find_acyclic_subgraph_paths(mol, start_idx, component, cyclic_atoms, sub_exclude)
    if not paths:
        return None
    best_paths, is_ring, is_bicycle, is_spiro, is_polycycle, xyz, polycycle_descriptor = select_principal_parent(
        mol, paths, [], [start_idx]
    )
    return best_paths, is_ring, is_bicycle, is_spiro, is_polycycle, xyz, polycycle_descriptor, False


def _retained_subgraph_ring(mol: Molecule, path: list[int], is_ring: bool, is_bicycle: bool, is_polycycle: bool):
    """Return a retained ring name and locant maps when valid for this subgraph.

    Blue Book references: P-52 and P-53 for retained names, and P-22/P-25 for
    supported heterocycle retained names.
    """

    temp_retained = retained.get_retained_ring(mol, path) if is_ring else None
    if not temp_retained:
        return None, None
    retained_name_val, locant_maps = temp_retained
    if locant_maps is None and (is_bicycle or is_polycycle):
        return None, None
    if any(mol.atoms[idx].symbol not in RETAINED_RING_ELEMENTS for idx in path):
        return None, None
    return retained_name_val, locant_maps


def _find_spiro_side_pair(
    mol: Molecule, c_idx: int, n_subs: list[int], main_set: set[int], sub_exclude: set[int]
) -> tuple[int, int] | None:
    """Find a side-ring pair that forms a spiro substituent at ``c_idx``.

    Blue Book references: P-24 and P-52.3 for spiro systems and spiro
    substituent citation.
    """

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


def _spiro_side_component(
    mol: Molecule, c_idx: int, side_start: int, main_set: set[int], sub_exclude: set[int]
) -> set[int]:
    """Return the atoms in a side ring used as a spiro substituent.

    Blue Book references: P-24 and P-52.3 for spiro ring construction.
    """

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


def _spiro_subgraph_name(mol: Molecule, c_idx: int, sub_comp: set[int]) -> str:
    """Name a side ring as a synthetic spiro substituent marker.

    Blue Book references: P-24 and P-52.3; the attachment atom is temporarily
    represented as silicon so the side ring can be named independently, then the
    silane marker is stripped back out.
    """

    sub_mol = Molecule()
    for n in sub_comp:
        atom = mol.atoms[n]
        symbol = "Si" if n == c_idx else atom.symbol
        sub_mol.add_atom(symbol=symbol, idx=n, charge=atom.charge, stereo=atom.stereo)
    for n in sub_comp:
        for nxt in mol.get_neighbors(n):
            if nxt in sub_comp and n < nxt:
                bond = mol.get_bond(n, nxt)
                sub_mol.add_bond(u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring)

    sub_name_raw = name_component(sub_mol, sub_comp, is_substituent=False)
    match = re.search(r"(?:(^|-)(\d+)-)?sil[a]?", sub_name_raw)
    if not match:
        return f"[SPIRO]-1-{sub_name_raw}"

    loc = match.group(2) if match.group(2) else "1"
    if match.group(2):
        sub_name_clean = re.sub(rf"(^|-){loc}-sil[a]?-?", r"\1", sub_name_raw)
    else:
        sub_name_clean = re.sub(r"sil[a]?-?", "", sub_name_raw)

    sub_name_clean = sub_name_clean.replace("--", "-").strip("-")
    sub_name_clean = sub_name_clean.replace("-cyclo", "cyclo")
    if not sub_name_clean:
        sub_name_clean = "methane"
    return f"[SPIRO]-{loc}-{sub_name_clean}"


def _collect_subgraph_substituents(
    mol: Molecule,
    candidate_path: list[int],
    sub_perceived: list[PerceivedGroup],
    sub_exclude: set[int],
) -> dict[int, list[str]]:
    """Collect prefixes attached to a recursive subgraph parent.

    Blue Book references: P-14.2, P-16.5, P-44, P-61 through P-67, and P-24
    for multiplicative prefixes, complex prefixes, parent substituents, and
    spiro side-ring substituents.
    """

    main_set = set(candidate_path)
    subst_mapping: dict[int, list[str]] = {}
    sub_handled_atoms = set()

    for group in sub_perceived:
        if group.attachment_carbon in main_set and not group.is_principal_candidate:
            name = substituents.get(group.key).prefix if group.key in substituents.SUBSTITUENTS else ""
            if name:
                subst_mapping.setdefault(group.attachment_carbon, []).append(name)
                sub_handled_atoms.update(group.atoms_involved)

    for c_idx in candidate_path:
        n_subs = [
            n_idx
            for n_idx in mol.get_neighbors(c_idx)
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude
        ]

        spiro_pair = _find_spiro_side_pair(mol, c_idx, n_subs, main_set, sub_exclude)
        if spiro_pair:
            sub_comp = _spiro_side_component(mol, c_idx, spiro_pair[0], main_set, sub_exclude)
            subst_mapping.setdefault(c_idx, []).append(_spiro_subgraph_name(mol, c_idx, sub_comp))
            sub_handled_atoms.update(sub_comp - {c_idx})
            n_subs = [n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude:
                branch_name = name_subgraph(mol, n_idx, sub_exclude | main_set, upstream_atom=c_idx)
                if branch_name:
                    subst_mapping.setdefault(c_idx, []).append(branch_name)

    return subst_mapping


def _choose_subgraph_numbering(
    mol: Molecule,
    candidate_paths: list[list[int]],
    start_idx: int,
    subst_mapping: dict[int, list[str]],
    locant_maps,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    fixed_start: bool,
    retained_name_val: str | None,
):
    """Choose a numbering and locant map for recursive substituent assembly.

    Blue Book references: P-14.4, P-44, and P-45; retained-ring locant maps are
    compared before falling back to normal parent numbering.
    """

    if locant_maps:

        def evaluate_map(lmap):
            def get_val(idx):
                return parse_locant(lmap[idx])

            principal = sorted([get_val(start_idx)])
            het_by_priority = {}
            for atom in mol:
                if atom.idx in lmap and not atom.is_carbon:
                    priority = atom.element.hw_priority or 99
                    het_by_priority.setdefault(priority, []).append(atom.idx)
            heteroatom_eval = tuple(
                sorted([get_val(idx) for idx in het_by_priority[priority]])
                for priority in sorted(het_by_priority.keys())
            )
            substituent_eval = sorted([get_val(idx) for idx in set(subst_mapping.keys()) if idx in lmap])
            return heteroatom_eval + (principal, substituent_eval)

        locant_map = min(locant_maps, key=evaluate_map)
        return list(locant_map.keys()), locant_map

    return (
        number_parent(
            mol,
            candidate_paths,
            {start_idx},
            subst_mapping,
            is_ring,
            is_bicycle,
            is_spiro,
            is_polycycle=is_polycycle,
            fixed_start=fixed_start,
            retained_name=retained_name_val,
        ),
        None,
    )


def _subgraph_locant_getter(numbered_path: list[int], locant_map):
    """Create a locant accessor for numbered recursive substituent atoms.

    Blue Book references: P-14.3 for locants.
    """

    def get_loc(idx):
        return locant_map[idx] if locant_map else str(numbered_path.index(idx) + 1)

    return get_loc


def _add_indicated_hydrogens(mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc) -> None:
    """Add indicated hydrogen locants for retained ring substituent names.

    Blue Book references: P-14.7, P-22.2, and P-25 for indicated hydrogen in
    retained heterocycle and hydrocarbon names.
    """

    if parts.retained_name not in INDICATED_H_RETAINED_NAMES:
        return
    for idx in numbered_path:
        atom = mol.atoms[idx]
        if atom.symbol in ["N", "C"]:
            ring_bonds = [mol.get_bond(idx, n) for n in mol.get_neighbors(idx) if n in numbered_path]
            if sum(b.order for b in ring_bonds) == 2:
                parts.indicated_hydrogens.append(get_loc(idx))


def _add_subgraph_substituents(parts: AssemblyParts, subst_mapping: dict[int, list[str]], get_loc) -> None:
    """Add collected substituent prefixes to assembly parts.

    Blue Book references: P-14.2 and P-16.5 for locants, multiplicative
    prefixes, and complex substituent citation.
    """

    for c_idx, names in subst_mapping.items():
        locant = get_loc(c_idx)
        for name in names:
            existing = next((s for s in parts.substituents if s.name == name), None)
            if existing:
                existing.locants.append(locant)
            else:
                parts.substituents.append(SubstituentItem(name=name, locants=[locant]))


def _add_subgraph_parent_features(
    mol: Molecule,
    parts: AssemblyParts,
    numbered_path: list[int],
    get_loc,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
) -> None:
    """Add replacement prefixes and unsaturation locants to assembly parts.

    Blue Book references: P-31 for unsaturation, P-44 for parent hydrides, and
    P-51 for skeletal replacement prefixes.
    """

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
                parts.a_prefixes.append(SubstituentItem(name=hw_stem, locants=[loc]))

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
                    else:
                        parts.unsaturations.append(UnsaturationItem(bond_key=bond_key, locants=[locant_str]))


def _build_subgraph_parts(
    mol: Molecule,
    start_idx: int,
    upstream_atom: int | None,
    numbered_path: list[int],
    get_loc,
    retained_name_val: str | None,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    xyz,
    polycycle_descriptor,
) -> AssemblyParts:
    """Create assembly parts for a recursive substituent parent.

    Blue Book references: P-13.6, P-14.3, P-31, P-44, P-45, and P-91/P-93 for
    substituent suffixes, locants, parent descriptors, and stereochemistry.
    """

    attach_locant = get_loc(start_idx)
    upstream_order = _upstream_bond_order(mol, start_idx, upstream_atom)
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
        is_double_attach=upstream_order == 2,
        is_triple_attach=upstream_order == 3,
        attachment_locant=attach_locant,
        retained_name=retained_name_val,
    )

    for atom_idx in numbered_path:
        if mol.atoms[atom_idx].stereo:
            parts.stereo_features.append((get_loc(atom_idx), mol.atoms[atom_idx].stereo))
    return parts


def _finalize_subgraph_name(name: str, parts: AssemblyParts) -> str:
    """Apply recursive-substituent wrapping rules to an assembled name.

    Blue Book references: P-13.6 and P-16.5 for substituent suffix citation and
    parentheses around complex substituent prefixes.
    """

    if name == "phenyl" and not parts.substituents:
        return name
    if (
        (name.endswith("yl") or name.endswith("ylidene") or name.endswith("ylidyne"))
        and not parts.substituents
        and not parts.unsaturations
        and str(parts.attachment_locant) == "1"
        and not name.startswith("bicyclo")
        and not name.startswith("spiro")
        and not name.startswith("tricyclo")
    ):
        return name
    return f"({name})"


def name_subgraph(mol: Molecule, start_idx: int, exclude_atoms: set[int], upstream_atom: int = None) -> str:
    """Name a recursive substituent subgraph attached to the current parent.

    Blue Book references: P-13.6, P-14.2, P-16.5, P-61, P-62, P-63, P-65,
    P-66, and P-67.  Extendable prefix vocabularies are loaded from
    ``data/namer_rules.json``.
    """

    start_atom = mol.atoms[start_idx]
    cyclic_atoms_global = get_cyclic_atoms(mol, exclude_atoms)

    if not start_atom.is_carbon and start_idx not in cyclic_atoms_global:
        heteroatom_handlers = {
            "O": _name_oxygen_subgraph,
            "N": _name_nitrogen_subgraph,
            "S": _name_sulfur_subgraph,
            "P": _name_phosphorus_subgraph,
            "Si": _name_group_13_14_subgraph,
            "B": _name_group_13_14_subgraph,
        }
        if start_atom.symbol == "Se":
            return _name_chalcogen_subgraph(
                mol, start_idx, exclude_atoms, upstream_atom, SIMPLE_SELANYL_PREFIXES, "selanyl", "selenoxo"
            )
        if start_atom.symbol in HALOGEN_PREFIXES:
            return _name_halogen_subgraph(mol, start_idx, exclude_atoms, upstream_atom)
        if start_atom.symbol in heteroatom_handlers:
            return heteroatom_handlers[start_atom.symbol](mol, start_idx, exclude_atoms, upstream_atom)

    component = _subgraph_component(mol, start_idx, exclude_atoms)
    direct_prefix = _direct_subgraph_prefix(mol, start_idx, component)
    if direct_prefix:
        return direct_prefix

    sub_exclude = set(mol.atoms.keys()) - component
    parent_selection = _select_subgraph_parent(mol, start_idx, component, sub_exclude)
    if parent_selection is None:
        return ""

    (
        candidate_paths,
        is_ring,
        is_bicycle,
        is_spiro,
        is_polycycle,
        xyz,
        polycycle_descriptor,
        fixed_start_val,
    ) = parent_selection

    retained_name_val, locant_maps = _retained_subgraph_ring(
        mol, candidate_paths[0], is_ring, is_bicycle, is_polycycle
    )
    sub_perceived = perceive_groups(mol)
    subst_mapping = _collect_subgraph_substituents(mol, candidate_paths[0], sub_perceived, sub_exclude)
    numbered_path, locant_map = _choose_subgraph_numbering(
        mol,
        candidate_paths,
        start_idx,
        subst_mapping,
        locant_maps,
        is_ring,
        is_bicycle,
        is_spiro,
        is_polycycle,
        fixed_start_val,
        retained_name_val,
    )
    get_loc = _subgraph_locant_getter(numbered_path, locant_map)

    parts = _build_subgraph_parts(
        mol,
        start_idx,
        upstream_atom,
        numbered_path,
        get_loc,
        retained_name_val,
        is_ring,
        is_bicycle,
        is_spiro,
        is_polycycle,
        xyz,
        polycycle_descriptor,
    )
    _emit_bond_stereo(mol, parts, numbered_path, get_loc, sub_exclude, upstream_atom)
    _add_indicated_hydrogens(mol, parts, numbered_path, get_loc)
    _add_subgraph_substituents(parts, subst_mapping, get_loc)
    _add_subgraph_parent_features(mol, parts, numbered_path, get_loc, is_bicycle, is_spiro, is_polycycle)

    return _finalize_subgraph_name(assemble_name(parts), parts)


def _single_atom_component_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Return the name for a one-atom ionic component, when supported.

    Blue Book references: P-72 for names of ionic components in salts.
    """

    if len(component_atoms) != 1:
        return ""
    atom = mol.atoms[list(component_atoms)[0]]
    if atom.symbol in SINGLE_ATOM_CATIONS:
        return atom.element.name
    if atom.symbol in SINGLE_ATOM_ANIONS:
        return SINGLE_ATOM_ANIONS[atom.symbol]
    return ""


def _component_groups(mol: Molecule, component_atoms: set[int]) -> list[PerceivedGroup]:
    """Return perceived groups whose attachment atom is inside a component.

    Blue Book references: P-41, P-44, and P-63 through P-67 for characteristic
    group and detachable-prefix recognition.
    """

    return [group for group in perceive_groups(mol) if group.attachment_carbon in component_atoms]


def _component_principal_key(perceived_groups: list[PerceivedGroup], is_substituent: bool) -> str | None:
    """Select the senior principal characteristic group for a component.

    Blue Book references: P-41 and P-44 for characteristic group seniority and
    suffix selection.
    """

    if is_substituent:
        return None
    candidates = [group.key for group in perceived_groups if group.is_principal_candidate]
    return suffixes.most_senior(candidates).key if candidates else None


def _anhydride_half_name(mol: Molecule, start_c: int, bridge_o: int) -> str:
    """Name one acid half of an anhydride component.

    Blue Book references: P-65.7 for acid anhydride names.
    """

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
                    u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring
                )

    return name_component(sub_mol, half_atoms).replace(" acid", "")


def _try_name_anhydride_component(mol: Molecule, perceived_groups: list[PerceivedGroup], principal_key: str | None) -> str:
    """Return an anhydride component name when the component is an anhydride.

    Blue Book references: P-65.7 for symmetrical and unsymmetrical acid
    anhydride names.
    """

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
        name1 = _anhydride_half_name(mol, c_neighbors[0], bridge_o)
        name2 = _anhydride_half_name(mol, c_neighbors[1], bridge_o)
        if name1 == name2:
            return f"{name1} anhydride"
        names = sorted([name1, name2])
        return f"{names[0]} {names[1]} anhydride"
    return ""


def _retarget_external_carbonyl_groups(
    mol: Molecule,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    exclude_atoms: set[int],
    cyclic_atoms_all: set[int],
) -> None:
    """Move exocyclic carbonyl group attachment onto the parent chain atom.

    Blue Book references: P-44.3 and P-65/P-66 for carbonyl-containing suffix
    groups attached to chains and rings.
    """

    for group in perceived_groups:
        if group.key == principal_key or group.key not in CHAIN_EXTERNAL_CARBONYL_GROUPS:
            continue
        group_c = group.attachment_carbon
        if group_c in cyclic_atoms_all:
            continue
        adj_c = [n for n in mol.get_neighbors(group_c) if mol.atoms[n].is_carbon and n not in group.atoms_involved]
        if len(adj_c) == 1:
            group.attachment_carbon = adj_c[0]
            group.atoms_involved.add(group_c)
            exclude_atoms.add(group_c)


def _exclude_nonparent_group_atoms(
    mol: Molecule, perceived_groups: list[PerceivedGroup], exclude_atoms: set[int], cyclic_atoms_all: set[int]
) -> None:
    """Exclude linker atoms that should not become part of the parent skeleton.

    Blue Book references: P-44.3 and P-65 through P-67 for suffix group atoms
    cited outside the parent hydride.
    """

    for group in perceived_groups:
        if group.key == "anhydride":
            atom_idx = next((o for o in group.atoms_involved if mol.degree(o) == 2 and mol.atoms[o].symbol == "O"), None)
        elif group.key in ESTER_LIKE_PREFIX_GROUPS:
            if group.key in PEROXY_ESTER_GROUPS:
                atom_idx = next(
                    (
                        o
                        for o in group.atoms_involved
                        if mol.atoms[o].symbol == "O"
                        and mol.get_bond(o, group.attachment_carbon) is None
                        and mol.degree(o) == 2
                    ),
                    None,
                )
            else:
                atom_idx = next((o for o in group.atoms_involved if mol.degree(o) == 2 or mol.atoms[o].charge == -1), None)
        elif group.key in SULFONYL_PREFIX_GROUPS:
            atom_idx = next((s for s in group.atoms_involved if mol.atoms[s].symbol == "S"), None)
        elif group.key in AMIDE_LIKE_PREFIX_GROUPS:
            atom_idx = next((n for n in group.atoms_involved if mol.atoms[n].symbol == "N"), None)
        else:
            atom_idx = None

        if atom_idx is not None and atom_idx not in cyclic_atoms_all:
            exclude_atoms.add(atom_idx)


def _partition_principal_and_prefix_groups(
    perceived_groups: list[PerceivedGroup], principal_key: str | None
) -> tuple[list[int], list[PerceivedGroup]]:
    """Split perceived groups into principal attachment atoms and prefixes.

    Blue Book references: P-41 and P-44 for principal groups and prefixes.
    """

    principal_carbons = []
    prefix_groups = []
    for group in perceived_groups:
        if group.key == principal_key:
            principal_carbons.append(group.attachment_carbon)
        else:
            prefix_groups.append(group)
    return principal_carbons, prefix_groups


def _select_component_parent(mol: Molecule, exclude_atoms: set[int], principal_carbons: list[int]):
    """Select the parent chain or ring system for a connected component.

    Blue Book references: P-44 and P-45 for parent hydride selection.
    """

    chains = find_all_carbon_paths(mol, exclude_atoms)
    ring_systems = find_ring_systems(mol, exclude_atoms)
    if not chains and not ring_systems:
        return None
    return select_principal_parent(mol, chains, ring_systems, principal_carbons)


def _filter_component_groups_to_parent(
    perceived_groups: list[PerceivedGroup], parent_set: set[int], is_substituent: bool
) -> tuple[list[PerceivedGroup], str | None, list[int], list[PerceivedGroup]]:
    """Keep only groups attached to the selected parent and recompute seniority.

    Blue Book references: P-44 for choosing the parent before assigning suffix
    and prefix roles.
    """

    valid_groups = [group for group in perceived_groups if group.attachment_carbon in parent_set]
    principal_key = _component_principal_key(valid_groups, is_substituent)
    principal_carbons, prefix_groups = _partition_principal_and_prefix_groups(valid_groups, principal_key)
    return valid_groups, principal_key, principal_carbons, prefix_groups


def _principal_involved_atoms(
    perceived_groups: list[PerceivedGroup], principal_key: str | None, parent_path: list[int]
) -> set[int]:
    """Return atoms already consumed by the principal group on the parent.

    Blue Book references: P-44 and P-65 through P-67; principal group atoms are
    not re-named as substituent branches.
    """

    atoms = set()
    if principal_key:
        for group in perceived_groups:
            if group.key == principal_key and group.attachment_carbon in parent_path:
                atoms.update(group.atoms_involved)
    return atoms


def _ester_prefix_from_group(mol: Molecule, group: PerceivedGroup, sub_exclude: set[int], suffix_text: str) -> str:
    """Return an alkoxycarbonyl or alkoxysulfonyl-style prefix.

    Blue Book references: P-65.6 and P-67.1 for ester and sulfonate prefixes
    when these groups are not principal.
    """

    if group.key in PEROXY_ESTER_GROUPS:
        single_o = next(
            (
                o
                for o in group.atoms_involved
                if mol.atoms[o].symbol == "O"
                and mol.get_bond(o, group.attachment_carbon) is None
                and mol.degree(o) == 2
            ),
            None,
        )
    else:
        single_o = next((o for o in group.atoms_involved if mol.degree(o) == 2 or mol.atoms[o].charge == -1), None)

    if single_o is None:
        return ""
    r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in group.atoms_involved), None)
    if r_group_c is None:
        return ""
    branch_name = name_subgraph(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
    if not branch_name:
        return ""
    return f"({_oxy_prefix_from_branch(branch_name)}{suffix_text})"


def _amide_prefix_from_group(mol: Molecule, group: PerceivedGroup, sub_exclude: set[int]) -> str:
    """Return carbamoyl or carbamothioyl prefix text for an amide-like group.

    Blue Book references: P-66.1 for amide and thioamide prefixes, including
    N-substituted prefix forms.
    """

    single_n = next((n for n in group.atoms_involved if mol.atoms[n].symbol == "N"), None)
    if single_n is None:
        return ""
    n_subs = [n for n in mol.get_neighbors(single_n) if n not in group.atoms_involved and mol.atoms[n].symbol != "H"]
    if not n_subs:
        return AMIDE_PREFIX_BASES[group.key]
    sub_names = [name_subgraph(mol, x, sub_exclude | {single_n}, upstream_atom=single_n) for x in n_subs]
    return f"({_format_counted_prefixes(sub_names)}{AMIDE_PREFIX_BASES[group.key]})"


def _collect_component_prefix_substituents(
    mol: Molecule,
    prefix_groups: list[PerceivedGroup],
    parent_path: list[int],
    sub_exclude: set[int],
) -> tuple[dict[int, list[str]], set[int]]:
    """Collect characteristic groups cited as prefixes on the component parent.

    Blue Book references: P-61 through P-67 for detachable prefixes and suffix
    groups cited as prefixes when they are not principal.
    """

    main_set = set(parent_path)
    subst_mapping: dict[int, list[str]] = {}
    handled_prefix_atoms = set()

    for group in prefix_groups:
        if group.key in PREFIX_GROUPS_TO_SKIP or group.attachment_carbon not in main_set:
            continue

        name = ""
        if group.key in ESTER_LIKE_PREFIX_GROUPS:
            name = _ester_prefix_from_group(mol, group, sub_exclude, "carbonyl")
        elif group.key in AMIDE_LIKE_PREFIX_GROUPS:
            name = _amide_prefix_from_group(mol, group, sub_exclude)
        elif group.key in CARBOXY_PREFIX_GROUPS:
            name = "carboxy"
        elif group.key in CYANO_PREFIX_GROUPS:
            name = "cyano"
        elif group.key in ACID_HALIDE_PREFIXES:
            name = ACID_HALIDE_PREFIXES[group.key]
        elif group.key in PEROXY_ACID_PREFIX_GROUPS:
            name = "carboperoxy"
        elif group.key in SULFONYL_PREFIX_GROUPS:
            name = _ester_prefix_from_group(mol, group, sub_exclude, "sulfonyl") or "sulfo"
        elif group.key in DIRECT_PREFIX_GROUPS:
            name = DIRECT_PREFIX_GROUPS[group.key]
        elif group.attachment_carbon in parent_path:
            name = suffixes.get(group.key).prefix if group.is_principal_candidate else substituents.get(group.key).prefix

        if name:
            subst_mapping.setdefault(group.attachment_carbon, []).append(name)
            handled_prefix_atoms.update(group.atoms_involved)

    return subst_mapping, handled_prefix_atoms


def _collect_component_branch_substituents(
    mol: Molecule,
    parent_path: list[int],
    subst_mapping: dict[int, list[str]],
    handled_prefix_atoms: set[int],
    principal_involved_atoms: set[int],
    base_exclude: set[int],
    sub_exclude: set[int],
) -> None:
    """Collect ordinary branch and spiro substituents from the component parent.

    Blue Book references: P-14.2, P-16.5, P-24, P-44, and P-61 through P-67.
    """

    main_set = set(parent_path)
    for c_idx in parent_path:
        n_subs = [
            n_idx
            for n_idx in mol.get_neighbors(c_idx)
            if n_idx not in main_set
            and n_idx not in principal_involved_atoms
            and n_idx not in handled_prefix_atoms
            and n_idx not in base_exclude
        ]

        spiro_pair = _find_spiro_side_pair(mol, c_idx, n_subs, main_set, base_exclude)
        if spiro_pair:
            sub_comp = _spiro_side_component(mol, c_idx, spiro_pair[0], main_set, base_exclude)
            subst_mapping.setdefault(c_idx, []).append(_spiro_subgraph_name(mol, c_idx, sub_comp))
            handled_prefix_atoms.update(sub_comp - {c_idx})
            n_subs = [n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in principal_involved_atoms and n_idx not in handled_prefix_atoms:
                branch_name = name_subgraph(mol, n_idx, sub_exclude | main_set, upstream_atom=c_idx)
                if branch_name:
                    subst_mapping.setdefault(c_idx, []).append(branch_name)


def _choose_component_numbering(
    mol: Molecule,
    best_paths: list[list[int]],
    principal_carbons: list[int],
    subst_mapping: dict[int, list[str]],
    locant_maps,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    retained_name_val: str | None,
):
    """Choose component numbering from retained maps or normal parent rules.

    Blue Book references: P-14.4, P-44, and P-45 for lowest locant sets and
    retained-name locant maps.
    """

    if locant_maps:

        def evaluate_map(lmap):
            def get_val(idx):
                return parse_locant(lmap[idx])

            principal_eval = sorted([get_val(c) for c in principal_carbons if c in lmap])
            het_by_priority = {}
            for atom in mol:
                if atom.idx in lmap and not atom.is_carbon:
                    priority = atom.element.hw_priority or 99
                    het_by_priority.setdefault(priority, []).append(atom.idx)
            heteroatom_eval = tuple(
                sorted([get_val(idx) for idx in het_by_priority[priority]])
                for priority in sorted(het_by_priority.keys())
            )
            substituent_eval = sorted([get_val(idx) for idx in set(subst_mapping.keys()) if idx in lmap])
            return heteroatom_eval + (principal_eval, substituent_eval)

        locant_map = min(locant_maps, key=evaluate_map)
        return list(locant_map.keys()), locant_map

    return (
        number_parent(
            mol,
            best_paths,
            principal_carbons,
            subst_mapping,
            is_ring,
            is_bicycle,
            is_spiro,
            is_polycycle=is_polycycle,
            retained_name=retained_name_val,
        ),
        None,
    )


def _build_component_parts(
    mol: Molecule,
    numbered_path: list[int],
    get_loc,
    retained_name_val: str | None,
    is_ring: bool,
    is_bicycle: bool,
    is_spiro: bool,
    is_polycycle: bool,
    xyz,
    polycycle_descriptor,
) -> AssemblyParts:
    """Create assembly parts for a complete connected component.

    Blue Book references: P-13, P-14, P-31, P-44, P-45, P-52/P-53, and P-91/P-93.
    """

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
    for atom_idx in numbered_path:
        if mol.atoms[atom_idx].stereo:
            parts.stereo_features.append((get_loc(atom_idx), mol.atoms[atom_idx].stereo))
    return parts


def _add_component_front_modifiers(
    mol: Molecule,
    parts: AssemblyParts,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    sub_exclude: set[int],
) -> None:
    """Add ester/sulfonate front modifiers such as the alcohol component name.

    Blue Book references: P-65.6 and P-67.1 for ester and sulfonate component
    names with front modifiers.
    """

    if principal_key not in FRONT_MODIFIER_PRINCIPAL_GROUPS:
        return
    for group in perceived_groups:
        if group.key != principal_key:
            continue
        c_idx = group.attachment_carbon
        if group.key in PEROXY_ESTER_GROUPS:
            single_o = next(
                (
                    o
                    for o in group.atoms_involved
                    if mol.atoms[o].symbol == "O" and mol.get_bond(o, c_idx) is None and mol.degree(o) == 2
                ),
                None,
            )
        else:
            single_o = next((o for o in group.atoms_involved if mol.degree(o) == 2 or mol.atoms[o].charge == -1), None)
        if single_o is None:
            continue
        r_group_c = next((n for n in mol.get_neighbors(single_o) if n not in group.atoms_involved), None)
        if r_group_c is None:
            continue
        branch_name = name_subgraph(mol, r_group_c, sub_exclude | {single_o}, upstream_atom=single_o)
        if branch_name:
            parts.front_modifiers.append(_strip_outer_parentheses(branch_name))


def _add_component_n_substituents(
    mol: Molecule,
    parts: AssemblyParts,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    numbered_path: list[int],
    get_loc,
    sub_exclude: set[int],
) -> None:
    """Add N-substituent prefixes and N/N' locants for principal groups.

    Blue Book references: P-62 and P-66 for amines, amides, imines,
    hydrazones, and hydrazines with N-substitution.
    """

    if principal_key not in N_SUBSTITUENT_PRINCIPAL_GROUPS:
        return
    principal_groups = [g for g in perceived_groups if g.key == principal_key and g.attachment_carbon in numbered_path]
    principal_groups.sort(key=lambda g: parse_locant(get_loc(g.attachment_carbon)))

    n_idx_global = 0
    for group in principal_groups:
        c_idx = group.attachment_carbon
        nitrogens = [n for n in group.atoms_involved if mol.atoms[n].symbol == "N"]
        nitrogens.sort(key=lambda n: mol.get_bond(n, c_idx) is not None, reverse=True)
        for single_n in nitrogens:
            n_substituents = [
                n
                for n in mol.get_neighbors(single_n)
                if n != c_idx and n not in group.atoms_involved and mol.atoms[n].symbol != "H"
            ]

            if principal_key == "hydrazine":
                loc_prefix = "N" if single_n == nitrogens[0] else "N'"
            elif principal_key in HYDRAZONE_PRINCIPAL_GROUPS:
                loc_prefix = "N"
            elif len(principal_groups) == 1 and len(nitrogens) == 1:
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


def _add_component_principal_group(
    parts: AssemblyParts, principal_key: str | None, principal_carbons: list[int], numbered_path: list[int], get_loc
) -> None:
    """Add the principal characteristic group suffix locants to assembly parts.

    Blue Book references: P-41 and P-44 for senior characteristic group suffixes.
    """

    if principal_key:
        locants = sorted([get_loc(c) for c in principal_carbons if c in numbered_path], key=parse_locant)
        parts.principal_group = PrincipalGroupItem(key=principal_key, locants=locants)


def _add_component_substituents(
    parts: AssemblyParts, subst_mapping: dict[int, list[str]], numbered_path: list[int], get_loc
) -> None:
    """Add collected component substituents to assembly parts.

    Blue Book references: P-14.2 and P-16.5 for substituent locants and complex
    prefix citation.
    """

    for c_idx, names in subst_mapping.items():
        if c_idx in numbered_path:
            locant = get_loc(c_idx)
            for name in names:
                existing = next((s for s in parts.substituents if s.name == name), None)
                if existing:
                    existing.locants.append(locant)
                else:
                    parts.substituents.append(SubstituentItem(name=name, locants=[locant]))


def name_component(mol: Molecule, component_atoms: set[int], is_substituent: bool = False) -> str:
    """Name one connected component or recursive component of a molecule.

    Blue Book references: P-44 and P-45 for parent selection, P-52/P-53 for
    retained names, P-61 through P-67 for prefixes and characteristic group
    suffixes, and P-72 for one-atom ionic components.
    """

    single_atom_name = _single_atom_component_name(mol, component_atoms)
    if single_atom_name:
        return single_atom_name

    perceived_groups = _component_groups(mol, component_atoms)
    principal_key = _component_principal_key(perceived_groups, is_substituent)

    anhydride_name = _try_name_anhydride_component(mol, perceived_groups, principal_key)
    if anhydride_name:
        return anhydride_name

    exclude_atoms = set(mol.atoms.keys()) - component_atoms
    cyclic_atoms_all = get_cyclic_atoms(mol, set())
    _retarget_external_carbonyl_groups(mol, perceived_groups, principal_key, exclude_atoms, cyclic_atoms_all)
    principal_carbons, _ = _partition_principal_and_prefix_groups(perceived_groups, principal_key)
    _exclude_nonparent_group_atoms(mol, perceived_groups, exclude_atoms, cyclic_atoms_all)

    parent_selection = _select_component_parent(mol, exclude_atoms, principal_carbons)
    if parent_selection is None:
        return "methane"

    best_paths, is_ring, is_bicycle, is_spiro, is_polycycle, xyz, polycycle_descriptor = parent_selection
    parent_path = best_paths[0]
    parent_set = set(parent_path)

    perceived_groups, principal_key, principal_carbons, prefix_groups = _filter_component_groups_to_parent(
        perceived_groups, parent_set, is_substituent
    )
    retained_name_val, locant_maps = _retained_subgraph_ring(mol, parent_path, is_ring, is_bicycle, is_polycycle)

    principal_involved_atoms = _principal_involved_atoms(perceived_groups, principal_key, parent_path)
    base_exclude = set(mol.atoms.keys()) - component_atoms
    sub_exclude = base_exclude | parent_set | principal_involved_atoms

    subst_mapping, handled_prefix_atoms = _collect_component_prefix_substituents(
        mol, prefix_groups, parent_path, sub_exclude
    )
    _collect_component_branch_substituents(
        mol,
        parent_path,
        subst_mapping,
        handled_prefix_atoms,
        principal_involved_atoms,
        base_exclude,
        sub_exclude,
    )

    numbered_path, locant_map = _choose_component_numbering(
        mol,
        best_paths,
        principal_carbons,
        subst_mapping,
        locant_maps,
        is_ring,
        is_bicycle,
        is_spiro,
        is_polycycle,
        retained_name_val,
    )
    get_loc = _subgraph_locant_getter(numbered_path, locant_map)

    parts = _build_component_parts(
        mol,
        numbered_path,
        get_loc,
        retained_name_val,
        is_ring,
        is_bicycle,
        is_spiro,
        is_polycycle,
        xyz,
        polycycle_descriptor,
    )
    _emit_bond_stereo(mol, parts, numbered_path, get_loc, base_exclude)
    _add_indicated_hydrogens(mol, parts, numbered_path, get_loc)
    _add_component_front_modifiers(mol, parts, perceived_groups, principal_key, sub_exclude)
    _add_component_n_substituents(mol, parts, perceived_groups, principal_key, numbered_path, get_loc, sub_exclude)
    _add_subgraph_parent_features(mol, parts, numbered_path, get_loc, is_bicycle, is_spiro, is_polycycle)
    _add_component_principal_group(parts, principal_key, principal_carbons, numbered_path, get_loc)
    _add_component_substituents(parts, subst_mapping, numbered_path, get_loc)

    name = assemble_name(parts)
    return SPECIAL_COMPONENT_NAMES.get(name, name)

def name_smiles(smiles: str) -> str:
    """Return an IUPAC-style name for a SMILES string.

    Blue Book references: P-13 for name construction, P-44/P-45 for parent
    selection and numbering, and P-72 for ordering disconnected ionic
    components.  Component-order metal names are data-backed in
    ``data/namer_rules.json``.
    """

    mol = read_smiles(smiles)
    if not mol.atoms:
        return ""
    components = get_connected_components(mol)

    names =[]
    for comp in components:
        comp_name = name_component(mol, comp)
        if comp_name:
            names.append(comp_name)

    def sort_key(name):
        return (0 if name in SALT_METAL_NAMES else 1, name)

    names.sort(key=sort_key)

    return " ".join(names)
