# openclatura/rules/retained.py
import itertools

from openclatura.hantzsch_widman import hw_spec_for_name, hw_spec_for_ring
from openclatura.molecule import Molecule
from openclatura.nomenclature import RULES
from openclatura.retained_fused_templates import match_retained_fused_templates


def _match_fused_templates(mol: Molecule, path: list[int]) -> tuple[str, list[dict[int, str]]] | None:
    """Resolve a fused retained parent from the locant-keyed graph templates."""

    # Strict first; relaxed reaches hydro derivatives but blurs tautomers.
    matches = match_retained_fused_templates(mol, path)
    if not matches and _is_plain_hydro_derivative(mol, path):
        matches = match_retained_fused_templates(mol, path, allow_nonaromatic=True)
    if not matches:
        return None
    name = matches[0].template.name
    maps = [match.atom_to_locant for match in matches if match.template.name == name]
    return name, maps


def _is_plain_hydro_derivative(mol: Molecule, path: list[int]) -> bool:
    """Whether the saturation is only hydrogen, so a hydro prefix can state it.

    A ring ketone or a charged ring atom saturates positions the emitter accounts
    for in its own way, and a mancude template match would then drop them.
    """

    ring = set(path)
    for idx in path:
        atom = mol.atoms[idx]
        if atom.charge:
            return False
        for neighbor in mol.get_neighbors(idx):
            bond = mol.get_bond(idx, neighbor)
            if neighbor not in ring and bond is not None and bond.order > 1:
                return False
    return True


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
    deg3_nodes = [u for u, d in internal_degrees.items() if d == 3]

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

    if any(symbol not in ("C", "N", "O", "S") for symbol in symbols):
        return _generated_monocycle(mol, path, size, total_bonds, double_bonds)

    # Fused systems only: the template table also holds monocycles, and those are
    # resolved below so a partly saturated ring still reaches its hydro name.
    if deg3_nodes:
        fused = _match_fused_templates(mol, path)
        if fused is not None:
            return fused

    if total_bonds == size:
        if size == 3 and double_bonds == 0:
            # One heteroatom in a three-ring: every carbon is equivalent, so the
            # numbering these names imply is the only one available.
            if n_count == 1 and o_count == 0 and s_count == 0:
                return "aziridine", None
            if o_count == 1 and n_count == 0 and s_count == 0:
                return "oxirane", None
            if s_count == 1 and n_count == 0 and o_count == 0:
                return "thiirane", None
        if size == 4 and double_bonds == 0:
            if n_count == 1 and o_count == 0 and s_count == 0:
                return "azetidine", None
        if size == 6 and double_bonds == 0:
            if n_count == 1 and o_count == 0 and s_count == 0:
                return "piperidine", None
            if o_count == 1 and n_count == 0 and s_count == 0:
                return "oxane", None
            if s_count == 1 and n_count == 0 and o_count == 0:
                return "thiane", None
            if n_count == 2 and o_count == 0 and s_count == 0:
                n_indices = [i for i in path if mol.atoms[i].symbol == "N"]
                dist = min(
                    abs(path.index(n_indices[0]) - path.index(n_indices[1])),
                    6 - abs(path.index(n_indices[0]) - path.index(n_indices[1])),
                )
                if dist == 3:
                    return "piperazine", None
            if n_count == 1 and o_count == 1 and s_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                o_idx = next(i for i in path if mol.atoms[i].symbol == "O")
                dist = min(abs(path.index(n_idx) - path.index(o_idx)), 6 - abs(path.index(n_idx) - path.index(o_idx)))
                if dist == 3:
                    return "morpholine", None
            if n_count == 1 and s_count == 1 and o_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                s_idx = next(i for i in path if mol.atoms[i].symbol == "S")
                dist = min(abs(path.index(n_idx) - path.index(s_idx)), 6 - abs(path.index(n_idx) - path.index(s_idx)))
                if dist == 3:
                    return "thiomorpholine", None
        if size == 5 and double_bonds == 0:
            if n_count == 1 and o_count == 0 and s_count == 0:
                return "pyrrolidine", None
            if o_count == 1 and n_count == 0 and s_count == 0:
                return "oxolane", None
            if s_count == 1 and n_count == 0 and o_count == 0:
                return "thiolane", None
            if n_count == 2 and o_count == 0 and s_count == 0:
                n_indices = [i for i in path if mol.atoms[i].symbol == "N"]
                dist = min(
                    abs(path.index(n_indices[0]) - path.index(n_indices[1])),
                    5 - abs(path.index(n_indices[0]) - path.index(n_indices[1])),
                )
                if dist == 1:
                    n1, n2 = n_indices
                    i1, i2 = path.index(n1), path.index(n2)
                    if (i1 + 1) % 5 == i2:
                        rot = path[i1:] + path[:i1]
                    else:
                        rot = path[i2:] + path[:i2]
                    locants = ["1", "2", "3", "4", "5"]
                    return "pyrazolidine", [{rot[i]: locants[i] for i in range(5)}]
                if dist == 2:
                    n1, n2 = n_indices
                    i1, i2 = path.index(n1), path.index(n2)
                    forward = (i1 + 2) % 5 == i2
                    if forward:
                        rot = path[i1:] + path[:i1]
                    else:
                        rot = path[i2:] + path[:i2]
                    locants = ["1", "2", "3", "4", "5"]
                    return "imidazolidine", [{rot[i]: locants[i] for i in range(5)}]
            if n_count == 1 and o_count == 1 and s_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                o_idx = next(i for i in path if mol.atoms[i].symbol == "O")
                dist = min(abs(path.index(n_idx) - path.index(o_idx)), 5 - abs(path.index(n_idx) - path.index(o_idx)))
                if dist == 1:
                    return "1,2-oxazolidine", None
                if dist == 2:
                    return "1,3-oxazolidine", None
            if n_count == 1 and s_count == 1 and o_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                s_idx = next(i for i in path if mol.atoms[i].symbol == "S")
                dist = min(abs(path.index(n_idx) - path.index(s_idx)), 5 - abs(path.index(n_idx) - path.index(s_idx)))
                if dist == 1:
                    return "1,2-thiazolidine", None
                if dist == 2:
                    return "1,3-thiazolidine", None

        if size == 6 and double_bonds == 3 and _has_no_cumulated_double_bonds(mol, path):
            if n_count == 0:
                return "benzene", None
            if n_count == 1:
                return "pyridine", None
            if n_count == 2:
                n_indices = [i for i in path if mol.atoms[i].symbol == "N"]
                dist = min(
                    abs(path.index(n_indices[0]) - path.index(n_indices[1])),
                    6 - abs(path.index(n_indices[0]) - path.index(n_indices[1])),
                )
                if dist == 1:
                    return "pyridazine", None
                if dist == 2:
                    return "pyrimidine", None
                if dist == 3:
                    return "pyrazine", None
            if n_count == 3:
                n_indices = [i for i in path if mol.atoms[i].symbol == "N"]
                idx_sorted = sorted([path.index(i) for i in n_indices])
                d1 = idx_sorted[1] - idx_sorted[0]
                d2 = idx_sorted[2] - idx_sorted[1]
                d3 = 6 - (idx_sorted[2] - idx_sorted[0])
                dists = sorted([d1, d2, d3])
                if dists == [1, 1, 4]:
                    return "1,2,3-triazine", None
                if dists == [1, 2, 3]:
                    return "1,2,4-triazine", None
                if dists == [2, 2, 2]:
                    return "1,3,5-triazine", None
                return None
        if size == 5 and double_bonds == 2 and _has_no_cumulated_double_bonds(mol, path):
            if o_count == 1 and n_count == 0 and s_count == 0:
                return "furan", None
            if s_count == 1 and n_count == 0 and o_count == 0:
                return "thiophene", None
            if n_count == 1 and o_count == 0 and s_count == 0:
                return "pyrrole", None
            if n_count == 2 and o_count == 0 and s_count == 0:
                n_indices = [i for i in path if mol.atoms[i].symbol == "N"]
                dist = min(
                    abs(path.index(n_indices[0]) - path.index(n_indices[1])),
                    5 - abs(path.index(n_indices[0]) - path.index(n_indices[1])),
                )
                if dist == 1:
                    return "pyrazole", None
                return "imidazole", None
            if n_count == 3 and o_count == 0 and s_count == 0:
                n_indices = [i for i in path if mol.atoms[i].symbol == "N"]
                idx_sorted = sorted([path.index(i) for i in n_indices])
                d1 = idx_sorted[1] - idx_sorted[0]
                d2 = idx_sorted[2] - idx_sorted[1]
                d3 = 5 - (idx_sorted[2] - idx_sorted[0])
                dists = sorted([d1, d2, d3])
                if dists == [1, 1, 3]:
                    return "1,2,3-triazole", None
                if dists == [1, 2, 2]:
                    return "1,2,4-triazole", None
                return None
            if n_count == 4 and o_count == 0 and s_count == 0:
                return "tetrazole", None
            if n_count == 1 and s_count == 1 and o_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                s_idx = next(i for i in path if mol.atoms[i].symbol == "S")
                dist = min(abs(path.index(n_idx) - path.index(s_idx)), 5 - abs(path.index(n_idx) - path.index(s_idx)))
                if dist == 1:
                    return "1,2-thiazole", None
                return "1,3-thiazole", None
            if n_count == 1 and o_count == 1 and s_count == 0:
                n_idx = next(i for i in path if mol.atoms[i].symbol == "N")
                o_idx = next(i for i in path if mol.atoms[i].symbol == "O")
                dist = min(abs(path.index(n_idx) - path.index(o_idx)), 5 - abs(path.index(n_idx) - path.index(o_idx)))
                if dist == 1:
                    return "1,2-oxazole", None
                return "1,3-oxazole", None

    # a partly saturated mancude heterocycle is that parent plus hydro prefixes.
    hydro_monocycle = _match_hydro_monocycle_retained(mol, path, size, total_bonds, double_bonds, symbols)
    if hydro_monocycle is not None:
        return hydro_monocycle, None

    # Only once every retained name has been ruled out: pyrazolidine is retained,
    # so the ring must not be spelled 1,2-diazolidine just because it can be.
    return _generated_monocycle(mol, path, size, total_bonds, double_bonds)


def _generated_monocycle(
    mol: Molecule, path: list[int], size: int, total_bonds: int, double_bonds: int
) -> tuple[str, None] | None:
    """The spelled-out Hantzsch-Widman parent for a monocycle, mancude or hydro."""

    if total_bonds != size:
        return None
    for hydro in (False, True):
        generated = _generated_monocycle_name(mol, path, double_bonds, hydro=hydro)
        if generated is not None:
            return generated, None
    return None


def _match_data_monocycle_retained(
    mol: Molecule,
    path: list[int],
    size: int,
    total_bonds: int,
    double_bonds: int,
    symbols: list[str],
) -> str | None:
    return _match_monocycle_spec(mol, path, size, total_bonds, double_bonds, symbols, hydro=False)


def _match_hydro_monocycle_retained(
    mol: Molecule,
    path: list[int],
    size: int,
    total_bonds: int,
    double_bonds: int,
    symbols: list[str],
) -> str | None:
    return _match_monocycle_spec(mol, path, size, total_bonds, double_bonds, symbols, hydro=True)


def _match_monocycle_spec(
    mol: Molecule,
    path: list[int],
    size: int,
    total_bonds: int,
    double_bonds: int,
    symbols: list[str],
    *,
    hydro: bool,
) -> str | None:
    if total_bonds != size:
        return None
    for spec in RULES.retained.monocycle_specs:
        if size != spec["size"]:
            continue
        if hydro:
            # Partly saturated heterocycles only: cyclohexene is not tetrahydrobenzene.
            if not spec.get("symbols") or not 0 < spec["double_bonds"] - double_bonds < spec["double_bonds"]:
                continue
        elif double_bonds != spec["double_bonds"]:
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
        expected_gap_cycle = spec.get("hetero_gap_cycle")
        if expected_gap_cycle is not None and _hetero_gap_cycle(mol, path) != list(expected_gap_cycle):
            continue
        expected_chalcogen = spec.get("chalcogen_nitrogen_distance_multiset")
        if expected_chalcogen is not None and _chalcogen_nitrogen_distance_multiset(mol, path) != sorted(
            expected_chalcogen
        ):
            continue
        if hydro and mancude_monocycle_hydro_plan(mol, path, spec["name"]) is None:
            continue
        return spec["name"]
    return None


def _generated_monocycle_name(mol: Molecule, path: list[int], double_bonds: int, *, hydro: bool) -> str | None:
    """Fall back to a spelled-out Hantzsch-Widman parent when no table entry fits."""

    spec = hw_spec_for_ring(mol, path)
    if spec is None:
        return None
    if hydro:
        if not 0 < spec["double_bonds"] - double_bonds < spec["double_bonds"]:
            return None
        if mancude_monocycle_hydro_plan(mol, path, spec["name"]) is None:
            return None
    elif double_bonds != spec["double_bonds"]:
        return None
    return spec["name"]


# Single-bonded in the mancude parent itself, so never a hydro position.
FIXED_SATURATED_RING_ELEMENTS = ("O", "S", "Se", "Te")


def mancude_monocycle_hydro_plan(
    mol: Molecule, path: list[int], retained_name: str | None
) -> tuple[int, int, list[int]] | None:
    """How a partly saturated retained mancude monocycle spells its saturation.

    Returns (indicated, added, citable): the ring atoms able to carry a cited
    hydrogen, and how many of them are the parent's own indicated hydrogen and
    how many are added hydrogen.  Anything left over takes a hydro prefix.
    None when the ring is not a partly saturated form of the parent.

    The two kinds are counted apart because they are numbered apart: indicated
    hydrogen outranks the principal group, added hydrogen does not.  A chalcogen
    is single-bonded in the mancude parent already, so it is neither.  Each
    exocyclic double bond buys one added position -- the ``(1H)`` of
    pyridin-2(1H)-one -- which is why 1H-pyrrole-2,5-dione needs no hydro prefix.
    """

    spec = next(
        (
            spec
            for spec in RULES.retained.monocycle_specs
            if spec["name"] == retained_name and spec.get("double_bonds") and len(path) == spec["size"]
        ),
        None,
    )
    if spec is None:
        spec = hw_spec_for_name(retained_name or "")
        if spec is None or not spec.get("double_bonds") or len(path) != spec["size"]:
            return None
    parent_double_bonds = int(spec["double_bonds"])
    if _count_double_bonds_in_ring(mol, path) >= parent_double_bonds:
        return None

    path_set = set(path)
    exocyclic = {idx for idx in path if _has_exocyclic_double_bond(mol, idx, path_set)}
    saturated = [idx for idx in path if idx not in exocyclic and _ring_bonds_all_single(mol, idx, path_set)]
    fixed = [idx for idx in saturated if mol.atoms[idx].symbol in FIXED_SATURATED_RING_ELEMENTS]
    citable = [idx for idx in saturated if idx not in fixed]

    parent_saturated = len(path) - 2 * parent_double_bonds
    indicated = min(max(parent_saturated - len(fixed), 0), len(citable))
    added = min(len(exocyclic), len(citable) - indicated)
    if (len(citable) - indicated - added) % 2:
        return None
    return indicated, added, citable


def _has_exocyclic_double_bond(mol: Molecule, atom_idx: int, path_set: set[int]) -> bool:
    return any(
        neighbor not in path_set and (bond := mol.get_bond(atom_idx, neighbor)) and bond.order > 1
        for neighbor in mol.get_neighbors(atom_idx)
    )


def _ring_bonds_all_single(mol: Molecule, atom_idx: int, path_set: set[int]) -> bool:
    bonds = [bond for n in mol.get_neighbors(atom_idx) if n in path_set and (bond := mol.get_bond(atom_idx, n))]
    return bool(bonds) and all(bond.order == 1 for bond in bonds)


def _symbol_counts_match(symbols: list[str], expected: dict[str, int]) -> bool:
    for symbol in set(expected) | {symbol for symbol in symbols if symbol != "C"}:
        if symbols.count(symbol) != int(expected.get(symbol, 0)):
            return False
    hetero_count = sum(int(value) for value in expected.values())
    return symbols.count("C") == len(symbols) - hetero_count


def _hetero_distance_multiset(mol: Molecule, path: list[int]) -> list[int]:
    hetero_indices = [idx for idx, atom_idx in enumerate(path) if mol.atoms[atom_idx].symbol != "C"]
    if len(hetero_indices) < 2:
        return []
    size = len(path)
    distances = []
    for left, right in itertools.combinations(hetero_indices, 2):
        distance = abs(left - right)
        distances.append(min(distance, size - distance))
    return sorted(distances)


def _chalcogen_nitrogen_distance_multiset(mol: Molecule, path: list[int]) -> list[int]:
    """Ring distances from each chalcogen to each nitrogen.

    The gap multiset is symmetric over all heteroatoms, so it cannot tell
    1,2,4-oxadiazole from 1,3,4-oxadiazole -- both are gaps 1,2,2.  Measuring
    from the chalcogen separates them: 1 and 2 against 2 and 2.
    """

    size = len(path)
    chalcogens = [idx for idx, atom_idx in enumerate(path) if mol.atoms[atom_idx].symbol in {"O", "S"}]
    nitrogens = [idx for idx, atom_idx in enumerate(path) if mol.atoms[atom_idx].symbol == "N"]
    distances = [min(abs(left - right), size - abs(left - right)) for left in chalcogens for right in nitrogens]
    return sorted(distances)


def _hetero_gap_multiset(mol: Molecule, path: list[int]) -> list[int]:
    return sorted(_hetero_gaps(mol, path))


def _hetero_gaps(mol: Molecule, path: list[int]) -> list[int]:
    hetero_indices = sorted(idx for idx, atom_idx in enumerate(path) if mol.atoms[atom_idx].symbol != "C")
    if len(hetero_indices) < 3:
        return []
    size = len(path)
    return [
        hetero_indices[(idx + 1) % len(hetero_indices)] - hetero_indices[idx]
        if idx + 1 < len(hetero_indices)
        else size - hetero_indices[idx] + hetero_indices[0]
        for idx in range(len(hetero_indices))
    ]


def _hetero_gap_cycle(mol: Molecule, path: list[int]) -> list[int]:
    """Canonical cyclic gap sequence between heteroatoms; the sorted multiset
    cannot separate 1,2,4,5- from 1,2,3,5-tetrazine, this can."""

    gaps = _hetero_gaps(mol, path)
    if not gaps:
        return []
    rotations = [
        tuple(sequence[offset:] + sequence[:offset]) for sequence in (gaps, gaps[::-1]) for offset in range(len(gaps))
    ]
    return list(min(rotations))
