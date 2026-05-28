"""Special component naming shortcuts."""

import re
from collections.abc import Callable

from .formatting import format_counted_prefixes, format_multiplier, oxy_prefix_from_branch, strip_outer_parentheses
from .molecule import Molecule
from .nomenclature import RULES
from .oxoacid_roles import CentralOxoRole, OxoLigandRole, central_oxo_roles
from .perception import PerceivedGroup
from .rules import multipliers, stems

ComponentNamer = Callable[..., str]
BranchNamer = Callable[..., str]


def single_atom_component_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Return the name for a one-atom ionic component, when supported."""

    if len(component_atoms) != 1:
        return ""
    atom = mol.atoms[list(component_atoms)[0]]
    if atom.symbol in RULES.ions.single_atom_cations:
        return atom.element.name
    if atom.symbol in RULES.ions.single_atom_anions:
        return RULES.ions.single_atom_anions[atom.symbol]
    hydride_name = RULES.components.mononuclear_parent_hydrides.get(atom.symbol)
    if hydride_name:
        return hydride_name
    return ""


def structural_replacement_parent_name(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: BranchNamer | None = None,
) -> str:
    """Return a replacement-parent hydride name from graph-derived specs."""

    return (
        simple_azine_parent_name(mol, component_atoms)
        or sulfonium_ylide_name(mol, component_atoms, branch_namer)
        or
        oxoacid_ester_name(mol, component_atoms, branch_namer)
        or oxoacid_parent_name(mol, component_atoms)
        or organophosphinic_acid_name(mol, component_atoms)
        or sulfoxide_parent_name(mol, component_atoms)
        or homonuclear_chain_parent_name(mol, component_atoms)
        or simple_central_parent_hydride_name(mol, component_atoms)
    )


def simple_azine_parent_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Name simple acyclic ketazines/aldazines from a graph C=N-N=C role."""

    for n1 in component_atoms:
        if mol.atoms[n1].symbol != "N":
            continue
        for n2 in mol.get_neighbors(n1):
            if n2 <= n1 or n2 not in component_atoms or mol.atoms[n2].symbol != "N":
                continue
            n_n_bond = mol.get_bond(n1, n2)
            if n_n_bond is None or n_n_bond.order != 1:
                continue
            c1 = _double_bonded_carbon(mol, n1, {n2})
            c2 = _double_bonded_carbon(mol, n2, {n1})
            if c1 is None or c2 is None:
                continue
            side1 = _component_atoms_until_blocked(mol, component_atoms, c1, {n1, n2})
            side2 = _component_atoms_until_blocked(mol, component_atoms, c2, {n1, n2})
            if not side1 or not side2 or side1 & side2:
                continue
            if side1 | side2 | {n1, n2} != component_atoms:
                continue
            parent1 = _simple_carbonyl_side_name(mol, side1, c1, as_ylidene=False)
            parent2 = _simple_carbonyl_side_name(mol, side2, c2, as_ylidene=False)
            ylidene1 = _simple_carbonyl_side_name(mol, side1, c1, as_ylidene=True)
            ylidene2 = _simple_carbonyl_side_name(mol, side2, c2, as_ylidene=True)
            if not parent1 or not parent2 or not ylidene1 or not ylidene2:
                continue
            # Prefer the longer carbonyl parent; this follows parent-size
            # preference and keeps the shorter side as the ylidene hydrazone
            # modifier.
            if (len(side2), parent2) > (len(side1), parent1):
                parent_name, modifier, parent_carbon, parent_n = parent2, ylidene1, c2, n2
            else:
                parent_name, modifier, parent_carbon, parent_n = parent1, ylidene2, c1, n1
            stereo = _hydrazone_stereo_prefix(mol, parent_carbon, parent_n, parent_name)
            return f"{stereo}{parent_name} {modifier}hydrazone"
    return ""


def _double_bonded_carbon(mol: Molecule, nitrogen: int, blocked: set[int]) -> int | None:
    candidates = [
        n
        for n in mol.get_neighbors(nitrogen)
        if n not in blocked
        and mol.atoms[n].is_carbon
        and (bond := mol.get_bond(nitrogen, n)) is not None
        and bond.order == 2
    ]
    return candidates[0] if len(candidates) == 1 else None


def _simple_carbonyl_side_name(mol: Molecule, side_atoms: set[int], carbonyl_carbon: int, *, as_ylidene: bool) -> str:
    if any(not mol.atoms[idx].is_carbon for idx in side_atoms):
        return ""
    if any(
        mol.get_bond(a, b).order not in {1, 2, 3}
        for a in side_atoms
        for b in mol.get_neighbors(a)
        if b in side_atoms and a < b
    ):
        return ""
    path = _longest_carbon_path_through(mol, side_atoms, carbonyl_carbon)
    if not path:
        return ""
    if len(path) != len(side_atoms):
        return ""
    loc = path.index(carbonyl_carbon) + 1
    reverse_loc = len(path) - loc + 1
    if reverse_loc < loc:
        loc = reverse_loc
    stem = stems.stem_for(len(path))
    if not stem:
        return ""
    unsaturation = _simple_path_unsaturation(mol, path)
    if as_ylidene:
        if len(path) == 1:
            return "methylidene"
        return f"{stem}{unsaturation or 'an'}-{loc}-ylidene"
    if len(path) == 1:
        return "formaldehyde"
    if loc == 1:
        return f"{stem}{unsaturation or 'an'}al"
    return f"{stem}{unsaturation or 'an'}-{loc}-one"


def _longest_carbon_path_through(mol: Molecule, side_atoms: set[int], required: int) -> list[int]:
    endpoints = [
        idx
        for idx in side_atoms
        if sum(1 for n in mol.get_neighbors(idx) if n in side_atoms) <= 1
    ] or list(side_atoms)
    best: list[int] = []

    def walk(curr: int, path: list[int], seen: set[int]) -> None:
        nonlocal best
        next_nodes = [n for n in mol.get_neighbors(curr) if n in side_atoms and n not in seen]
        if not next_nodes:
            if required in path and len(path) > len(best):
                best = path
            return
        for nxt in next_nodes:
            walk(nxt, path + [nxt], seen | {nxt})

    for endpoint in endpoints:
        walk(endpoint, [endpoint], {endpoint})
    return best


def _simple_path_unsaturation(mol: Molecule, path: list[int]) -> str:
    double_locs = []
    triple_locs = []
    for idx, (a, b) in enumerate(zip(path, path[1:]), start=1):
        order = mol.get_bond(a, b).order
        if order == 2:
            double_locs.append(idx)
        elif order == 3:
            triple_locs.append(idx)
    if not double_locs and not triple_locs:
        return ""
    parts = []
    if double_locs:
        suffix = "ene" if len(double_locs) == 1 else f"{multipliers.basic(len(double_locs))}ene"
        parts.append(f"{','.join(str(loc) for loc in double_locs)}-{suffix}")
    if triple_locs:
        suffix = "yne" if len(triple_locs) == 1 else f"{multipliers.basic(len(triple_locs))}yne"
        parts.append(f"{','.join(str(loc) for loc in triple_locs)}-{suffix}")
    return "-" + "-".join(parts)


def _hydrazone_stereo_prefix(mol: Molecule, carbon: int, nitrogen: int, parent_name: str) -> str:
    bond = mol.get_bond(carbon, nitrogen)
    if bond is None or bond.stereo not in {"E", "Z"}:
        return ""
    match = re.search(r"an-(\d+)-one$", parent_name)
    loc = match.group(1) if match else "1"
    return f"({loc}{bond.stereo})-"


def sulfonium_ylide_name(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: BranchNamer | None = None,
) -> str:
    """Name graph-proven sulfonium/carbanion ylides as full components."""

    sulfurs = [idx for idx in component_atoms if mol.atoms[idx].symbol == "S" and mol.atoms[idx].charge > 0]
    carbanions = [idx for idx in component_atoms if mol.atoms[idx].is_carbon and mol.atoms[idx].charge < 0]
    if len(sulfurs) != 1 or len(carbanions) != 1:
        return ""
    sulfur = sulfurs[0]
    ylide_carbon = carbanions[0]
    if mol.atoms[sulfur].total_h_count or mol.atoms[sulfur].explicit_h_count:
        return ""
    s_c_bond = mol.get_bond(sulfur, ylide_carbon)
    if s_c_bond is None or s_c_bond.order != 1:
        return ""
    sulfur_ligand_roots = [
        neighbor
        for neighbor in mol.get_neighbors(sulfur)
        if neighbor != ylide_carbon and neighbor in component_atoms and mol.atoms[neighbor].is_carbon
    ]
    if len(sulfur_ligand_roots) != 2:
        return ""
    sulfur_ligand_atoms: set[int] = set()
    sulfur_ligand_names = []
    for root in sulfur_ligand_roots:
        ligand_atoms = _carbon_ligand_atoms(mol, component_atoms, root, sulfur)
        if not ligand_atoms or sulfur_ligand_atoms & ligand_atoms:
            return ""
        sulfur_ligand_atoms.update(ligand_atoms)
        ligand_name = _alkyl_ligand_name(mol, component_atoms, root, sulfur)
        if not ligand_name:
            return ""
        sulfur_ligand_names.append(ligand_name)
    ylide_sub_roots = [
        neighbor
        for neighbor in mol.get_neighbors(ylide_carbon)
        if neighbor != sulfur and neighbor in component_atoms and mol.atoms[neighbor].symbol != "H"
    ]
    if len(ylide_sub_roots) > 1:
        return ""
    ylide_sub_atoms: set[int] = set()
    ylide_sub_name = ""
    if ylide_sub_roots:
        root = ylide_sub_roots[0]
        blocked = {sulfur, ylide_carbon} | sulfur_ligand_atoms
        ylide_sub_atoms = _component_atoms_until_blocked(mol, component_atoms, root, blocked)
        if not ylide_sub_atoms or ylide_sub_atoms & sulfur_ligand_atoms:
            return ""
        if branch_namer is not None:
            ylide_sub_name = strip_outer_parentheses(
                branch_namer(mol, root, set(mol.atoms) - component_atoms | blocked, upstream_atom=ylide_carbon)
            )
        if not ylide_sub_name:
            ylide_sub_name = _alkyl_ligand_name(mol, component_atoms, root, ylide_carbon)
        if not ylide_sub_name:
            return ""
    represented_atoms = {sulfur, ylide_carbon} | sulfur_ligand_atoms | ylide_sub_atoms
    if represented_atoms != component_atoms:
        return ""
    sulfur_prefix = format_counted_prefixes(sulfur_ligand_names)
    ylide_prefix = f"({ylide_sub_name}methanidyl)" if ylide_sub_name else "methanidyl"
    return f"{sulfur_prefix}{ylide_prefix}sulfonium"


def _component_atoms_until_blocked(
    mol: Molecule,
    component_atoms: set[int],
    root: int,
    blocked: set[int],
) -> set[int]:
    atoms = set()
    queue = [root]
    while queue:
        atom_idx = queue.pop(0)
        if atom_idx in atoms:
            continue
        if atom_idx not in component_atoms or atom_idx in blocked:
            return set()
        atoms.add(atom_idx)
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in blocked:
                continue
            if neighbor in component_atoms:
                queue.append(neighbor)
    return atoms


def oxoacid_parent_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Match functional parent hydrides by central atom and oxygen ligand counts."""

    roles = central_oxo_roles(mol, component_atoms)
    if len(roles) != 1:
        return ""
    role = roles[0]
    if role.has_organic_ester() or role.has_peroxy():
        return ""
    if {role.central, *role.oxygen_atoms} != component_atoms:
        return ""
    spec = _matching_oxoacid_spec_for_role(mol, role)
    if spec is None:
        return ""
    if role.has_anion() and spec.get("ester_suffix") and not role.count(OxoLigandRole.HYDROXY):
        return spec["ester_suffix"]
    return spec["name"]


def oxoacid_ester_name(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: BranchNamer | None = None,
) -> str:
    """Name organic esters of data-backed oxoacid parent hydrides."""

    matches = []
    for role in central_oxo_roles(mol, component_atoms):
        if role.has_peroxy() or role.count(OxoLigandRole.ALKOXY) != 1:
            continue
        ester_ligand = next(ligand for ligand in role.ligands if ligand.role == OxoLigandRole.ALKOXY)
        if ester_ligand.attachment_atom is None:
            continue
        spec = _matching_oxoacid_spec_for_role(mol, role)
        if spec is None or not spec.get("ester_suffix"):
            continue
        oxoacid_atoms = {role.central, *role.oxygen_atoms}
        ester_component_atoms = _ester_modifier_atoms(mol, component_atoms, ester_ligand.attachment_atom, oxoacid_atoms)
        if not ester_component_atoms or ester_component_atoms | oxoacid_atoms != component_atoms:
            continue
        ester_name = _ester_modifier_name(
            mol,
            component_atoms,
            ester_ligand.attachment_atom,
            ester_ligand.oxygen,
            oxoacid_atoms,
            branch_namer,
        )
        if not ester_name:
            continue
        matches.append(f"{ester_name} {_oxoacid_ester_suffix(spec, role)}")
    return matches[0] if len(matches) == 1 else ""


def _oxoacid_ester_suffix(spec: dict, role: CentralOxoRole) -> str:
    """Return an ester suffix that preserves remaining acid/anion oxygen roles."""

    base = spec["ester_suffix"]
    hydroxy_count = role.count(OxoLigandRole.HYDROXY)
    if hydroxy_count == 0 or "charge" in spec:
        return base
    hydrogen = "hydrogen" if hydroxy_count == 1 else f"{multipliers.basic(hydroxy_count)}hydrogen"
    return f"{hydrogen} {base}"


def _ester_modifier_atoms(
    mol: Molecule,
    component_atoms: set[int],
    root: int,
    acid_atoms: set[int],
) -> set[int]:
    atoms = set()
    queue = [root]
    while queue:
        atom_idx = queue.pop(0)
        if atom_idx in atoms:
            continue
        if atom_idx not in component_atoms or atom_idx in acid_atoms:
            return set()
        atoms.add(atom_idx)
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in acid_atoms:
                continue
            if neighbor in component_atoms:
                queue.append(neighbor)
    return atoms


def _ester_modifier_name(
    mol: Molecule,
    component_atoms: set[int],
    root: int,
    ester_oxygen: int,
    acid_atoms: set[int],
    branch_namer: BranchNamer | None,
) -> str:
    if branch_namer is not None:
        name = branch_namer(mol, root, set(mol.atoms) - component_atoms | acid_atoms, upstream_atom=ester_oxygen)
        if isinstance(name, tuple):
            name = name[0]
        if name:
            return strip_outer_parentheses(name)
    return _alkyl_ligand_name(mol, component_atoms, root, ester_oxygen)


def _oxoacid_oxygen_counts(mol: Molecule, central: int, oxygen_neighbors: list[int]) -> tuple[int, int]:
    role = central_oxo_roles(mol, {central, *oxygen_neighbors})
    if len(role) == 1:
        return role[0].spec_counts()
    return 0, 0


def _is_charge_normalized_halogen_oxo_ligand(
    mol: Molecule,
    central_symbol: str,
    central: int,
    oxygen: int,
) -> bool:
    """Return true for RDKit-normalized X(+)-O(-) oxo ligands on halogen oxoacids."""

    if central_symbol not in {"Cl", "Br", "I"}:
        return False
    bond = mol.get_bond(central, oxygen)
    return (
        bond is not None
        and bond.order == 1
        and mol.atoms[oxygen].charge < 0
        and sum(1 for _ in mol.get_neighbors(oxygen)) == 1
    )


def _matching_oxoacid_spec(central_symbol: str, single_o: int, double_o: int, charge: int) -> dict | None:
    for spec in RULES.components.replacement_parent_oxoacid_specs:
        if spec["central"] != central_symbol:
            continue
        if int(spec["single_o"]) != single_o or int(spec["double_o"]) != double_o:
            continue
        if "charge" in spec and int(spec["charge"]) != charge:
            continue
        return spec
    return None


def _matching_oxoacid_spec_for_role(mol: Molecule, role: CentralOxoRole) -> dict | None:
    single_o, double_o = role.spec_counts()
    return _matching_oxoacid_spec(role.central_symbol, single_o, double_o, mol.atoms[role.central].charge)


def organophosphinic_acid_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Name simple R-P(=O)(OH)H phosphinic acid parents."""

    phosphorus = [idx for idx in component_atoms if mol.atoms[idx].symbol == "P"]
    if len(phosphorus) != 1:
        return ""
    central = phosphorus[0]
    if (mol.atoms[central].total_h_count or mol.atoms[central].explicit_h_count) != 1:
        return ""
    double_oxygen = []
    hydroxy_oxygen = []
    carbon_roots = []
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms:
            return ""
        symbol = mol.atoms[neighbor].symbol
        bond = mol.get_bond(central, neighbor)
        if symbol == "O" and bond and bond.order == 2:
            double_oxygen.append(neighbor)
        elif symbol == "O" and bond and bond.order == 1 and mol.atoms[neighbor].charge == 0:
            hydroxy_oxygen.append(neighbor)
        elif symbol == "C" and bond and bond.order == 1:
            carbon_roots.append(neighbor)
        else:
            return ""
    if len(double_oxygen) != 1 or len(hydroxy_oxygen) != 1 or len(carbon_roots) != 1:
        return ""
    alkyl = _alkyl_ligand_name(mol, component_atoms, carbon_roots[0], central)
    return f"{alkyl}phosphinic acid" if alkyl else ""


def sulfoxide_parent_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Name simple dialkyl sulfoxides from a charged or neutral S-O graph."""

    sulfurs = [idx for idx in component_atoms if mol.atoms[idx].symbol == "S"]
    if len(sulfurs) != 1:
        return ""
    central = sulfurs[0]
    oxygens = []
    carbon_roots = []
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms:
            return ""
        symbol = mol.atoms[neighbor].symbol
        bond = mol.get_bond(central, neighbor)
        if symbol == "O" and bond and (bond.order == 2 or mol.atoms[neighbor].charge == -1):
            oxygens.append(neighbor)
        elif symbol == "C" and bond and bond.order == 1:
            carbon_roots.append(neighbor)
        else:
            return ""
    if len(oxygens) != 1 or len(carbon_roots) != 2:
        return ""
    left_atoms = _carbon_ligand_atoms(mol, component_atoms, carbon_roots[0], central)
    right_atoms = _carbon_ligand_atoms(mol, component_atoms, carbon_roots[1], central)
    if not left_atoms or not right_atoms or left_atoms & right_atoms:
        return ""
    ligands = [_alkyl_ligand_name(mol, component_atoms, root, central) for root in carbon_roots]
    if any(not ligand for ligand in ligands):
        return ""
    if ligands[0] == ligands[1]:
        return f"{multipliers.basic(2)}{ligands[0]} sulfoxide"
    return f"{' '.join(sorted(ligands))} sulfoxide"


def _carbon_ligand_atoms(mol: Molecule, component_atoms: set[int], root: int, central: int) -> set[int]:
    atoms = set()
    queue = [root]
    while queue:
        atom_idx = queue.pop(0)
        if atom_idx in atoms:
            continue
        if atom_idx not in component_atoms or mol.atoms[atom_idx].symbol != "C":
            return set()
        atoms.add(atom_idx)
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor == central:
                continue
            if neighbor in component_atoms:
                queue.append(neighbor)
    return atoms


def _alkyl_ligand_name(mol: Molecule, component_atoms: set[int], root: int, central: int) -> str:
    carbon_atoms = _carbon_ligand_atoms(mol, component_atoms, root, central)
    if not carbon_atoms:
        return ""
    for atom_idx in carbon_atoms:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in component_atoms and neighbor != central:
                bond = mol.get_bond(atom_idx, neighbor)
                if bond is None or bond.order != 1:
                    return ""
    if any(
        neighbor in component_atoms and neighbor != central and mol.atoms[neighbor].symbol != "C"
        for atom_idx in carbon_atoms
        for neighbor in mol.get_neighbors(atom_idx)
    ):
        return ""
    carbon_degrees = {
        atom_idx: sum(1 for neighbor in mol.get_neighbors(atom_idx) if neighbor in carbon_atoms)
        for atom_idx in carbon_atoms
    }
    if not _is_acyclic_carbon_tree(carbon_degrees):
        return ""
    return _systematic_alkyl_ligand_name(mol, carbon_atoms, root)


def _is_acyclic_carbon_tree(carbon_degrees: dict[int, int]) -> bool:
    """Return true for an acyclic alkyl ligand skeleton."""

    if not carbon_degrees:
        return False
    edge_count = sum(carbon_degrees.values()) // 2
    return edge_count == len(carbon_degrees) - 1


def _systematic_alkyl_ligand_name(mol: Molecule, carbon_atoms: set[int], root: int) -> str:
    """Name a rooted acyclic alkyl ligand by parent path and side branches."""

    if len(carbon_atoms) == 1:
        return "methyl"
    parent_path = _alkyl_parent_path(mol, carbon_atoms, root)
    if not parent_path:
        return ""
    path_set = set(parent_path)
    root_locant = parent_path.index(root) + 1
    branch_prefix = _alkyl_branch_prefixes(mol, carbon_atoms, parent_path, path_set)
    if branch_prefix is None:
        return ""
    stem = stems.stem_for(len(parent_path))
    parent = f"{stem}yl" if root_locant == 1 else f"{stem}an-{root_locant}-yl"
    return f"{branch_prefix}{parent}"


def _alkyl_parent_path(mol: Molecule, carbon_atoms: set[int], root: int) -> list[int]:
    paths = []
    atom_list = sorted(carbon_atoms)
    for left_index, left in enumerate(atom_list):
        for right in atom_list[left_index:]:
            path = _carbon_path_between(mol, carbon_atoms, left, right)
            if path and root in path:
                paths.append(path)
    if not paths:
        return []
    oriented = []
    for path in paths:
        variants = [path, list(reversed(path))]
        for variant in variants:
            root_locant = variant.index(root) + 1
            substituent_locants = _off_path_branch_locants(mol, carbon_atoms, variant)
            oriented.append((-len(variant), root_locant, substituent_locants, variant))
    return min(oriented)[3]


def _carbon_path_between(mol: Molecule, carbon_atoms: set[int], start: int, end: int) -> list[int]:
    queue: list[tuple[int, list[int]]] = [(start, [start])]
    visited = set()
    while queue:
        current, path = queue.pop(0)
        if current == end:
            return path
        if current in visited:
            continue
        visited.add(current)
        for neighbor in sorted(n for n in mol.get_neighbors(current) if n in carbon_atoms):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return []


def _off_path_branch_locants(mol: Molecule, carbon_atoms: set[int], path: list[int]) -> tuple[int, ...]:
    path_set = set(path)
    locants = []
    for locant, atom_idx in enumerate(path, start=1):
        if any(neighbor in carbon_atoms and neighbor not in path_set for neighbor in mol.get_neighbors(atom_idx)):
            locants.append(locant)
    return tuple(locants)


def _alkyl_branch_prefixes(
    mol: Molecule,
    carbon_atoms: set[int],
    parent_path: list[int],
    path_set: set[int],
) -> str | None:
    names_by_locant: list[tuple[int, str]] = []
    for locant, atom_idx in enumerate(parent_path, start=1):
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in carbon_atoms or neighbor in path_set:
                continue
            branch_atoms = _carbon_ligand_atoms(mol, carbon_atoms, neighbor, atom_idx)
            branch_name = _alkyl_ligand_name(mol, branch_atoms, neighbor, atom_idx) if branch_atoms else ""
            if not branch_name:
                return None
            names_by_locant.append((locant, branch_name))
    if not names_by_locant:
        return ""
    locants_by_name: dict[str, list[str]] = {}
    for locant, name in names_by_locant:
        locants_by_name.setdefault(name, []).append(str(locant))
    parts = []
    for name in sorted(locants_by_name):
        locants = ",".join(locants_by_name[name])
        parts.append(f"{locants}-{format_multiplier(name, len(locants_by_name[name]))}")
    return "".join(parts)


def _carbon_degree(mol: Molecule, carbon_atoms: set[int], atom_idx: int) -> int:
    return sum(1 for neighbor in mol.get_neighbors(atom_idx) if neighbor in carbon_atoms)


def homonuclear_chain_parent_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Name acyclic same-element parent hydride chains and simple ligands."""

    backbone_symbols = {
        mol.atoms[idx].symbol
        for idx in component_atoms
        if mol.atoms[idx].symbol in RULES.components.mononuclear_parent_hydrides
        and mol.atoms[idx].symbol not in {"O", "F", "Cl", "Br", "I"}
    }
    if len(backbone_symbols) != 1:
        return ""
    symbol = next(iter(backbone_symbols))
    backbone = [idx for idx in component_atoms if mol.atoms[idx].symbol == symbol]
    if len(backbone) < 2:
        return ""
    if any(mol.atoms[idx].symbol not in {symbol, "C", "F", "Cl", "Br", "I"} for idx in component_atoms):
        return ""
    chain = _ordered_backbone_chain(mol, backbone)
    if chain is None:
        return ""
    chain_set = set(chain)
    for atom_idx in component_atoms - chain_set:
        backbone_neighbors = [n for n in mol.get_neighbors(atom_idx) if n in chain_set]
        if len(backbone_neighbors) != 1:
            return ""
        bond = mol.get_bond(atom_idx, backbone_neighbors[0])
        if bond is None or bond.order != 1:
            return ""
        if not _terminal_ligand_name(mol, atom_idx, backbone_neighbors[0]):
            return ""
    bond_orders = [mol.get_bond(chain[idx], chain[idx + 1]).order for idx in range(len(chain) - 1)]
    parent = _same_element_parent_name(symbol, len(chain), bond_orders)
    if not parent:
        return ""
    prefixes = _simple_chain_ligand_prefixes(mol, component_atoms, chain)
    return f"{prefixes}{parent}" if prefixes else parent


def simple_central_parent_hydride_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Name simple mononuclear parent hydrides with halogen/alkoxy ligands."""

    central_candidates = [
        idx
        for idx in component_atoms
        if mol.atoms[idx].symbol in RULES.components.mononuclear_parent_hydrides
        and mol.atoms[idx].symbol not in {"O", "F", "Cl", "Br", "I"}
        and mol.degree(idx) >= 2
    ]
    if len(central_candidates) != 1:
        return ""
    central = central_candidates[0]
    central_symbol = mol.atoms[central].symbol
    ligand_names = []
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms:
            continue
        ligand = _central_ligand_name(mol, component_atoms, central, neighbor)
        if not ligand:
            return ""
        ligand_names.append(ligand)
    if not ligand_names:
        return ""
    prefix = _grouped_ligand_prefix(ligand_names)
    lambda_text = _lambda_text(mol, central)
    return f"{prefix}{lambda_text}{RULES.components.mononuclear_parent_hydrides[central_symbol]}"


def _ordered_backbone_chain(mol: Molecule, atoms: list[int]) -> list[int] | None:
    atom_set = set(atoms)
    degrees = {idx: sum(1 for n in mol.get_neighbors(idx) if n in atom_set) for idx in atoms}
    if any(degree > 2 for degree in degrees.values()):
        return None
    ends = [idx for idx, degree in degrees.items() if degree <= 1]
    if len(ends) != 2:
        return None
    chain = [min(ends)]
    previous = None
    while len(chain) < len(atoms):
        current = chain[-1]
        next_atoms = [n for n in mol.get_neighbors(current) if n in atom_set and n != previous]
        if len(next_atoms) != 1:
            return None
        previous = current
        chain.append(next_atoms[0])
    return chain


def _same_element_parent_name(symbol: str, length: int, bond_orders: list[int]) -> str:
    parent = RULES.components.mononuclear_parent_hydrides.get(symbol, "")
    if not parent or length < 2 or not parent.endswith("ane"):
        return ""
    base = f"{multipliers.basic(length)}{parent}"
    unsaturated = [order for order in bond_orders if order > 1]
    if not unsaturated:
        return base
    if len(unsaturated) != 1:
        return ""
    stem = base[:-3]
    order = unsaturated[0]
    if order == 2:
        return f"{stem}ene" if length == 2 else f"{stem}-{bond_orders.index(order) + 1}-ene"
    if order == 3:
        return f"{stem}yne" if length == 2 else f"{stem}-{bond_orders.index(order) + 1}-yne"
    return ""


def _simple_chain_ligand_prefixes(mol: Molecule, component_atoms: set[int], chain: list[int]) -> str:
    chain_set = set(chain)
    items = []
    for locant, atom_idx in enumerate(chain, start=1):
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in component_atoms or neighbor in chain_set:
                continue
            bond = mol.get_bond(atom_idx, neighbor)
            if bond is None or bond.order != 1:
                return ""
            name = _terminal_ligand_name(mol, neighbor, atom_idx)
            if not name:
                return ""
            items.append((locant, name))
    if not items:
        return ""
    names = {}
    locants_by_name = {}
    for locant, name in items:
        names[name] = name
        locants_by_name.setdefault(name, []).append(str(locant))
    parts = []
    for name in sorted(names):
        locants = ",".join(locants_by_name[name])
        count = len(locants_by_name[name])
        prefix = multipliers.basic(count) if count > 1 else ""
        parts.append(f"{locants}-{prefix}{name}")
    return "".join(parts)


def _terminal_ligand_name(mol: Molecule, atom_idx: int, parent_idx: int) -> str:
    atom = mol.atoms[atom_idx]
    halogens = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
    if atom.symbol in halogens:
        return halogens[atom.symbol]
    if atom.symbol == "C" and mol.degree(atom_idx) == 1:
        return "methyl"
    return ""


def _central_ligand_name(mol: Molecule, component_atoms: set[int], central: int, neighbor: int) -> str:
    bond = mol.get_bond(central, neighbor)
    if bond is None or bond.order != 1:
        return ""
    name = _terminal_ligand_name(mol, neighbor, central)
    if name:
        return name
    if mol.atoms[neighbor].symbol == "O":
        return _alkoxy_ligand_name(mol, component_atoms, neighbor, central)
    return ""


def _alkoxy_ligand_name(mol: Molecule, component_atoms: set[int], oxygen: int, central: int) -> str:
    carbon_neighbors = [n for n in mol.get_neighbors(oxygen) if n in component_atoms and n != central]
    if len(carbon_neighbors) != 1:
        return ""
    oxygen_atom = mol.atoms[oxygen]
    if oxygen_atom.charge != 0:
        return ""
    alkyl = _alkyl_ligand_name(mol, component_atoms, carbon_neighbors[0], oxygen)
    return oxy_prefix_from_branch(alkyl) if alkyl else ""


def _grouped_ligand_prefix(names: list[str]) -> str:
    groups = {}
    for name in names:
        groups[name] = groups.get(name, 0) + 1
    parts = []
    for name in sorted(groups):
        count = groups[name]
        parts.append(name if count == 1 else f"{format_multiplier(name, count)}-")
    return "".join(parts)


def _lambda_text(mol: Molecule, atom_idx: int) -> str:
    bond_order_sum = sum(
        (mol.get_bond(atom_idx, n).order if mol.get_bond(atom_idx, n) else 0) for n in mol.get_neighbors(atom_idx)
    )
    hydrogens = mol.atoms[atom_idx].total_h_count or mol.atoms[atom_idx].explicit_h_count
    bonding_number = bond_order_sum + hydrogens
    standard = mol.atoms[atom_idx].element.standard_valence
    if bonding_number == standard:
        return ""
    return f"lambda{bonding_number}-"


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
                sub_mol.add_bond(u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring)

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
