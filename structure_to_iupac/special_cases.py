"""Special component naming shortcuts."""

from collections.abc import Callable

from .formatting import format_multiplier, oxy_prefix_from_branch
from .molecule import Molecule
from .nomenclature import RULES
from .perception import PerceivedGroup
from .rules import multipliers, stems

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
    hydride_name = RULES.components.mononuclear_parent_hydrides.get(atom.symbol)
    if hydride_name:
        return hydride_name
    return ""


def structural_replacement_parent_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Return a replacement-parent hydride name from graph-derived specs."""

    return (
        oxoacid_parent_name(mol, component_atoms)
        or organophosphinic_acid_name(mol, component_atoms)
        or sulfoxide_parent_name(mol, component_atoms)
        or homonuclear_chain_parent_name(mol, component_atoms)
        or simple_central_parent_hydride_name(mol, component_atoms)
    )


def oxoacid_parent_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Match functional parent hydrides by central atom and oxygen ligand counts."""

    non_oxygen = [idx for idx in component_atoms if mol.atoms[idx].symbol != "O"]
    if len(non_oxygen) != 1:
        return ""
    central = non_oxygen[0]
    central_atom = mol.atoms[central]
    single_o = 0
    double_o = 0
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms or mol.atoms[neighbor].symbol != "O":
            return ""
        bond = mol.get_bond(central, neighbor)
        if bond and bond.order == 2:
            double_o += 1
        else:
            single_o += 1
    for spec in RULES.components.replacement_parent_oxoacid_specs:
        if spec["central"] != central_atom.symbol:
            continue
        if int(spec["single_o"]) != single_o or int(spec["double_o"]) != double_o:
            continue
        if "charge" in spec and int(spec["charge"]) != central_atom.charge:
            continue
        return spec["name"]
    return ""


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
    bond_orders = [
        mol.get_bond(chain[idx], chain[idx + 1]).order
        for idx in range(len(chain) - 1)
    ]
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
    bond_order_sum = sum((mol.get_bond(atom_idx, n).order if mol.get_bond(atom_idx, n) else 0) for n in mol.get_neighbors(atom_idx))
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
