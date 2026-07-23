"""Special component naming shortcuts."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from .assembly_parts import NameAtomBinding, NameTokenBinding
from .charge_pair_roles import charge_pair_roles
from .formatting import format_counted_prefixes, format_multiplier, oxy_prefix_from_branch, strip_outer_parentheses
from .molecule import Molecule
from .naming_protocols import RecursiveSubgraphNamer
from .nitrogen_roles import azine_roles
from .nomenclature import RULES
from .oxoacid_roles import CentralOxoRole, OxoLigandRole, central_oxo_roles
from .oxoacid_templates import OxoacidTemplateKind, oxoacid_role_template
from .perception import PerceivedGroup
from .retained_specs import retained_parent_spec
from .rules import multipliers, retained, stems

ComponentNamer = Callable[..., str]


@dataclass(frozen=True)
class AnhydrideComponentName:
    """Final anhydride name with graph-bound acid halves and bridge core."""

    name: str
    bindings: tuple[NameAtomBinding, ...]


@dataclass(frozen=True)
class SpecialComponentName:
    """A complete special component name with graph-bound renderer metadata."""

    name: str
    role: str
    bindings: tuple[NameAtomBinding, ...]


def _bond_ids_within_atoms(mol: Molecule, atom_ids: set[int]) -> set[int]:
    bond_ids = set()
    for atom_idx in atom_ids:
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in atom_ids and atom_idx < neighbor_idx:
                bond = mol.get_bond(atom_idx, neighbor_idx)
                if bond:
                    bond_ids.add(bond.idx)
    return bond_ids


def _charged_atoms(mol: Molecule, atom_ids: set[int]) -> set[int]:
    return {atom_idx for atom_idx in atom_ids if mol.atoms[atom_idx].charge != 0}


def _component_name_result(
    mol: Molecule,
    component_atoms: set[int],
    name: str,
    role: str,
    *,
    bindings: tuple[NameAtomBinding, ...] = (),
) -> SpecialComponentName:
    """Build a typed special-name result, using full-component binding as the fallback."""

    if not bindings:
        bindings = (
            NameAtomBinding(
                stage="shortcut",
                role=role,
                term=name,
                atom_ids=set(component_atoms),
                bond_ids=_bond_ids_within_atoms(mol, component_atoms),
                charge_atom_ids=_charged_atoms(mol, component_atoms),
            ),
        )
    return SpecialComponentName(name=name, role=role, bindings=bindings)


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
    branch_namer: RecursiveSubgraphNamer | None = None,
) -> str:
    """Return a replacement-parent hydride name from graph-derived specs."""

    result = structural_replacement_parent_result(mol, component_atoms, branch_namer)
    return result.name if result is not None else ""


def structural_replacement_parent_result(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: RecursiveSubgraphNamer | None = None,
) -> SpecialComponentName | None:
    """Return a graph-bound replacement-parent hydride name result."""

    renderers = (
        ("biphenyl_parent", lambda: biphenyl_parent_result(mol, component_atoms)),
        ("diazo_lambda_heteroring_parent", lambda: diazo_lambda_heteroring_parent_name(mol, component_atoms)),
        ("simple_azine_parent", lambda: simple_azine_parent_name(mol, component_atoms, branch_namer)),
        ("phosphane_borane_zwitterion", lambda: phosphane_borane_zwitterion_result(mol, component_atoms, branch_namer)),
        ("sulfonium_ylide", lambda: sulfonium_ylide_result(mol, component_atoms, branch_namer)),
        ("hydroxyurea_parent", lambda: hydroxyurea_parent_result(mol, component_atoms, branch_namer)),
        ("oxoacid_ester", lambda: oxoacid_ester_result(mol, component_atoms, branch_namer)),
        ("oxoacid_parent", lambda: oxoacid_parent_result(mol, component_atoms)),
        ("organophosphinic_acid", lambda: organophosphinic_acid_result(mol, component_atoms)),
        ("sulfoxide_parent", lambda: sulfoxide_parent_result(mol, component_atoms)),
        ("homonuclear_chain_parent", lambda: homonuclear_chain_parent_result(mol, component_atoms)),
        ("simple_central_parent_hydride", lambda: simple_central_parent_hydride_result(mol, component_atoms)),
    )
    for role, render in renderers:
        rendered = render()
        if isinstance(rendered, SpecialComponentName):
            return rendered
        if rendered:
            return _component_name_result(mol, component_atoms, rendered, role)
    return None


def biphenyl_parent_result(mol: Molecule, component_atoms: set[int]) -> SpecialComponentName | None:
    """Return retained biphenyl from a graph-proven pair of benzene rings."""

    if len(component_atoms) != 12 or any(not mol.atoms[idx].is_carbon for idx in component_atoms):
        return None
    inter_ring_bonds = []
    for atom_idx in component_atoms:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in component_atoms or atom_idx >= neighbor:
                continue
            bond = mol.get_bond(atom_idx, neighbor)
            if bond and not bond.in_small_ring:
                inter_ring_bonds.append((atom_idx, neighbor, bond.idx))
    if len(inter_ring_bonds) != 1:
        return None
    left_root, right_root, bridge_bond = inter_ring_bonds[0]
    ring_components = []
    for root, blocked in ((left_root, right_root), (right_root, left_root)):
        atoms = _component_atoms_until_blocked(mol, component_atoms, root, {blocked})
        if len(atoms) != 6:
            return None
        ring_path = _ordered_simple_ring(mol, atoms)
        if not ring_path:
            return None
        retained_match = retained.get_retained_ring(mol, ring_path)
        if retained_match is None or retained_match[0] != "benzene":
            return None
        ring_components.append(atoms)
    bindings = (
        NameAtomBinding(
            stage="shortcut",
            role="biphenyl_parent",
            term="biphenyl",
            atom_ids=set(component_atoms),
            bond_ids=_bond_ids_within_atoms(mol, component_atoms),
            locants=("1", "1'"),
            emitted_tokens=(
                NameTokenBinding(
                    text="1",
                    token_kind="locant",
                    source="shortcut_renderer",
                    grammar_role="biphenyl_attachment",
                    binding_key="shortcut:biphenyl_attachment:left",
                    atom_ids={left_root},
                    bond_ids={bridge_bond},
                    locants=("1",),
                    render_order=0,
                ),
                NameTokenBinding(
                    text="1",
                    token_kind="locant",
                    source="shortcut_renderer",
                    grammar_role="biphenyl_attachment",
                    binding_key="shortcut:biphenyl_attachment:right",
                    atom_ids={right_root},
                    bond_ids={bridge_bond},
                    locants=("1'",),
                    render_order=1,
                ),
                NameTokenBinding(
                    text="biphenyl",
                    token_kind="parent",
                    source="shortcut_renderer",
                    grammar_role="biphenyl_parent",
                    binding_key="shortcut:biphenyl_parent",
                    atom_ids=set(component_atoms),
                    bond_ids=_bond_ids_within_atoms(mol, component_atoms),
                    render_order=2,
                ),
            ),
        ),
        NameAtomBinding(
            stage="shortcut",
            role="biphenyl_ring",
            term="phenyl",
            atom_ids=ring_components[0],
            bond_ids=_bond_ids_within_atoms(mol, ring_components[0]),
            locants=("1",),
        ),
        NameAtomBinding(
            stage="shortcut",
            role="biphenyl_ring",
            term="phenyl",
            atom_ids=ring_components[1],
            bond_ids=_bond_ids_within_atoms(mol, ring_components[1]),
            locants=("1'",),
        ),
        NameAtomBinding(
            stage="shortcut",
            role="biphenyl_bridge",
            term="biphenyl",
            atom_ids={left_root, right_root},
            bond_ids={bridge_bond},
            locants=("1", "1'"),
        ),
    )
    return _component_name_result(mol, component_atoms, "1,1'-biphenyl", "biphenyl_parent", bindings=bindings)


def diazo_lambda_heteroring_parent_name(mol: Molecule, component_atoms: set[int]) -> str:
    """Name audited N(-)=N(+)=E monocyclic lambda parents as diazo rings."""

    match = _diazo_lambda_ring_match(mol, component_atoms)
    if match is None:
        return ""
    hetero, n_pos, n_neg, ring_atoms = match
    ring_path = _preferred_lambda_ring_path(mol, hetero, ring_atoms)
    if not ring_path:
        return ""
    lambda_value = sum(mol.get_bond(hetero, n).order for n in mol.get_neighbors(hetero)) + (
        mol.atoms[hetero].total_h_count or mol.atoms[hetero].explicit_h_count
    )
    replacement_prefixes = _lambda_ring_replacement_prefixes(mol, ring_path, hetero, lambda_value)
    if not replacement_prefixes:
        return ""
    unsaturation = _lambda_ring_unsaturation_suffix(mol, ring_path)
    replacement_text = (
        "-".join(replacement_prefixes[:-1]) + "-" + replacement_prefixes[-1]
        if len(replacement_prefixes) > 1
        else replacement_prefixes[0]
    )
    return f"1-diazo-{replacement_text}cyclo{stems.stem_for(len(ring_path))}{unsaturation}"


def _diazo_lambda_ring_match(
    mol: Molecule,
    component_atoms: set[int],
) -> tuple[int, int, int, set[int]] | None:
    for hetero in component_atoms:
        if mol.atoms[hetero].symbol not in {"P", "S", "Se", "B"}:
            continue
        n_pos_candidates = [
            n
            for n in mol.get_neighbors(hetero)
            if n in component_atoms
            and mol.atoms[n].symbol == "N"
            and mol.atoms[n].charge > 0
            and (bond := mol.get_bond(hetero, n)) is not None
            and bond.order == 2
        ]
        if len(n_pos_candidates) != 1:
            continue
        n_pos = n_pos_candidates[0]
        n_neg_candidates = [
            n
            for n in mol.get_neighbors(n_pos)
            if n in component_atoms
            and n != hetero
            and mol.atoms[n].symbol == "N"
            and mol.atoms[n].charge < 0
            and (bond := mol.get_bond(n_pos, n)) is not None
            and bond.order == 2
        ]
        if len(n_neg_candidates) != 1:
            continue
        n_neg = n_neg_candidates[0]
        if any(n != n_pos for n in mol.get_neighbors(n_neg) if n in component_atoms):
            continue
        ring_atoms = set(component_atoms) - {n_pos, n_neg}
        if hetero not in ring_atoms or not _is_single_simple_ring(mol, ring_atoms):
            continue
        return hetero, n_pos, n_neg, ring_atoms
    return None


def _is_single_simple_ring(mol: Molecule, ring_atoms: set[int]) -> bool:
    if len(ring_atoms) < 3:
        return False
    edge_count = 0
    for atom in ring_atoms:
        degree = sum(1 for neighbor in mol.get_neighbors(atom) if neighbor in ring_atoms)
        if degree != 2:
            return False
        edge_count += degree
    return edge_count // 2 == len(ring_atoms)


def _preferred_lambda_ring_path(mol: Molecule, hetero: int, ring_atoms: set[int]) -> list[int]:
    neighbors = [n for n in mol.get_neighbors(hetero) if n in ring_atoms]
    if len(neighbors) != 2:
        return []
    candidates = [_lambda_ring_path_from(mol, hetero, start, ring_atoms) for start in neighbors]
    candidates = [path for path in candidates if path]
    if not candidates:
        return []
    return min(
        candidates, key=lambda path: (_hetero_locants_for_path(mol, path), _double_bond_locants_for_path(mol, path))
    )


def _lambda_ring_path_from(mol: Molecule, hetero: int, start: int, ring_atoms: set[int]) -> list[int]:
    path = [hetero]
    previous = hetero
    current = start
    while current != hetero:
        if current in path:
            return []
        path.append(current)
        next_atoms = [n for n in mol.get_neighbors(current) if n in ring_atoms and n != previous]
        if len(next_atoms) != 1:
            return []
        previous, current = current, next_atoms[0]
    return path if set(path) == ring_atoms else []


def _hetero_locants_for_path(mol: Molecule, path: list[int]) -> tuple[int, ...]:
    return tuple(i + 1 for i, atom in enumerate(path) if mol.atoms[atom].symbol not in {"C", "H"})


def _double_bond_locants_for_path(mol: Molecule, path: list[int]) -> tuple[int, ...]:
    locants = []
    for i, atom in enumerate(path):
        nxt = path[(i + 1) % len(path)]
        bond = mol.get_bond(atom, nxt)
        if bond is not None and bond.order == 2:
            locants.append(i + 1)
    return tuple(locants)


def _lambda_ring_replacement_prefixes(mol: Molecule, path: list[int], hetero: int, lambda_value: int) -> list[str]:
    replacement_names = {
        "B": "bora",
        "N": "aza",
        "O": "oxa",
        "P": "phospha",
        "S": "thia",
        "Se": "selena",
    }
    prefixes = []
    for i, atom in enumerate(path, start=1):
        symbol = mol.atoms[atom].symbol
        if symbol == "C":
            continue
        replacement = replacement_names.get(symbol)
        if replacement is None:
            return []
        if atom == hetero:
            prefixes.append((i, f"{i}lambda^{lambda_value}-{replacement}"))
        else:
            prefixes.append((i, f"{i}-{replacement}"))
    noncentral = [prefix for atom, prefix in prefixes if path[atom - 1] != hetero]
    central = [prefix for atom, prefix in prefixes if path[atom - 1] == hetero]
    return noncentral + central


def _lambda_ring_unsaturation_suffix(mol: Molecule, path: list[int]) -> str:
    locants = _double_bond_locants_for_path(mol, path)
    if not locants:
        return "ane"
    count = len(locants)
    infix = "ene" if count == 1 else f"{multipliers.basic(count)}ene"
    stem_joiner = "" if infix.startswith("e") else "a"
    return f"{stem_joiner}-{','.join(str(locant) for locant in locants)}-{infix}"


def simple_azine_parent_name(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: RecursiveSubgraphNamer | None = None,
) -> str:
    """Name simple acyclic ketazines/aldazines from a graph C=N-N=C role."""

    for role in azine_roles(mol, component_atoms):
        n1 = role.left.nitrogen_atom
        n2 = role.right.nitrogen_atom
        c1 = role.left.carbon_atom
        c2 = role.right.carbon_atom
        side1 = set(role.left.side_atoms)
        side2 = set(role.right.side_atoms)
        parent1 = _carbonyl_equivalent_side_name(mol, side1, c1, as_ylidene=False)
        parent2 = _carbonyl_equivalent_side_name(mol, side2, c2, as_ylidene=False)
        ylidene1 = _carbonyl_equivalent_side_name(mol, side1, c1, as_ylidene=True)
        ylidene2 = _carbonyl_equivalent_side_name(mol, side2, c2, as_ylidene=True)
        if not parent1 or not parent2 or not ylidene1 or not ylidene2:
            if branch_namer is None:
                continue
            if parent1:
                branch_ylidene2 = strip_outer_parentheses(
                    branch_namer(mol, c2, set(mol.atoms) - component_atoms | {n2}, upstream_atom=n2)
                )
                if branch_ylidene2 and branch_ylidene2.endswith("ylidene"):
                    stereo = _hydrazone_stereo_prefix(mol, c1, n1, parent1)
                    return f"{stereo}{parent1} {branch_ylidene2}hydrazone"
            if parent2:
                branch_ylidene1 = strip_outer_parentheses(
                    branch_namer(mol, c1, set(mol.atoms) - component_atoms | {n1}, upstream_atom=n1)
                )
                if branch_ylidene1 and branch_ylidene1.endswith("ylidene"):
                    stereo = _hydrazone_stereo_prefix(mol, c2, n2, parent2)
                    return f"{stereo}{parent2} {branch_ylidene1}hydrazone"
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


def phosphane_borane_zwitterion_result(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: RecursiveSubgraphNamer | None = None,
) -> SpecialComponentName | None:
    """Name graph-proven B(-)(H)3-P(+) zwitterions with bound ligands and charges."""

    role = next(
        (role for role in charge_pair_roles(mol, component_atoms) if role.key == "phosphane_borane_zwitterion"), None
    )
    if role is None or not role.template_supported or not role.template_audit(mol).ok:
        return None
    phosphorus = role.positive_atom
    boron = role.negative_atom
    ligand_roots = [
        neighbor
        for neighbor in mol.get_neighbors(phosphorus)
        if neighbor in component_atoms and neighbor != boron and mol.atoms[neighbor].is_carbon
    ]
    if len(ligand_roots) != 3:
        return None
    ligand_atoms_seen: set[int] = set()
    ligands: list[tuple[str, set[int]]] = []
    core_atoms = {phosphorus, boron}
    for root in ligand_roots:
        ligand_atoms = _component_atoms_until_blocked(mol, component_atoms, root, core_atoms)
        if not ligand_atoms or ligand_atoms & ligand_atoms_seen:
            return None
        ligand_atoms_seen.update(ligand_atoms)
        name = ""
        if branch_namer is not None:
            rendered = branch_namer(mol, root, set(mol.atoms) - component_atoms | core_atoms, upstream_atom=phosphorus)
            if isinstance(rendered, tuple):
                rendered = rendered[0]
            name = strip_outer_parentheses(rendered)
        if not name:
            name = _alkyl_ligand_name(mol, component_atoms, root, phosphorus)
        if not name:
            return None
        ligands.append((name, ligand_atoms))
    if ligand_atoms_seen | core_atoms != component_atoms:
        return None
    ligand_names = [name for name, _atoms in ligands]
    prefix = format_counted_prefixes(ligand_names)
    name = f"({prefix}phosphaniumyl)boranuide"
    bindings = tuple(
        NameAtomBinding(
            stage="shortcut",
            role="phosphane_borane_ligand",
            term=ligand_name,
            atom_ids=set(ligand_atoms),
            bond_ids=_bond_ids_within_atoms(mol, ligand_atoms),
        )
        for ligand_name, ligand_atoms in ligands
    ) + (
        NameAtomBinding(
            stage="shortcut",
            role="phosphane_borane_zwitterion_core",
            term="phosphaniumyl boranuide",
            atom_ids=core_atoms,
            bond_ids=_bond_ids_within_atoms(mol, core_atoms),
            charge_atom_ids={phosphorus, boron},
        ),
    )
    return _component_name_result(mol, component_atoms, name, "phosphane_borane_zwitterion", bindings=bindings)


def _carbonyl_equivalent_side_name(
    mol: Molecule,
    side_atoms: set[int],
    carbonyl_carbon: int,
    *,
    as_ylidene: bool,
) -> str:
    return (
        _retained_ring_carbaldehyde_side_name(mol, side_atoms, carbonyl_carbon, as_ylidene=as_ylidene)
        or _simple_ring_carbaldehyde_side_name(mol, side_atoms, carbonyl_carbon, as_ylidene=as_ylidene)
        or _simple_ring_ylidene_side_name(mol, side_atoms, carbonyl_carbon, as_ylidene=as_ylidene)
        or _simple_carbonyl_side_name(mol, side_atoms, carbonyl_carbon, as_ylidene=as_ylidene)
    )


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
    internal_edges = [(a, b) for a in side_atoms for b in mol.get_neighbors(a) if b in side_atoms and a < b]
    if any(mol.get_bond(a, b).order not in {1, 2, 3} for a, b in internal_edges):
        return ""
    if len(internal_edges) != len(side_atoms) - 1:
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


def _retained_ring_carbaldehyde_side_name(
    mol: Molecule,
    side_atoms: set[int],
    carbonyl_carbon: int,
    *,
    as_ylidene: bool,
) -> str:
    """Name Ar-CH=N sides as retained-ring carbaldehyde/ylidene roles."""

    if carbonyl_carbon not in side_atoms or not mol.atoms[carbonyl_carbon].is_carbon:
        return ""
    ring_atoms = set(side_atoms) - {carbonyl_carbon}
    if len(ring_atoms) < 3:
        return ""
    carbon_neighbors = [
        neighbor
        for neighbor in mol.get_neighbors(carbonyl_carbon)
        if neighbor in ring_atoms and (bond := mol.get_bond(carbonyl_carbon, neighbor)) is not None and bond.order == 1
    ]
    if len(carbon_neighbors) != 1:
        return ""
    if any(neighbor in side_atoms and neighbor not in ring_atoms for neighbor in mol.get_neighbors(carbonyl_carbon)):
        return ""
    ring_path = _ordered_simple_ring(mol, ring_atoms)
    if not ring_path:
        return ""
    retained_match = retained.get_retained_ring(mol, ring_path)
    if retained_match is None:
        return ""
    retained_name, locant_maps = retained_match
    locant_map = _choose_retained_map_for_attachment(locant_maps, carbon_neighbors[0])
    if locant_map is None:
        return ""
    locant = locant_map.get(carbon_neighbors[0])
    if not locant:
        return ""
    if as_ylidene:
        return _retained_ring_methylidene_name(retained_name, locant)
    if retained_name == "benzene":
        return "benzaldehyde"
    return f"{retained_name}-{locant}-carbaldehyde"


def _ordered_simple_ring(mol: Molecule, ring_atoms: set[int]) -> list[int]:
    """Return an ordered monocyclic ring path, or empty for fused/bridged rings."""

    internal_neighbors = {
        atom_idx: sorted(neighbor for neighbor in mol.get_neighbors(atom_idx) if neighbor in ring_atoms)
        for atom_idx in ring_atoms
    }
    if any(len(neighbors) != 2 for neighbors in internal_neighbors.values()):
        return []
    edge_count = sum(len(neighbors) for neighbors in internal_neighbors.values()) // 2
    if edge_count != len(ring_atoms):
        return []
    start = min(ring_atoms)
    path = [start]
    previous = None
    current = start
    while True:
        candidates = [neighbor for neighbor in internal_neighbors[current] if neighbor != previous]
        if not candidates:
            return []
        nxt = candidates[0]
        if nxt == start:
            return path if len(path) == len(ring_atoms) else []
        if nxt in path:
            return []
        previous, current = current, nxt
        path.append(current)


def _choose_retained_map_for_attachment(
    locant_maps: list[dict[int, str]] | None,
    attachment_atom: int,
) -> dict[int, str] | None:
    if not locant_maps:
        return None
    return min(
        (locants for locants in locant_maps if attachment_atom in locants),
        key=lambda locants: _retained_locant_sort_key(locants[attachment_atom]),
        default=None,
    )


def _retained_locant_sort_key(locant: str) -> tuple[int, str]:
    match = re.match(r"(\d+)", locant)
    return (int(match.group(1)) if match else 999, locant)


def _retained_ring_methylidene_name(retained_name: str, locant: str) -> str:
    if retained_name == "benzene":
        return "benzylidene"
    spec = retained_parent_spec(retained_name)
    stem = spec.substituent_stem if spec and spec.substituent_stem else retained_name.rstrip("e")
    return f"{stem}-{locant}-ylmethylidene"


def _simple_ring_carbaldehyde_side_name(
    mol: Molecule,
    side_atoms: set[int],
    carbonyl_carbon: int,
    *,
    as_ylidene: bool,
) -> str:
    """Name cycloalkenyl-CH=N sides as cyclo...carbaldehyde roles."""

    if carbonyl_carbon not in side_atoms or not mol.atoms[carbonyl_carbon].is_carbon:
        return ""
    ring_atoms = set(side_atoms) - {carbonyl_carbon}
    if len(ring_atoms) < 3:
        return ""
    ring_neighbors = [
        neighbor
        for neighbor in mol.get_neighbors(carbonyl_carbon)
        if neighbor in ring_atoms and (bond := mol.get_bond(carbonyl_carbon, neighbor)) is not None and bond.order == 1
    ]
    if len(ring_neighbors) != 1:
        return ""
    if any(neighbor in side_atoms and neighbor not in ring_atoms for neighbor in mol.get_neighbors(carbonyl_carbon)):
        return ""
    ring_path = _ordered_simple_ring_from_start(mol, ring_atoms, ring_neighbors[0])
    if not ring_path:
        return ""
    ring_name = _simple_replacement_cycle_name(mol, ring_path)
    if not ring_name:
        return ""
    return f"{ring_name}-1-ylmethylidene" if as_ylidene else f"{ring_name}-1-carbaldehyde"


def _ordered_simple_ring_from_start(mol: Molecule, ring_atoms: set[int], start: int) -> list[int]:
    path = _ordered_simple_ring(mol, ring_atoms)
    if not path or start not in path:
        return []
    pos = path.index(start)
    forward = path[pos:] + path[:pos]
    reverse = [forward[0], *reversed(forward[1:])]
    return min((forward, reverse), key=lambda ordered: (_simple_cycle_unsaturation_locants(mol, ordered), ordered))


def _simple_ring_ylidene_side_name(
    mol: Molecule,
    side_atoms: set[int],
    carbonyl_carbon: int,
    *,
    as_ylidene: bool,
) -> str:
    """Name simple monocyclic C=N azine sides as cyclo...ylidene roles."""

    if not as_ylidene or carbonyl_carbon not in side_atoms:
        return ""
    ring_path = _ordered_simple_ring(mol, set(side_atoms))
    if not ring_path:
        return ""
    if any(mol.atoms[idx].symbol != "C" and not mol.atoms[idx].element.hw_stem for idx in ring_path):
        return ""
    variants = []
    for offset in range(len(ring_path)):
        rotated = ring_path[offset:] + ring_path[:offset]
        for oriented in (rotated, [rotated[0], *reversed(rotated[1:])]):
            score = _simple_ring_numbering_score(mol, oriented, carbonyl_carbon)
            if score is None:
                continue
            name = _simple_replacement_cycle_name(mol, oriented)
            if name:
                locant = oriented.index(carbonyl_carbon) + 1
                variants.append((score, f"{name}-{locant}-ylidene"))
    return min(variants)[1] if variants else ""


def _simple_ring_numbering_score(mol: Molecule, oriented: list[int], attachment_atom: int) -> tuple | None:
    hetero_priorities = [
        mol.atoms[idx].element.hw_priority
        for idx in oriented
        if mol.atoms[idx].symbol != "C" and mol.atoms[idx].element.hw_priority is not None
    ]
    if hetero_priorities:
        first = mol.atoms[oriented[0]]
        if first.symbol == "C" or first.element.hw_priority != min(hetero_priorities):
            return None
    hetero_locants = tuple(pos for pos, idx in enumerate(oriented, start=1) if mol.atoms[idx].symbol != "C")
    attachment_locant = oriented.index(attachment_atom) + 1
    return (hetero_locants, attachment_locant, tuple(oriented))


def _simple_replacement_cycle_name(mol: Molecule, oriented: list[int]) -> str:
    hetero = [
        (pos, mol.atoms[idx].element.hw_priority or 999, mol.atoms[idx].element.hw_stem or "")
        for pos, idx in enumerate(oriented, start=1)
        if mol.atoms[idx].symbol != "C"
    ]
    stem = stems.stem_for(len(oriented))
    if not stem:
        return ""
    unsaturation = _simple_cycle_unsaturation(mol, oriented)
    parent_stem = f"{stem}a" if unsaturation else stem
    parent = f"cyclo{parent_stem}{unsaturation or 'an'}"
    if not hetero:
        return parent
    groups: dict[tuple[int, str], list[int]] = {}
    for locant, priority, prefix in hetero:
        if not prefix:
            return ""
        groups.setdefault((priority, prefix), []).append(locant)
    prefix_parts = []
    for (_priority, prefix), locants in sorted(groups.items()):
        locant_text = ",".join(str(locant) for locant in sorted(locants))
        prefix_text = prefix if len(locants) == 1 else f"{multipliers.basic(len(locants))}{prefix}"
        prefix_parts.append(f"{locant_text}-{prefix_text}")
    return "".join(prefix_parts) + parent


def _simple_cycle_unsaturation(mol: Molecule, oriented: list[int]) -> str:
    double_locs = _simple_cycle_unsaturation_locants(mol, oriented)
    if not double_locs:
        return ""
    suffix = "en" if len(double_locs) == 1 else f"{multipliers.basic(len(double_locs))}en"
    return f"-{','.join(str(locant) for locant in double_locs)}-{suffix}"


def _simple_cycle_unsaturation_locants(mol: Molecule, oriented: list[int]) -> tuple[int, ...]:
    double_locs = []
    for pos, (left, right) in enumerate(zip(oriented, oriented[1:] + oriented[:1]), start=1):
        bond = mol.get_bond(left, right)
        if bond is not None and bond.order == 2:
            double_locs.append(pos)
    return tuple(double_locs)


def _longest_carbon_path_through(mol: Molecule, side_atoms: set[int], required: int) -> list[int]:
    endpoints = [idx for idx in side_atoms if sum(1 for n in mol.get_neighbors(idx) if n in side_atoms) <= 1] or list(
        side_atoms
    )
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
    branch_namer: RecursiveSubgraphNamer | None = None,
) -> str:
    """Name graph-proven sulfonium/carbanion ylides as full components."""

    sulfurs = [idx for idx in component_atoms if mol.atoms[idx].symbol == "S" and mol.atoms[idx].charge > 0]
    carbanions = [idx for idx in component_atoms if mol.atoms[idx].is_carbon and mol.atoms[idx].charge < 0]
    if len(sulfurs) != 1 or len(carbanions) != 1:
        return ""
    sulfur = sulfurs[0]
    ylide_carbon = carbanions[0]
    role = next(
        (
            role
            for role in charge_pair_roles(mol, component_atoms)
            if role.key == "sulfonium_ylide_single_bond"
            and role.positive_atom == sulfur
            and role.negative_atom == ylide_carbon
            and role.template_supported
        ),
        None,
    )
    if role is None:
        return ""
    if not role.template_audit(mol).ok:
        return ""
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
    ylide_sub_root: int | None = None
    ylide_sub_name = ""
    if ylide_sub_roots:
        root = ylide_sub_roots[0]
        ylide_sub_root = root
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
    sulfaniumyl = f"{sulfur_prefix}sulfaniumyl"
    ylide_parent = _sulfonium_ylide_carbanion_parent_name(
        mol,
        ylide_carbon,
        ylide_sub_atoms,
        ylide_sub_root,
        ylide_sub_name,
    )
    if not ylide_parent:
        return ""
    locant, parent = ylide_parent
    prefix = f"{locant}-" if locant else ""
    return f"{prefix}({sulfaniumyl}){parent}"


def sulfonium_ylide_result(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: RecursiveSubgraphNamer | None = None,
) -> SpecialComponentName | None:
    """Return a typed sulfonium ylide result for the graph-proven renderer."""

    name = sulfonium_ylide_name(mol, component_atoms, branch_namer)
    if not name:
        return None
    sulfurs = {idx for idx in component_atoms if mol.atoms[idx].symbol == "S" and mol.atoms[idx].charge > 0}
    carbanions = {idx for idx in component_atoms if mol.atoms[idx].is_carbon and mol.atoms[idx].charge < 0}
    bindings = (
        NameAtomBinding(
            stage="shortcut",
            role="sulfonium_ylide",
            term=name,
            atom_ids=set(component_atoms),
            bond_ids=_bond_ids_within_atoms(mol, component_atoms),
            charge_atom_ids=sulfurs | carbanions,
        ),
    )
    return _component_name_result(mol, component_atoms, name, "sulfonium_ylide", bindings=bindings)


def _sulfonium_ylide_carbanion_parent_name(
    mol: Molecule,
    ylide_carbon: int,
    ylide_sub_atoms: set[int],
    ylide_sub_root: int | None,
    ylide_sub_name: str,
) -> tuple[str, str] | None:
    """Return locant and parent suffix for the carbon-anion side of an ylide.

    Preferred ylide wording uses a sulfaniumyl substituent on the carbon
    anion parent, e.g. ``1-(dimethylsulfaniumyl)ethan-1-ide``.  The local
    implementation keeps the supported grammar conservative: graph-proven
    linear alkyl substituents form an alkane-1-ide parent, aryl substituents
    form arylmethanide, and the unsubstituted center forms methanide.
    """

    if ylide_sub_root is None:
        return "", "methanide"
    alkyl_parent = _linear_alkyl_carbanion_parent_name(mol, ylide_sub_atoms, ylide_sub_root)
    if alkyl_parent:
        stem = stems.stem_for(alkyl_parent + 1)
        return "1", f"{stem}an-1-ide"
    if ylide_sub_name.endswith("yl"):
        return "", f"{ylide_sub_name[:-2]}ylmethanide"
    return "", f"({ylide_sub_name})methanide"


def _linear_alkyl_carbanion_parent_name(mol: Molecule, carbon_atoms: set[int], root: int) -> int:
    """Return the ylide-side alkyl length when it is a graph-proven chain."""

    if not carbon_atoms or root not in carbon_atoms:
        return 0
    if any(not mol.atoms[idx].is_carbon for idx in carbon_atoms):
        return 0
    carbon_degrees = {
        atom_idx: sum(1 for neighbor in mol.get_neighbors(atom_idx) if neighbor in carbon_atoms)
        for atom_idx in carbon_atoms
    }
    if not _is_acyclic_carbon_tree(carbon_degrees):
        return 0
    if any(degree > 2 for degree in carbon_degrees.values()):
        return 0
    for atom_idx in carbon_atoms:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in carbon_atoms and (bond := mol.get_bond(atom_idx, neighbor)) is not None and bond.order != 1:
                return 0
    return len(carbon_atoms)


def hydroxyurea_parent_result(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: RecursiveSubgraphNamer | None = None,
) -> SpecialComponentName | None:
    """Name graph-proven N-hydroxyureas with core/hydroxy/ligand bindings."""

    carbonyl_carbons = [
        idx
        for idx in component_atoms
        if mol.atoms[idx].is_carbon
        and len(
            [
                n
                for n in mol.get_neighbors(idx)
                if n in component_atoms and mol.atoms[n].symbol == "O" and mol.get_bond(idx, n).order == 2
            ]
        )
        == 1
        and len(
            [
                n
                for n in mol.get_neighbors(idx)
                if n in component_atoms and mol.atoms[n].symbol == "N" and mol.get_bond(idx, n).order == 1
            ]
        )
        == 2
    ]
    if len(carbonyl_carbons) != 1:
        return None
    carbonyl = carbonyl_carbons[0]
    carbonyl_oxygen = next(
        n for n in mol.get_neighbors(carbonyl) if n in component_atoms and mol.atoms[n].symbol == "O"
    )
    nitrogens = [
        n
        for n in mol.get_neighbors(carbonyl)
        if n in component_atoms and mol.atoms[n].symbol == "N" and mol.get_bond(carbonyl, n).order == 1
    ]
    if any((bond := mol.get_bond(carbonyl, nitrogen)) is not None and bond.in_small_ring for nitrogen in nitrogens):
        return None
    hydroxy_nitrogens = []
    hydroxy_oxygens = {}
    for nitrogen in nitrogens:
        oxygen_neighbors = [
            n
            for n in mol.get_neighbors(nitrogen)
            if n in component_atoms
            and n != carbonyl
            and mol.atoms[n].symbol == "O"
            and (bond := mol.get_bond(nitrogen, n)) is not None
            and bond.order == 1
        ]
        if oxygen_neighbors:
            if len(oxygen_neighbors) != 1:
                return None
            oxygen = oxygen_neighbors[0]
            if any(n != nitrogen for n in mol.get_neighbors(oxygen) if n in component_atoms):
                return None
            hydroxy_nitrogens.append(nitrogen)
            hydroxy_oxygens[nitrogen] = oxygen
    if len(hydroxy_nitrogens) != 1:
        return None
    hydroxy_n = hydroxy_nitrogens[0]
    other_n = next(n for n in nitrogens if n != hydroxy_n)
    core_atoms = {carbonyl, carbonyl_oxygen, *nitrogens, hydroxy_oxygens[hydroxy_n]}
    n_ligands = _urea_n_ligand_names(mol, component_atoms, other_n, core_atoms, branch_namer)
    n_prime_ligands = _urea_n_ligand_names(mol, component_atoms, hydroxy_n, core_atoms, branch_namer)
    if n_ligands is None or n_prime_ligands is None:
        return None
    represented = set(core_atoms)
    for ligand_atoms, _name in n_ligands + n_prime_ligands:
        represented.update(ligand_atoms)
    if represented != component_atoms:
        return None
    prefixes = [f"N-{format_multiplier(name, 1)}" for _atoms, name in n_ligands]
    prefixes.extend(f"N'-{format_multiplier(name, 1)}" for _atoms, name in n_prime_ligands)
    prefixes.append("N'-hydroxy")
    name = f"{'-'.join(prefixes)}urea"
    bindings = []
    for ligand_atoms, ligand_name in n_ligands:
        bindings.append(
            NameAtomBinding(
                stage="shortcut",
                role="urea_n_ligand",
                term=ligand_name,
                atom_ids=set(ligand_atoms),
                bond_ids=_bond_ids_within_atoms(mol, set(ligand_atoms)),
                locants=("N",),
            )
        )
    for ligand_atoms, ligand_name in n_prime_ligands:
        bindings.append(
            NameAtomBinding(
                stage="shortcut",
                role="urea_n_prime_ligand",
                term=ligand_name,
                atom_ids=set(ligand_atoms),
                bond_ids=_bond_ids_within_atoms(mol, set(ligand_atoms)),
                locants=("N'",),
            )
        )
    hydroxy_atoms = {hydroxy_n, hydroxy_oxygens[hydroxy_n]}
    bindings.extend(
        [
            NameAtomBinding(
                stage="shortcut",
                role="hydroxyurea_hydroxy",
                term="hydroxy",
                atom_ids=hydroxy_atoms,
                bond_ids=_bond_ids_within_atoms(mol, hydroxy_atoms),
                locants=("N'",),
            ),
            NameAtomBinding(
                stage="shortcut",
                role="urea_core",
                term="urea",
                atom_ids={carbonyl, carbonyl_oxygen, *nitrogens},
                bond_ids=_bond_ids_within_atoms(mol, {carbonyl, carbonyl_oxygen, *nitrogens}),
            ),
        ]
    )
    return _component_name_result(mol, component_atoms, name, "hydroxyurea_parent", bindings=tuple(bindings))


def _urea_n_ligand_names(
    mol: Molecule,
    component_atoms: set[int],
    nitrogen: int,
    core_atoms: set[int],
    branch_namer: RecursiveSubgraphNamer | None,
) -> list[tuple[set[int], str]] | None:
    ligands = []
    for root in mol.get_neighbors(nitrogen):
        if root in core_atoms:
            continue
        if root not in component_atoms or mol.atoms[root].symbol == "H":
            continue
        if not mol.atoms[root].is_carbon:
            return None
        ligand_atoms = _component_atoms_until_blocked(mol, component_atoms, root, core_atoms)
        if not ligand_atoms:
            return None
        if any(not mol.atoms[atom_idx].is_carbon for atom_idx in ligand_atoms):
            return None
        if any(
            (bond := mol.get_bond(atom_idx, neighbor)) is not None and bond.order != 1
            for atom_idx in ligand_atoms
            for neighbor in mol.get_neighbors(atom_idx)
            if neighbor in ligand_atoms and atom_idx < neighbor
        ):
            return None
        name = ""
        if branch_namer is not None:
            name = branch_namer(mol, root, set(mol.atoms) - component_atoms | core_atoms, upstream_atom=nitrogen)
            if isinstance(name, tuple):
                name = name[0]
            name = strip_outer_parentheses(name)
        if not name:
            name = _alkyl_ligand_name(mol, component_atoms, root, nitrogen)
        if not name:
            return None
        ligands.append((ligand_atoms, name))
    return sorted(ligands, key=lambda item: item[1])


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


def oxoacid_parent_result(mol: Molecule, component_atoms: set[int]) -> SpecialComponentName | None:
    """Match functional parent hydrides with central-oxo role bindings."""

    roles = central_oxo_roles(mol, component_atoms)
    if len(roles) != 1:
        return None
    role = roles[0]
    if role.has_organic_ester() or role.has_peroxy():
        return None
    if {role.central, *role.oxygen_atoms} != component_atoms:
        return None
    spec = _matching_oxoacid_spec_for_role(mol, role)
    if spec is None:
        return None
    if role.has_anion() and spec.get("ester_suffix") and not role.count(OxoLigandRole.HYDROXY):
        name = spec["ester_suffix"]
    else:
        name = spec["name"]
    bindings = _central_oxo_role_bindings(mol, role, name, "oxoacid_parent")
    return _component_name_result(mol, component_atoms, name, "oxoacid_parent", bindings=bindings)


def oxoacid_ester_result(
    mol: Molecule,
    component_atoms: set[int],
    branch_namer: RecursiveSubgraphNamer | None = None,
) -> SpecialComponentName | None:
    """Name organic esters with separate modifier and central-oxo suffix bindings."""

    matches = []
    for role in central_oxo_roles(mol, component_atoms):
        spec = _matching_oxoacid_spec_for_role(mol, role)
        template = oxoacid_role_template(mol, role)
        if template is not None and template.kind == OxoacidTemplateKind.UNSUPPORTED:
            continue
        if role.has_peroxy():
            if template is None or template.kind != OxoacidTemplateKind.ESTER_SUFFIX:
                continue
            rendered = _peroxy_oxoacid_ester_result(mol, component_atoms, role, template.suffix, branch_namer)
            if rendered is not None:
                matches.append(rendered)
            continue
        if role.count(OxoLigandRole.ALKOXY) != 1:
            continue
        ester_ligand = next(ligand for ligand in role.ligands if ligand.role == OxoLigandRole.ALKOXY)
        if ester_ligand.attachment_atom is None:
            continue
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
        suffix = _oxoacid_ester_suffix_from_template_or_spec(mol, spec, role)
        if not suffix:
            continue
        name = f"{ester_name} {suffix}"
        oxoacid_atoms = {role.central, *role.oxygen_atoms}
        bindings = (
            NameAtomBinding(
                stage="shortcut",
                role="oxoacid_ester_modifier",
                term=ester_name,
                atom_ids=ester_component_atoms,
                bond_ids=_bond_ids_within_atoms(mol, ester_component_atoms),
                charge_atom_ids=_charged_atoms(mol, ester_component_atoms),
            ),
            *_central_oxo_role_bindings(mol, role, suffix, "oxoacid_ester_suffix"),
        )
        matches.append(_component_name_result(mol, component_atoms, name, "oxoacid_ester", bindings=bindings))
    return matches[0] if len(matches) == 1 else None


def _peroxy_oxoacid_ester_result(
    mol: Molecule,
    component_atoms: set[int],
    role: CentralOxoRole,
    suffix: str,
    branch_namer: RecursiveSubgraphNamer | None,
) -> SpecialComponentName | None:
    peroxy_ligands = [ligand for ligand in role.ligands if ligand.role == OxoLigandRole.PEROXY]
    if len(peroxy_ligands) != 1 or not suffix:
        return None
    ligand = peroxy_ligands[0]
    if ligand.attachment_atom is None:
        return None
    distal_oxygen = ligand.attachment_atom
    modifier_roots = [
        neighbor
        for neighbor in mol.get_neighbors(distal_oxygen)
        if neighbor in component_atoms and neighbor not in {role.central, ligand.oxygen}
    ]
    if len(modifier_roots) != 1:
        return None
    oxoacid_atoms = {role.central, *role.oxygen_atoms, distal_oxygen}
    modifier_atoms = _ester_modifier_atoms(mol, component_atoms, modifier_roots[0], oxoacid_atoms)
    if not modifier_atoms or modifier_atoms | oxoacid_atoms != component_atoms:
        return None
    modifier = _ester_modifier_name(
        mol,
        component_atoms,
        modifier_roots[0],
        distal_oxygen,
        oxoacid_atoms,
        branch_namer,
    )
    if not modifier:
        return None
    name = f"{modifier} {suffix}"
    peroxy_atoms = {ligand.oxygen, distal_oxygen}
    bindings = (
        NameAtomBinding(
            stage="shortcut",
            role="oxoacid_ester_modifier",
            term=modifier,
            atom_ids=modifier_atoms,
            bond_ids=_bond_ids_within_atoms(mol, modifier_atoms),
            charge_atom_ids=_charged_atoms(mol, modifier_atoms),
        ),
        NameAtomBinding(
            stage="shortcut",
            role="peroxy_bridge",
            term="peroxy",
            atom_ids=peroxy_atoms,
            bond_ids=_bond_ids_within_atoms(mol, peroxy_atoms),
        ),
        *_central_oxo_role_bindings(mol, role, suffix, "oxoacid_ester_suffix"),
    )
    return _component_name_result(mol, component_atoms, name, "oxoacid_ester", bindings=bindings)


def _central_oxo_role_bindings(
    mol: Molecule,
    role: CentralOxoRole,
    term: str,
    parent_role: str,
) -> tuple[NameAtomBinding, ...]:
    """Return graph-role bindings for a central oxoacid parent or suffix."""

    oxygen_atoms = set(role.oxygen_atoms)
    core_atoms = {role.central, *oxygen_atoms}
    bindings: list[NameAtomBinding] = [
        NameAtomBinding(
            stage="shortcut",
            role=parent_role,
            term=term,
            atom_ids=core_atoms,
            bond_ids=_bond_ids_within_atoms(mol, core_atoms),
            charge_atom_ids=_charged_atoms(mol, core_atoms),
        )
    ]
    for ligand in role.ligands:
        ligand_atoms = {role.central, ligand.oxygen}
        if ligand.attachment_atom is not None:
            ligand_atoms.add(ligand.attachment_atom)
        bindings.append(
            NameAtomBinding(
                stage="shortcut",
                role=f"oxoacid_ligand_{ligand.role.value}",
                term=f"oxoacid_ligand_{ligand.role.value}",
                atom_ids=ligand_atoms,
                bond_ids=_bond_ids_within_atoms(mol, ligand_atoms),
                charge_atom_ids={ligand.oxygen} if mol.atoms[ligand.oxygen].charge else set(),
            )
        )
    return tuple(bindings)


def _oxoacid_ester_suffix_from_template_or_spec(mol: Molecule, spec: dict, role: CentralOxoRole) -> str:
    template = oxoacid_role_template(mol, role)
    if template is not None and template.kind == OxoacidTemplateKind.ESTER_SUFFIX:
        if template.suffix:
            return template.suffix
        return spec.get("ester_suffix", "")
    return _oxoacid_ester_suffix(spec, role)


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
    branch_namer: RecursiveSubgraphNamer | None,
) -> str:
    if branch_namer is not None:
        name = branch_namer(mol, root, set(mol.atoms) - component_atoms | acid_atoms, upstream_atom=ester_oxygen)
        if isinstance(name, tuple):
            name = name[0]
        if name:
            return strip_outer_parentheses(name)
    return _alkyl_ligand_name(mol, component_atoms, root, ester_oxygen)


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


def organophosphinic_acid_result(mol: Molecule, component_atoms: set[int]) -> SpecialComponentName | None:
    """Name simple R-P(=O)(OH)H phosphinic acids with ligand/core bindings."""

    phosphorus = [idx for idx in component_atoms if mol.atoms[idx].symbol == "P"]
    if len(phosphorus) != 1:
        return None
    central = phosphorus[0]
    if (mol.atoms[central].total_h_count or mol.atoms[central].explicit_h_count) != 1:
        return None
    double_oxygen = []
    hydroxy_oxygen = []
    carbon_roots = []
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms:
            return None
        symbol = mol.atoms[neighbor].symbol
        bond = mol.get_bond(central, neighbor)
        if symbol == "O" and bond and bond.order == 2:
            double_oxygen.append(neighbor)
        elif symbol == "O" and bond and bond.order == 1 and mol.atoms[neighbor].charge == 0:
            hydroxy_oxygen.append(neighbor)
        elif symbol == "C" and bond and bond.order == 1:
            carbon_roots.append(neighbor)
        else:
            return None
    if len(double_oxygen) != 1 or len(hydroxy_oxygen) != 1 or len(carbon_roots) != 1:
        return None
    alkyl = _alkyl_ligand_name(mol, component_atoms, carbon_roots[0], central)
    if not alkyl:
        return None
    ligand_atoms = _carbon_ligand_atoms(mol, component_atoms, carbon_roots[0], central)
    core_atoms = {central, *double_oxygen, *hydroxy_oxygen}
    name = f"{alkyl}phosphinic acid"
    bindings = (
        NameAtomBinding(
            stage="shortcut",
            role="organophosphinic_ligand",
            term=alkyl,
            atom_ids=ligand_atoms,
            bond_ids=_bond_ids_within_atoms(mol, ligand_atoms),
        ),
        NameAtomBinding(
            stage="shortcut",
            role="organophosphinic_acid_core",
            term="phosphinic acid",
            atom_ids=core_atoms,
            bond_ids=_bond_ids_within_atoms(mol, core_atoms),
        ),
    )
    return _component_name_result(mol, component_atoms, name, "organophosphinic_acid", bindings=bindings)


def sulfoxide_parent_result(mol: Molecule, component_atoms: set[int]) -> SpecialComponentName | None:
    """Name simple dialkyl sulfoxides with ligand/core/stereo bindings."""

    sulfurs = [idx for idx in component_atoms if mol.atoms[idx].symbol == "S"]
    if len(sulfurs) != 1:
        return None
    central = sulfurs[0]
    oxygens = []
    carbon_roots = []
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms:
            return None
        symbol = mol.atoms[neighbor].symbol
        bond = mol.get_bond(central, neighbor)
        if symbol == "O" and bond and (bond.order == 2 or mol.atoms[neighbor].charge == -1):
            oxygens.append(neighbor)
        elif symbol == "C" and bond and bond.order == 1:
            carbon_roots.append(neighbor)
        else:
            return None
    if len(oxygens) != 1 or len(carbon_roots) != 2:
        return None
    left_atoms = _carbon_ligand_atoms(mol, component_atoms, carbon_roots[0], central)
    right_atoms = _carbon_ligand_atoms(mol, component_atoms, carbon_roots[1], central)
    if not left_atoms or not right_atoms or left_atoms & right_atoms:
        return None
    ligands = [_alkyl_ligand_name(mol, component_atoms, root, central) for root in carbon_roots]
    if any(not ligand for ligand in ligands):
        return None
    stereo = f"({mol.atoms[central].stereo})-" if mol.atoms[central].stereo else ""
    if ligands[0] == ligands[1]:
        name = f"{stereo}{multipliers.basic(2)}{ligands[0]} sulfoxide"
        ligand_bindings = (
            NameAtomBinding(
                stage="shortcut",
                role="sulfoxide_ligand",
                term=ligands[0],
                atom_ids=left_atoms | right_atoms,
                bond_ids=_bond_ids_within_atoms(mol, left_atoms | right_atoms),
            ),
        )
    else:
        ordered = sorted(zip(ligands, (left_atoms, right_atoms), strict=True), key=lambda item: item[0])
        name = f"{stereo}{' '.join(ligand for ligand, _atoms in ordered)} sulfoxide"
        ligand_bindings = tuple(
            NameAtomBinding(
                stage="shortcut",
                role="sulfoxide_ligand",
                term=ligand,
                atom_ids=set(atoms),
                bond_ids=_bond_ids_within_atoms(mol, set(atoms)),
            )
            for ligand, atoms in ordered
        )
    core_atoms = {central, *oxygens}
    core_binding = NameAtomBinding(
        stage="shortcut",
        role="sulfoxide_core",
        term="sulfoxide",
        atom_ids=core_atoms,
        bond_ids=_bond_ids_within_atoms(mol, core_atoms),
        charge_atom_ids=_charged_atoms(mol, core_atoms),
    )
    return _component_name_result(
        mol, component_atoms, name, "sulfoxide_parent", bindings=ligand_bindings + (core_binding,)
    )


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


def homonuclear_chain_parent_result(mol: Molecule, component_atoms: set[int]) -> SpecialComponentName | None:
    """Name acyclic same-element parent hydride chains and simple ligands."""

    backbone_symbols = {
        mol.atoms[idx].symbol
        for idx in component_atoms
        if mol.atoms[idx].symbol in RULES.components.mononuclear_parent_hydrides
        and mol.atoms[idx].symbol not in {"O", "F", "Cl", "Br", "I"}
    }
    if len(backbone_symbols) != 1:
        return None
    symbol = next(iter(backbone_symbols))
    backbone = [idx for idx in component_atoms if mol.atoms[idx].symbol == symbol]
    if len(backbone) < 2:
        return None
    if any(mol.atoms[idx].symbol not in {symbol, "C", "F", "Cl", "Br", "I"} for idx in component_atoms):
        return None
    chain = _ordered_backbone_chain(mol, backbone)
    if chain is None:
        return None
    chain_set = set(chain)
    ligand_bindings = []
    for atom_idx in component_atoms - chain_set:
        backbone_neighbors = [n for n in mol.get_neighbors(atom_idx) if n in chain_set]
        if len(backbone_neighbors) != 1:
            return None
        bond = mol.get_bond(atom_idx, backbone_neighbors[0])
        if bond is None or bond.order != 1:
            return None
        ligand_name = _terminal_ligand_name(mol, atom_idx, backbone_neighbors[0])
        if not ligand_name:
            return None
        ligand_bindings.append(
            NameAtomBinding(
                stage="shortcut",
                role="homonuclear_chain_ligand",
                term=ligand_name,
                atom_ids={atom_idx},
                bond_ids={bond.idx},
                locants=(str(chain.index(backbone_neighbors[0]) + 1),),
            )
        )
    bond_orders = [mol.get_bond(chain[idx], chain[idx + 1]).order for idx in range(len(chain) - 1)]
    parent = _same_element_parent_name(symbol, len(chain), bond_orders)
    if not parent:
        return None
    prefixes = _simple_chain_ligand_prefixes(mol, component_atoms, chain)
    name = f"{prefixes}{parent}" if prefixes else parent
    parent_binding = NameAtomBinding(
        stage="shortcut",
        role="homonuclear_chain_parent",
        term=parent,
        atom_ids=set(chain),
        bond_ids=_bond_ids_within_atoms(mol, set(chain)),
    )
    return _component_name_result(
        mol,
        component_atoms,
        name,
        "homonuclear_chain_parent",
        bindings=(parent_binding, *ligand_bindings),
    )


def simple_central_parent_hydride_result(mol: Molecule, component_atoms: set[int]) -> SpecialComponentName | None:
    """Name simple mononuclear parent hydrides with halogen/alkoxy ligands."""

    central_candidates = [
        idx
        for idx in component_atoms
        if mol.atoms[idx].symbol in RULES.components.mononuclear_parent_hydrides
        and mol.atoms[idx].symbol not in {"O", "F", "Cl", "Br", "I"}
        and mol.degree(idx) >= 2
    ]
    if len(central_candidates) != 1:
        return None
    central = central_candidates[0]
    central_symbol = mol.atoms[central].symbol
    ligand_names = []
    ligand_bindings = []
    for neighbor in mol.get_neighbors(central):
        if neighbor not in component_atoms:
            continue
        ligand = _central_ligand_name(mol, component_atoms, central, neighbor)
        if not ligand:
            return None
        ligand_names.append(ligand)
        ligand_atoms = _component_atoms_until_blocked(mol, component_atoms, neighbor, {central})
        ligand_atoms = ligand_atoms or {neighbor}
        bond = mol.get_bond(central, neighbor)
        ligand_bindings.append(
            NameAtomBinding(
                stage="shortcut",
                role="central_parent_hydride_ligand",
                term=ligand,
                atom_ids=set(ligand_atoms),
                bond_ids={bond.idx} if bond else set(),
            )
        )
    if not ligand_names:
        return None
    prefix = _grouped_ligand_prefix(ligand_names)
    lambda_text = _lambda_text(mol, central)
    parent = f"{lambda_text}{RULES.components.mononuclear_parent_hydrides[central_symbol]}"
    name = f"{prefix}{parent}"
    core_binding = NameAtomBinding(
        stage="shortcut",
        role="central_parent_hydride_core",
        term=parent,
        atom_ids={central},
        charge_atom_ids={central} if mol.atoms[central].charge else set(),
    )
    return _component_name_result(
        mol,
        component_atoms,
        name,
        "simple_central_parent_hydride",
        bindings=(core_binding, *ligand_bindings),
    )


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
    mixed_single_ligands = len(groups) > 1
    for name in sorted(groups):
        count = groups[name]
        if count == 1:
            parts.append(format_multiplier(name, 1, safe_enclose=mixed_single_ligands))
        else:
            parts.append(f"{format_multiplier(name, count)}-")
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


def _anhydride_half_atoms(mol: Molecule, start_c: int, bridge_o: int) -> set[int]:
    """Return original atoms belonging to one acid half of an anhydride."""

    half_atoms = set()
    queue = [start_c]
    visited = {bridge_o}
    while queue:
        curr = queue.pop(0)
        if curr not in half_atoms:
            half_atoms.add(curr)
            visited.add(curr)
            queue.extend([x for x in mol.get_neighbors(curr) if x not in visited])
    return half_atoms


def anhydride_half_name(mol: Molecule, start_c: int, bridge_o: int, component_namer: ComponentNamer) -> str:
    """Name one acid half of an anhydride component."""

    original_half_atoms = _anhydride_half_atoms(mol, start_c, bridge_o)
    half_atoms = set(original_half_atoms)
    sub_mol = Molecule()
    for n in half_atoms:
        atom = mol.atoms[n]
        sub_mol.add_atom(
            symbol=atom.symbol,
            idx=n,
            charge=atom.charge,
            stereo=atom.stereo,
            raw_stereo=atom.raw_stereo,
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


def _bond_ids_between(mol: Molecule, atom_pairs: set[tuple[int, int]]) -> set[int]:
    bond_ids = set()
    for a, b in atom_pairs:
        bond = mol.get_bond(a, b)
        if bond:
            bond_ids.add(bond.idx)
    return bond_ids


def _anhydride_core_atoms(mol: Molecule, bridge_o: int, carbonyl_carbons: list[int]) -> set[int]:
    core_atoms = {bridge_o, *carbonyl_carbons}
    for carbon in carbonyl_carbons:
        for neighbor in mol.get_neighbors(carbon):
            bond = mol.get_bond(carbon, neighbor)
            if bond and bond.order == 2 and mol.atoms[neighbor].symbol == "O":
                core_atoms.add(neighbor)
    return core_atoms


def _anhydride_core_bond_ids(
    mol: Molecule, bridge_o: int, carbonyl_carbons: list[int], core_atoms: set[int]
) -> set[int]:
    link_bonds = _bond_ids_between(mol, {(bridge_o, carbon) for carbon in carbonyl_carbons})
    carbonyl_bonds = set()
    for carbon in carbonyl_carbons:
        for neighbor in mol.get_neighbors(carbon):
            if neighbor not in core_atoms:
                continue
            bond = mol.get_bond(carbon, neighbor)
            if bond and bond.order == 2 and mol.atoms[neighbor].symbol == "O":
                carbonyl_bonds.add(bond.idx)
    return link_bonds | carbonyl_bonds


def try_name_anhydride_component_result(
    mol: Molecule,
    perceived_groups: list[PerceivedGroup],
    principal_key: str | None,
    component_namer: ComponentNamer,
) -> AnhydrideComponentName | None:
    """Return a graph-bound anhydride component name when supported."""

    if principal_key != "anhydride":
        return None
    for group in perceived_groups:
        if group.key != "anhydride":
            continue
        bridge_o = next((o for o in group.atoms_involved if mol.degree(o) == 2 and mol.atoms[o].symbol == "O"), None)
        if bridge_o is None:
            continue
        c_neighbors = [n for n in mol.get_neighbors(bridge_o) if mol.atoms[n].is_carbon]
        if len(c_neighbors) != 2:
            continue

        halves = []
        for carbon in c_neighbors:
            half_atoms = _anhydride_half_atoms(mol, carbon, bridge_o)
            halves.append(
                {
                    "name": anhydride_half_name(mol, carbon, bridge_o, component_namer),
                    "atoms": half_atoms,
                    "bonds": _bond_ids_within_atoms(mol, half_atoms),
                }
            )

        if halves[0]["name"] == halves[1]["name"]:
            name = f"{halves[0]['name']} anhydride"
            half_bindings = (
                NameAtomBinding(
                    stage="shortcut",
                    role="anhydride_half",
                    term=halves[0]["name"],
                    atom_ids=set(halves[0]["atoms"]) | set(halves[1]["atoms"]),
                    bond_ids=set(halves[0]["bonds"]) | set(halves[1]["bonds"]),
                ),
            )
        else:
            ordered_halves = sorted(halves, key=lambda half: str(half["name"]))
            name = f"{ordered_halves[0]['name']} {ordered_halves[1]['name']} anhydride"
            half_bindings = tuple(
                NameAtomBinding(
                    stage="shortcut",
                    role="anhydride_half",
                    term=str(half["name"]),
                    atom_ids=set(half["atoms"]),
                    bond_ids=set(half["bonds"]),
                )
                for half in ordered_halves
            )

        core_atoms = _anhydride_core_atoms(mol, bridge_o, c_neighbors)
        core_bonds = _anhydride_core_bond_ids(mol, bridge_o, c_neighbors, core_atoms)
        core_binding = NameAtomBinding(
            stage="shortcut",
            role="anhydride_core",
            term="anhydride",
            atom_ids=core_atoms,
            bond_ids=core_bonds,
        )
        return AnhydrideComponentName(name=name, bindings=half_bindings + (core_binding,))
    return None


