# bluenamer/api.py

import re
from dataclasses import dataclass, field

from .assembler import assemble_name_raw, post_process_name
from .assembly_parts import AssemblyParts, SubstituentItem
from .assembly_spiro import extract_spiro_side_prefixes
from .chains import find_ring_systems, get_cyclic_atoms
from .component_namer import name_component as _name_component_impl
from .engine import DEFAULT_NAMING_ENGINE
from .formatting import format_counted_prefixes, format_multiplier, strip_outer_parentheses
from .group_atom_roles import amide_nitrogen
from .heteroatom_subgraphs import name_heteroatom_subgraph
from .ionic_naming import apply_anionic_parent_names, apply_cationic_imino_names, apply_cationic_imino_parent_prefixes
from .molecule import DecisionTrace, Molecule, NameAnalysis, TracePhase
from .name_assembly import NameAssemblyResult, token_span_trace_data
from .naming_context import NamingIntent
from .nomenclature import RULES
from .parent_pipeline import build_parent_assembly_plan, resolve_retained_parent
from .parent_selection import select_principal_parent
from .perception import PerceivedGroup, perceive_groups
from .rules import elision, multipliers, stems
from .spiro_assembly import SpiroAssembly
from .subgraph_tools import (
    add_indicated_hydrogens as _add_indicated_hydrogens,
)
from .subgraph_tools import (
    add_parent_features as _add_subgraph_parent_features,
)
from .subgraph_tools import (
    emit_bond_stereo as _emit_bond_stereo,
)
from .subgraph_tools import (
    find_spiro_side_pair as _find_spiro_side_pair,
)
from .subgraph_tools import (
    spiro_side_component as _spiro_side_component,
)
from .subgraph_tools import (
    subgraph_component as _subgraph_component,
)
from .substituent_tokens import graph_bound_substituent_tokens
from .trace_helpers import (
    add_substituent_trace as _add_substituent_trace,
)
from .trace_helpers import assembly_substituent_tree as _assembly_substituent_tree
from .trace_helpers import (
    assembly_trace_segments as _assembly_trace_segments,
)
from .trace_helpers import (
    bond_ids_within as _bond_ids_within,
)
from .trace_helpers import decision_trace_data, trace_decision


@dataclass
class DirectSubgraphPrefix:
    """Structured result for a direct functional-prefix subgraph."""

    name: str
    group_key: str = ""
    attachment_atom: int | None = None
    group_atoms: set[int] = field(default_factory=set)
    core_atoms: set[int] = field(default_factory=set)
    group_bonds: set[int] = field(default_factory=set)
    ligand_trees: list[dict] = field(default_factory=list)
    ligand_trace_segments: list[dict] = field(default_factory=list)
    ligand_decisions: list[dict] = field(default_factory=list)
    source: str = "direct_subgraph_prefix"

    def __bool__(self) -> bool:
        return bool(self.name)

    def trace_data(self) -> dict:
        """Return JSON-safe metadata for decision traces and tree nodes."""

        return {
            "name": self.name,
            "group_key": self.group_key,
            "attachment_atom": self.attachment_atom,
            "group_atoms": sorted(self.group_atoms),
            "core_atoms": sorted(self.core_atoms),
            "group_bonds": sorted(self.group_bonds),
            "ligand_count": len(self.ligand_trees),
            "source": self.source,
        }


def _direct_subgraph_prefix(mol: Molecule, start_idx: int, component: set[int]) -> DirectSubgraphPrefix | None:
    """Return a direct functional-group prefix when the subgraph is one group.

    Blue Book references: P-63 through P-67 for direct detachable prefixes such
    as nitro, cyano, carboxy, carbamoyl, and halo-carbonyl prefixes.
    """

    if mol.atoms[start_idx].is_carbon and start_idx in component:
        terminal_sulfurs = [
            neighbor
            for neighbor in mol.get_neighbors(start_idx)
            if neighbor in component
            and mol.atoms[neighbor].symbol == "S"
            and (bond := mol.get_bond(start_idx, neighbor)) is not None
            and bond.order == 2
            and all(other == start_idx for other in mol.get_neighbors(neighbor))
        ]
        if len(terminal_sulfurs) == 1 and component == {start_idx, terminal_sulfurs[0]}:
            return DirectSubgraphPrefix(name="thioformyl", core_atoms=set(component), source="structural_direct_prefix")

        triple_phosphorus = [
            neighbor
            for neighbor in mol.get_neighbors(start_idx)
            if mol.atoms[neighbor].symbol == "P"
            and neighbor in component
            and (bond := mol.get_bond(start_idx, neighbor)) is not None
            and bond.order == 3
        ]
        if len(triple_phosphorus) == 1:
            phosphorus = triple_phosphorus[0]
            terminal_oxo = [
                neighbor
                for neighbor in mol.get_neighbors(phosphorus)
                if neighbor in component
                and neighbor != start_idx
                and mol.atoms[neighbor].symbol == "O"
                and (bond := mol.get_bond(phosphorus, neighbor)) is not None
                and bond.order == 2
                and all(other == phosphorus for other in mol.get_neighbors(neighbor))
            ]
            if component == {start_idx, phosphorus}:
                return DirectSubgraphPrefix(
                    name="phosphanylidynemethyl",
                    core_atoms=set(component),
                    source="structural_direct_prefix",
                )
            if len(terminal_oxo) == 1 and component == {start_idx, phosphorus, terminal_oxo[0]}:
                return DirectSubgraphPrefix(
                    name="oxophosphanylidynemethyl",
                    core_atoms=set(component),
                    source="structural_direct_prefix",
                )

    for group in perceive_groups(mol):
        if start_idx in group.atoms_involved and group.atoms_involved.issubset(component):
            if group.key in RULES.prefixes.amide_like_groups:
                substituted = _direct_amide_subgraph_prefix(mol, group, component)
                if substituted:
                    return substituted
            prefix = RULES.functional_groups.direct_subgraph_prefix_for(group.key)
            if prefix:
                return DirectSubgraphPrefix(
                    name=prefix,
                    group_key=group.key,
                    attachment_atom=group.attachment_carbon,
                    group_atoms=set(group.atom_ids),
                    core_atoms=set(group.atoms_involved),
                    group_bonds=set(group.bond_ids),
                )
    return None


def _direct_amide_subgraph_prefix(
    mol: Molecule, group: PerceivedGroup, component: set[int]
) -> DirectSubgraphPrefix | None:
    """Return substituted carbamoyl/carbamothioyl for direct amide subgraphs."""

    amide_n = amide_nitrogen(mol, group)
    if amide_n is None:
        return None
    n_substituents = [
        n
        for n in mol.get_neighbors(amide_n)
        if n in component and n not in group.atoms_involved and mol.atoms[n].symbol != "H"
    ]
    base = RULES.functional_groups.prefix_for(group.key) or ""
    if not base:
        return None
    if not n_substituents:
        return DirectSubgraphPrefix(
            name=base,
            group_key=group.key,
            attachment_atom=group.attachment_carbon,
            group_atoms=set(group.atom_ids),
            core_atoms=set(group.atoms_involved),
            group_bonds=set(group.bond_ids),
        )
    outside_component = set(mol.atoms.keys()) - component
    branch_names = []
    ligand_trees = []
    ligand_trace_segments = []
    ligand_decisions = []
    for n_sub in n_substituents:
        branch_decisions = DecisionTrace()
        branch, branch_trace, branch_tree = name_subgraph(
            mol,
            n_sub,
            outside_component | set(group.atoms_involved) | {amide_n},
            upstream_atom=amide_n,
            return_trace=True,
            return_tree=True,
            decision_trace=branch_decisions,
        )
        branch = strip_outer_parentheses(branch)
        if branch:
            branch_names.append(branch)
            ligand_trace_segments.extend(branch_trace)
            ligand_decisions.extend(decision_trace_data(branch_decisions))
            if branch_tree:
                ligand_trees.append(branch_tree)
    name = base if not branch_names else f"({format_counted_prefixes(branch_names)}{base})"
    return DirectSubgraphPrefix(
        name=name,
        group_key=group.key,
        attachment_atom=group.attachment_carbon,
        group_atoms=set(group.atom_ids),
        core_atoms=set(group.atoms_involved),
        group_bonds=set(group.bond_ids),
        ligand_trees=ligand_trees,
        ligand_trace_segments=ligand_trace_segments,
        ligand_decisions=ligand_decisions,
    )


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
        selection = select_principal_parent(mol, [], valid_rings, [start_idx])
        if selection is None:
            return None
        return selection.with_fixed_start(selection.requires_fixed_substituent_start)

    paths = _find_acyclic_subgraph_paths(mol, start_idx, component, cyclic_atoms, sub_exclude)
    if not paths:
        return None
    selection = select_principal_parent(mol, paths, [], [start_idx])
    if selection is None:
        return None
    return selection.with_fixed_start(False)


def _spiro_subgraph_assembly(mol: Molecule, c_idx: int, sub_comp: set[int]) -> SpiroAssembly:
    """Name a side ring as structured spiro assembly data.

    Blue Book references: P-24 and P-52.3; the attachment atom is temporarily
    represented as silicon so the side ring can be named independently, then the
    silane marker is stripped back out before assembly rendering.
    """

    retained_n_ring = _retained_n_ring_spiro_assembly(mol, c_idx, sub_comp)
    if retained_n_ring is not None:
        return retained_n_ring

    simple_side_ring = _simple_monocyclic_spiro_side_assembly(mol, c_idx, sub_comp)
    if simple_side_ring is not None:
        return simple_side_ring

    heteroaromatic_side = _heteroaromatic_spiro_side_assembly(mol, c_idx, sub_comp)
    if heteroaromatic_side is not None:
        return heteroaromatic_side

    sub_mol = Molecule()
    for n in sub_comp:
        atom = mol.atoms[n]
        symbol = "Si" if n == c_idx else atom.symbol
        sub_mol.add_atom(
            symbol=symbol,
            idx=n,
            charge=atom.charge,
            stereo=atom.stereo,
            raw_stereo=atom.raw_stereo,
            is_aromatic=atom.is_aromatic,
            explicit_h_count=atom.explicit_h_count,
            total_h_count=atom.total_h_count,
        )
    for n in sub_comp:
        for nxt in mol.get_neighbors(n):
            if nxt in sub_comp and n < nxt:
                bond = mol.get_bond(n, nxt)
                sub_mol.add_bond(u=n, v=nxt, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring)

    sub_name_raw = name_component(sub_mol, sub_comp, is_substituent=False)
    match = re.search(r"(?:(^|-)(\d+)-)?sil[a]?", sub_name_raw)
    if not match:
        side_prefixes, side_parent_name, side_suffixes = extract_spiro_side_prefixes(sub_name_raw)
        return SpiroAssembly(
            parent_locant="",
            side_locant="1",
            side_parent_name=side_parent_name,
            side_prefixes=tuple(side_prefixes),
            side_suffixes=tuple(side_suffixes),
        )

    loc = match.group(2) if match.group(2) else "1"
    if match.group(2):
        sub_name_clean = re.sub(rf"(^|-){loc}-sil[a]?-?", r"\1", sub_name_raw)
    else:
        sub_name_clean = re.sub(r"sil[a]?-?", "", sub_name_raw)

    sub_name_clean = sub_name_clean.replace("--", "-").strip("-")
    sub_name_clean = sub_name_clean.replace("-cyclo", "cyclo")
    if not sub_name_clean:
        raise ValueError("spiro side component marker removal left no named parent")
    side_prefixes, side_parent_name, side_suffixes = extract_spiro_side_prefixes(sub_name_clean)
    return SpiroAssembly(
        parent_locant="",
        side_locant=loc,
        side_parent_name=side_parent_name,
        side_prefixes=tuple(side_prefixes),
        side_suffixes=tuple(side_suffixes),
    )


def _simple_monocyclic_spiro_side_assembly(mol: Molecule, c_idx: int, sub_comp: set[int]) -> SpiroAssembly | None:
    """Name a monocyclic spiro side ring before naming its external branches."""

    ring = _simple_heterocycle_containing_atom(mol, c_idx, sub_comp)
    if ring is None:
        return None
    if _ring_double_bond_count(mol, ring) != 0:
        return None
    if not all(mol.atoms[idx].symbol == "C" for idx in ring):
        return None
    if not _side_ring_has_heteroaromatic_branch(mol, ring, sub_comp):
        return None
    locant_map = _number_simple_spiro_side_ring(mol, ring, c_idx, sub_comp)
    parent_name = f"cyclo{stems.stem_for(len(ring))}ane"
    side_prefixes = _spiro_side_ring_branch_prefixes(mol, ring, sub_comp, locant_map)
    if not side_prefixes:
        return None
    return SpiroAssembly(
        parent_locant="",
        side_locant=locant_map[c_idx],
        side_parent_name=parent_name,
        side_prefixes=tuple(side_prefixes),
    )


def _side_ring_has_heteroaromatic_branch(mol: Molecule, ring: list[int], sub_comp: set[int]) -> bool:
    ring_atoms = set(ring)
    for atom_idx in ring:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor in sub_comp and neighbor not in ring_atoms:
                branch_atoms = _subgraph_component(mol, neighbor, set(mol.atoms) - (set(sub_comp) - ring_atoms))
                if _component_has_unsaturated_heterocycle(mol, branch_atoms):
                    return True
    return False


def _component_has_unsaturated_heterocycle(mol: Molecule, component_atoms: set[int]) -> bool:
    for atom_idx in component_atoms:
        ring = _simple_heterocycle_containing_atom(mol, atom_idx, component_atoms)
        if (
            ring
            and _is_aromatic_ring(mol, ring)
            and any(mol.atoms[idx].symbol != "C" for idx in ring)
            and _ring_double_bond_count(mol, ring) > 0
        ):
            return True
    return False


def _heteroaromatic_spiro_side_assembly(mol: Molecule, c_idx: int, sub_comp: set[int]) -> SpiroAssembly | None:
    """Build a graph-numbered Hantzsch-Widman side component for spiro rings."""

    ring = _simple_heterocycle_containing_atom(mol, c_idx, sub_comp)
    if ring is None or not any(mol.atoms[idx].symbol != "C" for idx in ring):
        return None
    if not _is_aromatic_ring(mol, ring):
        return None
    if _ring_double_bond_count(mol, ring) == 0:
        return None
    resolved = _hantzsch_widman_side_parent(mol, ring)
    if resolved is None:
        return None
    parent_name, locant_map = resolved
    side_prefixes = _spiro_side_ring_branch_prefixes(mol, ring, sub_comp, locant_map)
    return SpiroAssembly(
        parent_locant="",
        side_locant=locant_map[c_idx],
        side_parent_name=parent_name,
        side_prefixes=tuple(side_prefixes),
    )


def _spiro_side_ring_branch_prefixes(
    mol: Molecule,
    ring: list[int],
    sub_comp: set[int],
    locant_map: dict[int, str],
) -> list[str]:
    ring_atoms = set(ring)
    allowed_branch_atoms = set(sub_comp) - ring_atoms
    branch_exclude = set(mol.atoms) - allowed_branch_atoms
    side_prefixes = []
    for atom_idx in sorted(ring_atoms, key=lambda idx: int(locant_map[idx])):
        for neighbor in sorted(mol.get_neighbors(atom_idx)):
            if neighbor not in allowed_branch_atoms:
                continue
            branch = _name_heteroaromatic_branch_substituent(mol, neighbor, atom_idx, allowed_branch_atoms)
            if not branch:
                branch = name_subgraph(mol, neighbor, branch_exclude, upstream_atom=atom_idx)
            if branch:
                side_prefixes.append(f"{locant_map[atom_idx]}'-{format_multiplier(branch, 1, safe_enclose=True)}")
    return side_prefixes


def _name_heteroaromatic_branch_substituent(
    mol: Molecule,
    start_idx: int,
    upstream_atom: int,
    allowed_atoms: set[int],
) -> str:
    """Name a heteroaromatic branch from its own graph-numbered ring."""

    branch_atoms = _subgraph_component(mol, start_idx, set(mol.atoms) - allowed_atoms)
    ring = _simple_heterocycle_containing_atom(mol, start_idx, branch_atoms)
    if ring is None:
        return ""
    if (
        not _is_aromatic_ring(mol, ring)
        or not any(mol.atoms[idx].symbol != "C" for idx in ring)
        or _ring_double_bond_count(mol, ring) == 0
    ):
        return ""
    resolved = _hantzsch_widman_side_parent(mol, ring)
    if resolved is None:
        return ""
    parent_name, locant_map = resolved
    if start_idx not in locant_map:
        return ""
    ring_atoms = set(ring)
    prefix_items = []
    branch_allowed = set(branch_atoms) - ring_atoms
    branch_exclude = set(mol.atoms) - branch_allowed
    for atom_idx in sorted(ring_atoms, key=lambda idx: int(locant_map[idx])):
        for neighbor in sorted(mol.get_neighbors(atom_idx)):
            if neighbor == upstream_atom or neighbor not in branch_allowed:
                continue
            substituent = name_subgraph(mol, neighbor, branch_exclude, upstream_atom=atom_idx)
            if substituent:
                prefix_items.append(f"{locant_map[atom_idx]}-{substituent}")
    stem = parent_name[:-1] if parent_name.endswith("e") else parent_name
    substituent_name = f"{stem}-{locant_map[start_idx]}-yl"
    if prefix_items:
        return "-".join(prefix_items + [substituent_name])
    return substituent_name


def _simple_heterocycle_containing_atom(mol: Molecule, atom_idx: int, component_atoms: set[int]) -> list[int] | None:
    """Return a simple monocyclic side ring containing atom_idx."""

    candidates = []
    neighbors = [n for n in mol.get_neighbors(atom_idx) if n in component_atoms]
    for pos, first in enumerate(neighbors):
        for second in neighbors[pos + 1 :]:
            path = _shortest_path_without_atom(mol, first, second, component_atoms, atom_idx, 9)
            if not path:
                continue
            ring = [atom_idx] + path
            if len(ring) < 3 or len(ring) > 10:
                continue
            ring_set = set(ring)
            if any(sum(1 for n in mol.get_neighbors(idx) if n in ring_set) != 2 for idx in ring):
                continue
            if any(mol.atoms[idx].element.hw_priority is None and mol.atoms[idx].symbol != "C" for idx in ring):
                continue
            candidates.append(_ordered_ring_from_member(mol, ring, atom_idx))
    if not candidates:
        return None
    return min(candidates, key=lambda ring: (len(ring), _hantzsch_widman_ring_score(mol, ring), ring))


def _ordered_ring_from_member(mol: Molecule, ring: list[int], start: int) -> list[int]:
    ring_set = set(ring)
    first = min(n for n in mol.get_neighbors(start) if n in ring_set)
    ordered = [start, first]
    prev = start
    current = first
    while True:
        choices = [n for n in mol.get_neighbors(current) if n in ring_set and n != prev]
        if not choices:
            return ring
        nxt = min(choices)
        if nxt == start:
            return ordered
        if nxt in ordered:
            return ring
        ordered.append(nxt)
        prev, current = current, nxt


def _number_simple_spiro_side_ring(
    mol: Molecule,
    ring: list[int],
    spiro_atom: int,
    sub_comp: set[int],
) -> dict[int, str]:
    """Number a monocyclic side ring from the spiro atom with branch tie-breaks."""

    spiro_pos = ring.index(spiro_atom)
    forward = ring[spiro_pos:] + ring[:spiro_pos]
    reverse = [forward[0], *reversed(forward[1:])]

    def score(order: list[int]) -> tuple:
        locants = {atom_idx: pos + 1 for pos, atom_idx in enumerate(order)}
        branch_locants = []
        ring_atoms = set(ring)
        for atom_idx in ring:
            if any(n in sub_comp and n not in ring_atoms for n in mol.get_neighbors(atom_idx)):
                branch_locants.append(locants[atom_idx])
        return (tuple(sorted(branch_locants)), order)

    best = min((forward, reverse), key=score)
    return {atom_idx: str(pos + 1) for pos, atom_idx in enumerate(best)}


def _hantzsch_widman_side_parent(mol: Molecule, ring: list[int]) -> tuple[str, dict[int, str]] | None:
    """Return a generic Hantzsch-Widman parent name and atom locants."""

    variants = []
    for offset in range(len(ring)):
        rotated = ring[offset:] + ring[:offset]
        for oriented in (rotated, [rotated[0], *reversed(rotated[1:])]):
            first = mol.atoms[oriented[0]]
            if first.symbol == "C" or first.element.hw_priority is None:
                continue
            highest = min(
                mol.atoms[idx].element.hw_priority
                for idx in oriented
                if mol.atoms[idx].symbol != "C" and mol.atoms[idx].element.hw_priority is not None
            )
            if first.element.hw_priority != highest:
                continue
            name = _generic_hantzsch_widman_name(mol, oriented)
            if not name:
                continue
            locant_map = {atom_idx: str(pos + 1) for pos, atom_idx in enumerate(oriented)}
            variants.append((_hantzsch_widman_ring_score(mol, oriented), name, locant_map))
    if not variants:
        return None
    _, name, locant_map = min(variants, key=lambda item: (item[0], item[1]))
    return name, locant_map


def _generic_hantzsch_widman_name(mol: Molecule, oriented: list[int]) -> str:
    hetero = [
        (pos, mol.atoms[idx].element.hw_priority or 999, mol.atoms[idx].element.hw_stem or "")
        for pos, idx in enumerate(oriented, start=1)
        if mol.atoms[idx].symbol != "C"
    ]
    if not hetero:
        return ""
    groups: dict[int, list[tuple[int, str]]] = {}
    for pos, priority, prefix in hetero:
        if not prefix:
            return ""
        groups.setdefault(priority, []).append((pos, prefix))
    locants = []
    prefix_parts = []
    for priority in sorted(groups):
        group = sorted(groups[priority])
        locants.extend(str(pos) for pos, _ in group)
        prefix_parts.append(_multiply_hw_prefix(group[0][1], len(group)))
    suffix = _hantzsch_widman_stem(
        len(oriented),
        _ring_double_bond_count(mol, oriented),
        any(mol.atoms[idx].symbol == "N" for idx in oriented),
    )
    if not suffix:
        return ""
    parent = _join_hw_parts(prefix_parts + [suffix])
    return f"{','.join(locants)}-{parent}" if len(hetero) > 1 else parent


def _multiply_hw_prefix(prefix: str, count: int) -> str:
    if count == 1:
        return prefix
    return elision.elide_terminal_a(multipliers.basic(count), prefix)


def _join_hw_parts(parts: list[str]) -> str:
    if not parts:
        return ""
    result = parts[0]
    for part in parts[1:]:
        result = elision.elide_terminal_a(result, part)
    return result


def _hantzsch_widman_stem(size: int, double_bonds: int, has_nitrogen: bool) -> str:
    unsaturated = double_bonds > 0
    if size == 3:
        if not unsaturated:
            return "iridine" if has_nitrogen else "irane"
        return "irine" if has_nitrogen else "irene"
    if size == 4:
        return ("etidine" if has_nitrogen else "etane") if not unsaturated else "ete"
    if size == 5:
        return ("olidine" if has_nitrogen else "olane") if not unsaturated else "ole"
    if size == 6:
        return ("inane" if has_nitrogen else "ane") if not unsaturated else "ine"
    larger = {
        7: ("epine", "epane"),
        8: ("ocine", "ocane"),
        9: ("onine", "onane"),
        10: ("ecine", "ecane"),
    }
    if size not in larger:
        return ""
    unsat, sat = larger[size]
    return unsat if unsaturated else sat


def _hantzsch_widman_ring_score(mol: Molecule, oriented: list[int]) -> tuple:
    hetero_positions = [
        (pos, mol.atoms[idx].element.hw_priority or 999)
        for pos, idx in enumerate(oriented, start=1)
        if mol.atoms[idx].symbol != "C"
    ]
    grouped = []
    for priority in sorted({priority for _, priority in hetero_positions}):
        grouped.append(tuple(pos for pos, item_priority in hetero_positions if item_priority == priority))
    double_positions = tuple(
        pos
        for pos, (a, b) in enumerate(zip(oriented, oriented[1:] + oriented[:1]), start=1)
        if (bond := mol.get_bond(a, b)) is not None and bond.order == 2
    )
    return (
        tuple(pos for pos, _ in hetero_positions),
        tuple(grouped),
        double_positions,
    )


def _ring_double_bond_count(mol: Molecule, ring: list[int]) -> int:
    return sum(
        1 for a, b in zip(ring, ring[1:] + ring[:1]) if (bond := mol.get_bond(a, b)) is not None and bond.order == 2
    )


def _is_aromatic_ring(mol: Molecule, ring: list[int]) -> bool:
    """Return true when every ring atom came from an aromatic input atom."""

    return bool(ring) and all(mol.atoms[idx].is_aromatic for idx in ring)


def _retained_n_ring_spiro_assembly(mol: Molecule, c_idx: int, sub_comp: set[int]) -> SpiroAssembly | None:
    """Name spiro side components whose shared atom is in a retained N-ring."""

    ring = _small_n_ring_containing_atom(mol, c_idx, sub_comp)
    if not ring:
        return None
    ring_atoms = set(ring)
    n_atom = next(atom_idx for atom_idx in ring if mol.atoms[atom_idx].symbol == "N")
    if mol.atoms[n_atom].charge <= 0:
        return None
    locant_map = _number_saturated_n_ring_for_spiro(mol, ring, n_atom, c_idx, sub_comp)
    ring_size = len(ring)
    ionic_parent = RULES.charges.saturated_n_ring_ionic_parents.get(ring_size)
    if not ionic_parent:
        return None
    neutral_parent = next(
        (neutral for neutral, ionic in RULES.charges.retained_ionic_n_parents.items() if ionic == ionic_parent),
        "",
    )
    if not neutral_parent:
        return None
    stem = neutral_parent[:-1] if neutral_parent.endswith("e") else neutral_parent
    parent_name = f"{stem}-1-ium"

    side_prefixes = []
    allowed_branch_atoms = set(sub_comp) - ring_atoms
    branch_exclude = set(mol.atoms) - allowed_branch_atoms
    for atom_idx in sorted(ring_atoms, key=lambda idx: int(locant_map[idx])):
        for neighbor in sorted(mol.get_neighbors(atom_idx)):
            if neighbor not in allowed_branch_atoms:
                continue
            branch = name_subgraph(mol, neighbor, branch_exclude, upstream_atom=atom_idx)
            if branch:
                side_prefixes.append(f"{locant_map[atom_idx]}'-{branch}")

    return SpiroAssembly(
        parent_locant="",
        side_locant=locant_map[c_idx],
        side_parent_name=parent_name,
        side_prefixes=tuple(side_prefixes),
    )


def _small_n_ring_containing_atom(mol: Molecule, atom_idx: int, component_atoms: set[int]) -> list[int] | None:
    """Return a small saturated N-ring containing atom_idx, if present."""

    supported_sizes = set(RULES.charges.saturated_n_ring_ionic_parents)
    neighbors = [n for n in mol.get_neighbors(atom_idx) if n in component_atoms]
    candidates = []
    for pos, first in enumerate(neighbors):
        for second in neighbors[pos + 1 :]:
            path = _shortest_path_without_atom(mol, first, second, component_atoms, atom_idx, max(supported_sizes) - 1)
            if not path:
                continue
            ring = [atom_idx] + path
            if len(ring) not in supported_sizes:
                continue
            if sum(1 for idx in ring if mol.atoms[idx].symbol == "N") != 1:
                continue
            if any(mol.atoms[idx].symbol not in {"C", "N"} for idx in ring):
                continue
            if any((bond := mol.get_bond(a, b)) is None or bond.order != 1 for a, b in zip(ring, ring[1:] + ring[:1])):
                continue
            candidates.append(ring)
    if not candidates:
        return None
    return min(candidates, key=lambda ring: (len(ring), sorted(ring)))


def _shortest_path_without_atom(
    mol: Molecule,
    start: int,
    end: int,
    allowed_atoms: set[int],
    excluded_atom: int,
    max_edges: int,
) -> list[int] | None:
    queue: list[list[int]] = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        if len(path) - 1 > max_edges:
            continue
        current = path[-1]
        if current == end:
            return path
        for neighbor in mol.get_neighbors(current):
            if neighbor == excluded_atom or neighbor not in allowed_atoms or neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(path + [neighbor])
    return None


def _number_saturated_n_ring_for_spiro(
    mol: Molecule,
    ring: list[int],
    n_atom: int,
    spiro_atom: int,
    sub_comp: set[int],
) -> dict[int, str]:
    """Number a saturated retained N-ring from N=1 with graph-feature tie-breaks."""

    n_pos = ring.index(n_atom)
    forward = ring[n_pos:] + ring[:n_pos]
    reverse = [forward[0], *reversed(forward[1:])]
    candidates = []
    ring_atoms = set(ring)
    for order in (forward, reverse):
        locants = {atom_idx: str(pos + 1) for pos, atom_idx in enumerate(order)}
        branch_locants = tuple(
            sorted(
                int(locants[atom_idx])
                for atom_idx in ring
                if atom_idx != spiro_atom
                and any(neighbor in sub_comp and neighbor not in ring_atoms for neighbor in mol.get_neighbors(atom_idx))
            )
        )
        candidates.append((branch_locants, -int(locants[spiro_atom]), tuple(order), locants))
    return min(candidates)[3]


def _spiro_subgraph_name(mol: Molecule, c_idx: int, sub_comp: set[int]) -> str:
    """Compatibility marker for older assembly callers."""

    spiro = _spiro_subgraph_assembly(mol, c_idx, sub_comp)
    return f"[SPIRO]-{spiro.side_locant}-{spiro.side_parent_name}"


def _collect_subgraph_substituents(
    mol: Molecule,
    candidate_path: list[int],
    sub_perceived: list[PerceivedGroup],
    sub_exclude: set[int],
) -> dict[int, list[SubstituentItem]]:
    """Collect prefixes attached to a recursive subgraph parent.

    Blue Book references: P-14.2, P-16.5, P-44, P-61 through P-67, and P-24
    for multiplicative prefixes, complex prefixes, parent substituents, and
    spiro side-ring substituents.
    """

    main_set = set(candidate_path)
    subst_mapping: dict[int, list[SubstituentItem]] = {}
    sub_handled_atoms = set()

    for group in sub_perceived:
        if group.attachment_carbon in main_set and not group.is_principal_candidate:
            rule = RULES.functional_groups.by_key.get(group.key)
            name = rule.prefix if rule and rule.role == "prefix" else ""
            if name:
                subst_mapping.setdefault(group.attachment_carbon, []).append(
                    SubstituentItem(
                        name=name,
                        locants=[],
                        atom_ids=set(group.atoms_involved),
                        bond_ids=_bond_ids_within(mol, set(group.atoms_involved) | {group.attachment_carbon}),
                        charge_atom_ids=_charged_atoms(mol, set(group.atoms_involved)),
                    )
                )
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
            subst_mapping.setdefault(c_idx, []).append(
                SubstituentItem(
                    name="",
                    locants=[],
                    atom_ids=sub_comp - {c_idx},
                    bond_ids=_bond_ids_within(mol, sub_comp),
                    charge_atom_ids=_charged_atoms(mol, sub_comp - {c_idx}),
                    spiro=_spiro_subgraph_assembly(mol, c_idx, sub_comp),
                )
            )
            sub_handled_atoms.update(sub_comp - {c_idx})
            n_subs = [n for n in n_subs if n not in sub_comp]

        for n_idx in n_subs:
            if n_idx not in main_set and n_idx not in sub_handled_atoms and n_idx not in sub_exclude:
                branch_decisions = DecisionTrace()
                branch_name, branch_trace, branch_tree = name_subgraph(
                    mol,
                    n_idx,
                    sub_exclude | main_set,
                    upstream_atom=c_idx,
                    return_trace=True,
                    return_tree=True,
                    decision_trace=branch_decisions,
                )
                if branch_name:
                    branch_exclude = sub_exclude | main_set
                    branch_atoms = _subgraph_component(mol, n_idx, branch_exclude)
                    branch_with_attachment = branch_atoms | {c_idx}
                    subst_mapping.setdefault(c_idx, []).append(
                        SubstituentItem(
                            name=branch_name,
                            locants=[],
                            atom_ids=branch_atoms,
                            bond_ids=_bond_ids_within(mol, branch_with_attachment),
                            charge_atom_ids=_charged_atoms(mol, branch_atoms),
                            emitted_tokens=graph_bound_substituent_tokens(
                                mol,
                                n_idx,
                                branch_atoms,
                                branch_name,
                                c_idx,
                                branch_exclude,
                                name_subgraph,
                            ),
                            trace_segments=branch_trace,
                            nested_decisions=decision_trace_data(branch_decisions),
                            substituent_tree=branch_tree,
                        )
                    )

    return subst_mapping


def _add_subgraph_substituents(parts: AssemblyParts, subst_mapping: dict[int, list[SubstituentItem]], get_loc) -> None:
    """Add collected substituent prefixes to assembly parts.

    Blue Book references: P-14.2 and P-16.5 for locants, multiplicative
    prefixes, and complex substituent citation.
    """

    for c_idx, items in subst_mapping.items():
        locant = get_loc(c_idx)
        for item in items:
            _add_substituent_trace(
                parts,
                item.name,
                locant,
                item.atom_ids,
                item.bond_ids,
                item.charge_atom_ids,
                item.trace_segments,
                item.nested_decisions,
                item.emitted_tokens,
                substituent_tree=item.substituent_tree,
                spiro=item.spiro,
            )


def _charged_atoms(mol: Molecule, atom_ids: set[int]) -> set[int]:
    """Return formally charged atoms from an already named graph fragment."""

    return {atom_idx for atom_idx in atom_ids if mol.atoms[atom_idx].charge != 0}


def _finalize_subgraph_name(name: str, parts: AssemblyParts) -> str:
    """Apply recursive-substituent wrapping rules to an assembled name.

    Blue Book references: P-13.6 and P-16.5 for substituent suffix citation and
    parentheses around complex substituent prefixes.
    """

    if parts.principal_group is not None and parts.principal_group.key in {"nitrile", "ring_nitrile"}:
        name = name.replace("nitrilo", "cyano")
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


def _assemble_parent_name(
    mol: Molecule,
    parts: AssemblyParts,
    numbered_path: list[int],
    get_loc,
    *,
    finalize_subgraph: bool = False,
) -> str:
    """Assemble a parent name and apply shared post-assembly charge rules."""

    name = assemble_name_raw(parts)
    rewrites = []
    if finalize_subgraph:
        rewrites.append(
            (
                "simple_rooted_carbanion_substituent_name",
                lambda text: _simple_rooted_carbanion_substituent_name(mol, parts, numbered_path, get_loc) or text,
            )
        )
        rewrites.append(("finalize_subgraph_name", lambda text: _finalize_subgraph_name(text, parts)))
    rewrites.extend(
        (
            (
                "apply_anionic_parent_names",
                lambda text: apply_anionic_parent_names(text, mol, numbered_path, get_loc, parts.retained_name),
            ),
            (
                "apply_cationic_imino_parent_prefixes",
                lambda text: apply_cationic_imino_parent_prefixes(text, mol, numbered_path, get_loc),
            ),
            ("apply_cationic_imino_names", lambda text: apply_cationic_imino_names(text, mol)),
        )
    )
    rewrites.append(("post_process_name", post_process_name))
    result = NameAssemblyResult.from_rewrite_pipeline(name, parts.name_atom_bindings, rewrites=tuple(rewrites))
    parts.name_atom_bindings = list(result.bindings)
    parts.name_token_spans = token_span_trace_data(result)
    parts.name_rewrite_history = [
        {
            "name": operation.name,
            "before": operation.before,
            "after": operation.after,
            "ownership": operation.ownership,
            "source": operation.source,
            "binding_count": operation.binding_count,
            "changed_binding_count": operation.changed_binding_count,
            "token_count": operation.token_count,
            "changed_token_count": operation.changed_token_count,
            "edits": [
                {
                    "before_start": edit.before_start,
                    "before_end": edit.before_end,
                    "after_start": edit.after_start,
                    "after_end": edit.after_end,
                    "before_text": edit.before_text,
                    "after_text": edit.after_text,
                    "segments": [
                        {
                            "before_start": segment.before_start,
                            "before_end": segment.before_end,
                            "after_start": segment.after_start,
                            "after_end": segment.after_end,
                            "before_text": segment.before_text,
                            "after_text": segment.after_text,
                            "ownership": segment.ownership,
                            "group": segment.group,
                        }
                        for segment in edit.segments
                    ],
                }
                for edit in operation.edits
            ],
        }
        for operation in result.rewrite_history
    ]
    return result.text


def _simple_rooted_carbanion_substituent_name(
    mol: Molecule, parts: AssemblyParts, numbered_path: list[int], get_loc
) -> str:
    """Render simple C- substituent roots as methanidyl ligand names.

    This is deliberately narrow: only an acyclic all-carbon substituent whose
    charged carbon is locant 1 is converted. More complex carbanions need a
    charge-pair role template rather than a global ``-ide`` suffix.
    """

    if parts.substituents or parts.principal_group is not None or parts.unsaturations:
        return ""
    charged = [
        atom_idx for atom_idx in numbered_path if mol.atoms[atom_idx].is_carbon and mol.atoms[atom_idx].charge < 0
    ]
    if len(charged) != 1 or str(get_loc(charged[0])) != "1":
        return ""
    if any(not mol.atoms[atom_idx].is_carbon for atom_idx in numbered_path):
        return ""
    if any(
        mol.get_bond(a, b).order != 1
        for a, b in zip(numbered_path, numbered_path[1:])
        if mol.get_bond(a, b) is not None
    ):
        return ""
    side_len = len(numbered_path) - 1
    if side_len == 0:
        return "methanidyl"
    side_stem = stems.stem_for(side_len)
    if not side_stem:
        return ""
    return f"{side_stem}ylmethanidyl"


def name_subgraph(
    mol: Molecule,
    start_idx: int,
    exclude_atoms: set[int],
    upstream_atom: int = None,
    return_trace: bool = False,
    return_tree: bool = False,
    decision_trace: DecisionTrace | None = None,
):
    """Name a recursive substituent subgraph attached to the current parent.

    Blue Book references: P-13.6, P-14.2, P-16.5, P-61, P-62, P-63, P-65,
    P-66, and P-67.  Extendable prefix vocabularies are loaded from
    ``data/namer_rules.json``.
    """

    start_atom = mol.atoms[start_idx]
    cyclic_atoms_global = get_cyclic_atoms(mol, exclude_atoms)
    component = _subgraph_component(mol, start_idx, exclude_atoms)
    trace_decision(
        decision_trace,
        TracePhase.COMPONENT,
        "selected substituent subgraph",
        "Recursive branch naming starts from the atom attached to the parent and stops at the parent boundary.",
        atoms=component,
        bonds=_bond_ids_within(mol, component),
        data={
            "start_atom": start_idx,
            "upstream_atom": upstream_atom,
            "excluded_atom_count": len(exclude_atoms),
        },
    )

    if not start_atom.is_carbon and start_idx not in cyclic_atoms_global:
        name = name_heteroatom_subgraph(mol, start_idx, exclude_atoms, upstream_atom, name_subgraph)
        if name is not None:
            trace_decision(
                decision_trace,
                TracePhase.ASSEMBLY,
                "assembled heteroatom substituent shortcut",
                "A terminal non-carbon branch was rendered by the heteroatom-subgraph renderer.",
                atoms=component,
                bonds=_bond_ids_within(mol, component),
                data={"name": name},
            )
            tree = _shortcut_substituent_tree(name, component, mol, decision_trace)
            if return_trace and return_tree:
                return name, [], tree
            if return_trace:
                return name, []
            if return_tree:
                return name, tree
            return name

    direct_prefix = _direct_subgraph_prefix(mol, start_idx, component)
    if direct_prefix:
        direct_prefix_name = direct_prefix.name
        trace_decision(
            decision_trace,
            TracePhase.ASSEMBLY,
            "assembled direct substituent prefix",
            "The complete recursive component matches a direct functional-prefix role.",
            atoms=component,
            bonds=_bond_ids_within(mol, component),
            data=direct_prefix.trace_data(),
        )
        tree = _shortcut_substituent_tree(direct_prefix_name, component, mol, decision_trace, direct_prefix)
        if return_trace and return_tree:
            return direct_prefix_name, [], tree
        if return_trace:
            return direct_prefix_name, []
        if return_tree:
            return direct_prefix_name, tree
        return direct_prefix_name

    sub_exclude = set(mol.atoms.keys()) - component
    parent_selection = _select_subgraph_parent(mol, start_idx, component, sub_exclude)
    if parent_selection is None:
        trace_decision(
            decision_trace,
            TracePhase.PARENT_SELECTION,
            "failed substituent parent selection",
            "No chain or ring parent could be selected inside the recursive component.",
            atoms=component,
            bonds=_bond_ids_within(mol, component),
        )
        if return_trace and return_tree:
            return "", [], None
        if return_trace:
            return "", []
        if return_tree:
            return "", None
        return ""
    trace_decision(
        decision_trace,
        TracePhase.PARENT_SELECTION,
        "selected substituent parent skeleton",
        "The recursive branch uses the same parent-candidate ranking as ordinary components, scoped to the branch graph.",
        atoms=set(parent_selection.primary_path),
        bonds=_bond_ids_within(mol, set(parent_selection.primary_path)),
        data={
            "primary_path": list(parent_selection.primary_path),
            "is_ring": parent_selection.is_ring,
            "is_bicycle": parent_selection.is_bicycle,
            "is_spiro": parent_selection.is_spiro,
            "is_polycycle": parent_selection.is_polycycle,
            "fixed_start_required": parent_selection.fixed_start_required,
        },
    )

    retained_name_val, locant_maps = resolve_retained_parent(
        mol,
        parent_selection.primary_path,
        parent_selection.is_ring,
        parent_selection.is_bicycle,
        parent_selection.is_polycycle,
    )
    sub_perceived = perceive_groups(mol)
    subst_mapping = _collect_subgraph_substituents(mol, parent_selection.primary_path, sub_perceived, sub_exclude)
    parent_plan = build_parent_assembly_plan(
        mol,
        parent_selection,
        NamingIntent.subgraph(
            start_idx,
            upstream_atom,
            fixed_start=parent_selection.fixed_start_required,
        ),
        subst_mapping,
        locant_maps,
        retained_name_val,
    )
    numbered_path = parent_plan.numbered_path
    get_loc = parent_plan.get_loc
    parts = parent_plan.parts
    trace_decision(
        decision_trace,
        TracePhase.NUMBERING,
        "selected substituent numbering",
        "The branch parent numbering was selected before replacement, unsaturation, suffix, and substituent locants were emitted.",
        atoms=set(numbered_path),
        bonds=_bond_ids_within(mol, set(numbered_path)),
        data={
            "numbered_path": list(numbered_path),
            "atom_to_locant": {atom_idx: get_loc(atom_idx) for atom_idx in numbered_path},
            "retained_name": retained_name_val,
        },
    )
    _emit_bond_stereo(mol, parts, numbered_path, get_loc, sub_exclude, upstream_atom)
    _add_indicated_hydrogens(mol, parts, numbered_path, get_loc)
    _add_subgraph_substituents(parts, subst_mapping, get_loc)
    _add_subgraph_parent_features(
        mol,
        parts,
        numbered_path,
        get_loc,
        parent_selection.is_bicycle,
        parent_selection.is_spiro,
        parent_selection.is_polycycle,
    )

    name = _assemble_parent_name(mol, parts, numbered_path, get_loc, finalize_subgraph=True)
    trace_decision(
        decision_trace,
        TracePhase.ASSEMBLY,
        "assembled substituent name",
        "The scoped branch assembly was rendered as a substituent name.",
        atoms=component,
        bonds=_bond_ids_within(mol, component),
        data={
            "name": name,
            "trace_segment_count": len(_assembly_trace_segments(parts)) if return_trace else 0,
        },
    )
    if return_trace:
        trace_segments = _assembly_trace_segments(parts)
        if return_tree:
            return (
                name,
                trace_segments,
                _assembly_substituent_tree(
                    parts,
                    name=name,
                    atom_ids=component,
                    bond_ids=_bond_ids_within(mol, component),
                    decisions=decision_trace_data(decision_trace),
                ),
            )
        return name, trace_segments
    if return_tree:
        return (
            name,
            _assembly_substituent_tree(
                parts,
                name=name,
                atom_ids=component,
                bond_ids=_bond_ids_within(mol, component),
                decisions=decision_trace_data(decision_trace),
            ),
        )
    return name


def _shortcut_substituent_tree(
    name: str,
    component: set[int],
    mol: Molecule,
    decision_trace: DecisionTrace | None,
    direct_prefix: DirectSubgraphPrefix | None = None,
) -> dict:
    """Return a minimal recursive tree node for shortcut substituent names."""

    node = {
        "kind": "substituent",
        "name": name,
        "atoms": sorted(component),
        "bonds": sorted(_bond_ids_within(mol, component)),
        "parent": None,
        "principal_group": None,
        "substituents": [],
        "replacement_prefixes": [],
        "unsaturations": [],
        "trace_segments": [],
        "nested_decisions": decision_trace_data(decision_trace),
    }
    if direct_prefix is not None:
        node["functional_prefix"] = {
            "kind": "functional_prefix",
            "name": direct_prefix.name,
            "group_key": direct_prefix.group_key,
            "attachment_atom": direct_prefix.attachment_atom,
            "group_atoms": sorted(direct_prefix.group_atoms),
            "core_atoms": sorted(direct_prefix.core_atoms),
            "group_bonds": sorted(direct_prefix.group_bonds),
            "source": direct_prefix.source,
            "ligands": list(direct_prefix.ligand_trees),
            "ligand_trace_segments": list(direct_prefix.ligand_trace_segments),
            "ligand_decisions": list(direct_prefix.ligand_decisions),
        }
        if direct_prefix.ligand_trees:
            node["substituents"] = list(direct_prefix.ligand_trees)
        if direct_prefix.ligand_trace_segments:
            node["trace_segments"] = list(direct_prefix.ligand_trace_segments)
    return node


def name_component(
    mol: Molecule,
    component_atoms: set[int],
    is_substituent: bool = False,
    return_trace: bool = False,
    return_tree: bool = False,
    decision_trace: DecisionTrace | None = None,
):
    """Name one connected component or recursive component of a molecule."""

    return _name_component_impl(
        mol,
        component_atoms,
        is_substituent=is_substituent,
        return_trace=return_trace,
        return_tree=return_tree,
        decision_trace=decision_trace,
        name_subgraph=name_subgraph,
        name_spiro_subgraph=_spiro_subgraph_assembly,
        assemble_parent_name=_assemble_parent_name,
    )


def name_smiles_with_trace(smiles: str) -> tuple[str, list[dict]]:
    """Return a generated name and AssemblyParts-derived trace annotations.

    Blue Book references are carried by each trace segment. This API keeps the
    original ``name_smiles`` function unchanged while exposing atom and bond IDs
    selected during parent, prefix, unsaturation, and suffix assembly.
    """

    return DEFAULT_NAMING_ENGINE.name_smiles_with_trace(smiles)


def analyze_smiles(smiles: str) -> NameAnalysis:
    """Return a generated name with structure annotations and decision traces.

    The default ``name_smiles`` path stays minimal. This analysis API turns on
    explainability and records the major choices in the pipeline: parsing,
    component splitting, functional-group perception, priority, parent
    selection, numbering, and final assembly.
    """

    return DEFAULT_NAMING_ENGINE.analyze_smiles(smiles)


def name_smiles(smiles: str) -> str:
    """Return an IUPAC-style name for a SMILES string.

    Blue Book references: P-13 for name construction, P-44/P-45 for parent
    selection and numbering, and P-72 for ordering disconnected ionic
    components.  Component-order metal names are data-backed in
    ``data/namer_rules.json``.
    """

    return DEFAULT_NAMING_ENGINE.name_smiles(smiles)
