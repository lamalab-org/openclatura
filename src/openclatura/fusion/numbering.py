"""Completed-system numbering and mancude bond models for fused parents."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key

from ..canonical_ranks import canonical_ranks
from ..locants import system_locant_sort_key
from ..molecule import Molecule
from ..polycycle_topology import normalize_edge
from .config import fusion_nomenclature_config
from .faces import BoundedFaceModel
from .model import (
    BondAssignment,
    Face,
    FaceModel,
    FusedLayout,
    FusionGraph,
    ParentBondModel,
    RejectedNumbering,
    SystemLocant,
)
from .rules import GENERAL_HETEROATOM_COUNT_PRECEDENCE


class MancudeSearchBudgetExceeded(RuntimeError):
    """Raised when exhaustive parent bond assignment exceeds its bound."""

    def __init__(self, budget: int) -> None:
        super().__init__(f"mancude assignment search exceeded its budget of {budget} states")
        self.budget = budget


_CONFIG = fusion_nomenclature_config()


@dataclass(frozen=True, slots=True)
class CompletedNumbering:
    """One preferred completed-system numbering candidate."""

    perimeter: tuple[int, ...]
    atom_to_locant: tuple[tuple[int, SystemLocant], ...]
    score: tuple
    layout_index: int | None = None
    start_face_id: int | None = None
    start_atom: int | None = None

    @property
    def string_map(self) -> dict[int, str]:
        return {atom: str(locant) for atom, locant in self.atom_to_locant}


@dataclass(frozen=True, slots=True)
class CompletedNumberingSelection:
    """Accepted layout-derived numberings and rejected orientation evidence."""

    accepted: tuple[CompletedNumbering, ...]
    rejected: tuple[RejectedNumbering, ...] = ()


def completed_system_numberings(
    mol: Molecule,
    faces: BoundedFaceModel,
    *,
    face_model: FaceModel | None = None,
    layouts: tuple[FusedLayout, ...] = (),
) -> tuple[CompletedNumbering, ...]:
    """Return every tied preferred numbering for an all-peripheral fused system.

    Carbon atoms shared by bounded faces receive letter-suffixed locants;
    fusion heteroatoms remain in the integer sequence. Interior atoms are a
    separate nomenclature tier and are deliberately rejected here.
    """

    return completed_system_numbering_selection(
        mol,
        faces,
        face_model=face_model,
        layouts=layouts,
    ).accepted


def completed_system_numbering_selection(
    mol: Molecule,
    faces: BoundedFaceModel,
    *,
    face_model: FaceModel | None = None,
    layouts: tuple[FusedLayout, ...] = (),
) -> CompletedNumberingSelection:
    """Select completed-system maps, optionally proving starts from layouts.

    With layouts supplied, every accepted perimeter starts at the most
    counterclockwise nonfusion atom of the uppermost/rightmost face and then
    follows the geometric perimeter clockwise. The established ordered locant
    criteria remain a second-stage filter, preserving nomenclatural choices
    among reflected layouts.
    """

    if not layouts:
        return CompletedNumberingSelection(_graph_numbering_candidates(mol, faces))
    if face_model is None:
        raise ValueError("layout-derived numbering requires the corresponding typed face model")
    candidates: list[CompletedNumbering] = []
    rejected: list[RejectedNumbering] = []
    fusion_atoms = _fusion_atoms(faces)
    for layout_index, layout in enumerate(layouts):
        derived = _numbering_from_layout(
            mol,
            faces,
            face_model,
            layout,
            layout_index,
            fusion_atoms,
        )
        if derived is None:
            rejected.append(
                RejectedNumbering(
                    orientation_score=layout.orientation_score,
                    reason=f"layout {layout_index} has no valid uppermost/rightmost clockwise perimeter",
                )
            )
            continue
        candidates.append(derived)
    if not candidates:
        return CompletedNumberingSelection((), tuple(rejected))
    best_score = min(candidate.score for candidate in candidates)
    accepted: dict[tuple[tuple[int, str], ...], CompletedNumbering] = {}
    for candidate in candidates:
        if candidate.score != best_score:
            rejected.append(
                RejectedNumbering(
                    orientation_score=(
                        layouts[candidate.layout_index if candidate.layout_index is not None else 0].orientation_score,
                        candidate.score,
                    ),
                    reason=(
                        f"layout {candidate.layout_index} start at face {candidate.start_face_id}, atom "
                        f"{candidate.start_atom} loses the ordered completed-system locant criteria"
                    ),
                )
            )
            continue
        accepted.setdefault(_numbering_key(candidate), candidate)
    return CompletedNumberingSelection(tuple(accepted.values()), tuple(rejected))


def _graph_numbering_candidates(mol: Molecule, faces: BoundedFaceModel) -> tuple[CompletedNumbering, ...]:
    boundary = faces.outer_boundary.atoms
    if set(boundary) != set(faces.atom_ids):
        return ()
    fusion_atoms = _fusion_atoms(faces)
    candidates: list[CompletedNumbering] = []
    for oriented in _cycle_orientations(boundary):
        locant_map = _number_completed_system(mol, faces, oriented, fusion_atoms)
        if locant_map is None:
            continue
        candidates.append(
            CompletedNumbering(
                perimeter=oriented,
                atom_to_locant=_ordered_locant_items(locant_map),
                score=_numbering_score(mol, locant_map, fusion_atoms),
            )
        )
    if not candidates:
        return ()
    # A fused perimeter starts at the first nonfusion atom after a fusion
    # junction and traverses the uninterrupted nonfusion run before assigning
    # the first lettered junction locant. Without a drawing-derived seed this
    # intrinsic run length is the graph-invariant orientation criterion.
    longest_initial_run = max(_first_fusion_base(candidate) for candidate in candidates)
    candidates = [candidate for candidate in candidates if _first_fusion_base(candidate) == longest_initial_run]
    best = min(candidate.score for candidate in candidates)
    unique: dict[tuple[tuple[int, str], ...], CompletedNumbering] = {}
    for candidate in candidates:
        if candidate.score != best:
            continue
        key = tuple(sorted((atom, str(locant)) for atom, locant in candidate.atom_to_locant))
        unique[key] = candidate
    return tuple(unique.values())


def _numbering_from_layout(
    mol: Molecule,
    faces: BoundedFaceModel,
    face_model: FaceModel,
    layout: FusedLayout,
    layout_index: int,
    fusion_atoms: set[int],
) -> CompletedNumbering | None:
    positions = {atom: (x, y) for atom, x, y in layout.atom_positions}
    centers = {face: (x, y) for face, x, y in layout.face_positions}
    face_by_id = {face.id: face for face in face_model.faces}
    if set(positions) != set(faces.atom_ids) or set(centers) != set(face_by_id):
        return None
    face_order = _clockwise_face_order(centers)
    if not face_order:
        return None
    start_face_id = next(
        (
            face_id
            for face_id in face_order
            if any(
                atom not in fusion_atoms and atom in faces.outer_boundary.atoms
                for atom in face_by_id[face_id].atom_cycle
            )
        ),
        None,
    )
    if start_face_id is None:
        return None
    clockwise = _clockwise_boundary(faces.outer_boundary.atoms, positions)
    if clockwise is None:
        return None
    start_atom = _counterclockwise_nonfusion_start(
        face_by_id[start_face_id],
        clockwise,
        fusion_atoms,
    )
    if start_atom is None:
        return None
    offset = clockwise.index(start_atom)
    perimeter = clockwise[offset:] + clockwise[:offset]
    locant_map = _number_completed_system(mol, faces, perimeter, fusion_atoms)
    if locant_map is None:
        return None
    return CompletedNumbering(
        perimeter=perimeter,
        atom_to_locant=_ordered_locant_items(locant_map),
        score=_numbering_score(mol, locant_map, fusion_atoms),
        layout_index=layout_index,
        start_face_id=start_face_id,
        start_atom=start_atom,
    )


def _clockwise_face_order(centers: dict[int, tuple[int, int]]) -> tuple[int, ...]:
    """Order faces clockwise from the uppermost, then rightmost face."""

    if not centers or len(set(centers.values())) != len(centers):
        return ()
    first = max(centers, key=lambda face: (centers[face][1], centers[face][0]))
    center_x = Fraction(sum(x for x, _ in centers.values()), len(centers))
    center_y = Fraction(sum(y for _, y in centers.values()), len(centers))

    def vector(face: int) -> tuple[Fraction, Fraction]:
        x, y = centers[face]
        return Fraction(x) - center_x, Fraction(y) - center_y

    def compare(left: int, right: int) -> int:
        left_vector = vector(left)
        right_vector = vector(right)
        left_half = _clockwise_half(*left_vector)
        right_half = _clockwise_half(*right_vector)
        if left_half != right_half:
            return -1 if left_half < right_half else 1
        cross = left_vector[0] * right_vector[1] - left_vector[1] * right_vector[0]
        if cross:
            return -1 if cross < 0 else 1
        return 0

    ordered = sorted(centers, key=cmp_to_key(compare))
    offset = ordered.index(first)
    return tuple(ordered[offset:] + ordered[:offset])


def _clockwise_half(dx: Fraction, dy: Fraction) -> int:
    return 0 if dx > 0 or (dx == 0 and dy >= 0) else 1


def _clockwise_boundary(
    boundary: tuple[int, ...],
    positions: dict[int, tuple[int, int]],
) -> tuple[int, ...] | None:
    if any(atom not in positions for atom in boundary):
        return None
    signed_area = sum(
        positions[left][0] * positions[right][1] - positions[right][0] * positions[left][1]
        for left, right in zip(boundary, boundary[1:] + boundary[:1])
    )
    if signed_area == 0:
        return None
    return boundary if signed_area < 0 else tuple(reversed(boundary))


def _counterclockwise_nonfusion_start(
    face: Face,
    clockwise_boundary: tuple[int, ...],
    fusion_atoms: set[int],
) -> int | None:
    """Return the start of the selected face's clockwise nonfusion arc.

    P-25.3.3.1.1 starts at the most counterclockwise nonfusion atom and then
    proceeds clockwise around the completed system.  On the outer perimeter,
    that atom is the first nonfusion atom after a fusion junction.  This
    topological definition also handles a three-or-more-atom arc, where a
    coordinate extremum can incorrectly select an atom in the middle.
    """

    face_atoms = set(face.atom_cycle)
    candidates = []
    for index, atom in enumerate(clockwise_boundary):
        if atom not in face_atoms or atom in fusion_atoms:
            continue
        previous = clockwise_boundary[index - 1]
        if previous in face_atoms and previous in fusion_atoms:
            candidates.append(atom)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    # More than one nonfusion arc requires a later complex-numbering tier;
    # selecting one by atom id or drawing coordinates would not be invariant.
    return None


def _fusion_atoms(faces: BoundedFaceModel) -> set[int]:
    return set(faces.fusion_atoms)


def _numbering_key(numbering: CompletedNumbering) -> tuple[tuple[int, str], ...]:
    return tuple(sorted((atom, str(locant)) for atom, locant in numbering.atom_to_locant))


def parent_bond_model(
    parent: FusionGraph | Molecule,
    atom_ids: Iterable[int] | None = None,
    *,
    search_budget: int = _CONFIG.search.mancude_states,
) -> ParentBondModel:
    """Build all maximum non-cumulative Kekule assignments for a parent graph."""

    if isinstance(parent, FusionGraph):
        sites = {atom.id: atom for atom in parent.atoms}
        edges = tuple(sorted(normalize_edge(*bond.atoms) for bond in parent.bonds))
        bond_classes = {normalize_edge(*bond.atoms): bond.bond_class for bond in parent.bonds}
        required_double = frozenset(edge for edge in edges if bond_classes[edge] == "double")
        occupied = frozenset(atom for edge in required_double for atom in edge)
        eligible = frozenset(
            edge
            for edge in edges
            if bond_classes[edge] in {"aromatic", "mancude", "fusion"}
            and not occupied.intersection(edge)
            and all(sites[atom].pi_capacity and not sites[atom].forced_single for atom in edge)
        )
    else:
        if atom_ids is None:
            raise TypeError("atom_ids are required for the Molecule compatibility path")
        atoms = frozenset(atom_ids)
        edges = tuple(
            sorted(
                normalize_edge(bond.u, bond.v) for bond in parent.bonds.values() if bond.u in atoms and bond.v in atoms
            )
        )
        eligible = frozenset(
            edge
            for edge in edges
            if not parent.atoms[edge[0]].element.mancude_forced_single
            and not parent.atoms[edge[1]].element.mancude_forced_single
        )
        required_double = frozenset()
    required = frozenset(edges) - eligible - required_double
    matchings = _maximum_matchings(eligible, search_budget=search_budget)
    assignments = tuple(
        BondAssignment(tuple((edge, 2 if edge in matching or edge in required_double else 1) for edge in edges))
        for matching in matchings
    )
    return ParentBondModel(
        allowed_kekule_assignments=assignments,
        required_single_bonds=required,
        pi_eligible_edges=eligible,
        maximum_non_cumulative_double_bonds=len(required_double)
        + max((len(matching) for matching in matchings), default=0),
        required_double_bonds=required_double,
    )


def observed_parent_matches_bond_model(mol: Molecule, model: ParentBondModel) -> bool:
    """Return whether actual parent bond orders equal an allowed assignment."""

    observed = {
        normalize_edge(bond.u, bond.v): bond.order
        for bond in mol.bonds.values()
        if normalize_edge(bond.u, bond.v)
        in model.required_single_bonds | model.required_double_bonds | model.pi_eligible_edges
    }
    return any(dict(assignment.orders) == observed for assignment in model.allowed_kekule_assignments)


def _cycle_orientations(cycle: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    variants = []
    for direction in (cycle, tuple(reversed(cycle))):
        for offset in range(len(direction)):
            variants.append(direction[offset:] + direction[:offset])
    return tuple(dict.fromkeys(variants))


def _number_perimeter(
    mol: Molecule,
    perimeter: tuple[int, ...],
    fusion_atoms: set[int],
) -> dict[int, SystemLocant] | None:
    if perimeter[0] in fusion_atoms and mol.atoms[perimeter[0]].symbol == "C":
        return None
    result: dict[int, SystemLocant] = {}
    integer = 0
    suffix_counts: dict[int, int] = {}
    for atom in perimeter:
        is_fusion_carbon = atom in fusion_atoms and mol.atoms[atom].symbol == "C"
        if is_fusion_carbon:
            if integer == 0:
                return None
            suffix_index = suffix_counts.get(integer, 0)
            if suffix_index >= 26:
                return None
            suffix_counts[integer] = suffix_index + 1
            result[atom] = SystemLocant(integer, chr(ord("a") + suffix_index))
        else:
            integer += 1
            result[atom] = SystemLocant(integer)
    return result


def _number_completed_system(
    mol: Molecule,
    faces: BoundedFaceModel,
    perimeter: tuple[int, ...],
    fusion_atoms: set[int],
) -> dict[int, SystemLocant] | None:
    """Number the perimeter, then extend the map to interior atoms."""

    result = _number_perimeter(mol, perimeter, fusion_atoms)
    if result is None:
        return None
    interior = set(faces.atom_ids) - set(perimeter)
    if not interior:
        return result

    distance, anchors = _interior_distances_and_anchors(mol, faces.atom_ids, perimeter)
    if set(distance) != set(faces.atom_ids):
        return None
    ranks = canonical_ranks(mol, faces.atom_ids)
    interior_hetero = sorted(
        (atom for atom in interior if mol.atoms[atom].symbol != "C"),
        key=lambda atom: (
            distance[atom],
            min(system_locant_sort_key(result[anchor]) for anchor in anchors[atom]),
            _heteroatom_rank(mol.atoms[atom].symbol),
            ranks[atom],
        ),
    )
    next_integer = max(locant.base for locant in result.values())
    for atom in interior_hetero:
        next_integer += 1
        result[atom] = SystemLocant(next_integer)

    for atom in sorted(interior - set(interior_hetero), key=lambda value: (distance[value], ranks[value])):
        anchor = min(anchors[atom], key=lambda value: system_locant_sort_key(result[value]))
        anchor_locant = result[anchor]
        result[atom] = SystemLocant(
            anchor_locant.base,
            fusion_suffix=anchor_locant.fusion_suffix,
            interior_distance=distance[atom],
        )
    if len(set(result.values())) != len(result):
        return None
    return result


def _ordered_locant_items(locants: Mapping[int, SystemLocant]) -> tuple[tuple[int, SystemLocant], ...]:
    return tuple(sorted(locants.items(), key=lambda item: system_locant_sort_key(item[1])))


def _interior_distances_and_anchors(
    mol: Molecule,
    atom_ids: frozenset[int],
    perimeter: tuple[int, ...],
) -> tuple[dict[int, int], dict[int, frozenset[int]]]:
    """Return shortest perimeter distances and every equally short anchor."""

    distance = {atom: 0 for atom in perimeter}
    anchors: dict[int, frozenset[int]] = {atom: frozenset((atom,)) for atom in perimeter}
    pending = deque(perimeter)
    while pending:
        atom = pending.popleft()
        next_distance = distance[atom] + 1
        for neighbor in mol.get_neighbors(atom):
            if neighbor not in atom_ids:
                continue
            if neighbor not in distance:
                distance[neighbor] = next_distance
                anchors[neighbor] = anchors[atom]
                pending.append(neighbor)
            elif distance[neighbor] == next_distance:
                merged = anchors[neighbor] | anchors[atom]
                if merged != anchors[neighbor]:
                    anchors[neighbor] = merged
                    pending.append(neighbor)
    return distance, anchors


def _heteroatom_rank(symbol: str) -> int:
    try:
        return GENERAL_HETEROATOM_COUNT_PRECEDENCE.index(symbol)
    except ValueError:
        return len(GENERAL_HETEROATOM_COUNT_PRECEDENCE)


def _numbering_score(mol: Molecule, locants: dict[int, SystemLocant], fusion_atoms: set[int]) -> tuple:
    hetero = [(atom, locant) for atom, locant in locants.items() if mol.atoms[atom].symbol != "C"]
    all_hetero = tuple(sorted(_locant_key(locant) for _, locant in hetero))
    by_element = tuple(
        tuple(sorted(_locant_key(locant) for atom, locant in hetero if mol.atoms[atom].symbol == symbol))
        for symbol in GENERAL_HETEROATOM_COUNT_PRECEDENCE
    )
    fusion_carbons = tuple(sorted(_locant_key(locants[atom]) for atom in fusion_atoms if mol.atoms[atom].symbol == "C"))
    fusion_hetero = tuple(sorted(_locant_key(locants[atom]) for atom in fusion_atoms if mol.atoms[atom].symbol != "C"))
    indicated_h = tuple(sorted(_locant_key(locants[atom]) for atom in indicated_hydrogen_candidate_atoms(mol, locants)))
    return all_hetero, by_element, fusion_carbons, fusion_hetero, indicated_h


def indicated_hydrogen_candidate_atoms(
    mol: Molecule,
    parent_locants: Mapping[int, SystemLocant],
) -> tuple[int, ...]:
    """Return parent atoms considered by the indicated-H numbering tie-break."""

    return tuple(atom for atom in parent_locants if _is_indicated_hydrogen_candidate(mol, atom, parent_locants))


def bond_model_indicated_hydrogen_atoms(
    mol: Molecule,
    bond_model: ParentBondModel,
    candidates: set[int],
) -> frozenset[int]:
    """Return saturated sites required by the closest mancude assignment.

    The symmetric bond-order delta is interpreted without reading a rendered
    parent name. A missing double bond with hydrogen-bearing atoms at both ends
    is ordinary additive hydrogenation. If only one endpoint is an eligible
    saturated site, that endpoint requires an indicated-hydrogen citation.
    """

    model_edges = bond_model.required_single_bonds | bond_model.required_double_bonds | bond_model.pi_eligible_edges
    observed = {
        tuple(sorted((bond.u, bond.v))): (
            None
            if mol.atoms[bond.u].is_aromatic
            and mol.atoms[bond.v].is_aromatic
            and tuple(sorted((bond.u, bond.v))) in bond_model.pi_eligible_edges
            else bond.order
        )
        for bond in mol.bonds.values()
        if tuple(sorted((bond.u, bond.v))) in model_edges
    }
    viable: list[frozenset[int]] = []
    for assignment in bond_model.allowed_kekule_assignments:
        allowed = {tuple(sorted(edge)): order for edge, order in assignment.orders}
        if set(observed) != set(allowed):
            continue
        indicated: set[int] = set()
        compatible = True
        for edge, order in observed.items():
            if order is None or order == allowed[edge]:
                continue
            if order != 1 or allowed[edge] != 2:
                compatible = False
                break
            endpoints = set(edge) & candidates
            if len(endpoints) == 1:
                indicated.update(endpoints)
            elif len(endpoints) != 2:
                compatible = False
                break
        if compatible:
            viable.append(frozenset(indicated))
    return min(viable, key=lambda atoms: (len(atoms), tuple(sorted(atoms)))) if viable else frozenset()


def _is_indicated_hydrogen_candidate(
    mol: Molecule,
    atom: int,
    parent_locants: Mapping[int, SystemLocant],
) -> bool:
    value = mol.atoms[atom]
    if value.total_h_count <= 0:
        return False
    if value.symbol != "C":
        return True
    parent_neighbors = [neighbor for neighbor in mol.get_neighbors(atom) if neighbor in parent_locants]
    return value.total_h_count > 1 and all(mol.get_bond(atom, neighbor).order == 1 for neighbor in parent_neighbors)


def _locant_key(locant: SystemLocant) -> tuple[int, str, int]:
    return locant.base, locant.fusion_suffix, locant.interior_distance or 0


def _first_fusion_base(numbering: CompletedNumbering) -> int:
    return min(locant.base for _, locant in numbering.atom_to_locant if locant.fusion_suffix)


def _maximum_matchings(
    edges: frozenset[tuple[int, int]], *, search_budget: int
) -> tuple[frozenset[tuple[int, int]], ...]:
    """Return every maximum matching using memoized residual vertex states.

    Branching on edges revisits the same residual graph many times, which is
    exponential even for an unbranched acene-like parent.  A matching state is
    completely determined by the vertices still available, so solving each
    such state once preserves exhaustive output while making long fused chains
    practical.
    """

    vertices = tuple(sorted({atom for edge in edges for atom in edge}))
    neighbors = {
        atom: tuple(sorted(edge[1] if edge[0] == atom else edge[0] for edge in edges if atom in edge))
        for atom in vertices
    }
    memo: dict[frozenset[int], tuple[frozenset[tuple[int, int]], ...]] = {}
    states = 0

    def search(available: frozenset[int]) -> tuple[frozenset[tuple[int, int]], ...]:
        nonlocal states
        cached = memo.get(available)
        if cached is not None:
            return cached
        states += 1
        if states > search_budget:
            raise MancudeSearchBudgetExceeded(search_budget)
        if not available:
            return (frozenset(),)

        atom = min(available)
        without_atom = available - {atom}
        candidates = list(search(without_atom))
        for neighbor in neighbors[atom]:
            if neighbor not in available:
                continue
            edge = normalize_edge(atom, neighbor)
            candidates.extend(matching | {edge} for matching in search(without_atom - {neighbor}))
        maximum = max(map(len, candidates), default=0)
        result = tuple(
            sorted(
                {matching for matching in candidates if len(matching) == maximum},
                key=lambda matching: tuple(sorted(matching)),
            )
        )
        memo[available] = result
        return result

    return search(frozenset(vertices))
