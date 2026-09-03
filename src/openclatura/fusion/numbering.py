"""Completed-system numbering and mancude bond models for fused parents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key

from ..molecule import Molecule
from .faces import BoundedFaceModel, normalize_edge
from .model import BondAssignment, FaceModel, FusedLayout, ParentBondModel, RejectedNumbering, SystemLocant
from .rules import GENERAL_HETEROATOM_COUNT_PRECEDENCE

_FIXED_SINGLE_ELEMENTS = frozenset({"O", "S", "Se", "Te"})


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
    if set(faces.outer_boundary.atoms) != set(faces.atom_ids):
        return CompletedNumberingSelection(())
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
        locant_map = _number_perimeter(mol, oriented, fusion_atoms)
        if locant_map is None:
            continue
        candidates.append(
            CompletedNumbering(
                perimeter=oriented,
                atom_to_locant=tuple((atom, locant_map[atom]) for atom in oriented),
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
    start_face = face_by_id[start_face_id]
    start_candidates = [
        atom for atom in start_face.atom_cycle if atom not in fusion_atoms and atom in faces.outer_boundary.atoms
    ]
    # For a face in its preferred orientation, the uppermost then leftmost
    # peripheral vertex is its most counterclockwise nonfusion position.
    start_atom = max(start_candidates, key=lambda atom: (positions[atom][1], -positions[atom][0]))
    perimeter = _clockwise_perimeter(faces.outer_boundary.atoms, positions, start_atom)
    if perimeter is None:
        return None
    locant_map = _number_perimeter(mol, perimeter, fusion_atoms)
    if locant_map is None:
        return None
    return CompletedNumbering(
        perimeter=perimeter,
        atom_to_locant=tuple((atom, locant_map[atom]) for atom in perimeter),
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


def _clockwise_perimeter(
    boundary: tuple[int, ...],
    positions: dict[int, tuple[int, int]],
    start_atom: int,
) -> tuple[int, ...] | None:
    if start_atom not in boundary or any(atom not in positions for atom in boundary):
        return None
    signed_area = sum(
        positions[left][0] * positions[right][1] - positions[right][0] * positions[left][1]
        for left, right in zip(boundary, boundary[1:] + boundary[:1])
    )
    if signed_area == 0:
        return None
    clockwise = boundary if signed_area < 0 else tuple(reversed(boundary))
    offset = clockwise.index(start_atom)
    return clockwise[offset:] + clockwise[:offset]


def _fusion_atoms(faces: BoundedFaceModel) -> set[int]:
    face_membership = Counter(atom for face in faces.faces for atom in face.atoms)
    return {atom for atom, count in face_membership.items() if count > 1}


def _numbering_key(numbering: CompletedNumbering) -> tuple[tuple[int, str], ...]:
    return tuple(sorted((atom, str(locant)) for atom, locant in numbering.atom_to_locant))


def parent_bond_model(mol: Molecule, atom_ids: Iterable[int], *, search_budget: int = 100_000) -> ParentBondModel:
    """Build all maximum non-cumulative Kekule assignments for a parent graph."""

    atoms = frozenset(atom_ids)
    edges = tuple(
        sorted(normalize_edge(bond.u, bond.v) for bond in mol.bonds.values() if bond.u in atoms and bond.v in atoms)
    )
    required = frozenset(
        edge
        for edge in edges
        if mol.atoms[edge[0]].symbol in _FIXED_SINGLE_ELEMENTS or mol.atoms[edge[1]].symbol in _FIXED_SINGLE_ELEMENTS
    )
    eligible = frozenset(edges) - required
    matchings = _maximum_matchings(eligible, search_budget=search_budget)
    assignments = tuple(
        BondAssignment(tuple((edge, 2 if edge in matching else 1) for edge in edges)) for matching in matchings
    )
    return ParentBondModel(
        allowed_kekule_assignments=assignments,
        required_single_bonds=required,
        pi_eligible_edges=eligible,
        maximum_non_cumulative_double_bonds=max((len(matching) for matching in matchings), default=0),
    )


def observed_parent_matches_bond_model(mol: Molecule, model: ParentBondModel) -> bool:
    """Return whether actual parent bond orders equal an allowed assignment."""

    observed = {
        normalize_edge(bond.u, bond.v): bond.order
        for bond in mol.bonds.values()
        if normalize_edge(bond.u, bond.v) in model.required_single_bonds | model.pi_eligible_edges
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


def _numbering_score(mol: Molecule, locants: dict[int, SystemLocant], fusion_atoms: set[int]) -> tuple:
    hetero = [(atom, locant) for atom, locant in locants.items() if mol.atoms[atom].symbol != "C"]
    all_hetero = tuple(sorted(_locant_key(locant) for _, locant in hetero))
    by_element = tuple(
        tuple(sorted(_locant_key(locant) for atom, locant in hetero if mol.atoms[atom].symbol == symbol))
        for symbol in GENERAL_HETEROATOM_COUNT_PRECEDENCE
    )
    fusion_carbons = tuple(sorted(_locant_key(locants[atom]) for atom in fusion_atoms if mol.atoms[atom].symbol == "C"))
    fusion_hetero = tuple(sorted(_locant_key(locants[atom]) for atom in fusion_atoms if mol.atoms[atom].symbol != "C"))
    indicated_h = tuple(
        sorted(_locant_key(locant) for atom, locant in locants.items() if mol.atoms[atom].total_h_count > 0)
    )
    return all_hetero, by_element, fusion_carbons, fusion_hetero, indicated_h


def _locant_key(locant: SystemLocant) -> tuple[int, str, int]:
    return locant.base, locant.fusion_suffix, locant.interior_distance or 0


def _first_fusion_base(numbering: CompletedNumbering) -> int:
    return min(locant.base for _, locant in numbering.atom_to_locant if locant.fusion_suffix)


def _maximum_matchings(
    edges: frozenset[tuple[int, int]], *, search_budget: int
) -> tuple[frozenset[tuple[int, int]], ...]:
    ordered = tuple(sorted(edges))
    best_size = -1
    best: set[frozenset[tuple[int, int]]] = set()
    states = 0

    def search(position: int, used: frozenset[int], selected: tuple[tuple[int, int], ...]) -> None:
        nonlocal best_size, states
        states += 1
        if states > search_budget:
            raise RuntimeError(f"mancude assignment search exceeded its budget of {search_budget} states")
        if position == len(ordered):
            size = len(selected)
            value = frozenset(selected)
            if size > best_size:
                best_size = size
                best.clear()
            if size == best_size:
                best.add(value)
            return
        if len(selected) + (len(ordered) - position) < best_size:
            return
        edge = ordered[position]
        search(position + 1, used, selected)
        if edge[0] not in used and edge[1] not in used:
            search(position + 1, used | frozenset(edge), selected + (edge,))

    search(0, frozenset(), ())
    return tuple(sorted(best, key=lambda matching: tuple(sorted(matching))))
