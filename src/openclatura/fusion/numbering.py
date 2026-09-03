"""Completed-system numbering and mancude bond models for fused parents."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ..molecule import Molecule
from .faces import BoundedFaceModel, normalize_edge
from .model import BondAssignment, ParentBondModel, SystemLocant
from .rules import GENERAL_HETEROATOM_COUNT_PRECEDENCE

_FIXED_SINGLE_ELEMENTS = frozenset({"O", "S", "Se", "Te"})


@dataclass(frozen=True, slots=True)
class CompletedNumbering:
    """One preferred completed-system numbering candidate."""

    perimeter: tuple[int, ...]
    atom_to_locant: tuple[tuple[int, SystemLocant], ...]
    score: tuple

    @property
    def string_map(self) -> dict[int, str]:
        return {atom: str(locant) for atom, locant in self.atom_to_locant}


def completed_system_numberings(mol: Molecule, faces: BoundedFaceModel) -> tuple[CompletedNumbering, ...]:
    """Return every tied preferred numbering for an all-peripheral fused system.

    Carbon atoms shared by bounded faces receive letter-suffixed locants;
    fusion heteroatoms remain in the integer sequence. Interior atoms are a
    separate nomenclature tier and are deliberately rejected here.
    """

    boundary = faces.outer_boundary.atoms
    if set(boundary) != set(faces.atom_ids):
        return ()
    face_membership = Counter(atom for face in faces.faces for atom in face.atoms)
    fusion_atoms = {atom for atom, count in face_membership.items() if count > 1}
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


def parent_bond_model(mol: Molecule, atom_ids: Iterable[int], *, search_budget: int = 100_000) -> ParentBondModel:
    """Build all maximum non-cumulative Kekule assignments for a parent graph."""

    atoms = frozenset(atom_ids)
    edges = tuple(
        sorted(
            normalize_edge(bond.u, bond.v)
            for bond in mol.bonds.values()
            if bond.u in atoms and bond.v in atoms
        )
    )
    required = frozenset(
        edge
        for edge in edges
        if mol.atoms[edge[0]].symbol in _FIXED_SINGLE_ELEMENTS
        or mol.atoms[edge[1]].symbol in _FIXED_SINGLE_ELEMENTS
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
    fusion_carbons = tuple(
        sorted(_locant_key(locants[atom]) for atom in fusion_atoms if mol.atoms[atom].symbol == "C")
    )
    fusion_hetero = tuple(
        sorted(_locant_key(locants[atom]) for atom in fusion_atoms if mol.atoms[atom].symbol != "C")
    )
    indicated_h = tuple(
        sorted(_locant_key(locant) for atom, locant in locants.items() if mol.atoms[atom].total_h_count > 0)
    )
    return all_hetero, by_element, fusion_carbons, fusion_hetero, indicated_h


def _locant_key(locant: SystemLocant) -> tuple[int, str, int]:
    return locant.base, locant.fusion_suffix, locant.interior_distance or 0


def _first_fusion_base(numbering: CompletedNumbering) -> int:
    return min(
        locant.base
        for _, locant in numbering.atom_to_locant
        if locant.fusion_suffix
    )


def _maximum_matchings(edges: frozenset[tuple[int, int]], *, search_budget: int) -> tuple[frozenset[tuple[int, int]], ...]:
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
