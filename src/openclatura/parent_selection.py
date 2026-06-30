# bluenamer/parent_selection.py
from dataclasses import dataclass

from .chains import RingSystem
from .molecule import Molecule
from .naming_data import values
from .ring_parent import RingParent

PARENT_SELECTION_CRITERIA = tuple(values("parent_selection_criteria"))
ELEMENT_SENIORITY = {"N": 1, "P": 2, "Si": 3, "B": 4, "O": 5, "S": 6, "C": 7}
HETEROATOM_SENIORITY = {"N": 1, "O": 2, "S": 3, "P": 4, "Si": 5, "B": 6}
HETEROATOM_COUNT_SENIORITY = ("O", "S", "N", "P", "Si", "B")


@dataclass(frozen=True)
class ParentSeniorityProfile:
    """Named parent-selection criteria for one candidate skeleton."""

    principal_group_count: int
    contains_principal_group: bool
    senior_element_vector: tuple[int, ...]
    polycycle_parent: bool
    bicycle_parent: bool
    spiro_parent: bool
    ring_parent: bool
    ring_count: int
    parent_atom_count: int
    heteroatom_count: int
    senior_heteroatom_vector: tuple[int, ...]
    senior_heteroatom_count_vector: tuple[int, ...]
    multiple_bond_count: int
    double_bond_count: int
    path_tiebreak: tuple[int, ...]

    def criterion_value(self, criterion: str):
        """Return the sortable value for one configured criterion."""

        if criterion == "principal_group_count":
            return -self.principal_group_count
        if criterion == "contains_principal_group":
            return -int(self.contains_principal_group)
        if criterion == "senior_element_vector":
            return self.senior_element_vector
        if criterion == "polycycle_parent":
            return -int(self.polycycle_parent)
        if criterion == "bicycle_parent":
            return -int(self.bicycle_parent)
        if criterion == "spiro_parent":
            return -int(self.spiro_parent)
        if criterion == "ring_parent":
            return -int(self.ring_parent)
        if criterion == "ring_count":
            return -self.ring_count
        if criterion == "parent_atom_count":
            return -self.parent_atom_count
        if criterion == "heteroatom_count":
            return -self.heteroatom_count
        if criterion == "senior_heteroatom_vector":
            return self.senior_heteroatom_vector
        if criterion == "senior_heteroatom_count_vector":
            return self.senior_heteroatom_count_vector
        if criterion == "ring_seniority":
            return self.ring_seniority_key()
        if criterion == "chain_seniority":
            return self.chain_seniority_key()
        if criterion == "multiple_bond_count":
            return -self.multiple_bond_count
        if criterion == "double_bond_count":
            return -self.double_bond_count
        if criterion == "path_tiebreak":
            return self.path_tiebreak
        raise KeyError(f"Unknown parent selection criterion: {criterion}")

    def score_tuple(self, criteria: tuple[str, ...] = PARENT_SELECTION_CRITERIA) -> tuple:
        """Return the current sortable score tuple from configured criteria."""

        return tuple(self.criterion_value(criterion) for criterion in criteria)

    def ring_seniority_key(self) -> tuple:
        """Return ring-system seniority criteria after ring-over-chain selection."""

        if not self.ring_parent:
            return ()
        return (
            self.senior_heteroatom_vector,
            -self.ring_count,
            -self.parent_atom_count,
            -self.heteroatom_count,
            self.senior_heteroatom_count_vector,
        )

    def chain_seniority_key(self) -> tuple:
        """Return Brief Guide section 6 chain criteria f.1 and related ties."""

        if self.ring_parent:
            return ()
        return (
            -self.parent_atom_count,
            -self.heteroatom_count,
            self.senior_heteroatom_count_vector,
        )


@dataclass
class ParentCandidate:
    """One parent skeleton candidate with explicit scoring metadata."""

    path: list[int]
    is_ring: bool
    is_bicycle: bool
    is_spiro: bool
    is_polycycle: bool
    xyz: tuple[int, int, int]
    principal_groups_count: int
    length: int
    seniority_profile: ParentSeniorityProfile
    score_tuple: tuple

    @classmethod
    def build(
        cls,
        path: list[int],
        *,
        is_ring: bool,
        is_bicycle: bool,
        is_spiro: bool,
        is_polycycle: bool,
        xyz: tuple[int, int, int],
        principal_groups_count: int,
        mol: Molecule | None = None,
        ring_count: int = 0,
    ) -> "ParentCandidate":
        """Build a candidate using the current parent-selection preference order."""

        profile = ParentSeniorityProfile(
            principal_group_count=principal_groups_count,
            contains_principal_group=principal_groups_count > 0,
            senior_element_vector=_senior_element_vector(mol, path, include_carbon=True),
            polycycle_parent=is_polycycle,
            bicycle_parent=is_bicycle,
            spiro_parent=is_spiro,
            ring_parent=is_ring,
            ring_count=ring_count,
            parent_atom_count=len(path),
            heteroatom_count=_heteroatom_count(mol, path),
            senior_heteroatom_vector=_senior_element_vector(mol, path, include_carbon=False),
            senior_heteroatom_count_vector=_senior_heteroatom_count_vector(mol, path),
            multiple_bond_count=_multiple_bond_count(mol, path, include_double=True),
            double_bond_count=_multiple_bond_count(mol, path, include_double=False),
            path_tiebreak=tuple(path),
        )
        score_tuple = profile.score_tuple()
        return cls(
            path=path,
            is_ring=is_ring,
            is_bicycle=is_bicycle,
            is_spiro=is_spiro,
            is_polycycle=is_polycycle,
            xyz=xyz,
            principal_groups_count=principal_groups_count,
            length=len(path),
            seniority_profile=profile,
            score_tuple=score_tuple,
        )

    def legacy_sort_key(self) -> tuple:
        """Return the previous max-sort key, kept for traceability."""

        return (
            self.principal_groups_count,
            self.is_polycycle,
            self.is_bicycle,
            self.is_spiro,
            self.is_ring,
            self.length,
        )


@dataclass
class ParentSelection:
    """Selected parent skeleton and its structural flags."""

    paths: list[list[int]]
    is_ring: bool
    is_bicycle: bool
    is_spiro: bool
    is_polycycle: bool
    xyz: tuple[int, int, int]
    polycycle_descriptor: str | None = None
    ring_parent: RingParent | None = None
    fixed_start_required: bool = False
    seniority_profile: ParentSeniorityProfile | None = None
    score_tuple: tuple = ()

    @property
    def primary_path(self) -> list[int]:
        """Return the first candidate path used by downstream numbering."""

        return self.paths[0]

    @property
    def atom_set(self) -> set[int]:
        """Return the atom set for the selected primary parent path."""

        return set(self.primary_path)

    @property
    def requires_fixed_substituent_start(self) -> bool:
        """Return whether subgraph numbering must keep its attachment fixed."""

        return self.fixed_start_required or self.is_bicycle or self.is_spiro or self.is_polycycle

    def with_fixed_start(self, fixed_start_required: bool) -> "ParentSelection":
        """Return a copy with subgraph fixed-start behavior attached."""

        return ParentSelection(
            paths=self.paths,
            is_ring=self.is_ring,
            is_bicycle=self.is_bicycle,
            is_spiro=self.is_spiro,
            is_polycycle=self.is_polycycle,
            xyz=self.xyz,
            polycycle_descriptor=self.polycycle_descriptor,
            ring_parent=self.ring_parent,
            fixed_start_required=fixed_start_required,
            seniority_profile=self.seniority_profile,
            score_tuple=self.score_tuple,
        )


def select_principal_parent(
    mol: Molecule,
    chains: list[list[int]],
    ring_systems: list[RingSystem],
    principal_carbons: list[int],
) -> ParentSelection | None:
    """Select the best parent skeleton from chain and ring candidates."""

    candidates: list[ParentCandidate] = []

    for c in chains:
        c_set = set(c)
        pg_count = sum(1 for a in principal_carbons if a in c_set)
        candidates.append(
            ParentCandidate.build(
                c,
                is_ring=False,
                is_bicycle=False,
                is_spiro=False,
                is_polycycle=False,
                xyz=(0, 0, 0),
                principal_groups_count=pg_count,
                mol=mol,
                ring_count=0,
            )
        )

    for rs in ring_systems:
        p = rs.paths[0]
        p_set = set(p)
        pg_count = sum(1 for a in principal_carbons if a in p_set)
        xyz = (rs.x, rs.y, rs.z) if rs.is_bicycle else (rs.x, rs.y, 0)
        candidates.append(
            ParentCandidate.build(
                p,
                is_ring=True,
                is_bicycle=rs.is_bicycle,
                is_spiro=rs.is_spiro,
                is_polycycle=rs.is_polycycle,
                xyz=xyz,
                principal_groups_count=pg_count,
                mol=mol,
                ring_count=_ring_count_for_system(rs),
            )
        )

    if not candidates:
        return None

    candidates = _prefer_spiro_backbone_components(candidates, ring_systems)
    candidates.sort(key=lambda m: m.score_tuple)
    best = candidates[0]

    if best.is_ring:
        winning_rs = next(rs for rs in ring_systems if rs.paths[0] == best.path)
        descriptor = winning_rs.polycycle_descriptor if winning_rs.is_polycycle else None
        return ParentSelection(
            paths=winning_rs.paths,
            is_ring=True,
            is_bicycle=best.is_bicycle,
            is_spiro=best.is_spiro,
            is_polycycle=best.is_polycycle,
            xyz=best.xyz,
            polycycle_descriptor=descriptor,
            ring_parent=winning_rs.ring_parent,
            seniority_profile=best.seniority_profile,
            score_tuple=best.score_tuple,
        )

    return ParentSelection(
        paths=[best.path],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        seniority_profile=best.seniority_profile,
        score_tuple=best.score_tuple,
    )


def _senior_element_vector(mol: Molecule | None, path: list[int], *, include_carbon: bool) -> tuple[int, ...]:
    if mol is None:
        return ()
    ranks = set()
    for atom_idx in path:
        symbol = mol.atoms[atom_idx].symbol
        if include_carbon:
            rank = ELEMENT_SENIORITY.get(symbol)
        else:
            rank = HETEROATOM_SENIORITY.get(symbol)
        if rank is not None:
            ranks.add(rank)
    if not include_carbon and ranks:
        return (min(ranks),)
    return tuple(sorted(ranks))


def _senior_heteroatom_count_vector(mol: Molecule | None, path: list[int]) -> tuple[int, ...]:
    """Return descending counts in the section 6 e.5/f order O, S, N, P, Si, B."""

    if mol is None:
        return ()
    path_symbols = [mol.atoms[atom_idx].symbol for atom_idx in path]
    return tuple(-path_symbols.count(symbol) for symbol in HETEROATOM_COUNT_SENIORITY)


def _heteroatom_count(mol: Molecule | None, path: list[int]) -> int:
    if mol is None:
        return 0
    return sum(1 for atom_idx in path if mol.atoms[atom_idx].symbol != "C")


def _multiple_bond_count(mol: Molecule | None, path: list[int], *, include_double: bool) -> int:
    if mol is None:
        return 0
    path_set = set(path)
    count = 0
    for atom_idx in path:
        for neighbor in mol.get_neighbors(atom_idx):
            if neighbor not in path_set or atom_idx >= neighbor:
                continue
            bond = mol.get_bond(atom_idx, neighbor)
            if bond is None:
                continue
            if include_double and bond.order > 1:
                count += 1
            elif not include_double and bond.order == 2:
                count += 1
    return count


def _prefer_spiro_backbone_components(
    candidates: list[ParentCandidate], ring_systems: list[RingSystem]
) -> list[ParentCandidate]:
    """Limit spiro-component competitions to the graph backbone component.

    Spiro-connected ring components are not independent whole-parent hydride
    candidates.  When principal-group coverage is tied, choose the component
    that represents the shared spiro backbone before applying ordinary parent
    seniority.  This prevents a small hetero side ring from stealing the parent
    slot only because senior-element ordering is evaluated before ring
    complexity.
    """

    if len(ring_systems) < 2:
        return candidates

    shared_counts: dict[tuple[int, ...], int] = {}
    for system in ring_systems:
        shared_count = sum(1 for other in ring_systems if other is not system and system.atoms & other.atoms)
        for path in system.paths:
            shared_counts[tuple(path)] = shared_count

    max_shared = max(shared_counts.values(), default=0)
    if max_shared == 0:
        return candidates

    best_principal_key = min(
        (
            candidate.seniority_profile.criterion_value("contains_principal_group"),
            candidate.seniority_profile.criterion_value("principal_group_count"),
        )
        for candidate in candidates
    )
    eligible = [
        candidate
        for candidate in candidates
        if (
            candidate.seniority_profile.criterion_value("contains_principal_group"),
            candidate.seniority_profile.criterion_value("principal_group_count"),
        )
        == best_principal_key
    ]
    eligible_ring_components = []
    for candidate in eligible:
        if not candidate.is_ring:
            continue
        path_key = tuple(candidate.path)
        if shared_counts.get(path_key, 0) <= 0:
            continue
        eligible_ring_components.append(candidate)
    if len(eligible_ring_components) < 2:
        return candidates

    backbone_keys = {
        tuple(candidate.path): _spiro_backbone_rank_key(candidate, shared_counts)
        for candidate in eligible_ring_components
    }
    best_backbone_key = min(backbone_keys.values())
    backbone_paths = {path for path, key in backbone_keys.items() if key == best_backbone_key}
    if not backbone_paths:
        return candidates
    competing_paths = set(backbone_keys)
    return [
        candidate
        for candidate in candidates
        if tuple(candidate.path) not in competing_paths or tuple(candidate.path) in backbone_paths
    ]


def _spiro_backbone_rank_key(candidate: ParentCandidate, shared_counts: dict[tuple[int, ...], int]) -> tuple:
    """Return the local rank key for competing spiro-connected ring components."""

    path_key = tuple(candidate.path)
    return (
        -shared_counts[path_key],
        -candidate.seniority_profile.ring_count,
        -candidate.seniority_profile.parent_atom_count,
        candidate.seniority_profile.path_tiebreak,
    )


def _ring_count_for_system(ring_system: RingSystem) -> int:
    if ring_system.is_polycycle:
        return max(3, len(ring_system.chord_edges) + 1)
    if ring_system.is_bicycle or ring_system.is_spiro:
        return 2
    return 1
