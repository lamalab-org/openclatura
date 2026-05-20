# structure-to-iupac/parent_selection.py
from dataclasses import dataclass

from .chains import RingSystem
from .molecule import Molecule
from .ring_parent import RingParent


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
    ) -> "ParentCandidate":
        """Build a candidate using the current parent-selection preference order."""

        score_tuple = (
            -principal_groups_count,
            -int(is_polycycle),
            -int(is_bicycle),
            -int(is_spiro),
            -int(is_ring),
            -len(path),
            tuple(path),
        )
        return cls(
            path=path,
            is_ring=is_ring,
            is_bicycle=is_bicycle,
            is_spiro=is_spiro,
            is_polycycle=is_polycycle,
            xyz=xyz,
            principal_groups_count=principal_groups_count,
            length=len(path),
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
            )
        )

    if not candidates:
        return None

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
            score_tuple=best.score_tuple,
        )

    return ParentSelection(
        paths=[best.path],
        is_ring=False,
        is_bicycle=False,
        is_spiro=False,
        is_polycycle=False,
        xyz=(0, 0, 0),
        score_tuple=best.score_tuple,
    )
