# structure-to-iupac/parent_selection.py
from dataclasses import dataclass
from .molecule import Molecule
from .chains import RingSystem

@dataclass
class ParentMetrics:
    path: list[int]
    is_ring: bool
    is_bicycle: bool
    is_spiro: bool
    is_polycycle: bool
    xyz: tuple[int, int, int]
    principal_groups_count: int
    length: int

    def sort_key(self) -> tuple:
        return (
            self.principal_groups_count,
            self.is_polycycle,
            self.is_bicycle,
            self.is_spiro,
            self.is_ring,
            self.length
        )

def select_principal_parent(mol: Molecule, chains: list[list[int]], ring_systems: list[RingSystem], principal_carbons: list[int]):
    candidates =[]

    for c in chains:
        c_set = set(c)
        pg_count = sum(1 for a in principal_carbons if a in c_set)
        candidates.append(ParentMetrics(c, False, False, False, False, (0,0,0), pg_count, len(c)))

    for rs in ring_systems:
        p = rs.paths[0]
        p_set = set(p)
        pg_count = sum(1 for a in principal_carbons if a in p_set)
        xyz = (rs.x, rs.y, rs.z) if rs.is_bicycle else (rs.x, rs.y, 0)
        candidates.append(ParentMetrics(p, True, rs.is_bicycle, rs.is_spiro, rs.is_polycycle, xyz, pg_count, len(p)))

    if not candidates:
        return[], False, False, False, False, (0,0,0), None

    candidates.sort(key=lambda m: m.sort_key(), reverse=True)
    best = candidates[0]

    if best.is_ring:
        winning_rs = next(rs for rs in ring_systems if rs.paths[0] == best.path)
        descriptor = winning_rs.polycycle_descriptor if winning_rs.is_polycycle else None
        return winning_rs.paths, True, best.is_bicycle, best.is_spiro, best.is_polycycle, best.xyz, descriptor

    return [best.path], False, False, False, False, (0,0,0), None