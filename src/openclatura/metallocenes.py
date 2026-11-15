"""Metallocene parents: ``ferrocene`` and its congeners.

A metallocene is written as three disconnected components -- a metal dication
and two cyclopentadienide rings -- which is also the structure OPSIN builds for
``ferrocene``.  Named component by component it would come out as a salt, so the
three are claimed here, before the engine splits the molecule, and named as one
parent hydride with ten substitutable positions: ``1``-``5`` on one ring and
``1'``-``5'`` on the other.

Everything attached to a ring is cited as a prefix on that parent
(``1,1'-bis(diphenylphosphanyl)ferrocene``).  Suffix names
(``ferrocenecarboxylic acid``) and the substituent form (``ferrocenyl``) are not
generated; the prefix names denote the same structures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

from .assembly_parts import split_rendered_substituent_name
from .assembly_prefixes import substituent_sort_key
from .formatting import format_multiplier, is_complex_prefix, is_composite_prefix
from .molecule import Molecule

# Every parent here was checked to parse in OPSIN to metal(2+) plus two
# cyclopentadienide rings.
METALLOCENE_PARENT_NAMES: dict[str, str] = {
    "Ti": "titanocene",
    "V": "vanadocene",
    "Cr": "chromocene",
    "Fe": "ferrocene",
    "Co": "cobaltocene",
    "Ni": "nickelocene",
    "Ru": "ruthenocene",
    "Os": "osmocene",
}

_RING_SIZE = 5
_METAL_CHARGE = 2

SubgraphNamer = Callable[..., object]
# A locant is ordered by number first and prime second: 1 < 1' < 2 < 2'.
_LocantKey = tuple[int, int]


class UnsupportedMetalError(ValueError):
    """A metallocene metal occurs outside a metallocene this module can name."""


@dataclass(frozen=True)
class Metallocene:
    """One perceived metallocene: its metal and the two rings as atom cycles."""

    metal: int
    rings: tuple[tuple[int, ...], tuple[int, ...]]
    atoms: frozenset[int]

    @property
    def ring_atoms(self) -> frozenset[int]:
        return frozenset(self.rings[0]) | frozenset(self.rings[1])


def _cyclopentadienide_ring(mol: Molecule, component: set[int]) -> tuple[int, ...] | None:
    """Return the ring of a substituted cyclopentadienide component, in cycle order.

    The ring is five carbons holding the component's only negative charge and
    two ring double bonds; nothing may be fused to it or doubly bonded to it.
    """

    anions = [idx for idx in component if mol.atoms[idx].charge < 0 and mol.atoms[idx].is_carbon]
    if sum(mol.atoms[idx].charge for idx in component) != -1:
        return None

    for start in anions:
        if mol.atoms[start].charge != -1:
            continue
        ring = _five_carbon_cycle_through(mol, start)
        if ring is None:
            continue
        ring_set = set(ring)
        if any(mol.atoms[idx].charge for idx in ring_set if idx != start):
            continue
        ring_bond_orders = [mol.get_bond(ring[i], ring[(i + 1) % _RING_SIZE]).order for i in range(_RING_SIZE)]
        if sorted(ring_bond_orders) != [1, 1, 1, 2, 2]:
            continue
        if not _ring_is_isolated(mol, ring):
            continue
        return ring
    return None


def _five_carbon_cycle_through(mol: Molecule, start: int) -> tuple[int, ...] | None:
    """Return the unique five-carbon cycle through ``start``, if there is exactly one."""

    found: set[frozenset[int]] = set()
    cycle: tuple[int, ...] | None = None

    def walk(path: list[int]) -> None:
        nonlocal cycle
        for neighbor in mol.get_neighbors(path[-1]):
            if not mol.atoms[neighbor].is_carbon:
                continue
            if len(path) == _RING_SIZE:
                if neighbor == start and frozenset(path) not in found:
                    found.add(frozenset(path))
                    cycle = tuple(path)
                continue
            if neighbor not in path:
                walk([*path, neighbor])

    walk([start])
    return cycle if len(found) == 1 else None


def _ring_is_isolated(mol: Molecule, ring: tuple[int, ...]) -> bool:
    """True when each ring atom carries at most one singly bonded branch and no branch rejoins the ring."""

    ring_set = set(ring)
    seen_branch_atoms: set[int] = set()
    for idx in ring:
        outside = [n for n in mol.get_neighbors(idx) if n not in ring_set]
        if len(outside) > 1:
            return False
        for neighbor in outside:
            if mol.get_bond(idx, neighbor).order != 1:
                return False
            branch = _branch_atoms(mol, neighbor, ring_set)
            if branch & seen_branch_atoms:
                return False
            seen_branch_atoms |= branch
    return True


def _branch_atoms(mol: Molecule, start: int, blocked: set[int]) -> set[int]:
    seen = {start}
    stack = [start]
    while stack:
        for neighbor in mol.get_neighbors(stack.pop()):
            if neighbor not in seen and neighbor not in blocked:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def find_metallocene(mol: Molecule, components: list[set[int]]) -> tuple[Metallocene | None, list[set[int]]]:
    """Claim one metal dication and two cyclopentadienide components as a metallocene.

    Returns the metallocene (or ``None``) and the components left for ordinary
    naming.  Raises :class:`UnsupportedMetalError` when a metallocene metal is
    present but the molecule is not exactly one recognisable metallocene: these
    metals have no other naming support, so a salt-style name would be wrong.
    """

    metal_components = [
        component
        for component in components
        if any(mol.atoms[idx].symbol in METALLOCENE_PARENT_NAMES for idx in component)
    ]
    if not metal_components:
        return None, components

    if len(metal_components) != 1 or len(metal_components[0]) != 1:
        raise UnsupportedMetalError("metallocene metals are only supported as one bare M(2+) centre per molecule")
    (metal,) = metal_components[0]
    if mol.atoms[metal].charge != _METAL_CHARGE:
        raise UnsupportedMetalError(f"{mol.atoms[metal].symbol} is only supported as the 2+ centre of a metallocene")

    rings: list[tuple[int, ...]] = []
    ring_components: list[set[int]] = []
    for component in components:
        if component is metal_components[0]:
            continue
        ring = _cyclopentadienide_ring(mol, component)
        if ring is not None:
            rings.append(ring)
            ring_components.append(component)
    if len(rings) != 2:
        raise UnsupportedMetalError(
            f"{mol.atoms[metal].symbol}(2+) needs exactly two cyclopentadienide rings to be named as a metallocene"
        )

    claimed = [metal_components[0], *ring_components]
    remaining = [component for component in components if not any(component is taken for taken in claimed)]
    atoms = frozenset({metal}) | frozenset(ring_components[0]) | frozenset(ring_components[1])
    return Metallocene(metal=metal, rings=(rings[0], rings[1]), atoms=atoms), remaining


def _ring_numberings(ring: tuple[int, ...]) -> list[dict[int, int]]:
    """Every numbering of a five-membered ring: each start atom, both directions."""

    numberings = []
    for start in range(_RING_SIZE):
        for step in (1, -1):
            numberings.append({ring[(start + step * offset) % _RING_SIZE]: offset + 1 for offset in range(_RING_SIZE)})
    return numberings


def _format_locant(key: _LocantKey) -> str:
    return f"{key[0]}{chr(39) * key[1]}"


def _enclose(name: str, count: int) -> str:
    """Cite ``count`` copies of a prefix, bracketing compound prefixes so they read as one unit."""

    if count == 1 and (is_complex_prefix(name) or is_composite_prefix(name)) and not name.startswith("("):
        return f"({name})"
    return format_multiplier(name, count)


def name_metallocene(mol: Molecule, metallocene: Metallocene, name_subgraph: SubgraphNamer) -> str:
    """Name a metallocene with every ring substituent cited as a prefix."""

    parent = METALLOCENE_PARENT_NAMES[mol.atoms[metallocene.metal].symbol]
    ring_atoms = set(metallocene.ring_atoms)

    # ring index -> [(ring atom, prefix name)]
    substituents: tuple[list[tuple[int, str]], list[tuple[int, str]]] = ([], [])
    for ring_index, ring in enumerate(metallocene.rings):
        for atom in ring:
            for neighbor in mol.get_neighbors(atom):
                if neighbor in ring_atoms:
                    continue
                rendered = name_subgraph(mol, neighbor, ring_atoms, upstream_atom=atom)
                prefix, _ = split_rendered_substituent_name(rendered)
                if not prefix:
                    raise ValueError(f"could not name a substituent of {parent}")
                substituents[ring_index].append((atom, prefix))

    if not substituents[0] and not substituents[1]:
        return parent

    def citation_order(item: tuple[str, _LocantKey]) -> tuple[str, str, _LocantKey]:
        return (substituent_sort_key(item[0]), item[0], item[1])

    best: tuple[tuple, list[tuple[str, _LocantKey]]] | None = None
    for unprimed in (0, 1):
        ring_order = (unprimed, 1 - unprimed)
        for numberings in product(*(_ring_numberings(metallocene.rings[index]) for index in ring_order)):
            located = sorted(
                (
                    (prefix, (numbering[atom], primes))
                    for primes, (index, numbering) in enumerate(zip(ring_order, numberings, strict=True))
                    for atom, prefix in substituents[index]
                ),
                key=citation_order,
            )
            # P-31.1.4: lowest locants to the substituents all together, then to
            # the substituent cited first in alphabetical order.
            rank = (sorted(key for _, key in located), [key for _, key in located])
            if best is None or rank < best[0]:
                best = (rank, located)

    assert best is not None
    located = best[1]
    if len(located) == 1:
        return f"{_enclose(located[0][0], 1)}{parent}"

    grouped: dict[str, list[_LocantKey]] = {}
    for prefix, key in located:
        grouped.setdefault(prefix, []).append(key)
    cited = [
        f"{','.join(_format_locant(key) for key in sorted(keys))}-{_enclose(prefix, len(keys))}"
        for prefix, keys in grouped.items()
    ]
    return "-".join(cited) + parent


__all__ = [
    "METALLOCENE_PARENT_NAMES",
    "Metallocene",
    "UnsupportedMetalError",
    "find_metallocene",
    "name_metallocene",
]
