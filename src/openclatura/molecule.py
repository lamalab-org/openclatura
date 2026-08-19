from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from .rules import elements


@dataclass
class Atom:
    idx: int
    symbol: str
    charge: int = 0
    isotope: int | None = None
    stereo: str | None = None  # 'R' or 'S'
    raw_stereo: str | None = None  # RDKit tetrahedral tag when CIP is unavailable: 'CW' or 'CCW'
    cip: str | None = None  # independent modern (rdCIPLabeler) CIP label; only populated during self-audit
    is_aromatic: bool = False
    explicit_h_count: int = 0
    total_h_count: int = 0

    def __post_init__(self):
        if not elements.is_known(self.symbol):
            raise ValueError(f"Unknown element symbol: {self.symbol}")

    @property
    def element(self) -> elements.Element:
        return elements.get(self.symbol)

    @property
    def is_carbon(self) -> bool:
        return self.symbol == "C"


@dataclass
class Bond:
    idx: int
    u: int
    v: int
    order: int = 1
    stereo: str | None = None  # 'E' or 'Z'
    in_small_ring: bool = False  # NEW: Tracks if bond is in a ring of size <= 7
    cip: str | None = None  # independent modern (rdCIPLabeler) E/Z label; only during self-audit


@dataclass(frozen=True)
class AtomBinding:
    """A named relationship between a nomenclature object and graph atoms."""

    role: str
    atom_ids: tuple[int, ...]


@dataclass(frozen=True)
class BondBinding:
    """A named relationship between a nomenclature object and graph bonds."""

    role: str
    bond_ids: tuple[int, ...]


@dataclass(frozen=True)
class FunctionalGroupMetadata:
    """Nomenclature metadata attached to a perceived functional group."""

    prefix: str | None = None
    suffix: str | None = None
    multi_suffix: object | None = None
    suffix_multiplier_positions: tuple[int, ...] = (0,)
    seniority: int | None = None
    suffix_with_locant: bool = False
    source: str = "perception"


class TracePhase(str, Enum):
    """High-level phases in the structure-to-name pipeline."""

    PARSE = "parse"
    COMPONENT = "component"
    PERCEPTION = "perception"
    PRIORITY = "priority"
    PARENT_SELECTION = "parent_selection"
    NUMBERING = "numbering"
    ASSEMBLY = "assembly"


class OperationClass(str, Enum):
    """High-level IUPAC operation classes represented by the naming pipeline."""

    SUBSTITUTIVE = "substitutive"
    REPLACEMENT = "replacement"
    ADDITIVE = "additive"
    SUBTRACTIVE = "subtractive"
    CONJUNCTIVE = "conjunctive"
    MULTIPLICATIVE = "multiplicative"
    FUSION = "fusion"


@dataclass(frozen=True)
class NomenclatureOperation:
    """Structured operation record derived from naming decisions."""

    operation_class: OperationClass
    detail: str
    locants: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceStep:
    """One explainable naming decision."""

    phase: TracePhase
    decision: str
    reason: str
    atoms: tuple[int, ...] = ()
    bonds: tuple[int, ...] = ()
    data: dict = field(default_factory=dict)


@dataclass
class DecisionTrace:
    """Append-only trace of major naming decisions."""

    steps: list[TraceStep] = field(default_factory=list)

    def add(
        self,
        phase: TracePhase,
        decision: str,
        reason: str,
        *,
        atoms: set[int] | list[int] | tuple[int, ...] = (),
        bonds: set[int] | list[int] | tuple[int, ...] = (),
        data: dict | None = None,
    ) -> None:
        self.steps.append(
            TraceStep(
                phase=phase,
                decision=decision,
                reason=reason,
                atoms=tuple(sorted(atoms)),
                bonds=tuple(sorted(bonds)),
                data=data or {},
            )
        )


@dataclass(frozen=True)
class NameAnalysis:
    """Full explainable result for a SMILES naming run."""

    name: str
    trace_segments: list[dict]
    decisions: list[TraceStep]
    substituent_tree: list[dict] = field(default_factory=list)
    operations: list[NomenclatureOperation] = field(default_factory=list)


class Molecule:
    def __init__(self):
        self.atoms: dict[int, Atom] = {}
        self.bonds: dict[int, Bond] = {}
        self._adj: dict[int, list[int]] = {}
        self._bond_lookup: dict[tuple[int, int], int] = {}
        self._cyclic_cache: set[int] | None = None  # full-molecule ring atoms; invalidated on mutation
        self._perception_cache: tuple | None = None  # perceived functional groups; invalidated on mutation
        self._canonical_rank_cache: dict[int, int] | None = None
        self._retained_fused_cache: dict[tuple, tuple] = {}
        self.audit_rdmol = None
        self.accurate_cip: dict[int, str] = {}
        self.substituted_symbols: frozenset[int] = frozenset()

    def add_atom(
        self,
        symbol: str,
        idx: int | None = None,
        charge: int = 0,
        stereo: str | None = None,
        *,
        isotope: int | None = None,
        raw_stereo: str | None = None,
        cip: str | None = None,
        is_aromatic: bool = False,
        explicit_h_count: int = 0,
        total_h_count: int = 0,
    ) -> Atom:
        if idx is None:
            idx = max(self.atoms.keys(), default=0) + 1
        if idx in self.atoms:
            raise ValueError(f"Atom with idx {idx} already exists.")
        atom = Atom(
            idx=idx,
            symbol=symbol,
            charge=charge,
            isotope=isotope,
            stereo=stereo,
            raw_stereo=raw_stereo,
            cip=cip,
            is_aromatic=is_aromatic,
            explicit_h_count=explicit_h_count,
            total_h_count=total_h_count,
        )
        self.atoms[idx] = atom
        self._adj[idx] = []
        self._cyclic_cache = None
        self._perception_cache = None
        self._canonical_rank_cache = None
        self._retained_fused_cache.clear()
        return atom

    def add_bond(
        self,
        u: int,
        v: int,
        order: int = 1,
        idx: int | None = None,
        stereo: str | None = None,
        in_small_ring: bool = False,
        cip: str | None = None,
    ) -> Bond:
        if u not in self.atoms or v not in self.atoms:
            raise ValueError("Both atoms must exist")
        if u == v:
            raise ValueError("Cannot bond an atom to itself.")
        bond_key = (u, v) if u < v else (v, u)
        if bond_key in self._bond_lookup:
            raise ValueError(f"Atoms {u} and {v} are already bonded.")
        if idx is None:
            idx = max(self.bonds.keys(), default=0) + 1
        bond = Bond(idx=idx, u=u, v=v, order=order, stereo=stereo, in_small_ring=in_small_ring, cip=cip)
        self.bonds[idx] = bond
        self._bond_lookup[bond_key] = idx
        self._adj[u].append(v)
        self._adj[v].append(u)
        self._cyclic_cache = None
        self._perception_cache = None
        self._canonical_rank_cache = None
        self._retained_fused_cache.clear()
        return bond

    def get_neighbors(self, atom_idx: int) -> list[int]:
        return self._adj.get(atom_idx, [])

    def get_bond(self, u: int, v: int) -> Bond | None:
        bond_key = (u, v) if u < v else (v, u)
        bond_idx = self._bond_lookup.get(bond_key)
        if bond_idx is not None:
            return self.bonds[bond_idx]
        return None

    def degree(self, atom_idx: int) -> int:
        return len(self.get_neighbors(atom_idx))

    def subgraph(self, atom_ids, *, symbols: dict[int, str] | None = None) -> "Molecule":
        """Return the induced subgraph over atom_ids, keeping the original indices."""

        fragment = Molecule()
        for idx in atom_ids:
            atom = self.atoms[idx]
            fragment.add_atom(
                symbol=(symbols or {}).get(idx, atom.symbol),
                idx=idx,
                charge=atom.charge,
                stereo=atom.stereo,
                raw_stereo=atom.raw_stereo,
                is_aromatic=atom.is_aromatic,
                explicit_h_count=atom.explicit_h_count,
                total_h_count=atom.total_h_count,
            )
        for idx in atom_ids:
            for neighbor in self.get_neighbors(idx):
                if neighbor in atom_ids and idx < neighbor:
                    bond = self.get_bond(idx, neighbor)
                    fragment.add_bond(
                        u=idx, v=neighbor, order=bond.order, stereo=bond.stereo, in_small_ring=bond.in_small_ring
                    )
        fragment.substituted_symbols = frozenset(
            idx for idx, symbol in (symbols or {}).items() if idx in fragment.atoms and symbol != self.atoms[idx].symbol
        )
        return fragment

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms.values())


def bond_ids_within(mol: Molecule, atom_ids: set[int]) -> set[int]:
    """Return bond IDs whose endpoints are both in atom_ids."""

    bond_ids = set()
    for atom_idx in atom_ids:
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in atom_ids and atom_idx < neighbor_idx:
                bond = mol.get_bond(atom_idx, neighbor_idx)
                if bond is not None:
                    bond_ids.add(bond.idx)
    return bond_ids


def edges_within_atoms(mol: Molecule, atoms: set[int]) -> set[tuple[int, int]]:
    edges = set()
    for atom_idx in atoms:
        for neighbor_idx in mol.get_neighbors(atom_idx):
            if neighbor_idx in atoms and atom_idx < neighbor_idx:
                edges.add((atom_idx, neighbor_idx))
    return edges


def component_atoms_until_blocked(
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


def has_non_h_multiple_bond_neighbor(mol: Molecule, atom_idx: int, allowed: set[int]) -> bool:
    for neighbor in mol.get_neighbors(atom_idx):
        if neighbor in allowed or mol.atoms[neighbor].symbol == "H":
            continue
        bond = mol.get_bond(atom_idx, neighbor)
        if bond is not None and bond.order != 1:
            return True
    return False


def double_bonded_carbon(mol: Molecule, nitrogen: int, blocked: set[int]) -> int | None:
    candidates = [
        n
        for n in mol.get_neighbors(nitrogen)
        if n not in blocked
        and mol.atoms[n].is_carbon
        and (bond := mol.get_bond(nitrogen, n)) is not None
        and bond.order == 2
    ]
    return candidates[0] if len(candidates) == 1 else None
