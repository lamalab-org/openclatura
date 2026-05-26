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

    @property
    def is_heteroatom(self) -> bool:
        return self.symbol not in ("C", "H")


@dataclass
class Bond:
    idx: int
    u: int
    v: int
    order: int = 1
    stereo: str | None = None  # 'E' or 'Z'
    in_small_ring: bool = False  # NEW: Tracks if bond is in a ring of size <= 7

    def get_other_atom(self, atom_idx: int) -> int:
        if atom_idx == self.u:
            return self.v
        if atom_idx == self.v:
            return self.u
        raise ValueError(f"Atom {atom_idx} is not part of bond {self.idx}")


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
    operations: list[NomenclatureOperation] = field(default_factory=list)


class Molecule:
    def __init__(self):
        self.atoms: dict[int, Atom] = {}
        self.bonds: dict[int, Bond] = {}
        self._adj: dict[int, list[int]] = {}
        self._bond_lookup: dict[tuple[int, int], int] = {}

    def add_atom(
        self,
        symbol: str,
        idx: int | None = None,
        charge: int = 0,
        stereo: str | None = None,
        *,
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
            stereo=stereo,
            is_aromatic=is_aromatic,
            explicit_h_count=explicit_h_count,
            total_h_count=total_h_count,
        )
        self.atoms[idx] = atom
        self._adj[idx] = []
        return atom

    def add_bond(
        self,
        u: int,
        v: int,
        order: int = 1,
        idx: int | None = None,
        stereo: str | None = None,
        in_small_ring: bool = False,
    ) -> Bond:
        if u not in self.atoms or v not in self.atoms:
            raise ValueError("Both atoms must exist")
        if u == v:
            raise ValueError("Cannot bond an atom to itself.")
        bond_key = tuple(sorted((u, v)))
        if bond_key in self._bond_lookup:
            raise ValueError(f"Atoms {u} and {v} are already bonded.")
        if idx is None:
            idx = max(self.bonds.keys(), default=0) + 1
        bond = Bond(idx=idx, u=u, v=v, order=order, stereo=stereo, in_small_ring=in_small_ring)
        self.bonds[idx] = bond
        self._bond_lookup[bond_key] = idx
        self._adj[u].append(v)
        self._adj[v].append(u)
        return bond

    def get_neighbors(self, atom_idx: int) -> list[int]:
        return self._adj.get(atom_idx, [])

    def get_bond(self, u: int, v: int) -> Bond | None:
        bond_key = tuple(sorted((u, v)))
        bond_idx = self._bond_lookup.get(bond_key)
        if bond_idx is not None:
            return self.bonds[bond_idx]
        return None

    def degree(self, atom_idx: int) -> int:
        return len(self.get_neighbors(atom_idx))

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms.values())
