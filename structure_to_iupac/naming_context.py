"""Context and result objects for structure-to-name decisions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .molecule import DecisionTrace, Molecule
from .parent_selection import ParentSelection
from .perception import PerceivedGroup


class NamingMode(str, Enum):
    """Supported naming contexts."""

    COMPONENT = "component"
    SUBGRAPH = "subgraph"


@dataclass(frozen=True)
class NamingIntent:
    """Mode-specific intent for parent numbering and assembly."""

    mode: NamingMode
    principal_atoms: tuple[int, ...] = ()
    root_atom: int | None = None
    upstream_atom: int | None = None
    fixed_start: bool = False
    is_substituent: bool = False

    @classmethod
    def component(cls, principal_atoms) -> "NamingIntent":
        return cls(mode=NamingMode.COMPONENT, principal_atoms=tuple(principal_atoms))

    @classmethod
    def subgraph(cls, root_atom: int, upstream_atom: int | None, *, fixed_start: bool) -> "NamingIntent":
        return cls(
            mode=NamingMode.SUBGRAPH,
            principal_atoms=(root_atom,),
            root_atom=root_atom,
            upstream_atom=upstream_atom,
            fixed_start=fixed_start,
            is_substituent=True,
        )


@dataclass
class NamingContext:
    """Shared context for a naming run or recursive naming branch."""

    mol: Molecule
    mode: NamingMode
    component_atoms: set[int] = field(default_factory=set)
    root_atom: int | None = None
    upstream_atom: int | None = None
    blocked_atoms: set[int] = field(default_factory=set)
    trace: DecisionTrace | None = None
    perceived_groups: list[PerceivedGroup] = field(default_factory=list)
    cyclic_atoms: set[int] = field(default_factory=set)


@dataclass
class SubgraphBoundary:
    """Explicit graph boundary for recursive branch naming."""

    root_atom: int
    upstream_atom: int | None
    blocked_atoms: set[int]
    component_atoms: set[int] = field(default_factory=set)
    parent_atoms: set[int] = field(default_factory=set)
    consumed_atoms: set[int] = field(default_factory=set)


@dataclass
class PrincipalGroupSelection:
    """Selected principal group key plus all matching group objects."""

    key: str | None
    groups: list[PerceivedGroup] = field(default_factory=list)
    prefix_groups: list[PerceivedGroup] = field(default_factory=list)

    @property
    def attachment_atoms(self) -> list[int]:
        return [group.attachment_carbon for group in self.groups]

    @property
    def involved_atoms(self) -> set[int]:
        atoms: set[int] = set()
        for group in self.groups:
            atoms.add(group.attachment_carbon)
            atoms.update(group.atoms_involved)
        return atoms


@dataclass
class ParentPlan:
    """Selected parent plus retained-name and principal-group metadata."""

    selection: ParentSelection
    retained_name: str | None = None
    locant_maps: list[dict[int, str]] | None = None
    principal: PrincipalGroupSelection | None = None


@dataclass
class NumberedParent:
    """Numbered parent path and locant lookup."""

    path: list[int]
    locant_map: dict[int, str] | None = None

    def locant_for(self, atom_idx: int) -> str | int:
        if self.locant_map:
            return self.locant_map[atom_idx]
        return self.path.index(atom_idx) + 1


@dataclass
class ParentAssemblyPlan:
    """Numbered parent plus assembly parts for the selected naming intent."""

    numbered_path: list[int]
    locant_map: dict[int, str] | None
    get_loc: Callable
    parts: object


@dataclass
class FragmentNamingResult:
    """Name plus graph and trace metadata for a named fragment."""

    name: str
    trace_segments: list[dict] = field(default_factory=list)
    atom_ids: set[int] = field(default_factory=set)
    bond_ids: set[int] = field(default_factory=set)


@dataclass
class ComponentNamingState:
    """Mutable state for the component naming decision tree."""

    component_atoms: set[int]
    is_substituent: bool = False
    perceived_groups: list[PerceivedGroup] = field(default_factory=list)
    principal_key: str | None = None
    exclude_atoms: set[int] = field(default_factory=set)
    cyclic_atoms_all: set[int] = field(default_factory=set)
    principal_carbons: list[int] = field(default_factory=list)
    prefix_groups: list[PerceivedGroup] = field(default_factory=list)
    parent_selection: ParentSelection | None = None
    retained_name: str | None = None
    locant_maps: list[dict[int, str]] | None = None
    principal_involved_atoms: set[int] = field(default_factory=set)
    base_exclude: set[int] = field(default_factory=set)
    sub_exclude: set[int] = field(default_factory=set)
    component_namer: Callable | None = None

    @property
    def parent_path(self) -> list[int]:
        if self.parent_selection is None:
            return []
        return self.parent_selection.primary_path

    @property
    def parent_set(self) -> set[int]:
        if self.parent_selection is None:
            return set()
        return self.parent_selection.atom_set
