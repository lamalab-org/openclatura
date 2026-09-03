"""Context and result objects for structure-to-name decisions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from .assembly_parts import RetainedParentMetadata
from .locant_sources import LocantMapSource
from .parent_selection import ParentSelection
from .perception import PerceivedGroup
from .ring_parent import RingParent


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
    omit_redundant_locants: bool = True

    @classmethod
    def component(cls, principal_atoms, *, omit_redundant_locants: bool = True) -> "NamingIntent":
        return cls(
            mode=NamingMode.COMPONENT,
            principal_atoms=tuple(principal_atoms),
            omit_redundant_locants=omit_redundant_locants,
        )

    @classmethod
    def subgraph(
        cls,
        root_atom: int,
        upstream_atom: int | None,
        *,
        fixed_start: bool,
        omit_redundant_locants: bool = True,
    ) -> "NamingIntent":
        return cls(
            mode=NamingMode.SUBGRAPH,
            principal_atoms=(root_atom,),
            root_atom=root_atom,
            upstream_atom=upstream_atom,
            fixed_start=fixed_start,
            is_substituent=True,
            omit_redundant_locants=omit_redundant_locants,
        )


@dataclass
class ParentAssemblyPlan:
    """Numbered parent plus assembly parts for the selected naming intent."""

    numbered_path: list[int]
    locant_map: dict[int, str] | None
    locant_map_source: LocantMapSource
    get_loc: Callable
    parts: object
    ring_parent: RingParent | None = None


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
    retained_parent_metadata: RetainedParentMetadata | None = None
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
