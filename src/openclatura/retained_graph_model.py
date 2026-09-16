"""Dependency-free value objects for graph-backed parent templates.

The retained-parent matcher and systematic-fusion planner share these records.
Keeping them outside either pipeline prevents the fusion policy layer from
depending on retained-parent assembly code and avoids parallel graph models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RetainedGraphAtomTemplate:
    locant: str
    symbol: str = "C"
    charge: int = 0
    aromatic: bool = True
    fusion: bool = False
    default_h: bool = False
    saturated: bool = False
    interior: bool = False
    pi_capacity: int | None = None
    forced_single: bool = False
    indicated_h_site: bool = False
    required_stereo: bool = False

    def __post_init__(self) -> None:
        if not self.locant or not self.symbol:
            raise ValueError("template atom locant and symbol must not be empty")
        if self.pi_capacity is not None and self.pi_capacity not in {0, 1}:
            raise ValueError("template atom pi_capacity must be zero or one")

    @property
    def resolved_pi_capacity(self) -> int:
        """Return template-local pi capacity, falling back to element policy."""

        if self.saturated or self.forced_single:
            return 0
        if self.pi_capacity is not None:
            return self.pi_capacity
        from .rules.elements import MANCUDE_FORCED_SINGLE_SYMBOLS

        return int(self.symbol not in MANCUDE_FORCED_SINGLE_SYMBOLS)


@dataclass(frozen=True, slots=True)
class RetainedGraphBondTemplate:
    locants: tuple[str, str]
    bond_class: str = "aromatic"

    def __post_init__(self) -> None:
        if len(self.locants) != 2 or self.locants[0] == self.locants[1]:
            raise ValueError("template bond must join two distinct locants")
        if not all(self.locants) or not self.bond_class:
            raise ValueError("template bond locants and class must not be empty")


@dataclass(frozen=True, slots=True)
class RetainedGraphTemplate:
    name: str
    pin: bool
    priority: int
    aliases: tuple[str, ...]
    attached_prefix: str | None
    derivative_stem: str | None
    default_indicated_h: tuple[str, ...]
    locants: tuple[str, ...]
    atoms: tuple[RetainedGraphAtomTemplate, ...]
    bonds: tuple[RetainedGraphBondTemplate, ...]
    rings: tuple[tuple[str, ...], ...]
    fusion_atoms: tuple[str, ...]
    peripheral_atoms: tuple[str, ...]
    interior_atoms: tuple[str, ...]
    family: str = "fused"
    numbering_policy: str = "retained_template"
    aromatic_equivalence_policy: str = "neutral_kekule_equivalent"
    charge_policy: str = "charge_layer"
    enforce_mancude_double_bonds: bool = False
    enabled: bool = False
    derivative_production_enabled: bool = False
    derivative_audit_enabled: bool = False
    implied_stereo: bool = False
    mancude_double_bonds: int | None = None
    indicated_hydrogen_count_override: int | None = None
    pre_descriptor_selection: bool = False

    @property
    def atom_by_locant(self) -> dict[str, RetainedGraphAtomTemplate]:
        return {atom.locant: atom for atom in self.atoms}

    @property
    def output_name(self) -> str:
        # Local import keeps the graph model independent of rendering policy at
        # module-import time while retaining the historical convenience API.
        from .retained_name_policy import retained_parent_output_name

        return retained_parent_output_name(self.name, "unsubstituted_parent")

    @property
    def indicated_hydrogen_count(self) -> int:
        if self.indicated_hydrogen_count_override is not None:
            return self.indicated_hydrogen_count_override
        if self.default_indicated_h:
            return len(self.default_indicated_h)
        from .rules.elements import MANCUDE_FORCED_SINGLE_SYMBOLS

        return sum(
            1
            for atom in self.atoms
            if not atom.aromatic and not atom.fusion and atom.symbol not in MANCUDE_FORCED_SINGLE_SYMBOLS
        )


@dataclass(frozen=True, slots=True)
class RetainedGraphTemplateMatch:
    template: RetainedGraphTemplate
    atom_to_locant: dict[int, str]
    locant_to_atom: dict[str, int]
    matched_atoms: frozenset[int]
    indicated_h: tuple[str, ...]
    trace: tuple[str, ...] = ()


# Historical names remain aliases, not separate models.
RetainedFusedAtomTemplate = RetainedGraphAtomTemplate
RetainedFusedBondTemplate = RetainedGraphBondTemplate
RetainedFusedGraphTemplate = RetainedGraphTemplate
RetainedFusedTemplateMatch = RetainedGraphTemplateMatch

_PI_CAPABLE_BOND_CLASSES = frozenset({"aromatic", "mancude", "fusion"})


def merge_parent_bond_classes(left: str, right: str) -> str | None:
    """Return the common parent-bond capability represented by two templates.

    ``aromatic``, ``mancude``, and ``fusion`` describe different template
    provenance for an edge that participates in the same delocalized parent
    bond model. Explicit single and double classes remain exact constraints.
    """

    if left == right:
        return left
    if left in _PI_CAPABLE_BOND_CLASSES and right in _PI_CAPABLE_BOND_CLASSES:
        return "mancude"
    return None


def monocyclic_graph_template(
    *,
    name: str,
    ring_size: int,
    symbol: str = "C",
    bond_class: Literal["single", "double", "aromatic", "mancude"] = "mancude",
    pin: bool = True,
) -> RetainedGraphTemplate:
    """Build a locant-complete monocyclic parent graph.

    This is shared graph infrastructure, not a retained-name lookup.  Callers
    remain responsible for nomenclatural eligibility, names, and attached
    forms; this factory only constructs the numbered graph they describe.
    """

    if not name:
        raise ValueError("monocyclic graph template requires a name")
    if ring_size < 3:
        raise ValueError("monocyclic graph template requires at least three atoms")
    if not symbol:
        raise ValueError("monocyclic graph template requires an atom symbol")
    if bond_class not in {"single", "double", "aromatic", "mancude"}:
        raise ValueError(f"unsupported monocyclic graph bond class {bond_class!r}")
    if type(pin) is not bool:
        raise TypeError("monocyclic graph template pin flag must be a boolean")
    locants = tuple(str(index) for index in range(1, ring_size + 1))
    edges = tuple(zip(locants, locants[1:] + locants[:1], strict=True))
    aromatic = bond_class in {"aromatic", "mancude"}
    return RetainedGraphTemplate(
        name=name,
        pin=pin,
        priority=1000,
        aliases=(),
        attached_prefix=None,
        derivative_stem=None,
        default_indicated_h=(),
        locants=locants,
        atoms=tuple(RetainedGraphAtomTemplate(locant=locant, symbol=symbol, aromatic=aromatic) for locant in locants),
        bonds=tuple(RetainedGraphBondTemplate(locants=edge, bond_class=bond_class) for edge in edges),
        rings=(locants,),
        fusion_atoms=(),
        peripheral_atoms=locants,
        interior_atoms=(),
        family="generated_monocycle",
        numbering_policy="generated_monocycle",
        charge_policy="exact",
        enabled=True,
        pre_descriptor_selection=True,
    )
