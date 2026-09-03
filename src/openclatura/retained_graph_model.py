"""Dependency-free value objects for graph-backed parent templates.

The retained-parent matcher and systematic-fusion planner share these records.
Keeping them outside either pipeline prevents the fusion policy layer from
depending on retained-parent assembly code and avoids parallel graph models.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    pi_capacity: int = 1
    forced_single: bool = False
    indicated_h_candidate: bool = False

    def __post_init__(self) -> None:
        if not self.locant or not self.symbol:
            raise ValueError("template atom locant and symbol must not be empty")
        if self.pi_capacity < 0:
            raise ValueError("pi_capacity must be non-negative")
        if self.forced_single and self.pi_capacity:
            raise ValueError("a forced-single atom cannot have positive pi_capacity")

    @property
    def formal_charge(self) -> int:
        """Compatibility spelling used by graph reconstruction proofs."""

        return self.charge


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
        from .rules import elements

        forced_single = frozenset(
            element.symbol for element in elements.ELEMENTS.values() if element.mancude_forced_single
        )
        return sum(
            1 for atom in self.atoms if not atom.aromatic and not atom.fusion and atom.symbol not in forced_single
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
