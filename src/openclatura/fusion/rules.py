"""Pure eligibility and component-seniority rules for fusion nomenclature."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..locants import retained_locant_sort_key
from ..rules import elements
from .config import fusion_nomenclature_config
from .model import FusionComponentMatch, FusionComponentSpec, FusionJoin, FusionMode, FusionRuleDecision

# P-25.3.2.4 gives these two criteria independent semantic identities and
# distinct element orders. Keep them separate so data changes to one rule
# cannot silently alter the other.
EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE = tuple(
    item.symbol
    for item in sorted(
        (item for item in elements.ELEMENTS.values() if item.fusion_special_priority is not None),
        key=lambda item: item.fusion_special_priority,
    )
)

GENERAL_HETEROATOM_COUNT_PRECEDENCE = tuple(
    item.symbol
    for item in sorted(
        (item for item in elements.ELEMENTS.values() if item.fusion_general_priority is not None),
        key=lambda item: item.fusion_general_priority,
    )
)

_CONFIG = fusion_nomenclature_config()
PIN_MINIMUM_LARGE_RING_SIZE = _CONFIG.rules.pin_minimum_ring_size
PIN_MINIMUM_LARGE_RING_COUNT = _CONFIG.rules.pin_minimum_ring_count


class FusionComponentLookup(Protocol):
    """Minimal registry interface needed by the pure comparison functions."""

    def get(self, key: str) -> FusionComponentSpec | None: ...

    def spec_for_match(self, match: FusionComponentMatch) -> FusionComponentSpec: ...


@dataclass(frozen=True, slots=True, order=True)
class ComponentSeniorityKey:
    """Normalized P-25.3.2.4 key; lower tuple values are preferred."""

    override: int
    earliest_special_heteroatom: int
    ring_count: int
    ring_size_vector: tuple[int, ...]
    heteroatom_count: int
    heteroatom_kind_count: int
    heteroatom_counts_by_priority: tuple[int, ...]
    horizontal_row_count: int
    all_heteroatom_locants: tuple[tuple[int, str], ...]
    per_element_locants: tuple[tuple[tuple[int, str], ...], ...]
    peripheral_fusion_carbon_locants: tuple[tuple[int, str], ...]
    deterministic_tiebreak: str

    def as_tuple(self) -> tuple:
        return (
            self.override,
            self.earliest_special_heteroatom,
            self.ring_count,
            self.ring_size_vector,
            self.heteroatom_count,
            self.heteroatom_kind_count,
            self.heteroatom_counts_by_priority,
            self.horizontal_row_count,
            self.all_heteroatom_locants,
            self.per_element_locants,
            self.peripheral_fusion_carbon_locants,
            self.deterministic_tiebreak,
        )


_CRITERION_LABELS = (
    "seniority_override",
    "earliest_special_heteroatom",
    "ring_count",
    "ring_size_vector",
    "heteroatom_count",
    "heteroatom_kind_count",
    "heteroatom_counts_by_priority",
    "horizontal_row_count",
    "all_heteroatom_locants",
    "per_element_locants",
    "peripheral_fusion_carbon_locants",
    "deterministic_tiebreak",
)


def pin_ring_size_gate(ring_sizes: tuple[int, ...]) -> bool:
    """Return whether a fused system satisfies the P-25 PIN ring-size gate."""

    return sum(size >= PIN_MINIMUM_LARGE_RING_SIZE for size in ring_sizes) >= PIN_MINIMUM_LARGE_RING_COUNT


def component_interface_orbit(
    spec: FusionComponentSpec,
    directed_path: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the canonical typed-automorphism orbit of one local path."""

    from ..retained_fused_templates import retained_graph_template_automorphisms

    variants = []
    for automorphism in retained_graph_template_automorphisms(spec.template):
        image = dict(automorphism)
        variants.extend(
            (
                tuple(image[locant] for locant in directed_path),
                tuple(image[locant] for locant in reversed(directed_path)),
            )
        )
    if not variants:
        raise ValueError(f"component template {spec.key!r} has no graph automorphism")
    return min(
        variants,
        key=lambda path: tuple(retained_locant_sort_key(locant) for locant in path),
    )


def multiplicative_attachment_key(
    spec: FusionComponentSpec,
    join: FusionJoin,
) -> tuple:
    """Identity required for attached components to share a multiplier."""

    return (
        component_variant_identity(spec),
        join.kind.value,
        join.order,
        "sides" if join.host_sides else "locants",
        component_interface_orbit(
            spec,
            tuple(locant.text for locant in join.interface.attached_path),
        ),
    )


def component_variant_identity(spec: FusionComponentSpec) -> tuple:
    """Stable typed identity of one exact component-policy/template variant."""

    template = spec.template
    return (
        spec.key,
        template.name,
        tuple(
            (atom.locant, atom.symbol, atom.charge, atom.aromatic, atom.fusion, atom.saturated, atom.interior)
            for atom in template.atoms
        ),
        tuple(sorted((tuple(sorted(bond.locants)), bond.bond_class) for bond in template.bonds)),
        tuple(sorted(tuple(sorted(ring, key=retained_locant_sort_key)) for ring in template.rings)),
        spec.multiplicative_prefix_style,
    )


def multiplicative_member_order_key(join: FusionJoin) -> tuple:
    """Canonical descriptor order within one multiplicative group."""

    return (
        tuple(tuple(ord(char) - ord("a") for char in side.letter) for side in join.host_sides),
        tuple(retained_locant_sort_key(locant.text) for locant in join.attached_locants),
    )


def fusion_mode_allows_planning(mode: FusionMode) -> bool:
    """Return whether a request mode permits invoking the new planner."""

    return mode in {FusionMode.AUDITED_PIN, FusionMode.GENERAL}


def component_seniority_key(
    component: FusionComponentMatch,
    registry: FusionComponentLookup | Mapping[str, FusionComponentSpec],
) -> ComponentSeniorityKey:
    """Return the explainable P-25.3.2.4 key for one matched occurrence."""

    return component_spec_seniority_key(_component_spec(component, registry))


def component_spec_seniority_key(spec: FusionComponentSpec) -> ComponentSeniorityKey:
    """Return the P-25.3.2.4 key for one resolved component variant.

    This avoids discarding an occurrence's exact graph-template variant by
    resolving the shared component key a second time.
    """

    heteroatoms = tuple(atom for atom in spec.atoms if atom.symbol != "C")
    counts = Counter(atom.symbol for atom in heteroatoms)
    special_ranks = [
        EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE.index(symbol)
        for symbol in counts
        if symbol in EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE
    ]
    earliest = min(special_ranks, default=len(EARLIEST_SPECIAL_HETEROATOM_PRECEDENCE))
    all_hetero_locants = tuple(sorted(retained_locant_sort_key(atom.locant) for atom in heteroatoms))
    per_element_locants = tuple(
        tuple(sorted(retained_locant_sort_key(atom.locant) for atom in heteroatoms if atom.symbol == symbol))
        for symbol in GENERAL_HETEROATOM_COUNT_PRECEDENCE
    )
    return ComponentSeniorityKey(
        override=spec.seniority_override if spec.seniority_override is not None else 1_000_000,
        earliest_special_heteroatom=earliest,
        ring_count=-len(spec.rings),
        ring_size_vector=tuple(-size for size in sorted(spec.ring_sizes, reverse=True)),
        heteroatom_count=-len(heteroatoms),
        heteroatom_kind_count=-len(counts),
        heteroatom_counts_by_priority=tuple(-counts.get(symbol, 0) for symbol in GENERAL_HETEROATOM_COUNT_PRECEDENCE),
        horizontal_row_count=-_maximum_horizontal_row_count(spec),
        all_heteroatom_locants=all_hetero_locants,
        per_element_locants=per_element_locants,
        peripheral_fusion_carbon_locants=tuple(
            sorted(retained_locant_sort_key(locant) for locant in spec.fusion_carbon_locants)
        ),
        deterministic_tiebreak=spec.key,
    )


def explain_component_comparison(
    left: FusionComponentMatch,
    right: FusionComponentMatch,
    registry: FusionComponentLookup | Mapping[str, FusionComponentSpec],
) -> FusionRuleDecision:
    """Explain the first seniority criterion that distinguishes two components."""

    left_key = component_seniority_key(left, registry)
    right_key = component_seniority_key(right, registry)
    left_values = left_key.as_tuple()
    right_values = right_key.as_tuple()
    for criterion, left_value, right_value in zip(_CRITERION_LABELS, left_values, right_values, strict=True):
        if left_value == right_value:
            continue
        winner = "left" if left_value < right_value else "right"
        return FusionRuleDecision(
            rule="P-25.3.2.4",
            criterion=criterion,
            outcome=winner,
            reason=f"{winner} component is senior by {criterion.replace('_', ' ')}.",
        )
    return FusionRuleDecision(
        rule="P-25.3.2.4",
        criterion="complete_tie",
        outcome="tie",
        reason="The component occurrences are tied by every implemented seniority criterion.",
    )


def _component_spec(
    component: FusionComponentMatch,
    registry: FusionComponentLookup | Mapping[str, FusionComponentSpec],
) -> FusionComponentSpec:
    resolver = getattr(registry, "spec_for_match", None)
    if resolver is not None:
        return resolver(component)
    spec = registry.get(component.spec_key)
    if spec is None:
        raise KeyError(f"Unknown fusion component spec: {component.spec_key}")
    return spec


def _maximum_horizontal_row_count(spec: FusionComponentSpec) -> int:
    """Return the graph-template-derived preferred horizontal ring count."""

    return spec.horizontal_ring_count
