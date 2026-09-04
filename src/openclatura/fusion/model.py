"""Immutable value objects for systematic fusion nomenclature.

This module deliberately has no dependency on the naming pipeline.  It defines
the graph, locant, proof, and planning records exchanged by future fusion
algorithms.  Search code may therefore construct and audit candidates without
mutating assembly state or rendering partially verified names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias

from ..locants import SystemLocant
from ..retained_graph_model import (
    RetainedGraphAtomTemplate as ComponentAtom,
)
from ..retained_graph_model import (
    RetainedGraphBondTemplate as ComponentBond,
)
from ..retained_graph_model import (
    RetainedGraphTemplate,
)

if TYPE_CHECKING:
    from ..assembly_parts import NameTokenBinding


class FusionMode(StrEnum):
    """Policy controlling whether systematic fusion planning may be used."""

    DISABLED = "disabled"
    LEGACY = "legacy"
    AUDITED_PIN = "audited_pin"
    GENERAL = "general"


class AuditStatus(StrEnum):
    """Outcome of an independent fusion-plan audit."""

    CONFIRMED = "confirmed"
    ABSTAIN = "abstain"
    MISMATCH = "mismatch"
    ERROR = "error"


class PinStatus(StrEnum):
    """Preferred-name status carried separately from graph correctness."""

    CONFIRMED = "confirmed"
    VALID_GENERAL_NAME = "valid_general_name"
    FALLBACK_NOT_VERIFIED_AS_PIN = "fallback_not_verified_as_pin"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class FusionJoinKind(StrEnum):
    """Supported relationship between two fusion component occurrences."""

    ORTHO = "ortho"
    ORTHO_PERI = "ortho_peri"
    HIGHER_ORDER = "higher_order"


def _require_nonempty(value: str, label: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{label} must not be empty")


def _require_nonnegative(value: int, label: str) -> None:
    if value < 0:
        raise ValueError(f"{label} must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class ComponentLocant:
    """A locant in one independently numbered fusion component."""

    component_id: int
    text: str
    prime_depth: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.component_id, "component_id")
        _require_nonempty(self.text, "component locant text")
        _require_nonnegative(self.prime_depth, "prime_depth")
        if "'" in self.text or "′" in self.text:
            raise ValueError("component locant text must not contain prime marks")

    def render(self, *, unicode_primes: bool = False) -> str:
        prime = "′" if unicode_primes else "'"
        return f"{self.text}{prime * self.prime_depth}"

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True, slots=True, order=True)
class FusionSide:
    """A lettered peripheral side in one fusion component."""

    component_id: int
    letter: str
    prime_depth: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.component_id, "component_id")
        _require_nonempty(self.letter, "fusion side letter")
        _require_nonnegative(self.prime_depth, "prime_depth")
        if not self.letter.isascii() or not self.letter.isalpha() or self.letter != self.letter.lower():
            raise ValueError("fusion side must be a lowercase ASCII letter sequence")

    def render(self, *, unicode_primes: bool = False) -> str:
        prime = "′" if unicode_primes else "'"
        return f"{self.letter}{prime * self.prime_depth}"

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True, slots=True)
class OrderedFusionInterface:
    """One fully ordered, graph-derived fusion interface.

    Component-local paths, descriptor projection, and input-graph evidence are
    kept together so construction and audit cannot silently choose different
    orientations.  ``cited_attached_locants`` is separate from the complete
    attached path because some fusion classes cite only a projection of the
    graph interface.
    """

    kind: FusionJoinKind
    attached_occurrence: int
    host_occurrence: int
    attached_path: tuple[ComponentLocant, ...]
    host_path: tuple[ComponentLocant, ...]
    cited_attached_locants: tuple[ComponentLocant, ...]
    host_sides: tuple[FusionSide, ...] = ()
    host_locants: tuple[ComponentLocant, ...] = ()
    ordered_input_atoms: tuple[int, ...] = ()
    ordered_input_edges: tuple[tuple[int, int], ...] = ()
    ordered_input_bonds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.attached_occurrence == self.host_occurrence:
            raise ValueError("fusion interface must connect distinct component occurrences")
        if len(self.attached_path) < 2 or len(self.host_path) < 2:
            raise ValueError("fusion interface paths require at least one shared edge")
        if len(self.attached_path) != len(self.host_path):
            raise ValueError("attached and host interface paths must have equal lengths")
        if len(self.ordered_input_atoms) != len(self.attached_path):
            raise ValueError("ordered input atoms must correspond exactly to interface paths")
        edge_count = len(self.ordered_input_atoms) - 1
        if len(self.ordered_input_edges) != edge_count or len(self.ordered_input_bonds) != edge_count:
            raise ValueError("ordered input edges and bonds must correspond exactly to the interface path")
        if len(set(self.ordered_input_atoms)) != len(self.ordered_input_atoms):
            raise ValueError("ordered fusion interface must be a simple path")
        if len(set(self.ordered_input_edges)) != len(self.ordered_input_edges):
            raise ValueError("ordered fusion interface edges must be unique")
        if len(set(self.ordered_input_bonds)) != len(self.ordered_input_bonds):
            raise ValueError("ordered fusion interface bonds must be unique")
        expected_edges = tuple(
            tuple(sorted((left, right)))
            for left, right in zip(self.ordered_input_atoms, self.ordered_input_atoms[1:])
        )
        if self.ordered_input_edges != expected_edges:
            raise ValueError("ordered input edges must follow the ordered input atom path")
        if any(locant.component_id != self.attached_occurrence for locant in self.attached_path):
            raise ValueError("attached path locants must belong to the attached occurrence")
        if any(locant.component_id != self.host_occurrence for locant in self.host_path):
            raise ValueError("host path locants must belong to the host occurrence")
        if not self.cited_attached_locants:
            raise ValueError("fusion interface requires cited attached-component locants")
        if any(
            locant.component_id != self.attached_occurrence
            for locant in self.cited_attached_locants
        ):
            raise ValueError("cited locants must belong to the attached occurrence")
        if not set(self.cited_attached_locants) <= set(self.attached_path):
            raise ValueError("cited attached locants must be drawn from the attached interface path")
        if bool(self.host_sides) == bool(self.host_locants):
            raise ValueError("fusion interface requires exactly one host descriptor representation")
        if any(side.component_id != self.host_occurrence for side in self.host_sides):
            raise ValueError("host sides must belong to the host occurrence")
        if any(locant.component_id != self.host_occurrence for locant in self.host_locants):
            raise ValueError("host locants must belong to the host occurrence")
        if self.kind is FusionJoinKind.ORTHO:
            if len(self.ordered_input_edges) != 1 or len(self.host_sides) != 1:
                raise ValueError("ordinary ortho fusion requires exactly one shared side")
        elif self.kind is FusionJoinKind.ORTHO_PERI:
            if len(self.ordered_input_edges) < 2 or len(self.host_sides) != edge_count:
                raise ValueError("ortho-peri fusion requires two or more ordered shared sides")
        elif not self.host_locants:
            raise ValueError("higher-order fusion requires numeric host locants")

    @property
    def shared_input_atoms(self) -> frozenset[int]:
        return frozenset(self.ordered_input_atoms)

    @property
    def shared_input_edges(self) -> frozenset[tuple[int, int]]:
        return frozenset(self.ordered_input_edges)

    @property
    def shared_input_bonds(self) -> frozenset[int]:
        return frozenset(self.ordered_input_bonds)


@dataclass(frozen=True, slots=True)
class FusionComponentSpec:
    """Fusion policy layered over the shared retained graph template."""

    key: str
    parent_name: str
    attached_prefix: str
    template: RetainedGraphTemplate
    usable_as_parent: bool
    usable_as_attached: bool
    rule_reference: str
    seniority_override: int | None = None
    horizontal_ring_count: int = 0

    def __post_init__(self) -> None:
        for value, label in (
            (self.key, "component key"),
            (self.parent_name, "parent name"),
            (self.rule_reference, "rule reference"),
        ):
            _require_nonempty(value, label)
        if self.usable_as_attached:
            _require_nonempty(self.attached_prefix, "attached prefix")
        if self.horizontal_ring_count < 0:
            raise ValueError("horizontal_ring_count must be non-negative")

    @property
    def derivative_stem(self) -> str | None:
        return self.template.derivative_stem

    @property
    def locants(self) -> tuple[str, ...]:
        return self.template.locants

    @property
    def atoms(self) -> tuple[ComponentAtom, ...]:
        return self.template.atoms

    @property
    def bonds(self) -> tuple[ComponentBond, ...]:
        return self.template.bonds

    @property
    def rings(self) -> tuple[tuple[str, ...], ...]:
        return self.template.rings

    @property
    def peripheral_order(self) -> tuple[str, ...]:
        return self.template.peripheral_atoms

    @property
    def pin_component(self) -> bool:
        return self.template.pin

    @property
    def ring_sizes(self) -> tuple[int, ...]:
        return tuple(len(ring) for ring in self.template.rings)

    @property
    def fusion_carbon_locants(self) -> tuple[str, ...]:
        atoms = self.template.atom_by_locant
        return tuple(locant for locant in self.template.fusion_atoms if atoms[locant].symbol == "C")


@dataclass(frozen=True, slots=True)
class FusionComponentMatch:
    """One graph occurrence and one exact local-locant mapping of a component."""

    occurrence_id: int
    spec_key: str
    covered_face_ids: frozenset[int]
    local_to_input_atom: tuple[tuple[str, int], ...]
    local_to_skeleton_atom: tuple[tuple[str, int], ...]
    topology_key: tuple
    template_name: str = ""

    def __post_init__(self) -> None:
        _require_nonnegative(self.occurrence_id, "occurrence_id")
        _require_nonempty(self.spec_key, "component spec key")
        if not self.covered_face_ids:
            raise ValueError("component match must cover at least one face")
        input_keys = _validate_bijective_map(self.local_to_input_atom, "local-to-input atom map")
        skeleton_keys = _validate_bijective_map(self.local_to_skeleton_atom, "local-to-skeleton atom map")
        if input_keys != skeleton_keys:
            raise ValueError("component match maps must cover the same complete local-locant set")

    @property
    def input_atom_by_locant(self) -> dict[str, int]:
        return dict(self.local_to_input_atom)

    @property
    def skeleton_atom_by_locant(self) -> dict[str, int]:
        return dict(self.local_to_skeleton_atom)


@dataclass(frozen=True, slots=True)
class FusionJoin:
    """An ordered, graph-bound fusion interface between two occurrences."""

    order: int
    interface: OrderedFusionInterface

    def __post_init__(self) -> None:
        if self.order <= 0:
            raise ValueError("fusion join order must be positive")

    @property
    def attached_occurrence(self) -> int:
        return self.interface.attached_occurrence

    @property
    def host_occurrence(self) -> int:
        return self.interface.host_occurrence

    @property
    def kind(self) -> FusionJoinKind:
        return self.interface.kind

    @property
    def attached_locants(self) -> tuple[ComponentLocant, ...]:
        return self.interface.cited_attached_locants

    @property
    def host_sides(self) -> tuple[FusionSide, ...]:
        return self.interface.host_sides

    @property
    def host_locants(self) -> tuple[ComponentLocant, ...]:
        return self.interface.host_locants

    @property
    def shared_input_atoms(self) -> frozenset[int]:
        return self.interface.shared_input_atoms

    @property
    def shared_input_bonds(self) -> frozenset[int]:
        return self.interface.shared_input_bonds


@dataclass(frozen=True, slots=True)
class FusionDescriptor:
    """Context-free descriptor data projected from one fusion join."""

    attached_locants: tuple[ComponentLocant, ...]
    parent_sides: tuple[FusionSide, ...] = ()
    parent_locants: tuple[ComponentLocant, ...] = ()
    kind: FusionJoinKind = FusionJoinKind.ORTHO

    def __post_init__(self) -> None:
        if not self.attached_locants:
            raise ValueError("fusion descriptor requires attached locants")
        if bool(self.parent_sides) == bool(self.parent_locants):
            raise ValueError("fusion descriptor requires exactly one parent interface form")
        if self.kind is FusionJoinKind.HIGHER_ORDER and not self.parent_locants:
            raise ValueError("higher-order descriptor requires parent locants")
        if self.kind is not FusionJoinKind.HIGHER_ORDER and not self.parent_sides:
            raise ValueError("ordinary descriptor requires parent side letters")

    def render(self) -> str:
        attached = ",".join(str(locant) for locant in self.attached_locants)
        if self.parent_sides:
            host = "".join(str(side) for side in self.parent_sides)
            return f"[{attached}-{host}]"
        host = ",".join(str(locant) for locant in self.parent_locants)
        return f"[{attached}:{host}]"

    @classmethod
    def from_interface(cls, interface: OrderedFusionInterface) -> FusionDescriptor:
        """Project context-free descriptor data from audited interface evidence."""

        return cls(
            attached_locants=interface.cited_attached_locants,
            parent_sides=interface.host_sides,
            parent_locants=interface.host_locants,
            kind=interface.kind,
        )


@dataclass(frozen=True, slots=True)
class FusionCitationNode:
    """One occurrence in the structured fusion-name citation tree."""

    occurrence_id: int
    children: tuple[FusionCitationNode, ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative(self.occurrence_id, "citation occurrence_id")


@dataclass(frozen=True, slots=True)
class FusionMultiplicityGroup:
    """A group of nomenclaturally identical attached occurrences."""

    occurrence_ids: tuple[int, ...]
    multiplier: str

    def __post_init__(self) -> None:
        if len(self.occurrence_ids) < 2 or len(self.occurrence_ids) != len(set(self.occurrence_ids)):
            raise ValueError("multiplicity group requires at least two unique occurrences")
        _require_nonempty(self.multiplier, "fusion multiplier")


@dataclass(frozen=True, slots=True, order=True)
class ParentLocationKey:
    """Ordered P-25 quality key for one component-parent location.

    Fields are normalized so the lexicographically smallest key wins.  The
    intrinsic seniority of the component type is evaluated separately before
    this location key is used.
    """

    incomplete_system: int
    maximum_attachment_order: int
    attachment_count_by_order: tuple[int, ...]
    multiplicative_grouping_score: tuple[int, ...]
    interparent_seniority: tuple = ()
    attached_component_preference: tuple = ()

    def __post_init__(self) -> None:
        if self.incomplete_system not in (0, 1):
            raise ValueError("incomplete_system must be a normalized boolean")
        _require_nonnegative(self.maximum_attachment_order, "maximum_attachment_order")


@dataclass(frozen=True, slots=True)
class FusionNameAst:
    """Structured fusion citation; rendering must be a pure function of this AST."""

    plan_kind: str
    parent_occurrences: tuple[int, ...]
    component_occurrences: tuple[FusionComponentMatch, ...]
    joins: tuple[FusionJoin, ...]
    citation_tree: FusionCitationNode
    multiplicative_groups: tuple[FusionMultiplicityGroup, ...] = ()
    descriptors: tuple[FusionDescriptor, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.plan_kind, "fusion plan kind")
        occurrence_ids = tuple(match.occurrence_id for match in self.component_occurrences)
        occurrence_set = set(occurrence_ids)
        if not occurrence_ids or len(occurrence_ids) != len(occurrence_set):
            raise ValueError("fusion AST requires unique component occurrences")
        if not self.parent_occurrences or not set(self.parent_occurrences) <= occurrence_set:
            raise ValueError("fusion AST parent occurrences must identify declared components")
        for join in self.joins:
            if {join.attached_occurrence, join.host_occurrence} - occurrence_set:
                raise ValueError("fusion join references an unknown occurrence")
        citation_ids = _citation_occurrence_ids(self.citation_tree)
        if citation_ids != occurrence_set:
            raise ValueError("citation tree must contain every component occurrence exactly once")
        grouped_ids = [occurrence for group in self.multiplicative_groups for occurrence in group.occurrence_ids]
        if len(grouped_ids) != len(set(grouped_ids)) or not set(grouped_ids) <= occurrence_set:
            raise ValueError("multiplicative groups must reference unique declared occurrences")
        if self.descriptors and len(self.descriptors) != len(self.joins):
            raise ValueError("descriptor count must match join count when descriptors are supplied")


@dataclass(frozen=True, slots=True)
class Face:
    id: int
    atom_cycle: tuple[int, ...]
    edge_cycle: tuple[int, ...]
    size: int

    def __post_init__(self) -> None:
        _require_nonnegative(self.id, "face id")
        if self.size < 3 or len(self.atom_cycle) != self.size or len(self.edge_cycle) != self.size:
            raise ValueError("face cycles must have the declared size")
        if len(set(self.atom_cycle)) != self.size or len(set(self.edge_cycle)) != self.size:
            raise ValueError("face cycles must be simple")


@dataclass(frozen=True, slots=True)
class FaceModel:
    faces: tuple[Face, ...]
    edge_to_faces: tuple[tuple[int, tuple[int, ...]], ...]
    perimeter_edges: frozenset[int]
    fusion_edges: frozenset[int]
    outer_boundary: tuple[int, ...]
    face_adjacency: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        face_ids = [face.id for face in self.faces]
        if not face_ids or len(face_ids) != len(set(face_ids)):
            raise ValueError("face model requires unique faces")
        if self.perimeter_edges & self.fusion_edges:
            raise ValueError("perimeter and fusion edge sets must be disjoint")
        edge_ids = [edge for edge, _ in self.edge_to_faces]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge_to_faces contains duplicate edges")
        known_faces = set(face_ids)
        if any(not set(owners) <= known_faces or not 1 <= len(owners) <= 2 for _, owners in self.edge_to_faces):
            raise ValueError("each face-model edge must belong to one or two known faces")


@dataclass(frozen=True, slots=True)
class FusedLayout:
    """One completed-system layout in integer quarter-grid units."""

    face_positions: tuple[tuple[int, int, int], ...]
    orientation: int = 0
    atom_positions: tuple[tuple[int, int, int], ...] = ()
    face_shapes: tuple[tuple[int, str], ...] = ()
    orientation_score: tuple[int, ...] = ()
    audit_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        face_ids = [face for face, _, _ in self.face_positions]
        if len(face_ids) != len(set(face_ids)):
            raise ValueError("fused layout contains duplicate face positions")
        atom_ids = [atom for atom, _, _ in self.atom_positions]
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError("fused layout contains duplicate atom positions")
        shape_faces = [face for face, _ in self.face_shapes]
        if len(shape_faces) != len(set(shape_faces)):
            raise ValueError("fused layout assigns more than one shape to a face")
        if self.face_shapes and set(shape_faces) != set(face_ids):
            raise ValueError("fused layout must assign a shape to every positioned face")


@dataclass(frozen=True, slots=True)
class RejectedNumbering:
    orientation_score: tuple
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty(self.reason, "rejected-numbering reason")


@dataclass(frozen=True, slots=True)
class FusionNumberingProof:
    """Completed-system numbering and every graph-preserving input map."""

    selected_face_model: FaceModel
    selected_layout: FusedLayout
    orientation_score: tuple
    abstract_atom_to_locant: tuple[tuple[int, SystemLocant], ...]
    input_locant_maps: tuple[tuple[tuple[int, SystemLocant], ...], ...]
    rejected_numberings: tuple[RejectedNumbering, ...] = ()

    def __post_init__(self) -> None:
        abstract_atoms = _validate_bijective_map(self.abstract_atom_to_locant, "abstract atom-to-locant map")
        abstract_locants = {locant for _, locant in self.abstract_atom_to_locant}
        if not self.input_locant_maps:
            raise ValueError("numbering proof requires at least one input locant map")
        for index, locant_map in enumerate(self.input_locant_maps):
            input_atoms = _validate_bijective_map(locant_map, f"input locant map {index}")
            if len(input_atoms) != len(abstract_atoms) or {locant for _, locant in locant_map} != abstract_locants:
                raise ValueError("every input locant map must completely cover the abstract numbering")

    def string_input_locant_maps(self) -> tuple[dict[int, str], ...]:
        return tuple({atom: str(locant) for atom, locant in locant_map} for locant_map in self.input_locant_maps)


@dataclass(frozen=True, slots=True)
class FusionGraphAtom:
    id: int
    symbol: str
    formal_charge: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative(self.id, "fusion graph atom id")
        _require_nonempty(self.symbol, "fusion graph atom symbol")


@dataclass(frozen=True, slots=True)
class FusionGraphBond:
    atoms: tuple[int, int]
    bond_class: str = "mancude"

    def __post_init__(self) -> None:
        if len(self.atoms) != 2 or self.atoms[0] == self.atoms[1]:
            raise ValueError("fusion graph bond must join two distinct atoms")
        _require_nonempty(self.bond_class, "fusion graph bond class")


@dataclass(frozen=True, slots=True)
class FusionGraph:
    atoms: tuple[FusionGraphAtom, ...]
    bonds: tuple[FusionGraphBond, ...]

    def __post_init__(self) -> None:
        atom_ids = [atom.id for atom in self.atoms]
        if not atom_ids or len(atom_ids) != len(set(atom_ids)):
            raise ValueError("fusion graph requires unique atoms")
        atom_set = set(atom_ids)
        seen_edges: set[frozenset[int]] = set()
        for bond in self.bonds:
            if not set(bond.atoms) <= atom_set:
                raise ValueError("fusion graph bond references an unknown atom")
            edge = frozenset(bond.atoms)
            if edge in seen_edges:
                raise ValueError("fusion graph contains a duplicate bond")
            seen_edges.add(edge)


@dataclass(frozen=True, slots=True)
class BondAssignment:
    """One complete allowed assignment of bond orders to parent edges."""

    orders: tuple[tuple[tuple[int, int], int], ...]

    def __post_init__(self) -> None:
        edges: set[frozenset[int]] = set()
        for edge, order in self.orders:
            if len(edge) != 2 or edge[0] == edge[1] or order <= 0:
                raise ValueError("bond assignment entries require distinct endpoints and positive order")
            canonical = frozenset(edge)
            if canonical in edges:
                raise ValueError("bond assignment contains a duplicate edge")
            edges.add(canonical)


@dataclass(frozen=True, slots=True)
class ParentBondModel:
    """All bonding implied by a named parent hydride."""

    allowed_kekule_assignments: tuple[BondAssignment, ...]
    required_single_bonds: frozenset[tuple[int, int]]
    pi_eligible_edges: frozenset[tuple[int, int]]
    maximum_non_cumulative_double_bonds: int

    def __post_init__(self) -> None:
        if self.maximum_non_cumulative_double_bonds < 0:
            raise ValueError("maximum non-cumulative double-bond count must be non-negative")
        required = _validate_edges(self.required_single_bonds, "required single bonds")
        eligible = _validate_edges(self.pi_eligible_edges, "pi-eligible edges")
        if required & eligible:
            raise ValueError("required-single and pi-eligible edges must be disjoint")
        known_edges = required | eligible
        for assignment in self.allowed_kekule_assignments:
            assignment_edges = {frozenset(edge) for edge, _ in assignment.orders}
            if assignment_edges != known_edges:
                raise ValueError("each Kekule assignment must cover every parent bond exactly once")
            if any(order != 1 for edge, order in assignment.orders if frozenset(edge) in required):
                raise ValueError("required-single bonds must have order one in every assignment")
            double_count = sum(order == 2 for _, order in assignment.orders)
            if double_count > self.maximum_non_cumulative_double_bonds:
                raise ValueError("Kekule assignment exceeds the declared double-bond maximum")


@dataclass(frozen=True, slots=True)
class FusionRuleDecision:
    """One explainable rule comparison or eligibility decision."""

    rule: str
    criterion: str
    outcome: str
    reason: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.rule, "rule"),
            (self.criterion, "criterion"),
            (self.outcome, "outcome"),
            (self.reason, "reason"),
        ):
            _require_nonempty(value, label)


@dataclass(frozen=True, slots=True)
class FusionAuditResult:
    """Independent reconstruction and nomenclature audit result."""

    status: AuditStatus
    checks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is AuditStatus.CONFIRMED and self.errors:
            raise ValueError("a confirmed audit cannot contain errors")
        if self.status in {AuditStatus.MISMATCH, AuditStatus.ERROR} and not self.errors:
            raise ValueError("a failed audit must explain at least one error")

    @property
    def confirmed(self) -> bool:
        return self.status is AuditStatus.CONFIRMED


@dataclass(frozen=True, slots=True)
class FusionParentPlan:
    """A complete systematic fusion parent that passed independent audit."""

    ast: FusionNameAst
    rendered_base_name: str
    abstract_parent_graph: FusionGraph
    numbering: FusionNumberingProof
    bond_model: ParentBondModel
    indicated_hydrogens: tuple[SystemLocant, ...]
    pin_status: PinStatus | str
    rule_trace: tuple[FusionRuleDecision, ...]
    audit: FusionAuditResult
    rendered_parts: tuple[NameTokenBinding, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.rendered_base_name, "rendered fusion base name")
        if not self.audit.confirmed:
            raise ValueError("FusionParentPlan requires a confirmed independent audit")
        if len(self.indicated_hydrogens) != len(set(self.indicated_hydrogens)):
            raise ValueError("fusion indicated-hydrogen locants must be unique")
        graph_atoms = {atom.id for atom in self.abstract_parent_graph.atoms}
        numbered_atoms = {atom for atom, _ in self.numbering.abstract_atom_to_locant}
        if graph_atoms != numbered_atoms:
            raise ValueError("fusion numbering must completely cover the abstract parent graph")


TypedLocantMap: TypeAlias = tuple[tuple[int, SystemLocant], ...]


@dataclass(frozen=True, slots=True)
class FusionConfirmed:
    plan: FusionParentPlan

    def __post_init__(self) -> None:
        if not self.plan.audit.confirmed:
            raise ValueError("FusionConfirmed requires a confirmed plan")


@dataclass(frozen=True, slots=True)
class FusionNotApplicable:
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty(self.reason, "not-applicable reason")


@dataclass(frozen=True, slots=True)
class FusionUnsupported:
    reason: str
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.reason, "unsupported reason")


@dataclass(frozen=True, slots=True)
class FusionAuditFailed:
    reason: str
    candidate_summary: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.reason, "audit-failure reason")


FusionPlanningResult: TypeAlias = FusionConfirmed | FusionNotApplicable | FusionUnsupported | FusionAuditFailed


def _validate_bijective_map(entries: tuple[tuple[object, object], ...], label: str) -> set[object]:
    if not entries:
        raise ValueError(f"{label} must not be empty")
    keys = [key for key, _ in entries]
    values = [value for _, value in entries]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} contains duplicate keys")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be bijective")
    return set(keys)


def _validate_edges(edges: frozenset[tuple[int, int]], label: str) -> set[frozenset[int]]:
    canonical: set[frozenset[int]] = set()
    for edge in edges:
        if len(edge) != 2 or edge[0] == edge[1]:
            raise ValueError(f"{label} must join distinct atom ids")
        normalized = frozenset(edge)
        if normalized in canonical:
            raise ValueError(f"{label} contains duplicate undirected edges")
        canonical.add(normalized)
    return canonical


def _citation_occurrence_ids(root: FusionCitationNode) -> set[int]:
    ids: list[int] = []
    stack = [root]
    while stack:
        node = stack.pop()
        ids.append(node.occurrence_id)
        stack.extend(node.children)
    if len(ids) != len(set(ids)):
        raise ValueError("citation tree contains a component occurrence more than once")
    return set(ids)
