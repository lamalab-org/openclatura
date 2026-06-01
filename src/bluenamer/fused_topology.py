"""Audited route objects for fused and related ring-system naming.

This module is intentionally a planning/audit layer.  It classifies ring
systems and carries graph-to-locant metadata, but it does not render production
names until a route has a verified renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from .molecule import Molecule
from .naming_data import load_json_table
from .grammar_snapshot_data import RetainedFusedToken, retained_fused_token, retained_fused_token_status
from .polycycle_topology import RingSystemTopology, edges_within_atoms, ring_system_topology
from .retained_fused_templates import (
    RetainedFusedGraphTemplate,
    RetainedFusedTemplateMatch,
    match_retained_fused_templates,
    retained_fused_graph_templates,
)


HETEROATOM_SENIORITY = {
    "F": 1,
    "Cl": 2,
    "Br": 3,
    "I": 4,
    "O": 5,
    "S": 6,
    "Se": 7,
    "Te": 8,
    "N": 9,
    "P": 10,
    "As": 11,
    "Sb": 12,
    "Bi": 13,
    "Si": 14,
    "Ge": 15,
    "Sn": 16,
    "Pb": 17,
    "B": 18,
}


PARENT_COMPONENT_HETEROATOM_SENIORITY = {
    "N": 1,
    "F": 2,
    "Cl": 3,
    "Br": 4,
    "I": 5,
    "O": 6,
    "S": 7,
    "Se": 8,
    "Te": 9,
    "P": 10,
    "As": 11,
    "Sb": 12,
    "Bi": 13,
    "Si": 14,
    "Ge": 15,
    "Sn": 16,
    "Pb": 17,
    "B": 18,
}


class RingTopologyRouteKind(StrEnum):
    RETAINED_FUSED = "retained_fused"
    SYSTEMATIC_FUSED = "systematic_fused"
    BRIDGED_FUSED = "bridged_fused"
    SPIRO = "spiro"
    VON_BAEYER = "von_baeyer"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class FusedEmissionExampleSet:
    name_policy: str
    stages: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class RingTopologyRoute:
    kind: RingTopologyRouteKind
    atoms: frozenset[int]
    topology: RingSystemTopology
    retained_matches: tuple[RetainedFusedTemplateMatch, ...] = ()
    reason: str = ""
    production_ready: bool = False

    @property
    def atom_to_locant(self) -> dict[int, str] | None:
        if not self.retained_matches:
            return None
        return dict(self.retained_matches[0].atom_to_locant)


@dataclass(frozen=True)
class FusedComponentCandidate:
    name: str
    atom_to_locant: dict[int, str]
    atoms: frozenset[int]
    source: str
    priority: int
    component_locants: tuple[str, ...]
    ring_count: int
    ring_size_vector: tuple[int, ...]
    heteroatom_symbols: tuple[str, ...]
    heteroatom_count: int
    heteroatom_variety: int
    senior_heteroatom_rank: int
    senior_heteroatom_count: int
    fusion_sides: tuple[tuple[str, tuple[str, str]], ...]
    attached_prefix: str | None = None
    derivative_stem: str | None = None
    opsin_token_status: str | None = None
    indicated_h: tuple[str, ...] = ()
    production_ready: bool = False

    @property
    def seniority_key(self) -> tuple[int, int, tuple[int, ...], int, int, int, tuple[int, ...], int, str]:
        return (
            0 if self.heteroatom_count else 1,
            -self.ring_count,
            tuple(-size for size in self.ring_size_vector),
            -self.heteroatom_count,
            -self.heteroatom_variety,
            self.senior_heteroatom_rank,
            tuple(PARENT_COMPONENT_HETEROATOM_SENIORITY.get(symbol, 10_000) for symbol in self.heteroatom_symbols),
            self.priority,
            self.name,
        )


@dataclass(frozen=True)
class FusedComponentRegistryEntry:
    component_id: str
    accepted_name: str
    fusion_prefix_name: str | None
    derivative_stem: str | None
    aliases: tuple[str, ...]
    graph_template: RetainedFusedGraphTemplate
    atom_locants: tuple[str, ...]
    fusion_side_letters: tuple[tuple[str, tuple[str, str]], ...]
    ring_count: int
    ring_size_sequence: tuple[int, ...]
    heteroatom_symbols: tuple[str, ...]
    heteroatom_count: int
    heteroatom_variety: int
    senior_heteroatom_rank: int
    retained_seniority_rank: int
    is_mancude: bool
    is_retained_parent_component: bool
    is_allowed_as_fusion_component: bool
    opsin_token_status: str | None
    opsin_parseable_names: tuple[str, ...]

    @property
    def parent_component_key(self) -> tuple[int, int, tuple[int, ...], int, int, int, tuple[int, ...], int, str]:
        return (
            0 if self.heteroatom_count else 1,
            -self.ring_count,
            tuple(-size for size in self.ring_size_sequence),
            -self.heteroatom_count,
            -self.heteroatom_variety,
            self.senior_heteroatom_rank,
            tuple(PARENT_COMPONENT_HETEROATOM_SENIORITY.get(symbol, 10_000) for symbol in self.heteroatom_symbols),
            self.retained_seniority_rank,
            self.accepted_name,
        )


@dataclass(frozen=True)
class FusedComponentRegistry:
    entries: tuple[FusedComponentRegistryEntry, ...]

    @property
    def by_name(self) -> dict[str, FusedComponentRegistryEntry]:
        names: dict[str, FusedComponentRegistryEntry] = {}
        for entry in self.entries:
            names[entry.accepted_name] = entry
            for alias in entry.aliases:
                names.setdefault(alias, entry)
        return names

    def parent_component_candidates(
        self,
        *,
        include_audit_only: bool = False,
    ) -> tuple[FusedComponentRegistryEntry, ...]:
        candidates = (
            entry
            for entry in self.entries
            if include_audit_only or entry.is_allowed_as_fusion_component
        )
        return tuple(sorted(candidates, key=lambda entry: entry.parent_component_key))


@dataclass(frozen=True)
class FusedNumberingCandidate:
    atom_to_locant: dict[int, str]
    locant_to_atom: dict[str, int]
    source: str
    audit_ok: bool
    audit_errors: tuple[str, ...] = ()
    peripheral_locants: tuple[str, ...] = ()
    fusion_atom_locants: tuple[str, ...] = ()
    heteroatom_locants: tuple[str, ...] = ()
    indicated_h: tuple[str, ...] = ()
    orientation_source: str = ""


@dataclass(frozen=True)
class BridgedFusedCandidate:
    core_atoms: frozenset[int]
    bridge_atoms: frozenset[int]
    bridge_attachment_atoms: tuple[int, int]
    source_route: RingTopologyRoute
    audit_ok: bool
    audit_errors: tuple[str, ...] = ()
    bridge_attachment_locants: tuple[str, str] | None = None
    bridge_length: int = 0


@dataclass(frozen=True)
class SpiroComponentReference:
    component_name: str
    atom_to_locant: dict[int, str]
    display_atom_to_locant: dict[int, str]
    spiro_atom: int
    atoms: frozenset[int]
    source: str
    audit_ok: bool
    audit_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChargedFusedTemplateGate:
    parent_name: str
    charged_atoms: dict[int, int]
    neutral_parent_verified: bool
    production_ready: bool
    reason: str


def classify_ring_topology_route(
    mol: Molecule,
    atoms: set[int] | frozenset[int],
    *,
    include_disabled_retained: bool = False,
) -> RingTopologyRoute:
    """Classify a ring system without forcing unsupported fused names."""

    atom_set = frozenset(atoms)
    topology = ring_system_topology(mol, atom_set)
    retained_matches = tuple(
        match_retained_fused_templates(mol, atom_set, include_disabled=include_disabled_retained)
    )
    if retained_matches:
        return RingTopologyRoute(
            kind=RingTopologyRouteKind.RETAINED_FUSED,
            atoms=atom_set,
            topology=topology,
            retained_matches=retained_matches,
            reason="matched retained fused graph template",
            production_ready=all(match.template.enabled for match in retained_matches),
        )
    if topology.classification in {"monospiro", "linear_dispiro", "complex_spiro"}:
        return RingTopologyRoute(
            kind=RingTopologyRouteKind.SPIRO,
            atoms=atom_set,
            topology=topology,
            reason="spiro topology; component names must be audited separately",
        )
    if topology.fused_edges:
        return RingTopologyRoute(
            kind=RingTopologyRouteKind.SYSTEMATIC_FUSED,
            atoms=atom_set,
            topology=topology,
            reason="fused topology must not be forced through von Baeyer",
        )
    if topology.classification in {"bicyclic", "complex_polycycle"}:
        return RingTopologyRoute(
            kind=RingTopologyRouteKind.VON_BAEYER,
            atoms=atom_set,
            topology=topology,
            reason="non-fused bridged topology may use audited von Baeyer",
        )
    return RingTopologyRoute(
        kind=RingTopologyRouteKind.UNSUPPORTED,
        atoms=atom_set,
        topology=topology,
        reason="no fused/spiro/von-Baeyer route proved",
    )


def fused_component_from_retained_match(
    mol: Molecule,
    match: RetainedFusedTemplateMatch,
) -> FusedComponentCandidate:
    """Project a retained fused template match into a fused component record."""

    ring_sizes = tuple(sorted((len(ring) for ring in match.template.rings), reverse=True))
    heteroatoms = tuple(
        atom.symbol
        for atom in match.template.atoms
        if atom.symbol != "C"
    )
    heteroatom_ranks = tuple(PARENT_COMPONENT_HETEROATOM_SENIORITY.get(symbol, 10_000) for symbol in heteroatoms)
    senior_rank = min(heteroatom_ranks, default=10_000)
    return FusedComponentCandidate(
        name=match.template.name,
        atom_to_locant=dict(match.atom_to_locant),
        atoms=match.matched_atoms,
        source="retained_fused_template",
        priority=match.template.priority,
        component_locants=match.template.locants,
        ring_count=len(match.template.rings),
        ring_size_vector=ring_sizes,
        heteroatom_symbols=heteroatoms,
        heteroatom_count=len(heteroatoms),
        heteroatom_variety=len(set(heteroatoms)),
        senior_heteroatom_rank=senior_rank,
        senior_heteroatom_count=sum(1 for rank in heteroatom_ranks if rank == senior_rank),
        fusion_sides=fused_parent_side_letters(match.template),
        attached_prefix=match.template.attached_prefix,
        derivative_stem=match.template.derivative_stem,
        opsin_token_status=retained_fused_token_status(match.template.name),
        indicated_h=match.indicated_h,
        production_ready=match.template.enabled,
    )


@lru_cache(maxsize=2)
def fused_component_registry(*, include_disabled: bool = False) -> FusedComponentRegistry:
    """Return retained fused component registry entries for fused descriptors.

    This is not a production renderer.  It gathers the data needed by the
    systematic fused descriptor engine from graph templates and OPSIN-visible
    token rows, while keeping indicated-H and audit-only templates out of live
    fused-component selection.
    """

    entries = tuple(
        _fused_component_registry_entry(template)
        for template in retained_fused_graph_templates(include_disabled=include_disabled)
    )
    return FusedComponentRegistry(entries=entries)


def _fused_component_registry_entry(template: RetainedFusedGraphTemplate) -> FusedComponentRegistryEntry:
    token = retained_fused_token(template.name)
    ring_sizes = tuple(sorted((len(ring) for ring in template.rings), reverse=True))
    heteroatoms = tuple(atom.symbol for atom in template.atoms if atom.symbol != "C")
    heteroatom_ranks = tuple(PARENT_COMPONENT_HETEROATOM_SENIORITY.get(symbol, 10_000) for symbol in heteroatoms)
    token_status = token.derivative_status if token is not None else None
    return FusedComponentRegistryEntry(
        component_id=f"retained_fused:{template.name}",
        accepted_name=template.name,
        fusion_prefix_name=_fusion_prefix_name(template, token),
        derivative_stem=template.derivative_stem,
        aliases=template.aliases,
        graph_template=template,
        atom_locants=template.locants,
        fusion_side_letters=fused_parent_side_letters(template),
        ring_count=len(template.rings),
        ring_size_sequence=ring_sizes,
        heteroatom_symbols=heteroatoms,
        heteroatom_count=len(heteroatoms),
        heteroatom_variety=len(set(heteroatoms)),
        senior_heteroatom_rank=min(heteroatom_ranks, default=10_000),
        retained_seniority_rank=template.priority,
        is_mancude=template.aromatic_equivalence_policy == "neutral_kekule_equivalent",
        is_retained_parent_component=True,
        is_allowed_as_fusion_component=_is_allowed_retained_fused_component(template, token_status),
        opsin_token_status=token_status,
        opsin_parseable_names=_opsin_parseable_names(template),
    )


def _fusion_prefix_name(template: RetainedFusedGraphTemplate, token: RetainedFusedToken | None) -> str | None:
    if template.attached_prefix:
        return template.attached_prefix
    if token is not None and token.fusion_stems:
        return token.fusion_stems[0]
    return None


def _is_allowed_retained_fused_component(
    template: RetainedFusedGraphTemplate,
    token_status: str | None,
) -> bool:
    if template.default_indicated_h:
        return False
    if template.aromatic_equivalence_policy != "neutral_kekule_equivalent":
        return False
    if token_status != "production_safe":
        return False
    return template.attached_prefix is not None


def _opsin_parseable_names(template: RetainedFusedGraphTemplate) -> tuple[str, ...]:
    seen: set[str] = set()
    names: list[str] = []
    for name in (template.name, *template.aliases):
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


@lru_cache(maxsize=1)
def fused_emission_examples() -> FusedEmissionExampleSet:
    """Return generic OPSIN grammar target examples for Stage 6-10."""

    raw = load_json_table("fused_emission_examples.json")
    return FusedEmissionExampleSet(
        name_policy=str(raw["name_policy"]),
        stages={
            str(stage): tuple(str(name) for name in names)
            for stage, names in raw.get("stages", {}).items()
        },
    )


def fused_parent_side_letters(template: RetainedFusedGraphTemplate) -> tuple[tuple[str, tuple[str, str]], ...]:
    """Return parent-side letters around the declared fused component periphery.

    This is the data projection used by systematic fused descriptors:
    side ``a`` is the bond between the first and second peripheral locants,
    side ``b`` the next bond, and so on.  Sides are emitted only for actual
    template bonds so interior/fusion shortcuts cannot become descriptor sides.
    """

    peripheral = tuple(template.peripheral_atoms)
    if len(peripheral) < 2:
        return ()
    bond_keys = {tuple(sorted(bond.locants)) for bond in template.bonds}
    sides: list[tuple[str, tuple[str, str]]] = []
    letter_ord = ord("a")
    for index, first in enumerate(peripheral):
        second = peripheral[(index + 1) % len(peripheral)]
        if tuple(sorted((first, second))) not in bond_keys:
            continue
        sides.append((chr(letter_ord), (first, second)))
        letter_ord += 1
    return tuple(sides)


def fused_numbering_from_retained_match(match: RetainedFusedTemplateMatch) -> FusedNumberingCandidate:
    locant_to_atom = dict(match.locant_to_atom)
    atom_to_locant = dict(match.atom_to_locant)
    errors = []
    if len(atom_to_locant) != len(match.matched_atoms):
        errors.append("retained fused numbering does not cover every matched atom")
    if set(atom_to_locant) != set(match.matched_atoms):
        errors.append("retained fused numbering atom set differs from matched atom set")
    if len(locant_to_atom) != len(atom_to_locant):
        errors.append("retained fused numbering has duplicate locants")
    return FusedNumberingCandidate(
        atom_to_locant=atom_to_locant,
        locant_to_atom=locant_to_atom,
        source=f"retained:{match.template.name}",
        audit_ok=not errors,
        audit_errors=tuple(errors),
        peripheral_locants=match.template.peripheral_atoms,
        fusion_atom_locants=match.template.fusion_atoms,
        heteroatom_locants=tuple(atom.locant for atom in match.template.atoms if atom.symbol != "C"),
        indicated_h=match.indicated_h,
        orientation_source=match.template.numbering_policy,
    )


def bridged_fused_candidates(mol: Molecule, route: RingTopologyRoute) -> tuple[BridgedFusedCandidate, ...]:
    """Return conservative bridged-fused candidates.

    A candidate is only emitted when removing a single non-fusion bridge atom
    leaves a connected fused core.  This is a data carrier for future rendering,
    not a production name path.
    """

    if not route.topology.fused_edges or not route.topology.bridgeheads:
        return ()
    atoms = set(route.atoms)
    edges = edges_within_atoms(mol, atoms)
    locants = route.atom_to_locant
    candidates: list[BridgedFusedCandidate] = []
    for atom in atoms:
        neighbors = [neighbor for neighbor in mol.get_neighbors(atom) if neighbor in atoms]
        if len(neighbors) != 2:
            continue
        core_atoms = frozenset(atoms - {atom})
        if len(core_atoms) < 3:
            continue
        core_topology = ring_system_topology(mol, core_atoms)
        errors = []
        if not core_topology.fused_edges:
            errors.append("bridge removal does not leave a fused core")
        if not _is_connected(core_atoms, edges - {tuple(sorted((atom, neighbor))) for neighbor in neighbors}):
            errors.append("bridge removal disconnects the fused core")
        candidates.append(
            BridgedFusedCandidate(
                core_atoms=core_atoms,
                bridge_atoms=frozenset({atom}),
                bridge_attachment_atoms=tuple(sorted(neighbors)),
                source_route=route,
                audit_ok=not errors,
                audit_errors=tuple(errors),
                bridge_attachment_locants=_bridge_attachment_locants(locants, tuple(sorted(neighbors))),
                bridge_length=1,
            )
        )
    return tuple(candidate for candidate in candidates if candidate.audit_ok)


def spiro_component_reference(
    component_name: str,
    atom_to_locant: dict[int, str] | None,
    spiro_atom: int,
    atoms: set[int] | frozenset[int],
    *,
    source: str,
    prime_count: int = 0,
) -> SpiroComponentReference:
    errors = []
    atom_set = frozenset(atoms)
    if atom_to_locant is None:
        errors.append("spiro component has no atom-to-locant map")
        atom_to_locant = {}
    if spiro_atom not in atom_set:
        errors.append("spiro atom is not in component")
    if set(atom_to_locant) != set(atom_set):
        errors.append("spiro component locant map does not cover component atoms")
    if len(set(atom_to_locant.values())) != len(atom_to_locant):
        errors.append("spiro component locants are not unique")
    return SpiroComponentReference(
        component_name=component_name,
        atom_to_locant=dict(atom_to_locant),
        display_atom_to_locant={
            atom: _prime_locant(locant, prime_count) for atom, locant in atom_to_locant.items()
        },
        spiro_atom=spiro_atom,
        atoms=atom_set,
        source=source,
        audit_ok=not errors,
        audit_errors=tuple(errors),
    )


def _bridge_attachment_locants(
    atom_to_locant: dict[int, str] | None,
    attachment_atoms: tuple[int, int],
) -> tuple[str, str] | None:
    if atom_to_locant is None:
        return None
    if any(atom not in atom_to_locant for atom in attachment_atoms):
        return None
    return tuple(atom_to_locant[atom] for atom in attachment_atoms)  # type: ignore[return-value]


def _prime_locant(locant: str, prime_count: int) -> str:
    if prime_count <= 0:
        return locant
    return locant + ("'" * prime_count)


def charged_fused_template_gate(
    parent_name: str,
    mol: Molecule,
    atoms: set[int] | frozenset[int],
    *,
    neutral_parent_verified: bool,
) -> ChargedFusedTemplateGate:
    charged_atoms = {atom: mol.atoms[atom].charge for atom in atoms if mol.atoms[atom].charge}
    production_ready = neutral_parent_verified and bool(charged_atoms)
    reason = (
        "charged fused parent can be rendered only after neutral parent numbering is production-verified"
        if not production_ready
        else "charged fused parent has a verified neutral numbering base"
    )
    return ChargedFusedTemplateGate(
        parent_name=parent_name,
        charged_atoms=charged_atoms,
        neutral_parent_verified=neutral_parent_verified,
        production_ready=production_ready,
        reason=reason,
    )


def _is_connected(atoms: frozenset[int], edges: set[tuple[int, int]]) -> bool:
    if not atoms:
        return False
    start = next(iter(atoms))
    seen = {start}
    stack = [start]
    while stack:
        atom = stack.pop()
        for first, second in edges:
            if first == atom and second in atoms and second not in seen:
                seen.add(second)
                stack.append(second)
            elif second == atom and first in atoms and first not in seen:
                seen.add(first)
                stack.append(first)
    return seen == set(atoms)
