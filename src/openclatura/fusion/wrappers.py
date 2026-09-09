"""Typed composition of fused parents with bridge and spiro nomenclature.

Ordinary fusion planning deliberately rejects bridged and spiro-only graphs.
This module keeps that boundary intact: it first proves an independently
named fused parent, then records the additional graph operation around it.
The wrapper never infers a junction locant from rendered name text.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import combinations

from ..assembly_parts import NameTokenBinding
from ..assembly_utils import needs_hyphen
from ..canonical_ranks import canonical_ranks
from ..locants import parse_system_locant, retained_locant_sort_key, system_locant_sort_key
from ..molecule import Molecule, bond_ids_within, edges_within_atoms
from ..name_operations import UnsaturationOperation
from ..polycycle_topology import connected_components, ring_system_topology
from ..retained_fused_templates import match_retained_fused_templates, retained_parent_metadata
from ..retained_name_policy import render_retained_hydrogen_state, retained_parent_output_name
from ..ring_parent import ParentHydrideKind, ParentHydrideMetadata, RingParent
from ..rules import multipliers, stems
from .composite_bridges import CompositeBridgeConstruction, composite_bridge_constructions
from .config import fusion_nomenclature_config
from .mancude import ParentDerivativeState, parent_derivative_state
from .model import FusionConfirmed, FusionMode, FusionParentPlan, ParentBondModel, PinDecision, PinStatus
from .numbering import retained_template_parent_bond_model
from .rules import fusion_ring_size_gate

_WRAPPER_SEARCH_STATES = fusion_nomenclature_config().search.component_selection_states
_COMPOSITE_BRIDGES = composite_bridge_constructions()


class WrapperParentKind(StrEnum):
    """Source of the independently proven component parent."""

    RETAINED = "retained"
    SYSTEMATIC_FUSION = "systematic_fusion"


class NondetachableBridgeKind(StrEnum):
    """Supported neutral, closed-shell nondetachable bridge classes."""

    CARBO = "carbo"
    EPOXY = "epoxy"
    EPITHIO = "epithio"
    EPIMINO = "epimino"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class WrapperParentPlan:
    """Compatibility projection over one canonical parent-hydride plan."""

    hydride: RingParent
    bond_models: tuple[ParentBondModel, ...]
    selected_locant_map: tuple[tuple[int, str], ...] | None = None
    selected_bond_model: ParentBondModel | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.atom_ids or not self.locant_maps:
            raise ValueError("wrapper parent requires a name, atoms, and locant maps")
        for entries in self.locant_maps:
            mapping = dict(entries)
            if set(mapping) != set(self.atom_ids) or len(set(mapping.values())) != len(mapping):
                raise ValueError("wrapper parent locant maps must be complete and bijective")
        if len(self.bond_models) != len(self.locant_maps):
            raise ValueError("wrapper parent requires one bond model per locant map")
        if (self.selected_locant_map is None) != (self.selected_bond_model is None):
            raise ValueError("wrapper parent locant and bond-model selections must be made together")
        if self.kind is WrapperParentKind.SYSTEMATIC_FUSION and self.fusion_plan is None:
            raise ValueError("systematic-fusion wrapper parent requires its fusion proof")

    @property
    def kind(self) -> WrapperParentKind:
        return (
            WrapperParentKind.SYSTEMATIC_FUSION
            if self.hydride.hydride_kind is ParentHydrideKind.SYSTEMATIC_FUSION
            else WrapperParentKind.RETAINED
        )

    @property
    def name(self) -> str:
        return self.hydride.base_name or ""

    @property
    def atom_ids(self) -> frozenset[int]:
        return self.hydride.atoms

    @property
    def locant_maps(self) -> tuple[tuple[tuple[int, str], ...], ...]:
        if self.selected_locant_map is not None:
            return (self.selected_locant_map,)
        return tuple(tuple(sorted(mapping.items())) for mapping in self.hydride.proof_locant_maps)

    @property
    def fusion_plan(self):
        return self.hydride.fusion_plan

    def select(self, index: int) -> WrapperParentPlan:
        """Select one aligned locant map and parent bond model."""

        entries = self.locant_maps[index]
        model = self.bond_models[index]
        hydride = self.hydride
        if self.fusion_plan is not None and self.fusion_plan.numbering_variants:
            variant = self.fusion_plan.numbering_variants[index]
            if (
                tuple(sorted(variant.numbering.string_input_locant_maps()[0].items())) != entries
                or variant.bond_model != model
            ):
                raise ValueError("wrapper numbering must select the aligned fusion proof and bond model")
            hydride = RingParent.from_fusion_plan(variant, pin_decision=self.hydride.pin_decision)
        return replace(
            self,
            hydride=replace(hydride, parent_bond_model=model),
            bond_models=(model,),
            selected_locant_map=entries,
            selected_bond_model=model,
        )


@dataclass(frozen=True, slots=True)
class NondetachableBridgeOperation:
    """One path outside a fused parent whose ends attach to that parent."""

    kind: NondetachableBridgeKind
    prefix: str
    atom_ids: tuple[int, ...]
    endpoint_atom_ids: tuple[int, int]
    endpoint_locants: tuple[str, str]
    bond_ids: frozenset[int]
    internal_bond_orders: tuple[int, ...] = ()
    unsaturation_locants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.internal_bond_orders) != max(0, len(self.atom_ids) - 1):
            raise ValueError("bridge bond orders must describe each internal path edge")
        if any(order not in {1, 2} for order in self.internal_bond_orders):
            raise ValueError("the audited bridge tier supports only single and double internal bonds")

    @property
    def rendered(self) -> str:
        return f"{','.join(self.endpoint_locants)}-{self.prefix}"


@dataclass(frozen=True, slots=True)
class BridgedFusionWrapperPlan:
    """A fused parent plus independently graph-proven bridge operations."""

    parent: WrapperParentPlan
    bridges: tuple[NondetachableBridgeOperation, ...]
    derivative_state: ParentDerivativeState
    rendered_name: str
    rendered_parts: tuple[NameTokenBinding, ...]
    audit_ok: bool
    audit_checks: tuple[str, ...] = ()
    search_states: int = 0
    bridge_unsaturation_operations: tuple[UnsaturationOperation, ...] = ()
    saturated_bridge_precursor: NondetachableBridgeOperation | None = None

    @property
    def atom_to_locant(self) -> dict[int, str]:
        """Number independent path bridges after the fused parent (P-25.4.4-5).

        Bridge prefix orientation is independent of completed-system numbering:
        the latter starts at the higher parent endpoint, highest bridges first.
        Stable sorting preserves citation order for identical endpoint pairs.
        """

        locants = dict(self.parent.locant_maps[0])
        next_locant = max(parse_system_locant(value).base for value in locants.values()) + 1
        bridges = sorted(
            self.bridges,
            key=lambda bridge: tuple(sorted(map(system_locant_sort_key, bridge.endpoint_locants), reverse=True)),
            reverse=True,
        )
        for bridge in bridges:
            path = bridge.atom_ids
            if system_locant_sort_key(bridge.endpoint_locants[0]) < system_locant_sort_key(bridge.endpoint_locants[1]):
                path = tuple(reversed(path))
            for atom in path:
                locants[atom] = str(next_locant)
                next_locant += 1
        return locants

    def __post_init__(self) -> None:
        if not self.bridges:
            raise ValueError("bridged fusion wrapper requires at least one bridge")
        if not self.audit_ok:
            raise ValueError("bridged fusion wrapper must pass its graph partition audit")
        if not self.audit_checks:
            raise ValueError("bridged fusion wrapper must record its successful audits")
        if self.search_states < 1:
            raise ValueError("bridged fusion wrapper must record bounded search effort")
        if "".join(part.text for part in self.rendered_parts) != self.rendered_name:
            raise ValueError("bridged fusion wrapper parts must reproduce the rendered name")
        if bool(self.bridge_unsaturation_operations) != (self.saturated_bridge_precursor is not None):
            raise ValueError("bridge dehydrogenation requires both a saturated precursor and operations")


@dataclass(frozen=True, slots=True)
class FusionSpiroSidePlan:
    """An audited fused component and its graph-derived spiro locant."""

    parent: WrapperParentPlan
    junction_atom_id: int
    junction_locant: str

    def __post_init__(self) -> None:
        if self.parent.kind is not WrapperParentKind.SYSTEMATIC_FUSION:
            raise ValueError("FusionSpiroSidePlan requires a systematic fusion parent")
        if self.junction_atom_id not in self.parent.atom_ids:
            raise ValueError("spiro junction lies outside the fused side parent")
        if self.junction_locant != dict(self.parent.locant_maps[0]).get(self.junction_atom_id):
            raise ValueError("spiro junction locant must come from the fusion proof map")

    def to_spiro_assembly(self, *, parent_locant: str = ""):
        """Adapt to the established spiro renderer without parsing parent text."""

        from ..spiro_assembly import SpiroAssembly

        return SpiroAssembly(
            parent_locant=parent_locant,
            side_locant=self.junction_locant,
            side_parent_name=self.parent.name,
        )


def plan_fusion_spiro_side(
    mol: Molecule,
    side_atom_ids: set[int] | frozenset[int],
    junction_atom_id: int,
    *,
    mode: FusionMode | str,
) -> FusionSpiroSidePlan | None:
    """Plan a fused spiro side only from a confirmed fusion-parent proof."""

    atoms = frozenset(side_atom_ids)
    if junction_atom_id not in atoms or any(mol.atoms[atom].charge for atom in atoms):
        return None
    parent = _systematic_fusion_parent(mol, atoms, FusionMode(mode))
    if parent is None:
        return None
    junction_locant = dict(parent.locant_maps[0]).get(junction_atom_id)
    if junction_locant is None:
        return None
    return FusionSpiroSidePlan(parent, junction_atom_id, junction_locant)


def plan_bridged_fusion_wrapper(
    mol: Molecule,
    atom_ids: set[int] | frozenset[int],
    *,
    mode: FusionMode | str,
    maximum_bridge_atoms: int | None = None,
) -> BridgedFusionWrapperPlan | None:
    """Find a retained or systematic fused parent beneath divalent path bridges.

    Candidate removal is restricted to neutral degree-two atoms.  Short
    combinations preserve support for multiple bridges, while connected path
    enumeration admits a single bridge of arbitrary configured length without
    an exponential all-subsets scan.  A candidate is accepted only when every removed
    component is a path with exactly two parent attachments and the remaining
    graph has a locant-complete retained or audited systematic-fusion parent.
    This search runs only for already bridged polycycles.
    """

    atoms = frozenset(atom_ids)
    policy = FusionMode(mode)
    if (
        policy in {FusionMode.DISABLED, FusionMode.LEGACY}
        or (maximum_bridge_atoms is not None and maximum_bridge_atoms < 1)
        or any(mol.atoms[atom].charge for atom in atoms)
    ):
        return None
    topology = ring_system_topology(mol, atoms)
    if topology.cycle_rank < 3 or not topology.bridgeheads:
        return None

    bridge_segments = _junction_path_interiors(mol, atoms, topology.internal_degrees)
    if not bridge_segments:
        return None
    visited_states = 0
    for parent_kind in (WrapperParentKind.RETAINED, WrapperParentKind.SYSTEMATIC_FUSION):
        candidates: list[tuple[tuple, BridgedFusionWrapperPlan]] = []
        candidate_groups = _bridge_removal_candidate_groups(bridge_segments, maximum_bridge_atoms)
        for count, removed_candidates in candidate_groups:
            for removed in removed_candidates:
                visited_states += 1
                if visited_states > _WRAPPER_SEARCH_STATES:
                    return None
                parent_atoms = atoms - removed
                parent = (
                    _retained_wrapper_parent(mol, parent_atoms)
                    if parent_kind is WrapperParentKind.RETAINED
                    else _systematic_fusion_parent(mol, parent_atoms, policy)
                )
                if parent is None:
                    continue
                prefer_bridge_dehydro = _prefer_completed_system_bridge_unsaturation(parent)
                bridge_paths = _bridge_path_components(mol, atoms, parent_atoms, removed)
                if bridge_paths is None:
                    continue
                for map_index, entries in enumerate(parent.locant_maps):
                    locants = dict(entries)
                    selected_parent = parent.select(map_index)
                    derivative_state = (
                        selected_parent.fusion_plan.derivative_state
                        if selected_parent.fusion_plan is not None
                        else parent_derivative_state(
                            mol,
                            parent_atoms,
                            selected_parent.selected_bond_model,
                            locants,
                            preserve_retained_parent_state=True,
                            allow_pi_redistribution=True,
                        )
                    )
                    if derivative_state is None:
                        continue
                    operations = _bridge_operations(mol, bridge_paths, parent_atoms, locants)
                    if operations is None:
                        continue
                    operations = tuple(
                        sorted(
                            operations, key=lambda operation: (operation.prefix.lstrip("("), operation.endpoint_locants)
                        )
                    )
                    bridge_unsaturation = ()
                    if prefer_bridge_dehydro:
                        bridge_unsaturation = _bridge_dehydrogenation(mol, operations, locants) or ()
                    precursor = _saturated_bridge_precursor(operations[0]) if bridge_unsaturation else None
                    audit_checks = _audit_bridge_plan(
                        mol,
                        atoms,
                        parent_atoms,
                        dict(entries),
                        operations,
                        selected_parent.selected_bond_model,
                        derivative_state,
                        bridge_unsaturation,
                        precursor,
                        fusion_plan=selected_parent.fusion_plan,
                        retained_parent_redistribution=selected_parent.fusion_plan is None,
                    )
                    if audit_checks is None:
                        continue
                    rendered_parts = _render_bridge_parts(mol, selected_parent, operations, bridge_unsaturation)
                    rendered = "".join(part.text for part in rendered_parts)
                    plan = BridgedFusionWrapperPlan(
                        selected_parent,
                        operations,
                        derivative_state,
                        rendered,
                        rendered_parts,
                        audit_ok=True,
                        audit_checks=audit_checks,
                        search_states=visited_states,
                        bridge_unsaturation_operations=bridge_unsaturation,
                        saturated_bridge_precursor=precursor,
                    )
                    rank = (
                        -len(parent_atoms),
                        tuple(
                            sorted(
                                retained_locant_sort_key(locant)
                                for operation in operations
                                for locant in operation.endpoint_locants
                            )
                        ),
                        tuple(
                            tuple(retained_locant_sort_key(locant) for locant in operation.endpoint_locants)
                            for operation in operations
                        ),
                        tuple(operation.unsaturation_locants for operation in operations),
                        tuple(
                            retained_locant_sort_key(locant)
                            for operation in derivative_state.hydro_operations
                            for locant in operation.locants
                        ),
                        tuple(operation.prefix for operation in operations),
                        map_index,
                        rendered,
                    )
                    candidates.append((rank, plan))
            if candidates:
                break
        if candidates:
            best = min(rank for rank, _ in candidates)
            tied = [plan for rank, plan in candidates if rank == best]
            if len(tied) == 1:
                return tied[0]
            # Equivalent bridge choices must not depend on input atom IDs or
            # candidate enumeration order. Reuse the graph's cached ranks.
            ranks = canonical_ranks(mol)
            return min(
                tied,
                key=lambda plan: tuple(
                    ranks[atom]
                    for atom, locant in sorted(
                        plan.atom_to_locant.items(), key=lambda item: system_locant_sort_key(item[1])
                    )
                ),
            )
    return None


def _prefer_completed_system_bridge_unsaturation(parent: WrapperParentPlan) -> bool:
    """Prefer delocalized bridge grammar for carbon-H-sensitive fusion components."""

    if parent.kind is WrapperParentKind.RETAINED:
        return False
    from .indicated_hydrogen import _component_carbon_h_locants
    from .registry import fusion_component_registry

    plan = parent.fusion_plan
    registry = fusion_component_registry()
    specs = {match.occurrence_id: registry.spec_for_match(match) for match in plan.ast.component_occurrences}
    # P-25.4.3.4.1 assigns pi bonds after bridge insertion. Localizing the
    # bridge double bonds can prevent OPSIN from resolving these components;
    # dehydrogenation lets it assign pi bonds over the completed system.
    return any(len(spec.rings) == 1 and _component_carbon_h_locants(spec) for spec in specs.values())


def _bridge_dehydrogenation(
    mol: Molecule,
    bridges: tuple[NondetachableBridgeOperation, ...],
    parent_locants: dict[int, str],
) -> tuple[UnsaturationOperation, ...] | None:
    """Express a conjugated carbon path as dehydrogenation of its saturated bridge."""

    if len(bridges) != 1:
        return None
    bridge = bridges[0]
    if bridge.kind is not NondetachableBridgeKind.CARBO or not bridge.unsaturation_locants:
        return None
    if not all(mol.atoms[atom].is_aromatic for atom in (*bridge.atom_ids, *bridge.endpoint_atom_ids)):
        return None
    if not all(mol.atoms[atom].symbol == "C" and mol.atoms[atom].is_aromatic for atom in parent_locants):
        return None
    # Bridge numbering continues the parent integers from the higher bridgehead.
    next_locant = max(retained_locant_sort_key(locant)[0] for locant in parent_locants.values()) + 1
    path = bridge.atom_ids
    if retained_locant_sort_key(bridge.endpoint_locants[0]) < retained_locant_sort_key(bridge.endpoint_locants[1]):
        path = tuple(reversed(path))
    locants = {atom: str(next_locant + index) for index, atom in enumerate(path)}
    result = []
    unsaturated_atoms = set()
    for left, right in zip(path, path[1:]):
        bond = mol.get_bond(left, right)
        if bond.order != 2:
            continue
        if unsaturated_atoms.intersection((left, right)):
            return None
        unsaturated_atoms.update((left, right))
        result.append(
            UnsaturationOperation(
                key=f"fusion:bridge:dehydro:{locants[left]},{locants[right]}",
                reason="completed-system conjugated bridge unsaturation",
                locants=(locants[left], locants[right]),
                atom_ids=(left, right),
                bond_id=bond.idx,
            )
        )
    return tuple(result)


def _saturated_bridge_precursor(bridge: NondetachableBridgeOperation) -> NondetachableBridgeOperation:
    return replace(
        bridge,
        prefix=_carbo_bridge_prefix(len(bridge.atom_ids), ()),
        internal_bond_orders=(1,) * len(bridge.internal_bond_orders),
        unsaturation_locants=(),
    )


def _bridge_removal_candidate_groups(
    bridge_segments: tuple[frozenset[int], ...],
    maximum_bridge_atoms: int | None,
):
    """Yield bounded unions of complete junction-to-junction path interiors."""

    by_size: dict[int, set[frozenset[int]]] = {}
    generated = 0
    for count in range(1, len(bridge_segments) + 1):
        for selected in combinations(bridge_segments, count):
            generated += 1
            if generated > _WRAPPER_SEARCH_STATES:
                break
            removed = frozenset().union(*selected)
            if maximum_bridge_atoms is not None and len(removed) > maximum_bridge_atoms:
                continue
            by_size.setdefault(len(removed), set()).add(removed)
        if generated > _WRAPPER_SEARCH_STATES:
            break
    for atom_count in sorted(by_size):
        yield atom_count, tuple(sorted(by_size[atom_count], key=lambda atoms: tuple(sorted(atoms))))


def _junction_path_interiors(
    mol: Molecule,
    atoms: frozenset[int],
    degrees: dict[int, int],
) -> tuple[frozenset[int], ...]:
    """Return maximal degree-two paths connecting distinct ring junctions."""

    junctions = {atom for atom, degree in degrees.items() if degree != 2}
    interiors: set[frozenset[int]] = set()
    for junction in sorted(junctions):
        for neighbor in sorted(item for item in mol.get_neighbors(junction) if item in atoms):
            if neighbor in junctions:
                continue
            path: list[int] = []
            previous, current = junction, neighbor
            while current not in junctions:
                if degrees.get(current) != 2 or current in path:
                    path = []
                    break
                path.append(current)
                following = [item for item in mol.get_neighbors(current) if item in atoms and item != previous]
                if len(following) != 1:
                    path = []
                    break
                previous, current = current, following[0]
            if path and current != junction:
                interiors.add(frozenset(path))
    return tuple(sorted(interiors, key=lambda path: (len(path), tuple(sorted(path)))))


def _retained_wrapper_parent(mol: Molecule, atoms: frozenset[int]) -> WrapperParentPlan | None:
    matches = match_retained_fused_templates(mol, set(atoms))
    if not matches:
        matches = match_retained_fused_templates(
            mol,
            set(atoms),
            allow_nonaromatic=True,
            allow_relocated_indicated_h=True,
        )
    if not matches:
        return None
    # The shared retained registry also contains monocyclic parents.  A bridge
    # wrapper requires an eligible fused base, not merely a retained match.
    matches = [match for match in matches if fusion_ring_size_gate(tuple(map(len, match.template.rings)))]
    if not matches:
        return None
    # Moving a carbon H site preserves the parent-hydride electron roles.
    # Moving it onto a heteroatom also changes donor/oxo composition and must
    # keep the established template state until that operation is proved.
    matches = [
        match
        if match.template.default_indicated_h
        and all(
            match.template.atom_by_locant[locant].symbol == "C"
            for locant in (*match.template.default_indicated_h, *match.indicated_h)
        )
        else replace(match, indicated_h=match.template.default_indicated_h)
        for match in matches
    ]
    first = matches[0]
    template_name = first.template.name
    same_parent = [
        match for match in matches if match.template.name == template_name and match.indicated_h == first.indicated_h
    ]
    candidates: dict[tuple[tuple[int, str], ...], ParentBondModel] = {}
    for match in same_parent:
        entries = tuple(sorted((atom, str(locant)) for atom, locant in match.atom_to_locant.items()))
        if entries not in candidates:
            candidates[entries] = retained_template_parent_bond_model(
                match.template, match.locant_to_atom, indicated_h=match.indicated_h
            )
    maps = tuple(candidates)
    metadata = retained_parent_metadata(template_name)
    return WrapperParentPlan(
        hydride=RingParent.from_retained_locant_maps(
            atoms=atoms,
            locant_maps=[dict(entries) for entries in maps],
            name=retained_parent_output_name(
                template_name,
                "wrapped_parent",
                default_indicated_h=first.template.default_indicated_h,
                indicated_h=first.indicated_h,
            ),
            metadata=(
                None
                if metadata is None
                else ParentHydrideMetadata(
                    default_indicated_h=first.indicated_h,
                    fusion_locants=metadata.fusion_locants,
                    derivative_stem=(
                        render_retained_hydrogen_state(
                            metadata.derivative_stem, first.template.default_indicated_h, first.indicated_h
                        )
                        if metadata.derivative_stem
                        else metadata.derivative_stem
                    ),
                    indicated_hydrogen_count=metadata.indicated_hydrogen_count,
                    mancude_double_bonds=metadata.mancude_double_bonds,
                    inherent_saturated_locants=metadata.inherent_saturated_locants
                    if first.indicated_h == first.template.default_indicated_h
                    else tuple(
                        sorted(
                            (set(metadata.inherent_saturated_locants) - set(first.template.default_indicated_h))
                            | {
                                locant
                                for locant in first.indicated_h
                                if first.template.atom_by_locant[locant].symbol == "C"
                            },
                            key=retained_locant_sort_key,
                        )
                    ),
                    relocated_indicated_h=first.indicated_h != first.template.default_indicated_h,
                )
            ),
        ),
        bond_models=tuple(candidates[entries] for entries in maps),
    )


def _systematic_fusion_parent(
    mol: Molecule,
    atoms: frozenset[int],
    mode: FusionMode,
) -> WrapperParentPlan | None:
    if mode in {FusionMode.DISABLED, FusionMode.LEGACY}:
        return None
    from .planner import plan_fusion_parent

    result = plan_fusion_parent(mol, atoms, mode=mode)
    if not isinstance(result, FusionConfirmed):
        return None
    return WrapperParentPlan(
        hydride=RingParent.from_fusion_plan(
            result.plan,
            pin_decision=PinDecision(
                PinStatus.VALID_GENERAL_NAME,
                ("fusion_rules_satisfied", *result.plan.audit.checks),
            ),
        ),
        bond_models=tuple(variant.bond_model for variant in (result.plan.numbering_variants or (result.plan,))),
    )


def _bridge_path_components(
    mol: Molecule,
    all_atoms: frozenset[int],
    parent_atoms: frozenset[int],
    removed: frozenset[int],
) -> tuple[tuple[int, ...], ...] | None:
    removed_edges = frozenset(
        edge for edge in edges_within_atoms(mol, set(all_atoms)) if edge[0] in removed and edge[1] in removed
    )
    components = connected_components(set(removed), removed_edges)
    paths = []
    for component in components:
        internal_degree = {
            atom: sum(neighbor in component for neighbor in mol.get_neighbors(atom)) for atom in component
        }
        if any(degree > 2 for degree in internal_degree.values()):
            return None
        attachments = {
            neighbor for atom in component for neighbor in mol.get_neighbors(atom) if neighbor in parent_atoms
        }
        if len(attachments) != 2:
            return None
        starts = sorted(atom for atom in component if internal_degree[atom] <= 1)
        if not starts:
            return None
        path = _walk_path(mol, component, starts[0])
        if set(path) != set(component):
            return None
        paths.append(path)
    return tuple(sorted(paths))


def _walk_path(mol: Molecule, atoms: set[int], start: int) -> tuple[int, ...]:
    path = [start]
    previous = None
    while len(path) < len(atoms):
        following = sorted(
            neighbor for neighbor in mol.get_neighbors(path[-1]) if neighbor in atoms and neighbor != previous
        )
        if not following:
            break
        previous, current = path[-1], following[0]
        path.append(current)
    return tuple(path)


def _bridge_operations(
    mol: Molecule,
    paths: tuple[tuple[int, ...], ...],
    parent_atoms: frozenset[int],
    locants: dict[int, str],
) -> tuple[NondetachableBridgeOperation, ...] | None:
    operations = []
    for path in paths:
        oriented = _orient_bridge_path(mol, path, parent_atoms, locants)
        if oriented is None:
            return None
        path, endpoints = oriented
        if any(
            mol.get_bond(path_atom, endpoint).order != 1 for path_atom, endpoint in zip((path[0], path[-1]), endpoints)
        ):
            return None
        # A path whose parent endpoints already share an edge is an annelated
        # ring component, not a nondetachable bridge over the parent.
        if mol.get_bond(*endpoints) is not None:
            return None
        bridge_class = _bridge_class(mol, path)
        if bridge_class is None:
            return None
        kind, prefix, internal_orders, unsaturation_locants = bridge_class
        endpoint_locants = tuple(locants[atom] for atom in endpoints)
        operation_atoms = set(path) | set(endpoints)
        operations.append(
            NondetachableBridgeOperation(
                kind=kind,
                prefix=prefix,
                atom_ids=path,
                endpoint_atom_ids=endpoints,
                endpoint_locants=endpoint_locants,
                bond_ids=frozenset(bond_ids_within(mol, operation_atoms) - bond_ids_within(mol, set(endpoints))),
                internal_bond_orders=internal_orders,
                unsaturation_locants=unsaturation_locants,
            )
        )
    return tuple(operations)


def _orient_bridge_path(
    mol: Molecule,
    path: tuple[int, ...],
    parent_atoms: frozenset[int],
    locants: dict[int, str],
) -> tuple[tuple[int, ...], tuple[int, int]] | None:
    """Orient a bridge from its preferred parent attachment and validate attachment topology."""

    attachments_by_atom = {
        atom: tuple(sorted(neighbor for neighbor in mol.get_neighbors(atom) if neighbor in parent_atoms))
        for atom in path
    }
    if len(path) == 1:
        endpoints = attachments_by_atom[path[0]]
        if len(endpoints) != 2:
            return None
        ordered = tuple(sorted(endpoints, key=lambda atom: retained_locant_sort_key(locants[atom])))
        return path, ordered
    if any(attachments_by_atom[atom] for atom in path[1:-1]):
        return None
    if len(attachments_by_atom[path[0]]) != 1 or len(attachments_by_atom[path[-1]]) != 1:
        return None
    forward_endpoints = (attachments_by_atom[path[0]][0], attachments_by_atom[path[-1]][0])
    reverse_path = tuple(reversed(path))
    reverse_endpoints = tuple(reversed(forward_endpoints))

    # Composite attachment order follows the senior fragment's actual prefix
    # (P-25.4.2.3 / P-25.4.3.2.2), ahead of endpoint or double-bond locants.
    for candidate_path, endpoints in ((path, forward_endpoints), (reverse_path, reverse_endpoints)):
        if _composite_bridge_construction(mol, candidate_path) is not None:
            return candidate_path, endpoints

    def orientation_key(candidate_path: tuple[int, ...], endpoints: tuple[int, int]) -> tuple:
        double_locants = tuple(
            index
            for index, (left, right) in enumerate(zip(candidate_path, candidate_path[1:]), start=1)
            if mol.get_bond(left, right).order == 2
        )
        return (
            double_locants,
            retained_locant_sort_key(locants[endpoints[0]]),
            retained_locant_sort_key(locants[endpoints[1]]),
        )

    return min(
        ((path, forward_endpoints), (reverse_path, reverse_endpoints)),
        key=lambda candidate: orientation_key(*candidate),
    )


def _composite_bridge_construction(
    mol: Molecule,
    path: tuple[int, ...],
) -> CompositeBridgeConstruction | None:
    if len(path) < 2 or mol.atoms[path[0]].symbol == "C":
        return None
    symbols = tuple(mol.atoms[atom].symbol for atom in path)
    orders = tuple(mol.get_bond(left, right).order for left, right in zip(path, path[1:]))
    construction = _COMPOSITE_BRIDGES.get((symbols, orders))
    if construction is not None and not any(mol.atoms[atom].charge for atom in path):
        return construction
    return None


def _bridge_class(
    mol: Molecule,
    path: tuple[int, ...],
) -> tuple[NondetachableBridgeKind, str, tuple[int, ...], tuple[str, ...]] | None:
    symbols = tuple(mol.atoms[atom].symbol for atom in path)
    internal_orders = tuple(mol.get_bond(left, right).order for left, right in zip(path, path[1:]))
    if all(symbol == "C" for symbol in symbols):
        double_locants = tuple(str(index) for index, order in enumerate(internal_orders, start=1) if order == 2)
        if any(order not in {1, 2} for order in internal_orders):
            return None
        prefix = _carbo_bridge_prefix(len(symbols), double_locants)
        return NondetachableBridgeKind.CARBO, prefix, internal_orders, double_locants
    composite = _composite_bridge_construction(mol, path)
    if composite is not None:
        return NondetachableBridgeKind.COMPOSITE, composite.prefix, internal_orders, composite.unsaturation_locants
    if internal_orders:
        return None
    if symbols == ("O",):
        return NondetachableBridgeKind.EPOXY, "epoxy", (), ()
    if symbols == ("S",):
        return NondetachableBridgeKind.EPITHIO, "epithio", (), ()
    if symbols == ("N",):
        return NondetachableBridgeKind.EPIMINO, "epimino", (), ()
    return None


def _carbo_bridge_prefix(length: int, double_locants: tuple[str, ...]) -> str:
    """Render one acyclic carbon bridge from its typed internal bond model."""

    stem = stems.get(length).stem
    if not double_locants:
        return f"{stem}ano"
    if length == 2 and double_locants == ("1",):
        return "etheno"
    multiplier = "" if len(double_locants) == 1 else multipliers.basic(len(double_locants))
    interfix = "a" if len(double_locants) > 1 else ""
    return f"{stem}{interfix}[{','.join(double_locants)}]{multiplier}eno"


def _render_bridge_parts(
    mol: Molecule,
    parent: WrapperParentPlan,
    operations: tuple[NondetachableBridgeOperation, ...],
    bridge_unsaturation: tuple[UnsaturationOperation, ...] = (),
) -> tuple[NameTokenBinding, ...]:
    parts: list[NameTokenBinding] = []
    if bridge_unsaturation:
        locants = tuple(locant for operation in bridge_unsaturation for locant in operation.locants)
        atoms = {atom for operation in bridge_unsaturation for atom in operation.atom_ids}
        parts.extend(
            (
                NameTokenBinding(
                    text=f"{','.join(locants)}-",
                    token_kind="locant",
                    source="fusion_wrapper_renderer",
                    grammar_role="bridge_dehydro_locants",
                    binding_key="fusion:bridge:dehydro:locants",
                    atom_ids=atoms,
                    locants=locants,
                ),
                NameTokenBinding(
                    text=f"{multipliers.basic(len(locants))}dehydro-",
                    token_kind="prefix",
                    source="fusion_wrapper_renderer",
                    grammar_role="bridge_dehydro",
                    binding_key="fusion:bridge:dehydro",
                    atom_ids=atoms,
                    bond_ids={operation.bond_id for operation in bridge_unsaturation},
                ),
            )
        )
    for index, operation in enumerate(operations):
        prefix = _carbo_bridge_prefix(len(operation.atom_ids), ()) if bridge_unsaturation else operation.prefix
        endpoint_atoms = set(operation.endpoint_atom_ids)
        parts.append(
            NameTokenBinding(
                text=f"{','.join(operation.endpoint_locants)}-",
                token_kind="locant",
                source="fusion_wrapper_renderer",
                grammar_role="nondetachable_bridge_locants",
                binding_key=f"fusion:bridge:{index}:locants",
                atom_ids=endpoint_atoms,
                locants=operation.endpoint_locants,
            )
        )
        parts.append(
            NameTokenBinding(
                text=prefix + ("-" if index + 1 < len(operations) or needs_hyphen(prefix, parent.name) else ""),
                token_kind="prefix",
                source="fusion_wrapper_renderer",
                grammar_role="nondetachable_bridge",
                binding_key=f"fusion:bridge:{index}:path",
                atom_ids=set(operation.atom_ids),
                bond_ids=set(operation.bond_ids),
            )
        )
    parts.append(
        NameTokenBinding(
            text=parent.name,
            token_kind="parent",
            source="fusion_wrapper_renderer",
            grammar_role="wrapped_fusion_parent",
            binding_key="fusion:bridge:parent",
            atom_ids=set(parent.atom_ids),
            bond_ids=bond_ids_within(mol, set(parent.atom_ids)),
        )
    )
    return tuple(parts)


def _audit_bridge_plan(
    mol: Molecule,
    all_atoms: frozenset[int],
    parent_atoms: frozenset[int],
    parent_locants: dict[int, str],
    operations: tuple[NondetachableBridgeOperation, ...],
    parent_bond_model: ParentBondModel,
    derivative_state: ParentDerivativeState,
    bridge_unsaturation: tuple[UnsaturationOperation, ...] = (),
    saturated_bridge_precursor: NondetachableBridgeOperation | None = None,
    *,
    fusion_plan: FusionParentPlan | None = None,
    retained_parent_redistribution: bool = False,
) -> tuple[str, ...] | None:
    if set(parent_locants) != set(parent_atoms) or len(set(parent_locants.values())) != len(parent_atoms):
        return None
    bridge_atoms = {atom for operation in operations for atom in operation.atom_ids}
    if (
        sum(len(operation.atom_ids) for operation in operations) != len(bridge_atoms)
        or parent_atoms & bridge_atoms
        or parent_atoms | bridge_atoms != all_atoms
    ):
        return None
    expected_edges = set(edges_within_atoms(mol, set(parent_atoms)))
    for operation in operations:
        path = operation.atom_ids
        if not path or any(mol.get_bond(left, right) is None for left, right in zip(path, path[1:])):
            return None
        attachments = {neighbor for atom in path for neighbor in mol.get_neighbors(atom) if neighbor in parent_atoms}
        if attachments != set(operation.endpoint_atom_ids):
            return None
        if operation.endpoint_locants != tuple(parent_locants[atom] for atom in operation.endpoint_atom_ids):
            return None
        classified = _bridge_class(mol, path)
        if classified is None:
            return None
        kind, prefix, internal_orders, unsaturation_locants = classified
        if (
            operation.kind is not kind
            or operation.prefix != prefix
            or operation.internal_bond_orders != internal_orders
            or operation.unsaturation_locants != unsaturation_locants
        ):
            return None
        scope = set(operation.atom_ids) | set(operation.endpoint_atom_ids)
        operation_edges = edges_within_atoms(mol, scope) - edges_within_atoms(mol, set(operation.endpoint_atom_ids))
        expected_edges.update(operation_edges)
        if operation.bond_ids != bond_ids_within(mol, scope) - bond_ids_within(mol, set(operation.endpoint_atom_ids)):
            return None
    if expected_edges != set(edges_within_atoms(mol, set(all_atoms))):
        return None
    indicated_h_atoms = frozenset()
    if fusion_plan is not None:
        if (
            fusion_plan.bond_model != parent_bond_model
            or fusion_plan.numbering.string_input_locant_maps()[0] != parent_locants
            or fusion_plan.derivative_state != derivative_state
        ):
            return None
        cited_locants = {str(locant) for locant in fusion_plan.indicated_hydrogens}
        indicated_h_atoms = frozenset(atom for atom, locant in parent_locants.items() if locant in cited_locants)
    if (
        parent_derivative_state(
            mol,
            parent_atoms,
            parent_bond_model,
            parent_locants,
            indicated_hydrogen_atom_ids=indicated_h_atoms,
            preserve_retained_parent_state=fusion_plan is None,
            allow_pi_redistribution=True if retained_parent_redistribution else None,
        )
        != derivative_state
    ):
        return None
    if bool(bridge_unsaturation) != (saturated_bridge_precursor is not None):
        return None
    if bridge_unsaturation:
        if len(operations) != 1 or saturated_bridge_precursor != _saturated_bridge_precursor(operations[0]):
            return None
        if bridge_unsaturation != _bridge_dehydrogenation(mol, operations, parent_locants):
            return None
        pi_degrees = dict.fromkeys(all_atoms, 0)
        double_edges = [edge for edge, order in derivative_state.bond_delta.assignment.orders if order == 2]
        double_edges.extend(operation.atom_ids for operation in bridge_unsaturation)
        for edge in double_edges:
            for atom in edge:
                pi_degrees[atom] += 1
        # A perfect matching reaches the carbon graph's upper bound and proves
        # the maximum noncumulative pi count without a second matching search.
        if any(degree != 1 for degree in pi_degrees.values()):
            return None
    return (
        "complete_bijective_parent_locants",
        "disjoint_parent_and_bridge_atoms",
        "simple_bridge_paths",
        "exact_bridge_endpoints_and_locants",
        "exact_bridge_bond_ownership",
        "typed_bridge_bond_and_prefix_model",
        "parent_derivative_state",
        *(("graph_bound_bridge_dehydrogenation", "complete_bridge_pi_assignment") if bridge_unsaturation else ()),
        "complete_wrapper_graph_reconstruction",
    )


__all__ = [
    "BridgedFusionWrapperPlan",
    "FusionSpiroSidePlan",
    "NondetachableBridgeKind",
    "NondetachableBridgeOperation",
    "WrapperParentKind",
    "WrapperParentPlan",
    "plan_bridged_fusion_wrapper",
    "plan_fusion_spiro_side",
]
